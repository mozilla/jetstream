import logging
import random
import string
from datetime import datetime, timedelta
from typing import Optional

import google.cloud.bigquery as bigquery
import google.cloud.storage as storage
import smart_open
from google.cloud.exceptions import BadRequest
from metric_config_parser.metric import AnalysisPeriod
from mozilla_nimbus_schemas.jetstream import AnalysisErrors

from jetstream import bq_normalize_name
from jetstream.logging import LogConfiguration

logger = logging.getLogger(__name__)

EXPERIMENT_LOG_PATH = "errors"
SKIP_ERROR_TYPES = ["EndedException", "EnrollmentNotCompleteException"]


def _get_statistics_tables_last_modified(
    client: bigquery.Client, bq_dataset: str, experiment_slug: Optional[str]
) -> dict[str, datetime]:
    """Returns statistics table names and their last modified timestamp as datetime object."""
    experiment_table = "%"
    if experiment_slug:
        experiment_table = bq_normalize_name(experiment_slug)

    periods = [f"'statistics_{experiment_table}_{p.table_suffix}'" for p in AnalysisPeriod]
    expression = " OR table_id LIKE ".join(periods)

    job = client.query(
        f"""
        SELECT table_id, TIMESTAMP_MILLIS(last_modified_time) as last_modified
        FROM {bq_dataset}.__TABLES__
        WHERE table_id LIKE {expression}
    """
    )

    result = job.result()
    return {row.table_id: row.last_modified for row in result}


def _get_gcs_blobs(
    storage_client: storage.Client, bucket: str, target_path: str
) -> dict[str, datetime]:
    """Return all blobs in the GCS location with their last modified timestamp."""
    blobs = storage_client.list_blobs(bucket, prefix=target_path + "/")

    return {blob.name.replace(".json", ""): blob.updated for blob in blobs}


def _export_table(
    client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    table: str,
    bucket: str,
    target_path: str,
    storage_client: storage.Client,
):
    """Export a single table or view to GCS as JSON."""
    try:
        # since views cannot get exported directly, write data into a temporary table
        job = client.query(
            f"""
            SELECT *
            FROM {dataset_id}.{table}
        """
        )

        job.result()
    except BadRequest as e:
        if "does not match any table" in e.message:
            logger.error(
                f"google.cloud.exceptions.BadRequest: {e.args[0]}. Skipping query and export..."
            )
            return
        else:
            raise e

    # add a random string to the identifier to prevent collision errors if there
    # happen to be multiple instances running that export data for the same experiment
    tmp = "".join(random.choices(string.ascii_lowercase, k=8))
    destination_uri = f"gs://{bucket}/{target_path}/{table}-{tmp}.ndjson"
    dataset_ref = bigquery.DatasetReference(project_id, job.destination.dataset_id)
    table_ref = dataset_ref.table(job.destination.table_id)

    logger.info(f"Export table {table} to {destination_uri}")

    job_config = bigquery.ExtractJobConfig()
    job_config.destination_format = "NEWLINE_DELIMITED_JSON"
    extract_job = client.extract_table(
        table_ref, destination_uri, location="US", job_config=job_config
    )
    extract_job.result()

    # convert ndjson to json
    _convert_ndjson_to_json(bucket, target_path, table, storage_client, tmp)


def _convert_ndjson_to_json(
    bucket_name: str, target_path: str, table: str, storage_client: storage.Client, tmp: str
):
    """Converts the provided ndjson file on GCS to json."""
    ndjson_blob_path = f"gs://{bucket_name}/{target_path}/{table}-{tmp}.ndjson"
    json_blob_path = f"gs://{bucket_name}/{target_path}/{table}-{tmp}.json"

    logger.info(f"Convert {ndjson_blob_path} to {json_blob_path}")

    # stream from GCS
    with smart_open.open(ndjson_blob_path) as fin:
        first_line = True

        with smart_open.open(json_blob_path, "w") as fout:
            fout.write("[")

            for line in fin:
                if not first_line:
                    fout.write(",")

                fout.write(line.replace("\n", ""))
                first_line = False

            fout.write("]")
            fout.close()
            fin.close()

    # delete ndjson file from bucket
    logger.info(f"Remove file {table}-{tmp}.ndjson")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{target_path}/{table}-{tmp}.ndjson")
    blob.delete()
    logger.info(f"Rename file {table}-{tmp}.json to {table}.json")
    bucket.rename_blob(
        bucket.blob(f"{target_path}/{table}-{tmp}.json"), f"{target_path}/{table}.json"
    )


def export_statistics_tables(
    project_id: str, dataset_id: str, bucket: str, experiment_slug: Optional[str] = None
):
    """Export statistics tables that have been modified or added to GCS as JSON."""
    bigquery_client = bigquery.Client(project_id)
    storage_client = storage.Client()
    target_path = "statistics"

    tables = _get_statistics_tables_last_modified(bigquery_client, dataset_id, experiment_slug)
    exported_json = _get_gcs_blobs(storage_client, bucket, target_path)

    for table, table_updated in tables.items():
        if table not in exported_json or table_updated > exported_json[table]:
            # table either new or updated since last export
            # so export new table data
            _export_table(
                bigquery_client, project_id, dataset_id, table, bucket, target_path, storage_client
            )


def _get_experiment_logs_as_json(
    client: bigquery.Client,
    dataset: str,
    table: str,
    experiment_slug: str,
    min_timestamp: datetime = None,
):
    """Retrieve records from a single table as JSON."""

    query_text = f"""
        SELECT *
        FROM {dataset}.{table}
        WHERE experiment = '{experiment_slug}'
    """
    if min_timestamp is not None:
        floored_timestamp = min_timestamp.replace(second=0, microsecond=0)
        query_text += f" AND timestamp >= TIMESTAMP('{floored_timestamp}')"

    for exception_type in SKIP_ERROR_TYPES:
        query_text += f" AND exception_type != '{exception_type}'"

    query_text += " ORDER BY timestamp ASC"

    results = client.query(query_text).result()

    # convert results to JSON
    records = [dict(row) for row in results]
    records_json = AnalysisErrors.parse_obj(records).json()

    return records_json, len(records)


def _upload_str_to_gcs(
    project_id: str,
    bucket_name: str,
    experiment_slug: str,
    base_name: str,
    str_to_upload: str,
):
    storage_client = storage.Client(project_id)
    bucket = storage_client.get_bucket(bucket_name)
    target_file = f"{base_name}_{bq_normalize_name(experiment_slug)}"
    target_path = base_name
    blob = bucket.blob(f"{target_path}/{target_file}.json")

    logger.info(f"Uploading {target_file} to {bucket_name}/{target_path}")

    blob.upload_from_string(
        data=str_to_upload,
        content_type="application/json",
    )


def export_experiment_logs(
    project_id: str,
    bucket_name: str,
    experiment_slug: str,
    log_project: str,
    log_dataset: str,
    log_table: str = "logs",
    analysis_start_time: datetime = None,
    enrollment_end: datetime = None,
    log_config: Optional[LogConfiguration] = None,
):
    """Export experiment logs to GCS."""

    if log_config is not None and log_config.log_to_bigquery:
        # explicitly flush the logs to bigquery so we know they will be available to the query
        logger.info("Flushing logs to BigQuery...")
        for handler in logger.root.handlers:
            try:
                handler.flush()
            except AttributeError:
                # ignore if log handler does not have 'flush'
                pass

    logger.info(f"Retrieving logs from BigQuery: {log_project}.{log_dataset}.{log_table}")

    bq_log_client = bigquery.Client(log_project)

    # Get errors before the last analysis run but still in the current weekly analysis
    weekly_analysis_start = analysis_start_time
    if analysis_start_time is not None and enrollment_end is not None:
        weekly_start_temp = enrollment_end + timedelta(days=1)
        while weekly_start_temp < analysis_start_time:
            weekly_analysis_start = weekly_start_temp
            weekly_start_temp += timedelta(weeks=1)

    experiment_logs, num_logs = _get_experiment_logs_as_json(
        bq_log_client, log_dataset, log_table, experiment_slug, weekly_analysis_start
    )

    log_text = f"Got {num_logs} logs for experiment {experiment_slug}"
    log_text += (
        f" (newer than {weekly_analysis_start})" if weekly_analysis_start is not None else ""
    )
    logger.info(log_text)

    if experiment_logs is not None:
        _upload_str_to_gcs(
            project_id, bucket_name, experiment_slug, EXPERIMENT_LOG_PATH, experiment_logs
        )
