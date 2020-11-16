import logging
from datetime import datetime
from typing import Dict, Optional

import smart_open
from google.cloud import bigquery, storage

from jetstream import AnalysisPeriod, bq_normalize_name

logger = logging.getLogger(__name__)


def _get_statistics_tables_last_modified(
    client: bigquery.Client, bq_dataset: str, experiment_slug: Optional[str]
) -> Dict[str, datetime]:
    """Returns statistics table names and their last modified timestamp as datetime object."""
    experiment_table = "%"
    if experiment_slug:
        experiment_table = bq_normalize_name(experiment_slug)

    periods = [f"'statistics_{experiment_table}_{p.adjective}'" for p in AnalysisPeriod]
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


def _get_gcs_blobs(storage_client: storage.Client, bucket: str) -> Dict[str, datetime]:
    """Return all blobs in the GCS location with their last modified timestamp."""
    blobs = storage_client.list_blobs(bucket)

    return {blob.name.replace(".json", ""): blob.updated for blob in blobs}


def _export_table(
    client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    table: str,
    bucket: str,
    storage_client: storage.Client,
):
    """Export a single table or view to GCS as JSON."""
    # since views cannot get exported directly, write data into a temporary table
    job = client.query(
        f"""
        SELECT *
        FROM {dataset_id}.{table}
    """
    )

    job.result()

    destination_uri = f"gs://{bucket}/{table}.ndjson"
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
    _convert_ndjson_to_json(bucket, table, storage_client)


def _convert_ndjson_to_json(bucket_name: str, table: str, storage_client: storage.Client):
    """Converts the provided ndjson file on GCS to json."""
    ndjson_blob_path = f"gs://{bucket_name}/{table}.ndjson"
    json_blob_path = f"gs://{bucket_name}/{table}.json"

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
    logger.info(f"Remove file {table}.ndjson")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{table}.ndjson")
    blob.delete()


def export_statistics_tables(
    project_id: str, dataset_id: str, bucket: str, experiment_slug: Optional[str] = None
):
    """Export statistics tables that have been modified or added to GCS as JSON."""
    bigquery_client = bigquery.Client(project_id)
    storage_client = storage.Client()

    tables = _get_statistics_tables_last_modified(bigquery_client, dataset_id, experiment_slug)
    exported_json = _get_gcs_blobs(storage_client, bucket)

    for table, table_updated in tables.items():
        if table not in exported_json or table_updated > exported_json[table]:
            # table either new or updated since last export
            # so export new table data
            _export_table(bigquery_client, project_id, dataset_id, table, bucket, storage_client)
