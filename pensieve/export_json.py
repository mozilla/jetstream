from google.cloud import bigquery
from google.cloud import storage
import logging

logging.getLogger(__name__)


def get_statistics_tables(client, bq_dataset: str):
    """Returns statistics table names and their last modified timestamp."""
    job = client.query(
        f"""
        SELECT table_id, TIMESTAMP_MILLIS(last_modified_time) as last_modified
        FROM {bq_dataset}.__TABLES__
        WHERE table_id LIKE 'statistics_%'
    """
    )

    result = job.result()
    return {row.table_id: row.last_modified for row in result}


def get_gcs_blobs(storage_client, bucket):
    """Return all blobs in the GCS location with their last modified timestamp."""
    blobs = storage_client.list_blobs(bucket)

    return {blob.name: blob.updated for blob in blobs}


def export_table(client, project_id, dataset_id, table, bucket):
    """Export a single table or view to GCS as JSON."""
    # since views cannot get exported directly, write data into a temporary table
    job = client.query(
        f"""
        SELECT *
        FROM {dataset_id}.{table}
    """
    )

    job.result()

    # get the temporary table results are written to
    tmp_table = job._properties["configuration"]["query"]["destinationTable"]

    destination_uri = f"gs://{bucket}/{table}.json"
    dataset_ref = bigquery.DatasetReference(project_id, tmp_table["datasetId"])
    table_ref = dataset_ref.table(tmp_table["tableId"])

    logging.info(f"Export table {table} to {destination_uri}")

    extract_job = client.extract_table(table_ref, destination_uri, location="US",)
    extract_job.result()


def export_statistics_tables(project_id, dataset_id, bucket):
    """Export statistics tables that have been modified or added to GCS as JSON."""
    bigquery_client = bigquery.Client(project_id)
    storage_client = storage.Client()

    tables = get_statistics_tables(bigquery_client, dataset_id)
    exported_json = get_gcs_blobs(storage_client, bucket)

    for table, table_updated in tables.items():
        if table not in exported_json or table_updated > exported_json[table]:
            # table either new or updated since last export
            # so export new table data
            export_table(bigquery_client, project_id, dataset_id, table, bucket)
