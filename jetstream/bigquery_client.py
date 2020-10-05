from datetime import datetime
import re
import time
from typing import Optional

import attr
import google.cloud.bigquery.client
import google.cloud.bigquery.dataset
import google.cloud.bigquery.job
import google.cloud.bigquery.table
from google.cloud.bigquery_storage import BigQueryReadClient


@attr.s(auto_attribs=True, slots=True)
class BigQueryClient:
    project: str
    dataset: str
    _client: Optional[google.cloud.bigquery.client.Client] = None
    _storage_client: Optional[BigQueryReadClient] = None

    @property
    def client(self):
        self._client = self._client or google.cloud.bigquery.client.Client(self.project)
        return self._client

    def table_to_dataframe(self, table: str):
        """Return all rows of the specified table as a dataframe."""
        self._storage_client = self._storage_client or BigQueryReadClient()

        table_ref = self.client.get_table(f"{self.project}.{self.dataset}.{table}")
        rows = self.client.list_rows(table_ref)
        return rows.to_dataframe(bqstorage_client=self._storage_client)

    def add_labels_to_table(self, table, labels):
        """Adds the provided labels to the table."""
        table_ref = self.client.dataset(self.dataset).table(table)
        table = self.client.get_table(table_ref)
        table.labels = labels

        self.client.update_table(table, ["labels"])

    def _current_timestamp_label(self):
        """Returns the current UTC timestamp as a valid BigQuery label."""
        return str(int(time.mktime(datetime.utcnow().timetuple())))

    def load_table_from_json(self, results, table, job_config):
        # wait for the job to complete
        destination_table = f"{self.project}.{self.dataset}.{table}"
        self.client.load_table_from_json(results, destination_table, job_config=job_config).result()

        # add a label with the current timestamp to the table
        self.add_labels_to_table(
            table,
            {"last_updated": self._current_timestamp_label()},
        )

    def execute(self, query: str, destination_table: Optional[str] = None) -> None:
        dataset = google.cloud.bigquery.dataset.DatasetReference.from_string(
            self.dataset,
            default_project=self.project,
        )
        kwargs = {}
        if destination_table:
            kwargs["destination"] = dataset.table(destination_table)
            kwargs["write_disposition"] = google.cloud.bigquery.job.WriteDisposition.WRITE_TRUNCATE
        config = google.cloud.bigquery.job.QueryJobConfig(default_dataset=dataset, **kwargs)
        job = self.client.query(query, config)
        # block on result
        job.result(max_results=1)

        if destination_table:
            # add a label with the current timestamp to the table
            self.add_labels_to_table(
                destination_table,
                {"last_updated": self._current_timestamp_label()},
            )

    def delete_tables_matching_regex(self, regex: str):
        """Delete all tables with names matching the specified pattern."""
        table_name_re = re.compile(regex)

        existing_tables = self.client.list_tables(self.dataset)
        for table in existing_tables:
            if table_name_re.match(table.table_id):
                self.client.delete_table(table.table_id, not_found_ok=True)
