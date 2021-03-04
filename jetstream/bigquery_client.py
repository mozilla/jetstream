import re
import time
from typing import Dict, Iterable, Mapping, Optional

import attr
import google.cloud.bigquery
import google.cloud.bigquery.client
import google.cloud.bigquery.dataset
import google.cloud.bigquery.job
import google.cloud.bigquery.table
import pandas as pd
from google.cloud.bigquery_storage import BigQueryReadClient

from . import AnalysisPeriod, bq_normalize_name


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

    def table_to_dataframe(self, table: str) -> pd.DataFrame:
        """Return all rows of the specified table as a dataframe."""
        self._storage_client = self._storage_client or BigQueryReadClient()

        table_ref = self.client.get_table(f"{self.project}.{self.dataset}.{table}")
        rows = self.client.list_rows(table_ref)
        return rows.to_dataframe(bqstorage_client=self._storage_client)

    def add_labels_to_table(self, table_name: str, labels: Mapping[str, str]) -> None:
        """Adds the provided labels to the table."""
        table_ref = self.client.dataset(self.dataset).table(table_name)
        table = self.client.get_table(table_ref)
        table.labels = labels

        self.client.update_table(table, ["labels"])

    def _current_timestamp_label(self) -> str:
        """Returns the current UTC timestamp as a valid BigQuery label."""
        return str(int(time.time()))

    def load_table_from_json(
        self, results: Iterable[Dict], table: str, job_config: google.cloud.bigquery.LoadJobConfig
    ):
        # wait for the job to complete
        destination_table = f"{self.project}.{self.dataset}.{table}"
        self.client.load_table_from_json(results, destination_table, job_config=job_config).result()

        # add a label with the current timestamp to the table
        self.add_labels_to_table(
            table,
            {"last_updated": self._current_timestamp_label()},
        )

    def execute(
        self,
        query: str,
        destination_table: Optional[str] = None,
        write_disposition: Optional[google.cloud.bigquery.job.WriteDisposition] = None,
    ) -> None:
        dataset = google.cloud.bigquery.dataset.DatasetReference.from_string(
            self.dataset,
            default_project=self.project,
        )
        kwargs = {}
        if destination_table:
            kwargs["destination"] = dataset.table(destination_table)
            kwargs["write_disposition"] = google.cloud.bigquery.job.WriteDisposition.WRITE_TRUNCATE

        if write_disposition:
            kwargs["write_disposition"] = write_disposition

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

    def tables_matching_regex(self, regex: str):
        """Returns a list of tables with names matching the specified pattern."""
        table_name_re = re.compile(regex)
        existing_tables = self.client.list_tables(self.dataset)
        return [table.table_id for table in existing_tables if table_name_re.match(table.table_id)]

    def touch_tables(self, normandy_slug: str):
        """Updates the last_updated timestamp on tables for a given experiment.

        Useful to prevent tables that we _didn't_ already touch from causing an experiment to look
        perpetually stale."""
        normalized_slug = bq_normalize_name(normandy_slug)
        analysis_periods = "|".join([p.value for p in AnalysisPeriod])
        table_name_re = f"^(statistics_)?{normalized_slug}_({analysis_periods})_.*$"
        tables = self.tables_matching_regex(table_name_re)
        timestamp = self._current_timestamp_label()
        for table in tables:
            self.add_labels_to_table(table, {"last_updated": timestamp})

    def delete_table(self, table_id: str) -> None:
        """Delete the table."""
        self.client.delete_table(table_id, not_found_ok=True)
