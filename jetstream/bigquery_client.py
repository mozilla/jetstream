import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional

import attr
import google.cloud.bigquery
import google.cloud.bigquery.client
import google.cloud.bigquery.dataset
import google.cloud.bigquery.job
import google.cloud.bigquery.table
import numpy as np
import pandas as pd
from google.cloud.bigquery_storage import BigQueryReadClient
from metric_config_parser.metric import AnalysisPeriod
from pytz import UTC

from . import bq_normalize_name


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

    def table_to_dataframe(self, table: str, nan_columns: List[str] = []) -> pd.DataFrame:
        """Return all rows of the specified table as a dataframe."""
        self._storage_client = self._storage_client or BigQueryReadClient()

        table_ref = self.client.get_table(f"{self.project}.{self.dataset}.{table}")
        rows = self.client.list_rows(table_ref)
        df = rows.to_dataframe(bqstorage_client=self._storage_client)

        # append null columns with the provided names
        for nan_col in nan_columns:
            if nan_col not in df.columns:
                df[nan_col] = np.nan

        return df

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
    ) -> google.cloud.bigquery.job.QueryJob:
        dataset = google.cloud.bigquery.dataset.DatasetReference.from_string(
            self.dataset,
            default_project=self.project,
        )
        kwargs: Dict[str, Any] = {}
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

        return job

    def tables_matching_regex(self, regex: str):
        """Returns a list of tables with names matching the specified pattern."""
        job = self.client.query(
            rf"""
            SELECT
                table_name
            FROM
                {self.dataset}.INFORMATION_SCHEMA.TABLES
            WHERE
                REGEXP_CONTAINS(table_name, r'{regex}')
            """
        )
        result = list(job.result())
        return [row.table_name for row in result]

    def touch_tables(self, normandy_slug: str):
        """Updates the last_updated timestamp on tables for a given experiment.

        Useful to prevent tables that we _didn't_ already touch from causing an experiment to look
        perpetually stale."""
        normalized_slug = bq_normalize_name(normandy_slug)
        analysis_periods = "|".join([p.value for p in AnalysisPeriod])
        table_name_re = f"^(statistics_|enrollments_)?{normalized_slug}(_({analysis_periods})_)?.*$"
        tables = self.tables_matching_regex(table_name_re)
        timestamp = self._current_timestamp_label()
        for table in tables:
            self.add_labels_to_table(table, {"last_updated": timestamp})

    def delete_table(self, table_id: str) -> None:
        """Delete the table."""
        self.client.delete_table(table_id, not_found_ok=True)

    def delete_experiment_tables(
        self, slug: str, analysis_periods: List[AnalysisPeriod], delete_enrollments: bool = False
    ):
        """Delete all tables associated with the specified experiment slug."""
        normalized_slug = bq_normalize_name(slug)
        analysis_periods_re = "|".join([p.value for p in analysis_periods])

        existing_tables = self.tables_matching_regex(
            f"^{normalized_slug}_.+_({analysis_periods_re}).*$"
        )
        existing_tables += self.tables_matching_regex(
            f"^statistics_{normalized_slug}_({analysis_periods_re}).*$"
        )

        if delete_enrollments:
            existing_tables += self.tables_matching_regex(f"^enrollments_{normalized_slug}$")

        for existing_table in existing_tables:
            self.delete_table(f"{self.project}.{self.dataset}.{existing_table}")

    def experiment_table_first_updated(self, slug: str) -> Optional[datetime]:
        """Get the timestamp for when an experiment related table was updated last."""
        if slug is None:
            return None

        table_prefix = bq_normalize_name(slug)

        job = self.client.query(
            rf"""
            SELECT
                table_name,
                REGEXP_EXTRACT_ALL(
                    option_value,
                    '.*STRUCT\\(\"last_updated\", \"([^\"]+)\"\\).*'
                ) AS last_updated
            FROM
            {self.dataset}.INFORMATION_SCHEMA.TABLE_OPTIONS
            WHERE option_name = 'labels' AND table_name LIKE "enrollments_{table_prefix}%"
            """
        )
        result = list(job.result())

        table_first_updated = None

        for row in result:
            if not len(row.last_updated):
                continue
            updated_timestamp = UTC.localize(datetime.utcfromtimestamp(int(row.last_updated[0])))

            if table_first_updated is None or updated_timestamp < table_first_updated:
                table_first_updated = updated_timestamp

        return table_first_updated
