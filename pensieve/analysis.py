from datetime import datetime, timedelta
import re
import logging
from textwrap import dedent
from typing import Optional

import attr
import google.cloud.bigquery.client
import google.cloud.bigquery.dataset
import google.cloud.bigquery.job
import google.cloud.bigquery.table
from google.cloud import bigquery
from google.cloud.bigquery_storage_v1beta1 import BigQueryStorageClient
import mozanalysis
from mozanalysis.experiment import TimeLimits
from mozanalysis.utils import add_days

from . import AnalysisPeriod
from pensieve.config import AnalysisConfiguration


@attr.s(auto_attribs=True)
class Analysis:
    """
    Wrapper for analysing experiments.
    """

    project: str
    dataset: str
    config: AnalysisConfiguration

    def __attrs_post_init__(self):
        self.logger = logging.getLogger(__name__)

    @property
    def bigquery(self):
        return BigQueryClient(project=self.project, dataset=self.dataset)

    def _get_timelimits_if_ready(
        self, period: AnalysisPeriod, current_date: datetime
    ) -> Optional[TimeLimits]:
        """
        Returns a TimeLimits instance if experiment is due for analysis.
        Otherwise returns None.
        """
        prior_date = current_date - timedelta(days=1)
        prior_date_str = prior_date.strftime("%Y-%m-%d")
        current_date_str = current_date.strftime("%Y-%m-%d")

        dates_enrollment = self.config.experiment.proposed_enrollment + 1

        if self.config.experiment.start_date is None:
            return None

        time_limits_args = {
            "first_enrollment_date": self.config.experiment.start_date.strftime("%Y-%m-%d"),
            "num_dates_enrollment": dates_enrollment,
        }

        if period != AnalysisPeriod.OVERALL:
            try:
                current_time_limits = TimeLimits.for_ts(
                    last_date_full_data=current_date_str,
                    time_series_period=period.adjective,
                    **time_limits_args,
                )
            except ValueError:
                # There are no analysis windows yet.
                # TODO: Add a more specific check.
                return None

            try:
                prior_time_limits = TimeLimits.for_ts(
                    last_date_full_data=prior_date_str,
                    time_series_period=period.adjective,
                    **time_limits_args,
                )
            except ValueError:
                # We have an analysis window today, and we didn't yesterday,
                # so we must have just closed our first window.
                return current_time_limits

            if len(current_time_limits.analysis_windows) == len(prior_time_limits.analysis_windows):
                # No new data today
                return None

            return current_time_limits

        # Period is OVERALL
        # The last full day of data for an experiment is the day before an operator switches it off.
        tomorrow = current_date + timedelta(days=1)
        if self.config.experiment.end_date != tomorrow:
            return None

        return TimeLimits.for_single_analysis_window(
            last_date_full_data=current_date_str,
            analysis_start_days=0,
            analysis_length_dates=(
                self.config.experiment.end_date - self.config.experiment.start_date
            ).days
            - dates_enrollment
            + 1,
            **time_limits_args,
        )

    def _normalize_name(self, name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]", "_", name)

    def _table_name(self, window_period: str, window_index: int) -> str:
        assert self.config.experiment.normandy_slug is not None
        normalized_slug = self._normalize_name(self.config.experiment.normandy_slug)
        return "_".join([normalized_slug, window_period, str(window_index)])

    def _publish_view(self, window_period: AnalysisPeriod, table_prefix=None):
        assert self.config.experiment.normandy_slug is not None
        normalized_slug = self._normalize_name(self.config.experiment.normandy_slug)
        view_name = "_".join([normalized_slug, window_period.adjective])
        wildcard_expr = "_".join([normalized_slug, window_period.value, "*"])

        if table_prefix:
            normalized_prefix = self._normalize_name(table_prefix)
            view_name = "_".join([normalized_prefix, view_name])
            wildcard_expr = "_".join([normalized_prefix, wildcard_expr])

        sql = dedent(
            f"""
            CREATE OR REPLACE VIEW `{self.project}.{self.dataset}.{view_name}` AS (
                SELECT
                    *,
                    CAST(_TABLE_SUFFIX AS int64) AS window_index
                FROM `{self.project}.{self.dataset}.{wildcard_expr}`
            )
            """
        )
        self.bigquery.execute(sql)

    def _calculate_metrics(
        self,
        exp: mozanalysis.experiment.Experiment,
        time_limits: TimeLimits,
        period: AnalysisPeriod,
        dry_run: bool,
    ):
        """
        Calculate metrics for a specific experiment.
        Returns the BigQuery table results are written to.
        """

        window = len(time_limits.analysis_windows)
        last_analysis_window = time_limits.analysis_windows[-1]
        # TODO: Add this functionality to TimeLimits.
        last_window_limits = attr.evolve(
            time_limits,
            analysis_windows=[last_analysis_window],
            first_date_data_required=add_days(
                time_limits.first_enrollment_date, last_analysis_window.start
            ),
        )

        res_table_name = self._table_name(period.value, window)

        sql = exp.build_query(
            {m.metric for m in self.config.metrics[period]},
            last_window_limits,
            "normandy",
            self.config.experiment.enrollment_query,
        )

        self.logger.info("Executing query for %s (%s)", self.config.experiment.slug, period.value)
        self.bigquery.execute(sql, res_table_name)
        self._publish_view(period)

        return res_table_name

    def _calculate_statistics(self, metrics_table: str, period: AnalysisPeriod):
        """
        Run statistics on metrics.
        """

        metrics_data = self.bigquery.table_to_dataframe(metrics_table)
        destination_table = f"{self.project}.{self.dataset}.statistics_{metrics_table}"

        results = []

        for m in self.config.metrics[period]:
            results += m.run(metrics_data).to_dict()["data"]

        job_config = bigquery.LoadJobConfig()
        job_config.schema = [
            bigquery.SchemaField("metric", "STRING"),
            bigquery.SchemaField("statistic", "STRING"),
            bigquery.SchemaField("parameter", "NUMERIC"),
            bigquery.SchemaField("branch", "STRING"),
            bigquery.SchemaField("comparison_to_control", "STRING"),
            bigquery.SchemaField("ci_width", "FLOAT64"),
            bigquery.SchemaField("point", "FLOAT64"),
            bigquery.SchemaField("lower", "FLOAT64"),
            bigquery.SchemaField("upper", "FLOAT64"),
        ]
        job_config.write_disposition = bigquery.job.WriteDisposition.WRITE_TRUNCATE

        # wait for the job to complete
        self.bigquery.client.load_table_from_json(
            results, destination_table, job_config=job_config
        ).result()

        self._publish_view(period, table_prefix="statistics")

    def run(self, current_date: datetime, dry_run: bool) -> None:
        """
        Run analysis using mozanalysis for a specific experiment.
        """
        self.logger.info("Analysis.run invoked for experiment %s", self.config.experiment.slug)

        if self.config.experiment.normandy_slug is None:
            self.logger.info("Skipping %s; no normandy_slug", self.config.experiment.slug)
            return  # some experiments do not have a normandy slug

        if not self.config.experiment.proposed_enrollment:
            self.logger.info("Skipping %s; no enrollment period", self.config.experiment.slug)
            return

        if self.config.experiment.start_date is None:
            self.logger.info("Skipping %s; no start_date", self.config.experiment.slug)
            return

        for period in self.config.metrics:
            time_limits = self._get_timelimits_if_ready(period, current_date)
            if time_limits is None:
                self.logger.info(
                    "Skipping %s (%s); not ready", self.config.experiment.slug, period.value
                )
                continue

            exp = mozanalysis.experiment.Experiment(
                experiment_slug=self.config.experiment.normandy_slug,
                start_date=self.config.experiment.start_date.strftime("%Y-%m-%d"),
            )

            if dry_run:
                self.logger.info(
                    "Not executing query for %s (%s); dry run",
                    self.config.experiment.slug,
                    period.value,
                )
                continue

            metrics_table = self._calculate_metrics(exp, time_limits, period, dry_run)
            self._calculate_statistics(metrics_table, period)
            self.logger.info(
                "Finished running query for %s (%s)", self.config.experiment.slug, period.value
            )


@attr.s(auto_attribs=True, slots=True)
class BigQueryClient:
    project: str
    dataset: str
    _client: Optional[google.cloud.bigquery.client.Client] = None
    _storage_client: Optional[BigQueryStorageClient] = None

    @property
    def client(self):
        self._client = self._client or google.cloud.bigquery.client.Client(self.project)
        return self._client

    def table_to_dataframe(self, table: str):
        """Return all rows of the specified table as a dataframe."""
        self._storage_client = self._storage_client or BigQueryStorageClient()

        table_ref = self.client.get_table(f"{self.project}.{self.dataset}.{table}")
        rows = self.client.list_rows(table_ref)
        return rows.to_dataframe(bqstorage_client=self._storage_client)

    def execute(self, query: str, destination_table: Optional[str] = None) -> None:
        dataset = google.cloud.bigquery.dataset.DatasetReference.from_string(
            self.dataset, default_project=self.project,
        )
        kwargs = {}
        if destination_table:
            kwargs["destination"] = dataset.table(destination_table)
            kwargs["write_disposition"] = google.cloud.bigquery.job.WriteDisposition.WRITE_TRUNCATE
        config = google.cloud.bigquery.job.QueryJobConfig(default_dataset=dataset, **kwargs)
        job = self.client.query(query, config)
        # block on result
        job.result(max_results=1)
