from datetime import datetime, timedelta
import re
from textwrap import dedent
import time
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
from jetstream.config import AnalysisConfiguration
from jetstream.dryrun import dry_run_query
from jetstream.statistics import Count, StatisticResult, StatisticResultCollection
from jetstream.logging import logger


class NoSlugException(Exception):
    """Experiment has no slug."""

    pass


class NoEnrollmentPeriodException(Exception):
    """Experiment has no enrollment period."""

    pass


class NoStartDateException(Exception):
    """Experiment has no start date."""

    pass


class EndedException(Exception):
    """Experiment has already ended."""

    pass


class EnrollmentLongerThanAnalysisException(Exception):
    """Enrollment period is longer than analysis dates"""

    pass


@attr.s(auto_attribs=True)
class Analysis:
    """
    Wrapper for analysing experiments.
    """

    project: str
    dataset: str
    config: AnalysisConfiguration

    def __attrs_post_init__(self):
        self.logger = logger

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

        assert period == AnalysisPeriod.OVERALL
        if (
            self.config.experiment.end_date != current_date
            or self.config.experiment.status != "Complete"
        ):
            return None

        analysis_length_dates = (
            (self.config.experiment.end_date - self.config.experiment.start_date).days
            - dates_enrollment
            + 1
        )

        if analysis_length_dates < 0:
            raise EnrollmentLongerThanAnalysisException(
                "Proposed enrollment longer than analysis dates length:"
                + f"{self.config.experiment.normandy_slug}"
            )

        return TimeLimits.for_single_analysis_window(
            last_date_full_data=prior_date_str,
            analysis_start_days=0,
            analysis_length_dates=analysis_length_dates,
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
            self.config.experiment.segments,
        )

        if dry_run:
            self.logger.info(
                "Dry run; not actually calculating %s metrics for %s",
                period.value,
                self.config.experiment.normandy_slug,
            )
        else:
            self.logger.info(
                "Executing query for %s (%s)",
                self.config.experiment.normandy_slug,
                period.value,
            )
            self.bigquery.execute(sql, res_table_name)
            self._publish_view(period)

        return res_table_name

    def _calculate_statistics(self, metrics_table: str, period: AnalysisPeriod):
        """
        Run statistics on metrics.
        """

        metrics_data = self.bigquery.table_to_dataframe(metrics_table)

        results = []

        reference_branch = self.config.experiment.reference_branch
        segment_labels = ["all"] + [s.name for s in self.config.experiment.segments]
        for segment in segment_labels:
            if segment != "all":
                if segment not in metrics_data.columns:
                    self.logger.error(
                        "Segment %s not in metrics table (%s)",
                        segment,
                        self.config.experiment.normandy_slug,
                    )
                    continue
                segment_data = metrics_data[metrics_data[segment]]
            else:
                segment_data = metrics_data
            for m in self.config.metrics[period]:
                stats = m.run(segment_data, reference_branch).set_segment(segment)
                results += stats.to_dict()["data"]

            counts = (
                Count().transform(segment_data, "*", "*").set_segment(segment).to_dict()["data"]
            )
            results += counts

            # add count=0 row to statistics table for missing branches
            missing_counts = StatisticResultCollection(
                [
                    StatisticResult(
                        metric="identity",
                        statistic="count",
                        parameter=None,
                        branch=b.slug,
                        comparison=None,
                        comparison_to_branch=None,
                        ci_width=None,
                        point=0,
                        lower=None,
                        upper=None,
                        segment=segment,
                    )
                    for b in self.config.experiment.branches
                    if b.slug not in {c["branch"] for c in counts}
                ]
            )

            results += missing_counts.to_dict()["data"]

        job_config = bigquery.LoadJobConfig()
        job_config.schema = StatisticResult.bq_schema
        job_config.write_disposition = bigquery.job.WriteDisposition.WRITE_TRUNCATE

        # wait for the job to complete
        self.bigquery.load_table_from_json(
            results, f"statistics_{metrics_table}", job_config=job_config
        )

        self._publish_view(period, table_prefix="statistics")

    def is_runnable(self, current_date: Optional[datetime] = None) -> bool:
        if self.config.experiment.normandy_slug is None:
            # some experiments do not have a normandy slug
            raise NoSlugException("Skipping %s; no slug", self.config.experiment.normandy_slug)

        if not self.config.experiment.proposed_enrollment:
            raise NoEnrollmentPeriodException(
                f"Skipping {self.config.experiment.normandy_slug}; no enrollment period"
            )

        if self.config.experiment.start_date is None:
            raise NoStartDateException(
                f"Skipping {self.config.experiment.normandy_slug}; no start_date"
            )

        if (
            current_date
            and self.config.experiment.end_date
            and self.config.experiment.end_date < current_date
        ):
            self.logger.info("Skipping %s; already ended", self.config.experiment.slug)
            return False

        return True

    def validate(self) -> None:
        if not self.is_runnable():
            raise Exception("Cannot validate experiment")

        dates_enrollment = self.config.experiment.proposed_enrollment + 1

        if self.config.experiment.end_date is not None:
            end_date = self.config.experiment.end_date
            analysis_length_dates = (
                (end_date - self.config.experiment.start_date).days - dates_enrollment + 1
            )
        else:
            analysis_length_dates = 21  # arbitrary
            end_date = self.config.experiment.start_date + timedelta(
                days=analysis_length_dates + dates_enrollment - 1
            )

        if analysis_length_dates < 0:
            logging.error(
                "Proposed enrollment longer than analysis dates length:"
                + f"{self.config.experiment.normandy_slug}"
            )
            raise Exception("Cannot validate experiment")

        limits = TimeLimits.for_single_analysis_window(
            last_date_full_data=end_date.strftime("%Y-%m-%d"),
            analysis_start_days=0,
            analysis_length_dates=analysis_length_dates,
            first_enrollment_date=self.config.experiment.start_date.strftime("%Y-%m-%d"),
            num_dates_enrollment=dates_enrollment,
        )

        exp = mozanalysis.experiment.Experiment(
            experiment_slug=self.config.experiment.normandy_slug,
            start_date=self.config.experiment.start_date.strftime("%Y-%m-%d"),
        )

        metrics = set()
        for v in self.config.metrics.values():
            metrics |= {m.metric for m in v}

        sql = exp.build_query(
            metrics,
            limits,
            "normandy",
            self.config.experiment.enrollment_query,
        )

        dry_run_query(sql)

    def run(self, current_date: datetime, dry_run: bool = False) -> None:
        """
        Run analysis using mozanalysis for a specific experiment.
        """
        self.logger.info(
            "Analysis.run invoked for experiment %s", self.config.experiment.normandy_slug
        )

        if not self.is_runnable(current_date):
            return

        for period in self.config.metrics:
            time_limits = self._get_timelimits_if_ready(period, current_date)

            if time_limits is None:
                self.logger.info(
                    "Skipping %s (%s); not ready",
                    self.config.experiment.normandy_slug,
                    period.value,
                )
                continue

            exp = mozanalysis.experiment.Experiment(
                experiment_slug=self.config.experiment.normandy_slug,
                start_date=self.config.experiment.start_date.strftime("%Y-%m-%d"),
            )

            metrics_table = self._calculate_metrics(exp, time_limits, period, dry_run)

            if dry_run:
                self.logger.info(
                    "Not calculating statistics %s (%s); dry run",
                    self.config.experiment.normandy_slug,
                    period.value,
                )
                continue

            self._calculate_statistics(metrics_table, period)
            self.logger.info(
                "Finished running query for %s (%s)",
                self.config.experiment.normandy_slug,
                period.value,
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
