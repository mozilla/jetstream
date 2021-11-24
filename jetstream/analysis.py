import logging
import os
import re
from datetime import datetime, timedelta
from textwrap import dedent
from typing import Any, Dict, List, Optional

import attr
import dask
import google
import mozanalysis
from dask.distributed import Client, LocalCluster
from google.cloud import bigquery
from google.cloud.exceptions import Conflict
from mozanalysis.experiment import AnalysisBasis, TimeLimits
from mozanalysis.utils import add_days
from pandas import DataFrame

import jetstream.errors as errors
from jetstream.bigquery_client import BigQueryClient
from jetstream.config import AnalysisConfiguration

# from jetstream.diagnostics.resource_profiling_plugin import ResourceProfilingPlugin
# from jetstream.diagnostics.task_monitoring_plugin import TaskMonitoringPlugin
from jetstream.dryrun import dry_run_query
from jetstream.logging import LogConfiguration, LogPlugin
from jetstream.statistics import (
    Count,
    StatisticResult,
    StatisticResultCollection,
    Summary,
)

from . import AnalysisPeriod, bq_normalize_name

logger = logging.getLogger(__name__)

DASK_DASHBOARD_ADDRESS = "127.0.0.1:8782"
DASK_N_PROCESSES = int(os.getenv("JETSTREAM_PROCESSES", 0)) or None  # Defaults to number of CPUs

_dask_cluster = None


@attr.s(auto_attribs=True)
class Analysis:
    """Wrapper for analysing experiments."""

    project: str
    dataset: str
    config: AnalysisConfiguration
    log_config: Optional[LogConfiguration] = None

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
                    time_series_period=period.mozanalysis_label,
                    **time_limits_args,
                )
            except ValueError:
                # There are no analysis windows yet.
                # TODO: Add a more specific check.
                return None

            try:
                prior_time_limits = TimeLimits.for_ts(
                    last_date_full_data=prior_date_str,
                    time_series_period=period.mozanalysis_label,
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
            self.config.experiment.end_date is None
            or self.config.experiment.end_date.date() != current_date.date()
            or self.config.experiment.status != "Complete"
        ):
            return None

        if self.config.experiment.end_date is None:
            return None

        analysis_length_dates = (
            (self.config.experiment.end_date - self.config.experiment.start_date).days
            - dates_enrollment
            + 1
        )

        if analysis_length_dates < 0:
            raise errors.EnrollmentLongerThanAnalysisException(self.config.experiment.normandy_slug)

        return TimeLimits.for_single_analysis_window(
            last_date_full_data=prior_date_str,
            analysis_start_days=0,
            analysis_length_dates=analysis_length_dates,
            **time_limits_args,
        )

    def _table_name(
        self, window_period: str, window_index: int, analysis_basis: Optional[AnalysisBasis] = None
    ) -> str:
        """
        Returns the Bigquery table name for statistics and metrics result tables.

        Tables names are based on analysis period, analysis window and optionally analysis basis.
        Metric aggregate tables should have analysis basis specified, while statistic tables
        should not.
        """
        assert self.config.experiment.normandy_slug is not None
        normalized_slug = bq_normalize_name(self.config.experiment.normandy_slug)

        if analysis_basis:
            return "_".join(
                [normalized_slug, analysis_basis.value, window_period, str(window_index)]
            )
        else:
            return "_".join([normalized_slug, window_period, str(window_index)])

    def _publish_view(self, window_period: AnalysisPeriod, table_prefix=None, analysis_basis=None):
        assert self.config.experiment.normandy_slug is not None
        normalized_slug = bq_normalize_name(self.config.experiment.normandy_slug)
        view_name = "_".join([normalized_slug, window_period.table_suffix])
        wildcard_expr = "_".join([normalized_slug, window_period.value, "*"])

        if analysis_basis:
            normalized_postfix = bq_normalize_name(analysis_basis)
            view_name = "_".join([normalized_slug, normalized_postfix, window_period.table_suffix])
            wildcard_expr = "_".join(
                [normalized_slug, normalized_postfix, window_period.value, "*"]
            )

        if table_prefix:
            normalized_prefix = bq_normalize_name(table_prefix)
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

    @dask.delayed
    def calculate_metrics(
        self,
        exp: mozanalysis.experiment.Experiment,
        time_limits: TimeLimits,
        period: AnalysisPeriod,
        analysis_basis: AnalysisBasis,
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

        res_table_name = self._table_name(period.value, window, analysis_basis=analysis_basis)
        normalized_slug = bq_normalize_name(self.config.experiment.normandy_slug)

        if dry_run:
            logger.info(
                "Dry run; not actually calculating %s metrics for %s",
                period.value,
                self.config.experiment.normandy_slug,
            )
        else:
            logger.info(
                "Executing query for %s (%s)",
                self.config.experiment.normandy_slug,
                period.value,
            )

            enrollments_table_name = f"enrollments_{normalized_slug}"
            exposure_signal = None

            if self.config.experiment.exposure_signal:
                # if a custom exposure signal has been defined in the config, we'll
                # need to pass it into the metrics computation
                exposure_signal = (
                    self.config.experiment.exposure_signal.to_mozanalysis_exposure_signal(
                        last_window_limits
                    )
                )

            metrics_sql = exp.build_metrics_query(
                {
                    m.metric.to_mozanalysis_metric()
                    for m in self.config.metrics[period]
                    if m.metric.analysis_bases == analysis_basis
                    or analysis_basis in m.metric.analysis_bases
                },
                last_window_limits,
                enrollments_table_name,
                analysis_basis,
                exposure_signal,
            )

            self.bigquery.execute(metrics_sql, res_table_name)
            self._publish_view(period, analysis_basis=analysis_basis.value)

        return res_table_name

    @dask.delayed
    def calculate_statistics(
        self, metric: Summary, segment_data: DataFrame, segment: str, analysis_basis: AnalysisBasis
    ) -> StatisticResultCollection:
        """
        Run statistics on metric.
        """
        return (
            metric.run(segment_data, self.config.experiment)
            .set_segment(segment)
            .set_analysis_basis(analysis_basis)
        )

    @dask.delayed
    def counts(
        self, segment_data: DataFrame, segment: str, analysis_basis: AnalysisBasis
    ) -> StatisticResultCollection:
        """Count and missing count statistics."""
        metric = "identity"
        counts = (
            Count()
            .transform(segment_data, metric, "*", self.config.experiment.normandy_slug)
            .set_segment(segment)
            .set_analysis_basis(analysis_basis)
        ).to_dict()["data"]

        return StatisticResultCollection(
            counts
            + [
                StatisticResult(
                    metric=metric,
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
                    analysis_basis=analysis_basis,
                )
                for b in self.config.experiment.branches
                if b.slug not in {c["branch"] for c in counts}
            ]
        )

    @dask.delayed
    def subset_to_segment(self, segment: str, metrics_data: DataFrame) -> DataFrame:
        """Return metrics data for segment"""
        if segment != "all":
            if segment not in metrics_data.columns:
                raise ValueError(f"Segment {segment} not in metrics table")
            segment_data = metrics_data[metrics_data[segment]]
        else:
            segment_data = metrics_data

        return segment_data

    def check_runnable(self, current_date: Optional[datetime] = None) -> bool:
        if self.config.experiment.normandy_slug is None:
            # some experiments do not have a normandy slug
            raise errors.NoSlugException()

        if self.config.experiment.skip:
            raise errors.ExplicitSkipException(self.config.experiment.normandy_slug)

        if self.config.experiment.is_high_population:
            raise errors.HighPopulationException(self.config.experiment.normandy_slug)

        if not self.config.experiment.proposed_enrollment:
            raise errors.NoEnrollmentPeriodException(self.config.experiment.normandy_slug)

        if self.config.experiment.start_date is None:
            raise errors.NoStartDateException(self.config.experiment.normandy_slug)

        if (
            current_date
            and self.config.experiment.end_date
            and self.config.experiment.end_date < current_date
        ):
            raise errors.EndedException(self.config.experiment.normandy_slug)

        return True

    def _app_id_to_bigquery_dataset(self, app_id: str) -> str:
        return re.sub(r"[^a-zA-Z0-9]", "_", app_id).lower()

    def validate(self) -> None:
        self.check_runnable()
        assert self.config.experiment.start_date is not None  # for mypy

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
            app_id=self._app_id_to_bigquery_dataset(self.config.experiment.app_id),
        )

        metrics = set()
        for v in self.config.metrics.values():
            metrics |= {m.metric.to_mozanalysis_metric() for m in v}

        exposure_signal = None
        if self.config.experiment.exposure_signal:
            exposure_signal = self.config.experiment.exposure_signal.to_mozanalysis_exposure_signal(
                limits
            )

        enrollments_sql = exp.build_enrollments_query(
            limits,
            self.config.experiment.platform.enrollments_query_type,
            self.config.experiment.enrollment_query,
            None,
            exposure_signal,
            self.config.experiment.segments,
        )

        dry_run_query(enrollments_sql)

        metrics_sql = exp.build_metrics_query(
            metrics, limits, "enrollments_table", AnalysisBasis.ENROLLMENTS
        )

        # enrollments_table doesn't get created when performing a dry run;
        # the metrics SQL is modified to include a subquery for a mock enrollments_table
        # A UNION ALL is required here otherwise the dry run fails with
        # "cannot query over table without filter over columns"
        metrics_sql = metrics_sql.replace(
            "WITH analysis_windows AS (",
            """WITH enrollments_table AS (
                SELECT '00000' AS client_id,
                    'test' AS branch,
                    DATE('2020-01-01') AS enrollment_date,
                    DATE('2020-01-01') AS exposure_date,
                    1 AS num_enrollment_events,
                    1 AS num_exposure_events
                UNION ALL
                SELECT '00000' AS client_id,
                    'test' AS branch,
                    DATE('2020-01-01') AS enrollment_date,
                    DATE('2020-01-01') AS exposure_date,
                    1 AS num_enrollment_events,
                    1 AS num_exposure_events
            ), analysis_windows AS (""",
        )

        dry_run_query(metrics_sql)

    @dask.delayed
    def save_statistics(
        self,
        period: AnalysisPeriod,
        segment_results: List[Dict[str, Any]],
        metrics_table: str,
    ):
        """Write statistics to BigQuery."""
        job_config = bigquery.LoadJobConfig()
        job_config.schema = StatisticResult.bq_schema
        job_config.write_disposition = bigquery.job.WriteDisposition.WRITE_TRUNCATE

        # wait for the job to complete
        self.bigquery.load_table_from_json(
            segment_results, f"statistics_{metrics_table}", job_config=job_config
        )

        self.bigquery.add_labels_to_table(
            f"statistics_{metrics_table}", {"schema_version": StatisticResult.SCHEMA_VERSION}
        )

        self._publish_view(period, table_prefix="statistics")

    def run(self, current_date: datetime, dry_run: bool = False) -> None:
        """
        Run analysis using mozanalysis for a specific experiment.
        """
        global _dask_cluster
        logger.info("Analysis.run invoked for experiment %s", self.config.experiment.normandy_slug)

        self.check_runnable(current_date)
        assert self.config.experiment.start_date is not None  # for mypy

        self.ensure_enrollments(current_date)

        # set up dask
        _dask_cluster = _dask_cluster or LocalCluster(
            dashboard_address=DASK_DASHBOARD_ADDRESS,
            processes=True,
            threads_per_worker=1,
            n_workers=DASK_N_PROCESSES,
        )
        client = Client(_dask_cluster)

        results = []

        if self.log_config:
            log_plugin = LogPlugin(self.log_config)
            client.register_worker_plugin(log_plugin)

            # add profiling plugins
            # resource_profiling_plugin = ResourceProfilingPlugin(
            #     scheduler=_dask_cluster.scheduler,
            #     project_id=self.log_config.log_project_id,
            #     dataset_id=self.log_config.log_dataset_id,
            #     table_id=self.log_config.task_profiling_log_table_id,
            #     experiment=self.config.experiment.normandy_slug,
            # )
            # _dask_cluster.scheduler.add_plugin(resource_profiling_plugin)

            # task_monitoring_plugin = TaskMonitoringPlugin(
            #     scheduler=_dask_cluster.scheduler,
            #     project_id=self.log_config.log_project_id,
            #     dataset_id=self.log_config.log_dataset_id,
            #     table_id=self.log_config.task_monitoring_log_table_id,
            #     experiment=self.config.experiment.normandy_slug,
            # )
            # _dask_cluster.scheduler.add_plugin(task_monitoring_plugin)

        table_to_dataframe = dask.delayed(self.bigquery.table_to_dataframe)

        for period in self.config.metrics:
            segment_results = []
            time_limits = self._get_timelimits_if_ready(period, current_date)

            if time_limits is None:
                logger.info(
                    "Skipping %s (%s); not ready",
                    self.config.experiment.normandy_slug,
                    period.value,
                )
                continue

            exp = mozanalysis.experiment.Experiment(
                experiment_slug=self.config.experiment.normandy_slug,
                start_date=self.config.experiment.start_date.strftime("%Y-%m-%d"),
                app_id=self._app_id_to_bigquery_dataset(self.config.experiment.app_id),
            )

            analysis_bases = []

            for m in self.config.metrics[period]:
                for analysis_basis in m.metric.analysis_bases:
                    analysis_bases.append(analysis_basis)

            analysis_bases = list(set(analysis_bases))

            if len(analysis_bases) == 0:
                continue

            for analysis_basis in analysis_bases:
                metrics_table = self.calculate_metrics(
                    exp, time_limits, period, analysis_basis, dry_run
                )

                if dry_run:
                    results.append(metrics_table)
                else:
                    metrics_dataframe = table_to_dataframe(metrics_table)

                if dry_run:
                    logger.info(
                        "Not calculating statistics %s (%s); dry run",
                        self.config.experiment.normandy_slug,
                        period.value,
                    )
                    continue

                segment_labels = ["all"] + [s.name for s in self.config.experiment.segments]
                for segment in segment_labels:
                    segment_data = self.subset_to_segment(segment, metrics_dataframe)
                    for m in self.config.metrics[period]:
                        segment_results += self.calculate_statistics(
                            m,
                            segment_data,
                            segment,
                            analysis_basis,
                        ).to_dict()["data"]

                    segment_results += self.counts(segment_data, segment, analysis_basis).to_dict()[
                        "data"
                    ]

            results.append(
                self.save_statistics(
                    period,
                    segment_results,
                    self._table_name(period.value, len(time_limits.analysis_windows)),
                )
            )

        result_futures = client.compute(results)
        client.gather(result_futures)  # block until futures have finished

    def ensure_enrollments(self, current_date: datetime) -> None:
        """Ensure that enrollment tables for experiment are up-to-date or re-create."""
        time_limits = self._get_timelimits_if_ready(AnalysisPeriod.DAY, current_date)

        if time_limits is None:
            logger.info(
                "Skipping enrollments for %s; not ready", self.config.experiment.normandy_slug
            )
            return

        if self.config.experiment.start_date is None:
            raise errors.NoStartDateException(self.config.experiment.normandy_slug)

        normalized_slug = bq_normalize_name(self.config.experiment.normandy_slug)
        enrollments_table = f"enrollments_{normalized_slug}"

        logger.info(f"Create {enrollments_table}")
        exp = mozanalysis.experiment.Experiment(
            experiment_slug=self.config.experiment.normandy_slug,
            start_date=self.config.experiment.start_date.strftime("%Y-%m-%d"),
            app_id=self._app_id_to_bigquery_dataset(self.config.experiment.app_id),
        )

        exposure_signal = None
        if self.config.experiment.exposure_signal:
            exposure_signal = self.config.experiment.exposure_signal.to_mozanalysis_exposure_signal(
                time_limits
            )

        enrollments_sql = exp.build_enrollments_query(
            time_limits,
            self.config.experiment.platform.enrollments_query_type,
            self.config.experiment.enrollment_query,
            None,
            exposure_signal,
            self.config.experiment.segments,
        )

        try:
            self.bigquery.execute(
                enrollments_sql,
                enrollments_table,
                google.cloud.bigquery.job.WriteDisposition.WRITE_EMPTY,
            )
        except Conflict:
            pass
