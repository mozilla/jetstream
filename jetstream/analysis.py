import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from textwrap import dedent
from typing import Any

import attr
import dask
import google
import mozanalysis
import pytz
from dask.distributed import Client, LocalCluster
from google.cloud import bigquery
from google.cloud.exceptions import Conflict
from metric_config_parser import metric
from metric_config_parser.analysis import AnalysisConfiguration
from metric_config_parser.metric import AnalysisPeriod
from mozanalysis.experiment import TimeLimits
from mozanalysis.utils import add_days
from mozilla_nimbus_schemas.jetstream import AnalysisBasis
from pandas import DataFrame

import jetstream.errors as errors
from jetstream.bigquery_client import BigQueryClient

# from jetstream.diagnostics.resource_profiling_plugin import ResourceProfilingPlugin
# from jetstream.diagnostics.task_monitoring_plugin import TaskMonitoringPlugin
from jetstream.dryrun import dry_run_query
from jetstream.exposure_signal import ExposureSignal
from jetstream.logging import LogConfiguration, LogPlugin
from jetstream.metric import Metric
from jetstream.platform import PLATFORM_CONFIGS
from jetstream.segment import Segment
from jetstream.statistics import (
    Count,
    StatisticResult,
    StatisticResultCollection,
    Summary,
)

from . import bq_normalize_name

logger = logging.getLogger(__name__)

DASK_DASHBOARD_ADDRESS = "127.0.0.1:8782"
DASK_N_PROCESSES = int(os.getenv("JETSTREAM_PROCESSES", 0)) or None  # Defaults to number of CPUs
COST_PER_SLOT_MS = 1 / 1000 / 60 / 60 * 0.06

_dask_cluster = None


@attr.s(auto_attribs=True)
class Analysis:
    """Wrapper for analysing experiments."""

    project: str
    dataset: str
    config: AnalysisConfiguration
    log_config: LogConfiguration | None = None
    start_time: datetime | None = None
    analysis_periods: list[AnalysisPeriod] = [
        AnalysisPeriod.DAY,
        AnalysisPeriod.WEEK,
        AnalysisPeriod.DAYS_28,
        AnalysisPeriod.OVERALL,
        AnalysisPeriod.PREENROLLMENT_WEEK,
        AnalysisPeriod.PREENROLLMENT_DAYS_28,
    ]
    sql_output_dir: str | None = None

    @property
    def bigquery(self):
        return BigQueryClient(project=self.project, dataset=self.dataset)

    def _get_timelimits_if_ready(
        self, period: AnalysisPeriod, current_date: datetime
    ) -> TimeLimits | None:
        """
        Returns a TimeLimits instance if experiment is due for analysis.
        Otherwise returns None.
        """
        prior_date = current_date - timedelta(days=1)
        prior_date_str = prior_date.strftime("%Y-%m-%d")
        current_date_str = current_date.strftime("%Y-%m-%d")

        dates_enrollment = self.config.experiment.enrollment_period + 1

        if self.config.experiment.start_date is None:
            return None

        time_limits_args = {
            "first_enrollment_date": self.config.experiment.start_date.strftime("%Y-%m-%d"),
            "num_dates_enrollment": dates_enrollment,
        }

        if period not in [
            AnalysisPeriod.OVERALL,
            AnalysisPeriod.PREENROLLMENT_WEEK,
            AnalysisPeriod.PREENROLLMENT_DAYS_28,
        ]:
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

        elif period in [AnalysisPeriod.PREENROLLMENT_WEEK, AnalysisPeriod.PREENROLLMENT_DAYS_28]:
            enrollment_end_date = self.config.experiment.start_date + timedelta(
                days=dates_enrollment
            )

            if enrollment_end_date != current_date:
                return None

            if period == AnalysisPeriod.PREENROLLMENT_WEEK:
                analysis_start_days = -7
                analysis_length_dates = 7
            elif period == AnalysisPeriod.PREENROLLMENT_DAYS_28:
                analysis_start_days = -7 * 4
                analysis_length_dates = 28
            else:
                return None

            return TimeLimits.for_single_analysis_window(
                last_date_full_data=prior_date_str,
                analysis_start_days=analysis_start_days,
                analysis_length_dates=analysis_length_dates,
                **time_limits_args,
            )

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

    def _write_sql_output(self, destination: str, sql: str):
        """Write SQL query to local file named after `destination`."""
        if self.sql_output_dir:
            Path(self.sql_output_dir).mkdir(parents=True, exist_ok=True)
            (Path(self.sql_output_dir) / destination).write_text(sql)

    def _table_name(
        self, window_period: str, window_index: int, analysis_basis: AnalysisBasis | None = None
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
    ) -> str:
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
                exposure_signal = ExposureSignal.from_exposure_signal_config(
                    self.config.experiment.exposure_signal
                )
                exposure_signal = exposure_signal.to_mozanalysis_exposure_signal(last_window_limits)

            # convert metric configurations to mozanalysis metrics
            metrics = {
                Metric.from_metric_config(m.metric).to_mozanalysis_metric()
                for m in self.config.metrics[period]
                if (
                    m.metric.analysis_bases == analysis_basis
                    or analysis_basis in m.metric.analysis_bases
                )
                and m.metric.select_expression is not None
            }

            metrics_sql = exp.build_metrics_query(
                metrics,
                last_window_limits,
                enrollments_table_name,
                analysis_basis,
                exposure_signal,
            )

            results = self.bigquery.execute(
                metrics_sql, res_table_name, experiment_slug=self.config.experiment.normandy_slug
            )
            logger.info(
                f"Metric query cost: {results.slot_millis * COST_PER_SLOT_MS}",
            )
            self._write_sql_output(res_table_name, metrics_sql)
            self._publish_view(period, analysis_basis=analysis_basis.value)

        return res_table_name

    @dask.delayed
    def calculate_statistics(
        self,
        metric: metric.Summary,
        segment_data: DataFrame,
        segment: str,
        analysis_basis: AnalysisBasis,
        analysis_length_dates: int,
        period: AnalysisPeriod,
    ) -> StatisticResultCollection:
        """
        Run statistics on metric.
        """
        return (
            Summary.from_config(metric, analysis_length_dates, period)
            .run(
                segment_data,
                self.config.experiment,
                analysis_basis,
                segment,
            )
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
            .transform(
                segment_data,
                metric,
                "*",
                self.config.experiment.normandy_slug,
                analysis_basis,
                segment,
            )
            .set_segment(segment)
            .set_analysis_basis(analysis_basis)
        )

        other_counts = [
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
            if b.slug not in {c.branch for c in counts.root}
        ]

        return StatisticResultCollection.model_validate(counts.root + other_counts)

    @dask.delayed
    def subset_metric_table(
        self,
        metrics_table_name: str,
        segment: str,
        summary: metric.Summary,
        analysis_basis: AnalysisBasis,
        period: AnalysisPeriod,
    ) -> DataFrame:
        """Pulls the metric data for this segment/analysis basis"""

        query = self._create_subset_metric_table_query(
            metrics_table_name, segment, summary, analysis_basis, period
        )

        results = self.bigquery.execute(query).to_dataframe()

        return results

    def _create_subset_metric_table_query(
        self,
        metrics_table_name: str,
        segment: str,
        summary: metric.Summary,
        analysis_basis: AnalysisBasis,
        period: AnalysisPeriod,
    ) -> str:
        query = ""
        if covariate_params := summary.statistic.params.get("covariate_adjustment", False):
            covariate_metric_name = covariate_params.get("metric", summary.metric.name)
            covariate_period = AnalysisPeriod(covariate_params["period"])
            if covariate_period != period:
                # when we configure a metric, all statistics are applied to all periods
                # however, to perform covariate adjustment we must use data from a different
                # period. So the metric will be configured with analysis periods like
                # [preenrollment_week, weekly, overall] but covariate adjustment should
                # only be applied on weekly and overall when using preenrollment_week
                # as the covariate.
                query = self._create_subset_metric_table_query_covariate(
                    metrics_table_name,
                    segment,
                    summary.metric,
                    analysis_basis,
                    covariate_period,
                    covariate_metric_name,
                )

        if not query:
            query = self._create_subset_metric_table_query_univariate(
                metrics_table_name, segment, summary.metric, analysis_basis
            )

        return query

    def _create_subset_metric_table_query_univariate(
        self,
        metrics_table_name: str,
        segment: str,
        metric: Metric,
        analysis_basis: AnalysisBasis,
    ) -> str:
        """Creates a SQL query string to pull a single metric for a segment/analysis"""

        metric_names = []
        # select placeholder column for metrics without select statement
        # since metrics that don't appear in the df are skipped
        # e.g., metrics with depends on such as population ratio metrics
        empty_metric_names = []
        if metric.depends_on:
            empty_metric_names.append(f"NULL AS {metric.name}")
            for dependency in metric.depends_on:
                metric_names.append(dependency.metric.name)
        else:
            metric_names.append(metric.name)

        query = dedent(
            f"""
        SELECT branch, {', '.join(metric_names + empty_metric_names)}
        FROM `{metrics_table_name}`
        WHERE {' IS NOT NULL AND '.join(metric_names + [''])[:-1]}
        """
        )

        if analysis_basis == AnalysisBasis.ENROLLMENTS:
            basis_filter = """enrollment_date IS NOT NULL"""
        elif analysis_basis == AnalysisBasis.EXPOSURES:
            basis_filter = """enrollment_date IS NOT NULL AND exposure_date IS NOT NULL"""
        else:
            raise ValueError(
                f"AnalysisBasis {analysis_basis} not valid"
                + f"Allowed values are: {[AnalysisBasis.ENROLLMENTS, AnalysisBasis.EXPOSURES]}"
            )

        query += basis_filter

        if segment != "all":
            segment_filter = dedent(
                f"""
            AND {segment} = TRUE"""
            )
            query += segment_filter

        return query

    def _create_subset_metric_table_query_covariate(
        self,
        metrics_table_name: str,
        segment: str,
        metric: Metric,
        analysis_basis: AnalysisBasis,
        covariate_period: AnalysisPeriod,
        covariate_metric_name: str,
    ) -> str:
        """Creates a SQL query string to pull a during-experiment metric and join on a
        pre-enrollment covariate for a segment/analysis"""

        if metric.depends_on:
            raise ValueError(
                "metrics with dependencies are not currently supported for covariate adjustment"
            )

        covariate_table_name = self._table_name(
            covariate_period.value, 1, analysis_basis=AnalysisBasis.ENROLLMENTS
        )

        if not self.bigquery.table_exists(covariate_table_name):
            normalized_slug = bq_normalize_name(self.config.experiment.normandy_slug)
            logger.warning(
                f"Covariate adjustment table {covariate_table_name} does not exist, falling back to unadjusted inferences",  # noqa:E501
                extra={
                    "experiment": normalized_slug,
                    "metric": metric.name,
                    "analysis_basis": analysis_basis.value,
                    "segment": segment,
                },
            )
            return self._create_subset_metric_table_query_univariate(
                metrics_table_name, segment, metric, analysis_basis
            )

        preenrollment_metric_select = f"pre.{covariate_metric_name} AS {covariate_metric_name}_pre"
        from_expression = dedent(
            f"""`{metrics_table_name}` during
            LEFT JOIN `{covariate_table_name}` pre
            USING (analysis_id, branch)"""
        )

        query = dedent(
            f"""
        SELECT
            during.branch,
            during.{metric.name},
            {preenrollment_metric_select}
        FROM (
            {from_expression}
        )
        WHERE during.{metric.name} IS NOT NULL AND
        """
        )

        if analysis_basis == AnalysisBasis.ENROLLMENTS:
            basis_filter = """during.enrollment_date IS NOT NULL"""
        elif analysis_basis == AnalysisBasis.EXPOSURES:
            basis_filter = (
                """during.enrollment_date IS NOT NULL AND during.exposure_date IS NOT NULL"""
            )
        else:
            raise ValueError(
                f"AnalysisBasis {analysis_basis} not valid"
                + f"Allowed values are: {[AnalysisBasis.ENROLLMENTS, AnalysisBasis.EXPOSURES]}"
            )

        query += basis_filter

        if segment != "all":
            segment_filter = dedent(
                f"""
            AND during.{segment} = TRUE"""
            )
            query += segment_filter

        return query

    def check_runnable(self, current_date: datetime | None = None) -> bool:
        if self.config.experiment.normandy_slug is None:
            # some experiments do not have a normandy slug
            raise errors.NoSlugException()

        if self.config.experiment.skip:
            raise errors.ExplicitSkipException(self.config.experiment.normandy_slug)

        if self.config.experiment.is_high_population:
            raise errors.HighPopulationException(self.config.experiment.normandy_slug)

        if not self.config.experiment.enrollment_period:
            raise errors.NoEnrollmentPeriodException(self.config.experiment.normandy_slug)

        if self.config.experiment.start_date is None:
            raise errors.NoStartDateException(self.config.experiment.normandy_slug)

        if (
            current_date
            and self.config.experiment.end_date
            and self.config.experiment.end_date < current_date
        ):
            raise errors.EndedException(self.config.experiment.normandy_slug)

        if self.config.experiment.is_rollout:
            raise errors.RolloutSkipException(self.config.experiment.normandy_slug)

        return True

    def _app_id_to_bigquery_dataset(self, app_id: str) -> str:
        return re.sub(r"[^a-zA-Z0-9]", "_", app_id).lower()

    def validate(self) -> None:
        self.check_runnable()
        assert self.config.experiment.start_date is not None  # for mypy

        dates_enrollment = self.config.experiment.enrollment_period + 1

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
            analysis_unit=self.config.experiment.analysis_unit,
        )

        metrics = set()
        for v in self.config.metrics.values():
            for metric_config in v:
                if metric_config.metric.select_expression:
                    metrics.add(
                        Metric.from_metric_config(metric_config.metric).to_mozanalysis_metric()
                    )

        exposure_signal = None
        if self.config.experiment.exposure_signal:
            exposure_signal = ExposureSignal.from_exposure_signal_config(
                self.config.experiment.exposure_signal
            )
            exposure_signal = exposure_signal.to_mozanalysis_exposure_signal(limits)

        segments = []
        for segment in self.config.experiment.segments:
            segments.append(Segment.from_segment_config(segment).to_mozanalysis_segment())

        enrollments_sql = exp.build_enrollments_query(
            limits,
            PLATFORM_CONFIGS[self.config.experiment.app_name].enrollments_query_type,
            self.config.experiment.enrollment_query,
            None,
            exposure_signal,
            segments,
            self.config.experiment.sample_size or None,
        )

        self._write_sql_output(
            f"enrollments_{bq_normalize_name(self.config.experiment.normandy_slug)}",
            enrollments_sql,
        )

        dry_run_query(enrollments_sql)
        print(f"Dry running enrollments query for {self.config.experiment.normandy_slug}:")
        print(enrollments_sql)

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
                SELECT '00000' AS analysis_id,
                    'test' AS branch,
                    DATE('2020-01-01') AS enrollment_date,
                    DATE('2020-01-01') AS exposure_date,
                    1 AS num_enrollment_events,
                    1 AS num_exposure_events
                UNION ALL
                SELECT '00000' AS analysis_id,
                    'test' AS branch,
                    DATE('2020-01-01') AS enrollment_date,
                    DATE('2020-01-01') AS exposure_date,
                    1 AS num_enrollment_events,
                    1 AS num_exposure_events
            ), analysis_windows AS (""",
        )

        self._write_sql_output(
            f"metrics_{bq_normalize_name(self.config.experiment.normandy_slug)}", metrics_sql
        )

        dry_run_query(metrics_sql)
        print(f"Dry running metrics query for {self.config.experiment.normandy_slug}:")
        print(metrics_sql)

    @dask.delayed
    def save_statistics(
        self,
        period: AnalysisPeriod,
        segment_results: list[dict[str, Any]],
        metrics_table: str,
    ):
        """Write statistics to BigQuery."""
        job_config = bigquery.LoadJobConfig()
        job_config.schema = StatisticResult.bq_schema
        job_config.write_disposition = bigquery.job.WriteDisposition.WRITE_TRUNCATE

        try:
            # wait for the job to complete
            self.bigquery.load_table_from_json(
                segment_results,
                f"statistics_{metrics_table}",
                job_config=job_config,
                experiment_slug=self.config.experiment.normandy_slug,
            )
        except google.api_core.exceptions.BadRequest as e:
            # There was a mismatch between the segment_results root dict
            # structure and the schema expected by bigquery. This error is
            # rather opaque, so we will do some extra manual logging to help
            # debugging these cases before re-raising the original exception.
            error_msg = """
            A BadRequest error from BigQuery likely indicates a mismatch between
            the statistics results data and the expected schema.
            """
            # logger.error(f"Expected schema: {StatisticResult.bq_schema}")
            # logger.error(f"Data received: {segment_results}")
            ve = ValueError(error_msg)
            raise ve from e

        self.bigquery.add_metadata_to_table(
            f"statistics_{metrics_table}", {"schema_version": StatisticResult.SCHEMA_VERSION}
        )

        self._publish_view(period, table_prefix="statistics")

    def run(
        self, current_date: datetime, dry_run: bool = False, statistics_only: bool = False
    ) -> None:
        """
        Run analysis using mozanalysis for a specific experiment.
        """
        global _dask_cluster
        self.start_time = datetime.now(tz=pytz.utc)
        logger.info(
            "Analysis.run invoked for experiment %s at %s",
            self.config.experiment.normandy_slug,
            self.start_time,
        )

        self.check_runnable(current_date)
        assert self.config.experiment.start_date is not None  # for mypy

        # make sure enrollment is actually ended (and enrollment is not manually overridden)
        if (
            self.config.experiment.experiment.type == "v6"
            and self.config.experiment.enrollment_end_date is None
        ) and (
            self.config.experiment.proposed_enrollment
            == self.config.experiment.experiment.proposed_enrollment
            and self.config.experiment.enrollment_end_date
            == self.config.experiment.experiment.enrollment_end_date
            and self.config.experiment.experiment_spec.enrollment_period is None
        ):
            raise errors.EnrollmentNotCompleteException(self.config.experiment.normandy_slug)

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

        for period in self.config.metrics:
            if period not in self.analysis_periods:
                logger.info(f"Skipping {period};")
                continue

            segment_results = StatisticResultCollection.model_validate([])
            time_limits = self._get_timelimits_if_ready(period, current_date)

            if time_limits is None:
                logger.info(
                    "Skipping %s (%s); not ready [START: %s, CURRENT: %s]",
                    self.config.experiment.normandy_slug,
                    period.value,
                    (
                        self.config.experiment.start_date.strftime("%Y-%m-%d")
                        if self.config.experiment.start_date is not None
                        else "None"
                    ),
                    current_date.strftime("%Y-%m-%d"),
                )
                continue

            exp = mozanalysis.experiment.Experiment(
                experiment_slug=self.config.experiment.normandy_slug,
                start_date=self.config.experiment.start_date.strftime("%Y-%m-%d"),
                app_id=self._app_id_to_bigquery_dataset(self.config.experiment.app_id),
                analysis_unit=self.config.experiment.analysis_unit,
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
                    exp, time_limits, period, analysis_basis, dry_run or statistics_only
                )

                if dry_run:
                    results.append(metrics_table)

                    logger.info(
                        "Not calculating statistics %s (%s); dry run",
                        self.config.experiment.normandy_slug,
                        period.value,
                    )
                    continue

                if statistics_only:
                    metrics_table_name = self._table_name(
                        period.value,
                        len(time_limits.analysis_windows),
                        analysis_basis=analysis_basis,
                    )
                    if not self.bigquery.table_exists(metrics_table_name):
                        logger.warning(
                            f"Cannot compute only statistics for period {period.value}; "
                            "metrics table does not exist!",
                            extra={
                                "experiment": self.config.experiment.normandy_slug,
                                "analysis_basis": analysis_basis.value,
                            },
                        )
                        continue

                segment_labels = ["all"] + [s.name for s in self.config.experiment.segments]
                for segment in segment_labels:
                    for summary in self.config.metrics[period]:
                        if (
                            summary.metric.analysis_bases != analysis_basis
                            and analysis_basis not in summary.metric.analysis_bases
                        ):
                            continue

                        segment_data = self.subset_metric_table(
                            metrics_table, segment, summary, analysis_basis, period
                        )

                        analysis_length_dates = 1
                        if period.value == AnalysisPeriod.OVERALL:
                            analysis_length_dates = time_limits.analysis_length_dates
                        elif period.value == AnalysisPeriod.WEEK:
                            analysis_length_dates = 7

                        segment_results.root += self.calculate_statistics(
                            summary,
                            segment_data,
                            segment,
                            analysis_basis,
                            analysis_length_dates,
                            period,
                        ).model_dump(warnings=False)

                    segment_results.root += self.counts(
                        segment_data, segment, analysis_basis
                    ).model_dump(warnings=False)

            results.append(
                self.save_statistics(
                    period,
                    segment_results.model_dump(warnings=False),
                    self._table_name(period.value, len(time_limits.analysis_windows)),
                )
            )

        result_futures = client.compute(results)
        client.gather(result_futures)  # block until futures have finished

    def enrollments_query(self, time_limits: TimeLimits) -> str:
        """Returns the enrollments SQL query."""
        exp = mozanalysis.experiment.Experiment(
            experiment_slug=self.config.experiment.normandy_slug,
            start_date=self.config.experiment.start_date.strftime("%Y-%m-%d"),
            app_id=self._app_id_to_bigquery_dataset(self.config.experiment.app_id),
            analysis_unit=self.config.experiment.analysis_unit,
        )

        exposure_signal = None
        if self.config.experiment.exposure_signal:
            exposure_signal = ExposureSignal.from_exposure_signal_config(
                self.config.experiment.exposure_signal
            )
            exposure_signal = exposure_signal.to_mozanalysis_exposure_signal(time_limits)

        segments = []
        for segment in self.config.experiment.segments:
            segments.append(Segment.from_segment_config(segment).to_mozanalysis_segment())

        return exp.build_enrollments_query(
            time_limits,
            PLATFORM_CONFIGS[self.config.experiment.app_name].enrollments_query_type,
            self.config.experiment.enrollment_query,
            None,
            exposure_signal,
            segments,
            self.config.experiment.sample_size or None,
        )

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
        enrollments_sql = self.enrollments_query(time_limits=time_limits)

        try:
            self._write_sql_output(enrollments_table, enrollments_sql)
            results = self.bigquery.execute(
                enrollments_sql,
                enrollments_table,
                google.cloud.bigquery.job.WriteDisposition.WRITE_EMPTY,
                experiment_slug=self.config.experiment.normandy_slug,
            )
            logger.info(
                "Enrollment query cost: " + f"{results.slot_millis * COST_PER_SLOT_MS}",
            )
        except Conflict:
            pass
