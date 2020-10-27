from datetime import datetime, timedelta
import logging
from textwrap import dedent
from typing import Optional, List, Any, Dict

import attr
import dask
from google.cloud import bigquery
import mozanalysis
from mozanalysis.experiment import TimeLimits
from mozanalysis.utils import add_days
from pandas import DataFrame

from . import AnalysisPeriod, bq_normalize_name
from jetstream.config import AnalysisConfiguration, MetricsConfigurationType
from jetstream.dryrun import dry_run_query
import jetstream.errors as errors
from jetstream.statistics import Count, StatisticResult, StatisticResultCollection, Summary
from jetstream.bigquery_client import BigQueryClient
from jetstream.experimenter import Branch

logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True)
class Analysis:
    """
    Wrapper for analysing experiments.
    """

    project: str
    dataset: str

    @property
    def bigquery(self):
        return BigQueryClient(project=self.project, dataset=self.dataset)

    def _get_timelimits_if_ready(
        self, period: AnalysisPeriod, current_date: datetime, config: AnalysisConfiguration
    ) -> Optional[TimeLimits]:
        """
        Returns a TimeLimits instance if experiment is due for analysis.
        Otherwise returns None.
        """
        prior_date = current_date - timedelta(days=1)
        prior_date_str = prior_date.strftime("%Y-%m-%d")
        current_date_str = current_date.strftime("%Y-%m-%d")

        dates_enrollment = config.experiment.proposed_enrollment + 1

        if config.experiment.start_date is None:
            return None

        time_limits_args = {
            "first_enrollment_date": config.experiment.start_date.strftime("%Y-%m-%d"),
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
        if config.experiment.end_date != current_date or config.experiment.status != "Complete":
            return None

        analysis_length_dates = (
            (config.experiment.end_date - config.experiment.start_date).days - dates_enrollment + 1
        )

        if analysis_length_dates < 0:
            raise errors.EnrollmentLongerThanAnalysisException(config.experiment.normandy_slug)

        return TimeLimits.for_single_analysis_window(
            last_date_full_data=prior_date_str,
            analysis_start_days=0,
            analysis_length_dates=analysis_length_dates,
            **time_limits_args,
        )

    def _table_name(self, window_period: str, window_index: int, normandy_slug: str) -> str:
        assert normandy_slug is not None
        normalized_slug = bq_normalize_name(normandy_slug)
        return "_".join([normalized_slug, window_period, str(window_index)])

    def _publish_view(self, window_period: AnalysisPeriod, normandy_slug: str, table_prefix=None):
        assert normandy_slug is not None
        normalized_slug = bq_normalize_name(normandy_slug)
        view_name = "_".join([normalized_slug, window_period.adjective])
        wildcard_expr = "_".join([normalized_slug, window_period.value, "*"])

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

    def _calculate_metrics(
        self,
        exp: mozanalysis.experiment.Experiment,
        time_limits: TimeLimits,
        period: AnalysisPeriod,
        dry_run: bool,
        enrollment_query: str,
        segments: List[mozanalysis.segments.Segment],
        normandy_slug: str,
        metrics: MetricsConfigurationType,
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

        res_table_name = self._table_name(period.value, window, normandy_slug)

        sql = exp.build_query(
            {m.metric for m in metrics[period]},
            last_window_limits,
            "normandy",
            enrollment_query,
            segments,
        )

        if dry_run:
            logger.info(
                "Dry run; not actually calculating %s metrics for %s",
                period.value,
                normandy_slug,
            )
        else:
            logger.info(
                "Executing query for %s (%s)",
                normandy_slug,
                period.value,
            )
            self.bigquery.execute(sql, res_table_name)
            self._publish_view(period, normandy_slug)

        return res_table_name

    def _calculate_statistics(
        self,
        metric: Summary,
        segment_data: DataFrame,
        reference_branch: str,
        normandy_slug: str,
        segment: str,
    ):
        """
        Run statistics on metric.
        """
        stats = metric.run(segment_data, reference_branch, normandy_slug).set_segment(segment)
        return stats.to_dict()["data"]

    def _counts(self, segment_data: DataFrame, normandy_slug: str, segment: str):
        """Count statistic."""
        return (
            Count()
            .transform(segment_data, "*", "*", normandy_slug)
            .set_segment(segment)
            .to_dict()["data"]
        )

    def _missing_counts(
        self,
        counts: List[Dict[str, Any]],
        segment_data: DataFrame,
        normandy_slug: str,
        segment: str,
        branches: List[Branch],
    ):
        """Missing count statistic."""
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
                for b in branches
                if b.slug not in {c["branch"] for c in counts}
            ]
        )

        return missing_counts.to_dict()["data"]

    def _segment_metrics(self, segment: str, metrics_data: DataFrame, normandy_slug):
        """Return metrics data for segment"""
        if segment != "all":
            if segment not in metrics_data.columns:
                logger.error(
                    f"Segment {segment} not in metrics table",
                    extra={"experiment": normandy_slug},
                )
                return
            segment_data = metrics_data[metrics_data[segment]]
        else:
            segment_data = metrics_data

        return segment_data

    def check_runnable(
        self, config: AnalysisConfiguration, current_date: Optional[datetime] = None
    ) -> bool:
        if config.experiment.normandy_slug is None:
            # some experiments do not have a normandy slug
            raise errors.NoSlugException()

        if config.experiment.is_high_population:
            raise errors.HighPopulationException(config.experiment.normandy_slug)

        if not config.experiment.proposed_enrollment:
            raise errors.NoEnrollmentPeriodException(config.experiment.normandy_slug)

        if config.experiment.start_date is None:
            raise errors.NoStartDateException(config.experiment.normandy_slug)

        if (
            current_date
            and config.experiment.end_date
            and config.experiment.end_date < current_date
        ):
            raise errors.EndedException(config.experiment.normandy_slug)

        return True

    def validate(self, config: AnalysisConfiguration) -> None:
        self.check_runnable(config)

        dates_enrollment = config.experiment.proposed_enrollment + 1

        if config.experiment.end_date is not None:
            end_date = config.experiment.end_date
            analysis_length_dates = (
                (end_date - config.experiment.start_date).days - dates_enrollment + 1
            )
        else:
            analysis_length_dates = 21  # arbitrary
            end_date = config.experiment.start_date + timedelta(
                days=analysis_length_dates + dates_enrollment - 1
            )

        if analysis_length_dates < 0:
            logging.error(
                "Proposed enrollment longer than analysis dates length:"
                + f"{config.experiment.normandy_slug}"
            )
            raise Exception("Cannot validate experiment")

        limits = TimeLimits.for_single_analysis_window(
            last_date_full_data=end_date.strftime("%Y-%m-%d"),
            analysis_start_days=0,
            analysis_length_dates=analysis_length_dates,
            first_enrollment_date=config.experiment.start_date.strftime("%Y-%m-%d"),
            num_dates_enrollment=dates_enrollment,
        )

        exp = mozanalysis.experiment.Experiment(
            experiment_slug=config.experiment.normandy_slug,
            start_date=config.experiment.start_date.strftime("%Y-%m-%d"),
        )

        metrics = set()
        for v in config.metrics.values():
            metrics |= {m.metric for m in v}

        sql = exp.build_query(
            metrics,
            limits,
            "normandy",
            config.experiment.enrollment_query,
        )

        dry_run_query(sql)

    def _save_statistics(
        self,
        period: AnalysisPeriod,
        segment_results: List[Dict[str, Any]],
        metrics_table: str,
        normandy_slug: str,
    ):
        """Write statistics to BigQuery."""
        job_config = bigquery.LoadJobConfig()
        job_config.schema = StatisticResult.bq_schema
        job_config.write_disposition = bigquery.job.WriteDisposition.WRITE_TRUNCATE

        # wait for the job to complete
        self.bigquery.load_table_from_json(
            segment_results, f"statistics_{metrics_table}", job_config=job_config
        )

        self._publish_view(period, normandy_slug=normandy_slug, table_prefix="statistics")

    def run(
        self, current_date: datetime, config: AnalysisConfiguration, dry_run: bool = False
    ) -> None:
        """
        Run analysis using mozanalysis for a specific experiment.
        """
        logger.info("Analysis.run invoked for experiment %s", config.experiment.normandy_slug)

        self.check_runnable(config, current_date)

        client = dask.distributed.Client(threads_per_worker=2, n_workers=4)

        # dask config
        results = []
        calculate_metrics = dask.delayed(self._calculate_metrics)
        calculate_statistics = dask.delayed(self._calculate_statistics)
        segment_metrics = dask.delayed(self._segment_metrics)
        missing_counts = dask.delayed(self._missing_counts)
        counts = dask.delayed(self._counts)
        table_to_dataframe = dask.delayed(self.bigquery.table_to_dataframe)
        save_statistics = dask.delayed(self._save_statistics)

        for period in config.metrics:
            time_limits = self._get_timelimits_if_ready(period, current_date, config)

            if time_limits is None:
                logger.info(
                    "Skipping %s (%s); not ready",
                    config.experiment.normandy_slug,
                    period.value,
                )
                continue

            exp = mozanalysis.experiment.Experiment(
                experiment_slug=config.experiment.normandy_slug,
                start_date=config.experiment.start_date.strftime("%Y-%m-%d"),
            )

            metrics_table = calculate_metrics(
                exp,
                time_limits,
                period,
                dry_run,
                config.experiment.enrollment_query,
                config.experiment.segments,
                config.experiment.normandy_slug,
                config.metrics,
            )

            if dry_run:
                logger.info(
                    "Not calculating statistics %s (%s); dry run",
                    config.experiment.normandy_slug,
                    period.value,
                )
                continue

            metrics_data = table_to_dataframe(metrics_table)

            segment_results = []

            segment_labels = ["all"] + [s.name for s in config.experiment.segments]
            for segment in segment_labels:
                segment_data = segment_metrics(
                    segment, metrics_data, config.experiment.normandy_slug
                )
                for m in config.metrics[period]:
                    segment_results += calculate_statistics(
                        m,
                        segment_data,
                        config.experiment.reference_branch,
                        config.experiment.normandy_slug,
                        segment,
                    )

                c = counts(segment_data, config.experiment.normandy_slug, segment)
                segment_results += c
                segment_results += missing_counts(
                    c,
                    segment_data,
                    config.experiment.normandy_slug,
                    segment,
                    config.experiment.branches,
                )

            results.append(
                save_statistics(
                    period, segment_results, metrics_table, config.experiment.normandy_slug
                )
            )

        dask.compute(*results)
