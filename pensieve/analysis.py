from datetime import datetime, timedelta
import re
import logging
from textwrap import dedent
from typing import Optional

import attr
import enum
import google.cloud.bigquery.client
import google.cloud.bigquery.dataset
import google.cloud.bigquery.job
import google.cloud.bigquery.table
import mozanalysis
from mozanalysis.experiment import TimeLimits
from mozanalysis.utils import add_days

from . import config


class AnalysisPeriod(enum.Enum):
    DAY = "day"
    WEEK = "week"
    OVERALL = "overall"

    @property
    def adjective(self) -> str:
        d = {"day": "daily", "week": "weekly", "overall": "overall"}
        return d[self.value]


@attr.s(auto_attribs=True)
class Analysis:
    """
    Wrapper for analysing experiments.
    """

    project: str
    dataset: str
    config: config.AnalysisConfiguration

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

        if not self.config.experiment.proposed_enrollment:
            self.logger.info("Skipping %s; no enrollment period", self.config.experiment.slug)
            return None

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
        if self.config.experiment.end_date != prior_date:
            return None

        return TimeLimits.for_single_analysis_window(
            last_date_full_data=prior_date_str,
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

    def _publish_view(self, window_period: AnalysisPeriod):
        assert self.config.experiment.normandy_slug is not None
        normalized_slug = self._normalize_name(self.config.experiment.normandy_slug)
        view_name = "_".join([normalized_slug, window_period.adjective])
        wildcard_expr = "_".join([normalized_slug, window_period.value, "*"])
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

    def run(self, current_date: datetime, dry_run: bool):
        """
        Run analysis using mozanalysis for a specific experiment.
        """
        self.logger.info("Analysis.run invoked for experiment %s", self.config.experiment.slug)

        if self.config.experiment.normandy_slug is None:
            self.logger.info("Skipping %s; no normandy_slug", self.config.experiment.slug)
            return  # some experiments do not have a normandy slug

        if self.config.experiment.start_date is None:
            self.logger.info("Skipping %s; no start_date", self.config.experiment.slug)
            return

        for period in AnalysisPeriod:
            time_limits = self._get_timelimits_if_ready(period, current_date)
            if time_limits is None:
                self.logger.info(
                    "Skipping %s (%s); not ready", self.config.experiment.slug, period.value
                )
                return

            exp = mozanalysis.experiment.Experiment(
                experiment_slug=self.config.experiment.normandy_slug,
                start_date=self.config.experiment.start_date.strftime("%Y-%m-%d"),
            )

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
                getattr(self.config.metrics, period.adjective),
                last_window_limits,
                "normandy",
                self.config.experiment.enrollment_query,
            )

            if dry_run:
                self.logger.info(
                    "Not executing query for %s (%s); dry run",
                    self.config.experiment.slug,
                    period.value,
                )
                return

            self.logger.info(
                "Executing query for %s (%s)", self.config.experiment.slug, period.value
            )
            self.bigquery.execute(sql, res_table_name)
            self._publish_view(period)
            self.logger.info(
                "Finished running query for %s (%s)", self.config.experiment.slug, period.value
            )


@attr.s(auto_attribs=True, slots=True)
class BigQueryClient:
    project: str
    dataset: str
    _client: Optional[google.cloud.bigquery.client.Client] = None

    @property
    def client(self):
        self._client = self._client or google.cloud.bigquery.client.Client(self.project)
        return self._client

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
