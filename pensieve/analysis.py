from datetime import datetime, timedelta
import re
import logging
from textwrap import dedent
from typing import Optional

import attr
import mozanalysis
from mozanalysis.bq import BigQueryContext
from mozanalysis.experiment import TimeLimits
import mozanalysis.metrics.desktop as mmd
from mozanalysis.utils import add_days

from . import experimenter


@attr.s(auto_attribs=True)
class Analysis:
    """
    Wrapper for analysing experiments.
    """

    project: str
    dataset: str
    experiment: experimenter.Experiment

    # list of standard metrics to be computed
    STANDARD_METRICS = [
        mmd.active_hours,
        mmd.uri_count,
        mmd.ad_clicks,
        mmd.search_count,
    ]

    def __attrs_post_init__(self):
        self.logger = logging.getLogger(__name__)

    @property
    def bq_context(self):
        return BigQueryContext(project_id=self.project, dataset_id=self.dataset)

    def _get_timelimits_if_ready(self, current_date: datetime) -> Optional[TimeLimits]:
        """
        Returns a TimeLimits instance if experiment is due for analysis.
        Otherwise returns None.
        """
        prior_date_str = (current_date - timedelta(days=1)).strftime("%Y-%m-%d")
        current_date_str = current_date.strftime("%Y-%m-%d")

        if not self.experiment.proposed_enrollment:
            self.logger.info("Skipping %s; no enrollment period", self.experiment.slug)
            return None

        dates_enrollment = self.experiment.proposed_enrollment + 1

        if self.experiment.start_date is None:
            return None

        time_limits_args = {
            "first_enrollment_date": self.experiment.start_date.strftime("%Y-%m-%d"),
            "time_series_period": "weekly",
            "num_dates_enrollment": dates_enrollment,
        }

        try:
            current_time_limits = TimeLimits.for_ts(
                last_date_full_data=current_date_str, **time_limits_args
            )
        except ValueError:
            # There are no analysis windows yet.
            # TODO: Add a more specific check.
            return None

        try:
            prior_time_limits = TimeLimits.for_ts(
                last_date_full_data=prior_date_str, **time_limits_args
            )
        except ValueError:
            # We have an analysis window today, and we didn't yesterday,
            # so we must have just closed our first window.
            return current_time_limits

        if len(current_time_limits.analysis_windows) == len(prior_time_limits.analysis_windows):
            # No new data today
            return None

        return current_time_limits

    def _normalize_name(self, name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]", "_", name)

    def _table_name(self, window_period: str, window_index: int) -> str:
        assert self.experiment.normandy_slug is not None
        normalized_slug = self._normalize_name(self.experiment.normandy_slug)
        return "_".join([normalized_slug, window_period, str(window_index)])

    def _publish_view(self, window_period: str):
        assert self.experiment.normandy_slug is not None
        mapping = {"day": "daily", "week": "weekly", "all": "all"}
        normalized_slug = self._normalize_name(self.experiment.normandy_slug)
        view_name = "_".join([normalized_slug, mapping[window_period]])
        wildcard_expr = "_".join([normalized_slug, window_period, "*"])
        sql = dedent(
            f"""
            CREATE OR REPLACE VIEW {view_name} AS (
                SELECT
                    *,
                    CAST(_TABLE_SUFFIX AS int64) AS window_index
                FROM `{self.project}.{self.dataset}.{wildcard_expr}`
            )
            """
        )
        self.bq_context.run_query(sql)

    def run(self, current_date: datetime, dry_run: bool):
        """
        Run analysis using mozanalysis for a specific experiment.
        """
        self.logger.info("Analysis.run invoked for experiment %s", self.experiment.slug)

        if self.experiment.normandy_slug is None:
            self.logger.info("Skipping %s; no normandy_slug", self.experiment.slug)
            return  # some experiments do not have a normandy slug

        if self.experiment.start_date is None:
            self.logger.info("Skipping %s; no start_date", self.experiment.slug)
            return

        time_limits = self._get_timelimits_if_ready(current_date)
        if time_limits is None:
            self.logger.info("Skipping %s; not ready", self.experiment.slug)
            return

        exp = mozanalysis.experiment.Experiment(
            experiment_slug=self.experiment.normandy_slug,
            start_date=self.experiment.start_date.strftime("%Y-%m-%d"),
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

        res_table_name = self._table_name("week", window)

        # todo additional experiment specific metrics from Experimenter
        sql = exp.build_query(self.STANDARD_METRICS, last_window_limits, "normandy", None)

        if dry_run:
            self.logger.info("Not executing query for %s; dry run", self.experiment.slug)
            return

        self.logger.info("Executing query for %s", self.experiment.slug)
        self.bq_context.run_query(sql, res_table_name)
        self._publish_view("week")
        self.logger.info("Finished running query for %s", self.experiment.slug)
