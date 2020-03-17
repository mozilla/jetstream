from datetime import datetime, timedelta
from typing import Optional

import attr
import mozanalysis
from mozanalysis.bq import BigQueryContext
from mozanalysis.bq import sanitize_table_name_for_bq
from mozanalysis.experiment import TimeLimits
import mozanalysis.metrics.desktop as mmd

from . import experimenter


class Analysis:
    """
    Wrapper for analysing experiments.
    """

    # list of standard metrics to be computed
    STANDARD_METRICS = [
        mmd.active_hours,
        mmd.uri_count,
        mmd.ad_clicks,
        mmd.search_count,
    ]

    def __init__(self, project, dataset):
        self.project = project
        self.dataset = dataset
        self.bq_context = None

    def _get_timelimits_if_ready(
        self, experiment: experimenter.Experiment, current_date: datetime
    ) -> Optional[TimeLimits]:
        """
        Returns a TimeLimits instance if experiment is due for analysis.
        Otherwise returns None.
        """
        prior_date_str = (current_date - timedelta(days=1)).strftime("%Y-%m-%d")
        current_date_str = current_date.strftime("%Y-%m-%d")

        dates_enrollment = 0
        if experiment.proposed_enrollment:
            dates_enrollment = experiment.proposed_enrollment + 1

        time_limits_args = {
            "first_enrollment_date": experiment.start_date.strftime("%Y-%m-%d"),
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

    def run(self, experiment: experimenter.Experiment, current_date: datetime):
        """
        Run analysis using mozanalysis for a specific experiment.
        """

        if experiment.normandy_slug is None:
            return  # some experiments do not have a normandy slug

        time_limits = self._get_timelimits_if_ready(experiment, current_date)
        if time_limits is None:
            return

        exp = mozanalysis.experiment.Experiment(
            experiment_slug=experiment.normandy_slug,
            start_date=experiment.start_date.strftime("%Y-%m-%d"),
        )

        window = len(time_limits.analysis_windows)
        last_analysis_window = time_limits.analysis_windows[-1:]
        # TODO: Add this functionality to TimeLimits.
        last_window_limits = attr.evolve(time_limits, analysis_windows=last_analysis_window,)

        res_table_name = sanitize_table_name_for_bq(
            "_".join([experiment.normandy_slug, "window", window])
        )

        # todo additional experiment specific metrics from Experimenter
        sql = exp.build_query(self.STANDARD_METRICS, last_window_limits, "normandy", None)

        if self.bq_context is None:
            self.bq_context = BigQueryContext(project_id=self.project, dataset_id=self.dataset)

        self.bq_context.run_query(sql, res_table_name)
