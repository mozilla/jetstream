from datetime import datetime, timedelta

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

    # todo: where should this come from? experiments, metrics?
    ANALYSIS_PERIOD = 7  # 1 week

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

    def _should_analyse_experiment(
        self, experiment: experimenter.Experiment, current_date: datetime
    ):
        """
        Returns True if a passed experiment should be analysed based
        on its start date and the last time it got analysed.
        """
        date_delta = current_date - experiment.start_date
        next_analysis_date = current_date + timedelta(days=date_delta.days % self.ANALYSIS_PERIOD)

        return (
            date_delta.days > 0 and current_date == next_analysis_date
        ) or experiment.end_date <= current_date

    def run(self, experiment: experimenter.Experiment, current_date: datetime):
        """
        Run analysis using mozanalysis for a specific experiment.
        """

        if experiment.normandy_slug is None:
            return  # some experiments do not have a normandy slug

        if self._should_analyse_experiment(experiment, current_date):
            exp = mozanalysis.experiment.Experiment(
                experiment_slug=experiment.normandy_slug,
                start_date=experiment.start_date.strftime("%Y-%m-%d"),
            )

            date_delta = current_date - experiment.start_date

            # data from the current day aren't available yet, so we use the previous day
            last_date_full_data = current_date

            if experiment.end_date < current_date:
                last_date_full_data = experiment.end_date

            # build and execute the BigQuery query
            last_date_full_data = last_date_full_data.strftime("%Y-%m-%d")
            analysis_start_days = max(0, date_delta.days - self.ANALYSIS_PERIOD)
            window = str(int(date_delta.days / self.ANALYSIS_PERIOD))
            res_table_name = sanitize_table_name_for_bq(
                "_".join([experiment.normandy_slug, "window", window])
            )

            time_limits = TimeLimits.for_single_analysis_window(
                exp.start_date,
                last_date_full_data,
                analysis_start_days,
                self.ANALYSIS_PERIOD,
                exp.num_dates_enrollment,
            )

            # todo additional experiment specific metrics from Experimenter
            sql = exp.build_query(self.STANDARD_METRICS, time_limits, "normandy", None)

            if self.bq_context is None:
                self.bq_context = BigQueryContext(project_id=self.project, dataset_id=self.dataset)

            self.bq_context.run_query(sql, res_table_name)
