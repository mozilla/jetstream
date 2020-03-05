import mozanalysis.metrics.desktop as mmd
from mozanalysis.experiment import Experiment, TimeLimits
from mozanalysis.bq import BigQueryContext
from datetime import datetime, timedelta
import pytz
import re


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
        self.bq_context = BigQueryContext(project_id=project, dataset_id=dataset)

    def _current_date(self):
        """Returns the current date with UTC timezone and time set to 00:00:00."""
        return datetime.combine(datetime.today(), datetime.min.time()).replace(tzinfo=pytz.utc)

    def _should_analyse_experiment(self, experiment, current_date):
        """
        Returns True if a passed experiment should be analysed based
        on its start date and the last time it got analysed.
        """
        date_delta = current_date - experiment.start_date
        next_analysis_date = current_date + timedelta(days=date_delta.days % self.ANALYSIS_PERIOD)

        return (
            date_delta.days > 0 and current_date == next_analysis_date
        ) or experiment.end_date < current_date

    # copied from mozanalyis
    # https://github.com/mozilla/mozanalysis/blob/579119e2f34a9259d5a79d03d86cea89eda34280/src/mozanalysis/bq.py#L9
    def _sanitize_table_name_for_bq(self, table_name):
        of_good_character_but_possibly_verbose = re.sub(r"[^a-zA-Z_0-9]", "_", table_name)

        if len(of_good_character_but_possibly_verbose) <= 1024:
            return of_good_character_but_possibly_verbose

        return (
            of_good_character_but_possibly_verbose[:500]
            + "___"
            + of_good_character_but_possibly_verbose[-500:]
        )

    def run(self, experiment):
        """
        Run analysis using mozanalysis for a specific experiment.
        """

        print(experiment.normandy_slug)

        if experiment.normandy_slug is None:
            return  # some experiments do not have a normandy slug

        current_date = self._current_date()

        if self._should_analyse_experiment(experiment, current_date):
            exp = Experiment(
                experiment_slug=experiment.normandy_slug,
                start_date=experiment.start_date.strftime("%Y-%m-%d"),
            )

            date_delta = current_date - experiment.start_date
            last_date_full_data = current_date

            if experiment.end_date < current_date:
                last_date_full_data = experiment.end_date

            # build and execute the BigQuery query
            last_date_full_data = last_date_full_data.strftime("%Y-%m-%d")
            analysis_start_days = max(0, date_delta.days - self.ANALYSIS_PERIOD)
            res_table_name = self._sanitize_table_name_for_bq(experiment.normandy_slug)

            time_limits = TimeLimits.for_single_analysis_window(
                exp.start_date,
                last_date_full_data,
                analysis_start_days,
                self.ANALYSIS_PERIOD,
                exp.num_dates_enrollment,
            )

            # todo additional experiment specific metrics from Experimenter
            # todo: use build_query once new version of mozanalysis lands
            sql = exp._build_query(self.STANDARD_METRICS, time_limits, "normandy", None)

            # todo: add option to append to table; right now the table gets reused
            self.bq_context.run_query(sql, res_table_name)
