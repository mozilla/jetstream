import mozanalysis.metrics.desktop as mmd
import mozanalysis.bayesian_stats.binary as mabsbin
import attr
from mozanalysis.experiment import Experiment
from mozanalysis.bq import BigQueryContext
from datetime import datetime, timedelta
import pytz

from pensieve.experimenter import Variant


class Analysis:
    # todo: where should this come from? experiments, metrics?
    ANALYSIS_PERIOD = 7  # 1 week

    # list of standard to be computed
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

    def run(self, experiment):
        # check if experiment should be analysed
        # todo: can UTC be assumed?
        current_date = datetime.combine(datetime.today(), datetime.min.time()).replace(
            tzinfo=pytz.utc
        )
        date_delta = current_date - experiment.start_date
        next_analysis_date = current_date + timedelta(days=date_delta.days % self.ANALYSIS_PERIOD)

        if current_date == next_analysis_date or experiment.end_date < current_date:
            exp = Experiment(
                experiment_slug=experiment.slug,
                start_date=experiment.start_date,
                num_dates_enrollment=14,  # todo
            )

            last_date_full_data = current_date

            if experiment.end_date < current_date:
                last_date_full_data = experiment.end_date

            # todo additional experiment specific metrics from Experimenter in addition to standard ones
            # todo output to custom table
            # ts_res = exp.get_single_window_data(
            #     bq_context=self.bq_context,
            #     metric_list=self.STANDARD_METRICS,
            #     last_date_full_data=last_date_full_data,
            #     analysis_start_days=max(0, date_delta - self.ANALYSIS_PERIOD),
            #     analysis_length_days=self.ANALYSIS_PERIOD
            # )
