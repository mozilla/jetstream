import attr

from metric_config_parser.metric import AnalysisPeriod
from .analysis import AnalysisConfiguration
from datetime import datetime
from .logging import LogConfiguration

from mozanalysis.inflight import InflightSummary
from mozanalysis.bq import BigQueryContext


@attr.s(auto_attribs=True)
class InflightAnalysis:
    project_id: str
    dataset_id: str
    config: AnalysisConfiguration
    log_config: LogConfiguration | None
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

    def run(self, date: datetime | None = None) -> None:
        context = BigQueryContext(self.dataset_id, self.project_id)
        summaries = self.config.metrics[AnalysisPeriod.INFLIGHT]
        for summary in summaries:
            moz_summary = InflightSummary.from_summary(summary, self.config.experiment)
            moz_summary.publish_views(context)
