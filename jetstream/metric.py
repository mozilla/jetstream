from typing import List, Optional

import attr
import mozanalysis.experiment
import mozanalysis.metrics
from jetstream_config_parser import data_source, metric


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Metric(metric.Metric):
    """
    Jetstream metric representation.

    Jetstream metrics are supersets of mozanalysis metrics with additional
    metadata required for analysis.
    """

    def __attrs_post_init__(self):
        # Print warning if exposures is used
        if metric.AnalysisBasis.EXPOSURES in self.analysis_bases:
            print(f"Using exposures analysis basis for {self.name}. Not supported in Experimenter")

    def to_mozanalysis_metric(self) -> mozanalysis.metrics.Metric:
        """Return Jetstream metric as mozanalysis metric."""
        return mozanalysis.metrics.Metric(
            name=self.name,
            data_source=mozanalysis.metrics.DataSource(
                name=self.data_source.name,
                from_expr=self.data_source.from_expression,
                experiments_column_type=self.data_source.experiments_column_type,
                client_id_column=self.data_source.client_id_column,
                submission_date_column=self.data_source.submission_date_column,
                default_dataset=self.data_source.default_dataset,
            ),
            select_expr=self.select_expression,
            friendly_name=self.friendly_name,
            description=self.description,
            bigger_is_better=self.bigger_is_better,
        )

    @classmethod
    def from_mozanalysis_metric(
        cls,
        mozanalysis_metric: mozanalysis.metrics.Metric,
        analysis_bases: Optional[List[metric.AnalysisBasis]] = [metric.AnalysisBasis.ENROLLMENTS],
    ) -> "Metric":
        return cls(
            name=mozanalysis_metric.name,
            data_source=data_source.DataSource(
                name=mozanalysis_metric.data_source.name,
                from_expr=mozanalysis_metric.data_source.from_expression,
                experiments_column_type=mozanalysis_metric.data_source.experiments_column_type,
                client_id_column=mozanalysis_metric.data_source.client_id_column,
                submission_date_column=mozanalysis_metric.data_source.submission_date_column,
                default_dataset=mozanalysis_metric.data_source.default_dataset,
            ),
            select_expression=mozanalysis_metric.select_expr,
            friendly_name=mozanalysis_metric.friendly_name,
            description=mozanalysis_metric.description,
            bigger_is_better=mozanalysis_metric.bigger_is_better,
            analysis_bases=analysis_bases or [metric.AnalysisBasis.ENROLLMENTS],
        )
