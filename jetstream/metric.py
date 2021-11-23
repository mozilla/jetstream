from typing import List, Optional

import attr
import mozanalysis.experiment
import mozanalysis.metrics


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Metric:
    """
    Jetstream metric representation.

    Jetstream metrics are supersets of mozanalysis metrics with additional
    metadata required for analysis.
    """

    name: str
    data_source: mozanalysis.metrics.DataSource
    select_expression: str
    friendly_name: Optional[str] = None
    description: Optional[str] = None
    bigger_is_better: bool = True
    analysis_bases: List[mozanalysis.experiment.AnalysisBasis] = [
        mozanalysis.experiment.AnalysisBasis.ENROLLMENTS
    ]

    def __attrs_post_init__(self):
        # Print warning if exposures is used
        if mozanalysis.experiment.AnalysisBasis.EXPOSURES in self.analysis_bases:
            print(f"Using exposures analysis basis for {self.name}. Not supported in Experimenter")

    def to_mozanalysis_metric(self) -> mozanalysis.metrics.Metric:
        """Return Jetstream metric as mozanalysis metric."""
        return mozanalysis.metrics.Metric(
            name=self.name,
            data_source=self.data_source,
            select_expr=self.select_expression,
            friendly_name=self.friendly_name,
            description=self.description,
            bigger_is_better=self.bigger_is_better,
        )

    @classmethod
    def from_mozanalysis_metric(
        cls,
        mozanalysis_metric: mozanalysis.metrics.Metric,
        analysis_bases: Optional[List[mozanalysis.experiment.AnalysisBasis]] = [
            mozanalysis.experiment.AnalysisBasis.ENROLLMENTS
        ],
    ) -> "Metric":
        return cls(
            name=mozanalysis_metric.name,
            data_source=mozanalysis_metric.data_source,
            select_expression=mozanalysis_metric.select_expr,
            friendly_name=mozanalysis_metric.friendly_name,
            description=mozanalysis_metric.description,
            bigger_is_better=mozanalysis_metric.bigger_is_better,
            analysis_bases=analysis_bases or [mozanalysis.experiment.AnalysisBasis.ENROLLMENTS],
        )
