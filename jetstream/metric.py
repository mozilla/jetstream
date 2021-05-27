import attr

import mozanalysis.experiment
import mozanalysis.metrics

from typing import Optional


@attr.s(auto_attribs=True)
class Metric(mozanalysis.metrics.Metric):
    """
    Jetstream metric representation.

    Jetstream metrics are supersets of mozanalysis metrics with additional
    metadata required for analysis.
    """

    analysis_basis: mozanalysis.experiment.AnalysisBasis = (
        mozanalysis.experiment.AnalysisBasis.ENROLLMENT
    )

    def to_mozanalysis_metric(self) -> mozanalysis.metrics.Metric:
        """Return Jetstream metric as mozanalysis metric."""
        return super(self.__class__, self)

    @classmethod
    def from_mozanalysis_metric(
        cls,
        mozanalysis_metric: mozanalysis.metrics.Metric,
        analysis_basis: Optional[mozanalysis.experiment.AnalysisBasis] = None,
    ) -> "Metric":
        return cls(
            name=mozanalysis_metric.name,
            data_source=mozanalysis_metric.data_source,
            select_expr=mozanalysis_metric.select_expr,
            friendly_name=mozanalysis_metric.friendly_name,
            description=mozanalysis_metric.description,
            bigger_is_better=mozanalysis_metric.bigger_is_better,
            analysis_basis=analysis_basis or mozanalysis.experiment.AnalysisBasis.ENROLLMENT,
        )
