from typing import List, Optional

import attr
import mozanalysis.experiment
import mozanalysis.metrics
from metric_config_parser import data_source
from metric_config_parser import metric as parser_metric
from mozilla_nimbus_schemas.jetstream import AnalysisBasis


class Metric(parser_metric.Metric):
    """
    Jetstream metric representation.

    Jetstream metrics are supersets of mozanalysis metrics with additional
    metadata required for analysis.
    """

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
        analysis_bases: Optional[List[AnalysisBasis]] = [
            AnalysisBasis.ENROLLMENTS,
            AnalysisBasis.EXPOSURES,
        ],
    ) -> "Metric":
        return cls(
            name=mozanalysis_metric.name,
            data_source=data_source.DataSource(
                name=mozanalysis_metric.data_source.name,
                from_expression=mozanalysis_metric.data_source._from_expr,
                experiments_column_type=mozanalysis_metric.data_source.experiments_column_type,
                client_id_column=mozanalysis_metric.data_source.client_id_column,
                submission_date_column=mozanalysis_metric.data_source.submission_date_column,
                default_dataset=mozanalysis_metric.data_source.default_dataset,
            ),
            select_expression=mozanalysis_metric.select_expr,
            friendly_name=mozanalysis_metric.friendly_name,
            description=mozanalysis_metric.description,
            bigger_is_better=mozanalysis_metric.bigger_is_better,
            analysis_bases=analysis_bases or [AnalysisBasis.ENROLLMENTS, AnalysisBasis.EXPOSURES],
        )

    @classmethod
    def from_metric_config(cls, metric_config: parser_metric.Metric) -> "Metric":
        """Create a metric class instance from a metric config."""
        args = attr.asdict(metric_config)
        if metric_config.data_source:
            args["data_source"] = data_source.DataSource(**attr.asdict(metric_config.data_source))
        return cls(**args)
