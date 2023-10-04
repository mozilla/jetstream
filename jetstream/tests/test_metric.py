from metric_config_parser.data_source import DataSource
from mozanalysis.metrics.fenix import uri_count
from mozilla_nimbus_schemas.jetstream import AnalysisBasis

from jetstream.metric import Metric


class TestMetric:
    def test_to_mozanalysis_metric(self):
        metric = Metric(
            name="test",
            data_source=DataSource(name="test_data_source", from_expression="test.test"),
            select_expression="test",
            analysis_bases=[AnalysisBasis.EXPOSURES],
        )

        mozanalysis_metric = metric.to_mozanalysis_metric()

        assert mozanalysis_metric
        assert mozanalysis_metric.name == metric.name
        assert metric.analysis_bases == [AnalysisBasis.EXPOSURES]

    def test_from_mozanalysis_metric(self):
        metric = Metric.from_mozanalysis_metric(uri_count)

        assert metric
        assert metric.name == "uri_count"
        assert metric.analysis_bases == [AnalysisBasis.ENROLLMENTS, AnalysisBasis.EXPOSURES]
