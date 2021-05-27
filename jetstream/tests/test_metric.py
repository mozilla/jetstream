from jetstream import AnalysisPeriod
from mozanalysis.experiment import AnalysisBasis
from mozanalysis.metrics import DataSource
from mozanalysis.metrics.fenix import uri_count

from jetstream.metric import Metric


class TestMetric:
    def test_to_mozanalysis_metric(self):
        metric = Metric(
            name="test",
            data_source=DataSource(name="test_data_source", from_expr="test.test"),
            select_expr="test",
            analysis_basis=AnalysisBasis.EXPOSURES,
        )

        mozanalysis_metric = metric.to_mozanalysis_metric()

        assert mozanalysis_metric
        assert mozanalysis_metric.name == metric.name

    def test_from_mozanalysis_metric(self):
        metric = Metric.from_mozanalysis_metric(uri_count)

        assert metric
        assert metric.name == "uri_count"
        assert metric.analysis_basis == AnalysisBasis.ENROLLMENTS
