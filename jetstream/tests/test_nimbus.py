from unittest.mock import Mock
from typing import List

import mozanalysis.metrics
import pytest

from jetstream.nimbus import FeatureEventTelemetry, FeatureScalarTelemetry


@pytest.fixture
def fake_probe_lister(monkeypatch):
    class FakeDesktopProbeLister:
        def columns_for_scalar(self, slug: str) -> List[str]:
            return ["a.fancy.column.definitely_not_real", "a.fancier.column.definitely_not_real"]

    fake = FakeDesktopProbeLister()
    monkeypatch.setattr("jetstream.nimbus.ProbeLister", fake)
    yield fake


@pytest.mark.usefixtures("fake_probe_lister")
class TestFeature:
    def test_feature_event_telemetry(self):
        et = FeatureEventTelemetry(event_category="a", event_method="b")
        fake_config = Mock()
        fake_config.reference_branch = None
        summaries = et.to_summaries("bonus_slug", fake_config)
        assert len(summaries)
        for s in summaries:
            assert isinstance(s.metric, mozanalysis.metrics.Metric)

    def test_scalar_event_telemetry(self):
        st = FeatureScalarTelemetry(name="definitely.not.real")
        fake_config = Mock()
        fake_config.reference_branch = None
        summaries = st.to_summaries("bonus_slug", fake_config)
        assert len(summaries)
        for s in summaries:
            assert isinstance(s.metric, mozanalysis.metrics.Metric)
            assert "definitely_not_real" in s.metric.select_expr
