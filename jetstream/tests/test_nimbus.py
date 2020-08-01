from unittest.mock import Mock
from typing import Set

import mozanalysis.metrics
import pytest

from jetstream.nimbus import FeatureEventTelemetry, FeatureScalarTelemetry


@pytest.fixture
def fake_desktop_probe_info(monkeypatch):
    class FakeDesktopProbeInfo:
        def processes_for_scalar(self, slug: str) -> Set[str]:
            return {"main", "cheese"}

    fake = FakeDesktopProbeInfo()
    monkeypatch.setattr("jetstream.nimbus.DesktopProbeInfo", fake)
    yield fake


@pytest.mark.usefixtures("fake_desktop_probe_info")
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
