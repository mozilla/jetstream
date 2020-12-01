from typing import List

import mozanalysis.metrics
import pytest

from jetstream.probe_sets import EventProbe, ScalarProbe


@pytest.fixture
def fake_probe_lister(monkeypatch):
    class FakeDesktopProbeLister:
        def columns_for_scalar(self, slug: str) -> List[str]:
            return ["a.fancy.column.definitely_not_real", "a.fancier.column.definitely_not_real"]

    fake = FakeDesktopProbeLister()
    monkeypatch.setattr("jetstream.probe_sets.ProbeLister", fake)
    yield fake


@pytest.mark.usefixtures("fake_probe_lister")
class TestProbe:
    def test_feature_event_telemetry(self):
        et = EventProbe(name="a", event_category="a", event_method="b")
        summaries = et.to_summaries("bonus_slug")
        assert len(summaries)
        for s in summaries:
            assert isinstance(s.metric, mozanalysis.metrics.Metric)

    def test_scalar_event_telemetry(self):
        st = ScalarProbe(name="definitely.not.real", event_category="b")
        summaries = st.to_summaries("bonus_slug")
        assert len(summaries)
        for s in summaries:
            assert isinstance(s.metric, mozanalysis.metrics.Metric)
            assert "definitely_not_real" in s.metric.select_expr
