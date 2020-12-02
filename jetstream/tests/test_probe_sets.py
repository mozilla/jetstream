from typing import List

import json
import mozanalysis.metrics
import pytest
import requests
from unittest.mock import patch

from jetstream.probe_sets import (
    TelemetryEventProbe,
    TelemetryScalarProbe,
    _ProbeSetsResolver,
    ProbeSet,
)

EXPERIMENTER_FIXTURE_PROBESETS = r"""
[
  {
    "name": "Test Probe Set",
    "slug": "test-probe-set",
    "probes": [
      {
        "kind": "scalar",
        "name": "Test Probe 1",
        "event_category": "test"
      }
    ]
  },
  {
    "name": "Pinned Tabs",
    "slug": "pinned_tabs",
    "probes": [
      {
        "kind": "event",
        "name": "Test Probe",
        "event_category": "test",
        "event_method": null,
        "event_object": null,
        "event_value": null
      }
    ]
  }
]
"""


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
        et = TelemetryEventProbe(name="a", event_category="a", event_method="b")
        summaries = et.to_summaries("bonus_slug")
        assert len(summaries)
        for s in summaries:
            assert isinstance(s.metric, mozanalysis.metrics.Metric)

    def test_scalar_event_telemetry(self):
        st = TelemetryScalarProbe(name="definitely.not.real", event_category="b")
        summaries = st.to_summaries("bonus_slug")
        assert len(summaries)
        for s in summaries:
            assert isinstance(s.metric, mozanalysis.metrics.Metric)
            assert "definitely_not_real" in s.metric.select_expr


@patch.object(requests.Session, "get")
def test_from_experimenter(mock_get):
    mock_get.return_value.json.return_value = json.loads(EXPERIMENTER_FIXTURE_PROBESETS)

    probe_sets_resolver = _ProbeSetsResolver()
    assert len(probe_sets_resolver.data) == 2
    mock_get.assert_any_call(_ProbeSetsResolver.EXPERIMENTER_API_URL_PROBESETS)
    assert isinstance(probe_sets_resolver.data["pinned_tabs"], ProbeSet)
    assert isinstance(probe_sets_resolver.data["test-probe-set"], ProbeSet)
    assert probe_sets_resolver.resolve("pinned_tabs").name == "Pinned Tabs"
