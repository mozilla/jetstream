from unittest.mock import Mock

from jetstream.nimbus import Feature, FeatureEventTelemetry, FeatureResolver


def test_feature_resolver():
    pip = FeatureResolver.resolve("picture_in_picture")
    assert isinstance(pip, Feature)
    fake_config = Mock()
    fake_config.reference_branch = None
    summaries = pip.to_summaries(fake_config)
    assert len(summaries)
    assert len(pip.telemetry)
    assert len(
        [
            probe
            for probe in pip.telemetry
            if isinstance(probe, FeatureEventTelemetry)
            and probe.event_category == "pictureinpicture"
        ]
    )
