from jetstream.nimbus import Feature, FeatureEventTelemetry, FeatureResolver


def test_feature_resolver():
    pip = FeatureResolver.resolve("picture_in_picture")
    assert isinstance(pip, Feature)
    summaries = pip.to_summaries()
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
