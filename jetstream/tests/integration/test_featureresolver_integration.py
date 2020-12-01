from jetstream.probe_sets import ProbeSet, EventProbe, ProbeSetsResolver


def test_feature_resolver():
    pip = ProbeSetsResolver.resolve("picture_in_picture")
    assert isinstance(pip, ProbeSet)
    summaries = pip.to_summaries()
    assert len(summaries)
    assert len(pip.telemetry)
    assert len(
        [
            probe
            for probe in pip.telemetry
            if isinstance(probe, EventProbe) and probe.event_category == "pictureinpicture"
        ]
    )
