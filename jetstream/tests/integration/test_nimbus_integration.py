from jetstream.nimbus import ProbeLister


class TestNimbusIntegration:
    def test_probelister(self):
        columns = ProbeLister.columns_for_scalar("telemetry.accumulate_unknown_histogram_keys")
        for process in ("content", "dynamic", "extension", "gpu", "parent", "socket"):
            assert (
                f"payload.processes.{process}.scalars.telemetry_accumulate_unknown_histogram_keys"
                in columns
            )

