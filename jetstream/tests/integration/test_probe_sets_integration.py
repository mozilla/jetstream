import os

import pytest

from jetstream.probe_sets import ProbeLister


class TestProbeSetsIntegration:
    @pytest.mark.skipif("CI" in os.environ, reason="CI doesn't have permissions for this table")
    def test_probelister(self):
        columns = ProbeLister.columns_for_scalar("telemetry.discarded.accumulations")
        for process in ("content", "dynamic", "extension", "gpu", "socket"):
            assert (
                f"payload.processes.{process}.scalars.telemetry_discarded_accumulations" in columns
            )

        columns = ProbeLister.columns_for_scalar("telemetry.data_upload_optin")
        assert columns == ["payload.processes.parent.scalars.telemetry_data_upload_optin"]
