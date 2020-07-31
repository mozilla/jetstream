import jetstream.probeinfo


class TestProbeInfoIntegration:
    def test_all_scalars(self):
        obj = jetstream.probeinfo.DesktopProbeInfo
        scalar_keys = {i for i in obj.data.keys() if i.startswith("scalar/")}
        assert len(scalar_keys)
        for scalar in scalar_keys:
            assert len(obj.processes_for_scalar(scalar[7:])) > 0
