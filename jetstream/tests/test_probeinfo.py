import gzip
import json
import pathlib

import jetstream.probeinfo


class TestProbeInfo:
    def test_all_scalars(self):
        root = pathlib.Path(__file__).parent
        with gzip.open(root / "data/probe_data.json.gz") as f:
            probe_info = json.load(f)

        obj = jetstream.probeinfo._DesktopProbeInfo(probe_info)

        scalar_keys = {i for i in probe_info.keys() if i.startswith("scalar/")}

        for scalar in scalar_keys:
            assert len(obj.processes_for_scalar(scalar[7:])) > 0
