from typing import Set

from glom import glom, Flatten, Iter, Path, T
import requests


class _DesktopProbeInfo:
    PROBE_INFO_URL = "https://probeinfo.telemetry.mozilla.org/firefox/all/main/all_probes"

    @property
    def data(self):
        if data := getattr(self, "_data", None):
            return data
        self._data = requests.get(self.PROBE_INFO_URL).json()
        return self._data

    def processes_for_scalar(self, name) -> Set[str]:
        scalar_key = f"scalar/{name}"
        return glom(
            self.data,
            (
                Path(scalar_key, "history"),
                T.values(),  # all release channels
                Flatten(),  # all history
                Iter("details.record_in_processes").flatten(),
                set,
            ),
        )


DesktopProbeInfo = _DesktopProbeInfo()
