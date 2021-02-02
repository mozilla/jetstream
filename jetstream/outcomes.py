from typing import Dict

import attr

from . import external_config


@attr.s(auto_attribs=True)
class _OutcomesResolver:
    """Fetches and resolves external outcome snippet definitions."""

    @property
    def data(self) -> Dict[str, external_config.ExternalOutcome]:

        if data := getattr(self, "_data", None):
            return data

        external_configs = external_config.ExternalConfigCollection.from_github_repo()
        self._data = {outcome.slug: outcome for outcome in external_configs.outcomes}
        return self._data

    def resolve(self, slug: str) -> external_config.ExternalOutcome:
        return self.data[slug]


OutcomesResolver = _OutcomesResolver()
