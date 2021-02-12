from typing import Dict, Optional

import attr

from . import external_config


@attr.s(auto_attribs=True)
class _OutcomesResolver:
    """Fetches and resolves external outcome snippet definitions."""

    external_configs: Optional[external_config.ExternalConfigCollection] = attr.ib(None)

    @property
    def data(self) -> Dict[str, external_config.ExternalOutcome]:

        if data := getattr(self, "_data", None):
            return data

        if self.external_configs is None:
            self.external_configs = external_config.ExternalConfigCollection.from_github_repo()
        self._data = {outcome.slug: outcome for outcome in self.external_configs.outcomes}
        return self._data

    def with_external_configs(
        self, external_configs: Optional[external_config.ExternalConfigCollection]
    ) -> "_OutcomesResolver":
        self.external_configs = external_configs
        return self

    def resolve(self, slug: str) -> external_config.ExternalOutcome:
        return self.data[slug]


OutcomesResolver = _OutcomesResolver()
