from typing import Dict, Optional

import attr

from . import external_config


@attr.s(auto_attribs=True, frozen=True)
class _OutcomesResolver:
    """Fetches and resolves external outcome snippet definitions."""

    _external_configs: Optional[external_config.ExternalConfigCollection] = None

    @property
    def data(self) -> Dict[str, external_config.ExternalOutcome]:
        if data := getattr(self, "_data", None):
            return data

        external_configs = (
            self._external_configs or external_config.ExternalConfigCollection.from_github_repo()
        )
        object.__setattr__(
            self, "_data", {outcome.slug: outcome for outcome in external_configs.outcomes}
        )
        return self._data  # type: ignore

    def resolve(self, slug: str) -> external_config.ExternalOutcome:
        return self.data[slug]


OutcomesResolver = _OutcomesResolver()
OutcomesResolverType = _OutcomesResolver
