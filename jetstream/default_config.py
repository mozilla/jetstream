import copy
from typing import Dict, Optional

import attr

from . import external_config


@attr.s(auto_attribs=True)
class _DefaultConfigsResolver:
    """Fetches and resolves external default configurations."""

    external_configs: Optional[external_config.ExternalConfigCollection] = attr.ib(None)

    @property
    def data(self) -> Dict[str, external_config.ExternalDefaultConfig]:
        if data := getattr(self, "_data", None):
            return data

        if self.external_configs is None:
            self.external_configs = external_config.ExternalConfigCollection.from_github_repo()
        self._data = {config.slug: config for config in self.external_configs.defaults}
        return self._data

    def with_external_configs(
        self, external_configs: Optional[external_config.ExternalConfigCollection]
    ) -> "_DefaultConfigsResolver":
        self.external_configs = external_configs
        return self

    def resolve(self, slug: str) -> Optional[external_config.ExternalDefaultConfig]:
        return copy.deepcopy(self.data.get(slug, None))


DefaultConfigsResolver = _DefaultConfigsResolver()
