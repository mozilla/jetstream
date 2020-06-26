"""
Retrieves external configuration files for specific experiments.

Experiment-specific configuration files are stored in https://github.com/mozilla/pensieve-config/
"""

import datetime as dt
import attr
from github import Github
from google.cloud import bigquery
import os
import toml
from typing import List, Optional

from pensieve.config import AnalysisSpec


@attr.s(auto_attribs=True)
class ExternalConfig:
    """Represent an external config file."""

    normandy_slug: str
    spec: AnalysisSpec
    last_modified: dt.datetime

    def updated(self, bq_project: str, bq_dataset: str) -> bool:
        """
        Check whether the config file has been updated/added and
        associated BigQuery tables are out of date.
        """
        client = bigquery.Client(bq_project, bq_dataset)
        job = client.query(
            f"""
            SELECT COUNT(*) AS n FROM {bq_dataset}.__TABLES__
            WHERE table_id LIKE '{self.normandy_slug}%'
            AND TIMESTAMP_MILLIS(last_modified_time) <
                '{dt.strptime(self.last_modified, "%Y-%m-%d %H:%M:%s")}'
        """
        )

        result = job.result()
        for row in result:
            if row.n > 0:
                return True

        return False


@attr.s(auto_attribs=True)
class ExternalConfigCollection:
    """
    Collection of experiment-specific configurations pulled in
    from an external GitHub repository.
    """

    configs: List[ExternalConfig] = attr.Factory(list)

    PENSIEVE_CONFIG_REPO = "mozilla/pensieve-config"

    @classmethod
    def from_github_repo(cls) -> "ExternalConfigCollection":
        """Pull in external config files."""

        g = Github()
        repo = g.get_repo(cls.PENSIEVE_CONFIG_REPO)
        files = repo.get_contents("")

        configs = []

        for file in files:
            if file.name.endswith(".toml"):
                normandy_slug = os.path.splitext(file.name)[0]
                spec = AnalysisSpec.from_dict(toml.loads(file.decoded_content))
                configs.append(ExternalConfig(normandy_slug, spec, file.last_modified))

        return cls(configs)

    def spec_for_experiment(self, normandy_slug: str) -> Optional[AnalysisSpec]:
        """Return the spec for a specific experiment."""
        for config in self.configs:
            if config.normandy_slug == normandy_slug:
                return config.spec

        return None
