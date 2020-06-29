"""
Retrieves external configuration files for specific experiments.

Experiment-specific configuration files are stored in https://github.com/mozilla/pensieve-config/
"""

import datetime as dt
import attr
from dateutil import parser
from github import Github
from github.ContentFile import ContentFile
from google.cloud import bigquery
import os
import re
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
        client = bigquery.Client(bq_project)
        job = client.query(
            f"""
            SELECT COUNT(*) AS n FROM {bq_dataset}.__TABLES__
            WHERE table_id LIKE '{re.sub(r"[^a-zA-Z0-9_]", "_", self.normandy_slug)}%'
            AND TIMESTAMP_MILLIS(last_modified_time) <
                '{self.last_modified.strftime("%Y-%m-%d %H:%M:%S")}'
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

        if isinstance(files, ContentFile):
            files = [files]

        configs = []

        for file in files:
            if file.name.endswith(".toml"):
                normandy_slug = os.path.splitext(file.name)[0]
                spec = AnalysisSpec.from_dict(toml.loads(file.decoded_content.decode("utf-8")))
                last_modified = parser.parse(str(file.last_modified))
                configs.append(ExternalConfig(normandy_slug, spec, last_modified))

        return cls(configs)

    def spec_for_experiment(self, normandy_slug: str) -> Optional[AnalysisSpec]:
        """Return the spec for a specific experiment."""
        for config in self.configs:
            if config.normandy_slug == normandy_slug:
                return config.spec

        return None
