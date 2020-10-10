"""
Retrieves external configuration files for specific experiments.

Experiment-specific configuration files are stored in https://github.com/mozilla/jetstream-config/
"""

import datetime as dt
import os
from typing import List, Optional

import attr
from github import Github
from github.ContentFile import ContentFile
from google.cloud import bigquery
from pytz import UTC
import toml

from . import bq_normalize_name
from jetstream.config import AnalysisSpec


@attr.s(auto_attribs=True)
class ExternalConfig:
    """Represent an external config file."""

    slug: str
    spec: AnalysisSpec
    last_modified: dt.datetime


@attr.s(auto_attribs=True)
class ExternalConfigCollection:
    """
    Collection of experiment-specific configurations pulled in
    from an external GitHub repository.
    """

    configs: List[ExternalConfig] = attr.Factory(list)

    JETSTREAM_CONFIG_REPO = "mozilla/jetstream-config"

    @classmethod
    def from_github_repo(cls) -> "ExternalConfigCollection":
        """Pull in external config files."""

        g = Github()
        repo = g.get_repo(cls.JETSTREAM_CONFIG_REPO)
        files = repo.get_contents("")

        if isinstance(files, ContentFile):
            files = [files]

        configs = []

        for file in files:
            if file.name.endswith(".toml"):
                commits = repo.get_commits(path=file.path)

                if commits.totalCount:
                    slug = os.path.splitext(file.name)[0]
                    spec = AnalysisSpec.from_dict(toml.loads(file.decoded_content.decode("utf-8")))
                    last_modified = UTC.localize(commits[0].commit.committer.date)
                    configs.append(ExternalConfig(slug, spec, last_modified))

        return cls(configs)

    def spec_for_experiment(self, slug: str) -> Optional[AnalysisSpec]:
        """Return the spec for a specific experiment."""
        for config in self.configs:
            if config.slug == slug:
                return config.spec

        return None

    def updated_configs(self, bq_project: str, bq_dataset: str) -> List[ExternalConfig]:
        """
        Return external configs that have been updated/added and
        with associated BigQuery tables being out of date.
        """
        client = bigquery.Client(bq_project)
        job = client.query(
            fr"""
            SELECT
                table_name,
                REGEXP_EXTRACT_ALL(
                    option_value,
                    '.*STRUCT\\(\"last_updated\", \"(.+)\"\\).*'
                ) AS last_updated
            FROM
            {bq_dataset}.INFORMATION_SCHEMA.TABLE_OPTIONS
            WHERE option_name = 'labels'
            """
        )

        result = list(job.result())

        updated_configs = []

        for config in self.configs:
            table_prefix = bq_normalize_name(config.slug)
            for row in result:
                if not row.table_name.startswith(table_prefix):
                    continue
                if not len(row.last_updated):
                    continue
                table_last_updated = UTC.localize(
                    dt.datetime.fromtimestamp(int(row.last_updated[0]))
                )
                if table_last_updated < config.last_modified:
                    updated_configs.append(config)
                    break

        return updated_configs
