"""
Parses configuration specifications into concrete objects.

Users should write something like:
    my_config = (
        config.AnalysisSpec
        .from_dict(toml.load(my_config_file))
        .resolve(an_experimenter_object, ConfigLoader.configs)
    )
to obtain a concrete AnalysisConfiguration object.

Spec objects are direct representations of the configuration and contain unresolved references
to metrics and data sources.

Calling .resolve(config_spec) on a Spec object produces a concrete resolved Configuration class.

Definition and Reference classes are also direct representations of the configuration,
which produce concrete mozanalysis classes when resolved.
"""

import copy
import datetime as dt
from typing import List, Optional

from google.cloud import bigquery
from jetstream_config_parser.analysis import AnalysisSpec
from jetstream_config_parser.config import Config, ConfigCollection, Outcome
from jetstream_config_parser.data_source import DataSource
from pytz import UTC

from . import bq_normalize_name


class _ConfigLoader:
    """
    Loads config files from an external repository.

    Config objects are converted into jetstream native types.
    """

    config_collection: Optional[ConfigCollection] = None

    @property
    def configs(self) -> ConfigCollection:
        configs = getattr(self, "_configs", None)
        if configs:
            return configs

        if self.config_collection is None:
            self.config_collection = ConfigCollection.from_github_repo()
        self._configs = self.config_collection
        return self._configs

    def updated_configs(self, bq_project: str, bq_dataset: str) -> List[Config]:
        """
        Return external configs that have been updated/added and
        with associated BigQuery tables being out of date.
        """
        client = bigquery.Client(bq_project)
        job = client.query(
            rf"""
            SELECT
                table_name,
                REGEXP_EXTRACT_ALL(
                    option_value,
                    '.*STRUCT\\(\"last_updated\", \"([^\"]+)\"\\).*'
                ) AS last_updated
            FROM
            {bq_dataset}.INFORMATION_SCHEMA.TABLE_OPTIONS
            WHERE option_name = 'labels' AND table_name LIKE "statistics_%"
            """
        )

        result = list(job.result())

        updated_configs = []

        for config in self.configs.configs:
            seen = False
            table_prefix = bq_normalize_name(config.slug)
            for row in result:
                if not row.table_name.startswith(f"statistics_{table_prefix}"):
                    continue
                seen = True
                if not len(row.last_updated):
                    continue
                table_last_updated = UTC.localize(
                    dt.datetime.utcfromtimestamp(int(row.last_updated[0]))
                )
                if table_last_updated < config.last_modified:
                    updated_configs.append(config)
                    break
            if not seen:
                updated_configs.append(config)

        return updated_configs

    def updated_defaults(self, bq_project: str, bq_dataset: str) -> List[str]:
        """
        Return experiment slugs that are linked to default configs that have
        been updated/added or updated.

        Only return configs for experiments that are currently live.
        """
        client = bigquery.Client(bq_project)
        job = client.query(
            rf"""
            WITH live_experiments AS (
                SELECT
                    normandy_slug,
                    app_name
                FROM
                `{bq_project}.monitoring.experimenter_experiments_v1`
                WHERE status = 'Live'
                AND start_date IS NOT NULL
                AND (end_date IS NULL OR end_date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 DAY))
                AND start_date > DATE('2022-05-18')
            )
            SELECT
                table_name,
                app_name,
                normandy_slug,
                REGEXP_EXTRACT_ALL(
                    option_value,
                    '.*STRUCT\\(\"last_updated\", \"([^\"]+)\"\\).*'
                ) AS last_updated
            FROM
            {bq_dataset}.INFORMATION_SCHEMA.TABLE_OPTIONS
            JOIN live_experiments
            ON table_name LIKE CONCAT("%statistics_", REPLACE(normandy_slug, "-", "_"), "%")
            WHERE option_name = 'labels'
            """
        )

        result = list(job.result())

        updated_experiments = []

        for default_config in self.configs.defaults:
            app_name = default_config.slug
            for row in result:
                if row.app_name != app_name:
                    continue
                if not len(row.last_updated):
                    continue
                table_last_updated = UTC.localize(
                    dt.datetime.utcfromtimestamp(int(row.last_updated[0]))
                )
                if table_last_updated < default_config.last_modified:
                    updated_experiments.append(row.normandy_slug)

        return list(set(updated_experiments))

    def spec_for_experiment(self, slug: str) -> Optional[AnalysisSpec]:
        """Return the spec for a specific experiment."""
        for config in self.configs.configs:
            if config.slug == slug:
                return copy.deepcopy(config.spec)

        return None

    def get_outcome(self, outcome_slug: str, app_name: str) -> Optional[Outcome]:
        """Return the outcome matching the specified slug."""
        for outcome in self.configs.outcomes:
            if outcome.slug == outcome_slug and app_name == outcome.platform:
                return copy.deepcopy(outcome)

        return None

    def get_data_source(self, data_source_slug: str, app_name: str) -> Optional[DataSource]:
        """Return the data source matching the specified slug."""
        data_source_definition = self.configs.get_data_source_definition(data_source_slug, app_name)
        if data_source_definition is None:
            raise Exception(f"Could not find definition for data source {data_source_slug}")

        return DataSource(
            name=data_source_definition.name,
            from_expression=data_source_definition.from_expression,
            client_id_column=data_source_definition.client_id_column,
            submission_date_column=data_source_definition.submission_date_column,
            experiments_column_type=None
            if data_source_definition.experiments_column_type == "none"
            else data_source_definition.experiments_column_type,
            default_dataset=data_source_definition.default_dataset,
        )


ConfigLoader = _ConfigLoader()
