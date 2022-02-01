"""
Retrieves external configuration files for specific experiments.

Experiment-specific configuration files are stored in https://github.com/mozilla/jetstream-config/
"""

import datetime as dt
from pathlib import Path
from typing import Any, List, MutableMapping, Optional, Union

import attr
import toml
from git import Repo
from google.cloud import bigquery
from pytz import UTC

import jetstream.experimenter
from jetstream.analysis import Analysis
from jetstream.config import PLATFORM_CONFIGS, AnalysisSpec, OutcomeSpec
from jetstream.errors import (
    MetricsConfigurationException,
    SegmentsConfigurationException,
    UnexpectedKeyConfigurationException,
)
from jetstream.util import TemporaryDirectory

from . import bq_normalize_name

OUTCOMES_DIR = "outcomes"


@attr.s(auto_attribs=True)
class ExternalConfig:
    """Represent an external config file."""

    slug: str
    spec: AnalysisSpec
    last_modified: dt.datetime

    def validate(self, experiment: jetstream.experimenter.Experiment) -> None:
        spec = AnalysisSpec.default_for_experiment(experiment)
        spec.merge(self.spec)
        conf = spec.resolve(experiment)
        Analysis("no project", "no dataset", conf).validate()


def check_all_keys_lowercase(config: MutableMapping[str, Any]) -> List[str]:
    """
    Recursively check that configuration keys are lowercase.

        Returns a dictionary with a dict key path to key which is invalid.
    """

    invalid_keys = list()
    for key, value in config.items():
        if isinstance(value, dict):
            invalid_keys.extend(check_all_keys_lowercase(value))

        if key.lower() != key:
            only_uppercase_letters = list(filter(str.isupper, key))
            if (
                "".join(sorted(set(only_uppercase_letters), key=only_uppercase_letters.index))
                == "GB"
            ):
                # handling rare occasion where uppercase is used:
                # example: https://github.com/mozilla/jetstream-config/blob/main/bug-1722551-pref-full-js-parsing-experiment-nightly-94-94.toml#L803  # noqa: E501
                continue

            invalid_keys.append(key)

    return invalid_keys

def validate_config_settings(config_file: Path) -> None:
    """
    Implemented to resolve a Github issue:
        - https://github.com/mozilla/jetstream/issues/843

    Loads external config file and runs a number of validation steps on it:
    - Checks that all config keys/settings are lowercase
    - Checks for missing core config keys
    - Checks for unexpected core configuration keys
    - Checks that all segments defined under experiment have configuration in segments section
    - Checks if metric with custom config is defined in metrics.weekly or metrics.overall fields

    Returns None, if issues found with the configuration an Exception is raised
    """

    config = toml.loads(config_file.read_text())

    # if config_errors := check_all_keys_lowercase(config):
    #     err_msg = (
    #         "It appears some config settings/keys"
    #         f"contain unexpected uppercase letters: {config_errors}"
    #     )
    #     raise WrongCaseConfigurationException(err_msg)

    optional_core_config_keys = (
        "metrics",
        "experiment",
        "segments",
        "data_sources",
        "friendly_name",
        "description",
    )  # TODO: double check that friendly_name and description are to be expected here

    core_config_keys_specified = config.keys()

    # checks for unexpected core configuration keys
    if unexpected_config_keys := (set(core_config_keys_specified) - set(optional_core_config_keys)):
        err_msg = f"Unexpected config key[s] found: {unexpected_config_keys}"
        raise UnexpectedKeyConfigurationException(err_msg)

    # checks that all segments defined under experiment have configuration in segments section
    if experiment_config := config.get("experiment", dict()):
        expected_segment_configuration = experiment_config.get("segments", list())

        if expected_segment_configuration:
            segments_config_keys = config.get("segments", dict()).keys()

            # TODO: will desktop include all possible segments?
            import mozanalysis.segments.desktop as desktop_segments

            mozanalysis_segments = [
                val
                for val in desktop_segments.__dict__.keys()
                if not (val.startswith("_") or val.startswith("Segment"))
            ]

            if missing_segment_configuration := (
                set(expected_segment_configuration)
                - set.union(set(segments_config_keys), set(mozanalysis_segments))
            ):
                docs_url = "https://experimenter.info/jetstream/configuration#defining-segments"
                err_msg = (
                    "It appears some configuration for specified "
                    f"segments is missing: {missing_segment_configuration}. "
                    f"Please refer to {docs_url} for Segments configuration guide."
                )
                raise SegmentsConfigurationException(err_msg)

    # Checks if metric with custom config is defined in metrics.weekly or metrics.overall fields
    if metrics_config := config.get("metrics"):
        daily_metrics = metrics_config.get("daily", list())
        weekly_metrics = metrics_config.get("weekly", list())
        overall_metrics = metrics_config.get("overall", list())

        specified_metrics = list(
            set.union(set(weekly_metrics), set(overall_metrics), set(daily_metrics))
        )
        configured_metrics = list(set(metrics_config.keys()) - set(["daily", "weekly", "overall"]))

        # TODO: will desktop include all possible metrics?
        import mozanalysis.metrics.desktop as desktop_metrics

        mozanalysis_metrics = [
            val
            for val in desktop_metrics.__dict__.keys()
            if not (val.startswith("_") or val.startswith("Metric"))
        ]

        if metrics_missing_configuration := (
            set(configured_metrics) - set.union(set(specified_metrics), set(mozanalysis_metrics))
        ):
            err_msg = (
                "It appears some configuration for specified "
                f"metrics is missing: {metrics_missing_configuration}"
            )
            raise MetricsConfigurationException(err_msg)

    return None
def check_all_keys_lowercase(config: dict) -> dict:
    """
    Recursively check that configuration keys are lowercase.

        Returns a dictionary with a dict key path to key which is invalid.
    """

    invalid_keys = list()
    for key, value in config.items():
        if isinstance(value, dict):
            invalid_keys.extend(check_all_keys_lowercase(value))

        if key.lower() != key:
            only_uppercase_letters = list(filter(str.isupper, key))
            if (
                "".join(sorted(set(only_uppercase_letters), key=only_uppercase_letters.index))
                == "GB"
            ):
                # handling rare occasion where uppercase is used:
                # example: https://github.com/mozilla/jetstream-config/blob/main/bug-1722551-pref-full-js-parsing-experiment-nightly-94-94.toml#L803  # noqa: E501
                continue

            invalid_keys.append(key)

    return invalid_keys

def validate_config_settings(config_file: Path) -> None:
    """
    Implemented to resolve a Github issue:
        - https://github.com/mozilla/jetstream/issues/843

    Loads external config file and runs a number of validation steps on it:
    - Checks that all config keys/settings are lowercase
    - Checks for missing core config keys
    - Checks for unexpected core configuration keys
    - Checks that all segments defined under experiment have configuration in segments section
    - Checks if metric with custom config is defined in metrics.weekly or metrics.overall fields

    Returns None, if issues found with the configuration an Exception is raised
    """

    config = toml.loads(config_file.read_text())

    # if config_errors := check_all_keys_lowercase(config):
    #     err_msg = (
    #         "It appears some config settings/keys"
    #         f"contain unexpected uppercase letters: {config_errors}"
    #     )
    #     raise WrongCaseConfigurationException(err_msg)

    optional_core_config_keys = (
        "metrics",
        "experiment",
        "segments",
        "data_sources",
        "friendly_name",
        "description",
    )  # TODO: double check that friendly_name and description are to be expected here

    core_config_keys_specified = config.keys()

    # checks for unexpected core configuration keys
    if unexpected_config_keys := (set(core_config_keys_specified) - set(optional_core_config_keys)):
        err_msg = f"Unexpected config key[s] found: {unexpected_config_keys}"
        raise UnexpectedKeyConfigurationException(err_msg)

    # checks that all segments defined under experiment have configuration in segments section
    if experiment_config := config.get("experiment", dict()):
        expected_segment_configuration = experiment_config.get("segments", list())

        if expected_segment_configuration:
            segments_config_keys = config.get("segments", dict()).keys()

            # TODO: will desktop include all possible segments?
            import mozanalysis.segments.desktop as desktop_segments

            mozanalysis_segments = [
                val
                for val in desktop_segments.__dict__.keys()
                if not (val.startswith("_") or val.startswith("Segment"))
            ]

            if missing_segment_configuration := (
                set(expected_segment_configuration)
                - set.union(set(segments_config_keys), set(mozanalysis_segments))
            ):
                docs_url = "https://experimenter.info/jetstream/configuration#defining-segments"
                err_msg = (
                    "It appears some configuration for specified "
                    f"segments is missing: {missing_segment_configuration}. "
                    f"Please refer to {docs_url} for Segments configuration guide."
                )
                raise SegmentsConfigurationException(err_msg)

    # Checks if metric with custom config is defined in metrics.weekly or metrics.overall fields
    if metrics_config := config.get("metrics"):
        daily_metrics = metrics_config.get("daily", list())
        weekly_metrics = metrics_config.get("weekly", list())
        overall_metrics = metrics_config.get("overall", list())

        specified_metrics = list(
            set.union(set(weekly_metrics), set(overall_metrics), set(daily_metrics))
        )
        configured_metrics = list(set(metrics_config.keys()) - set(["daily", "weekly", "overall"]))

        # TODO: will desktop include all possible metrics?
        import mozanalysis.metrics.desktop as desktop_metrics

        mozanalysis_metrics = [
            val
            for val in desktop_metrics.__dict__.keys()
            if not (val.startswith("_") or val.startswith("Metric"))
        ]

        if metrics_missing_configuration := (
            set(configured_metrics) - set.union(set(specified_metrics), set(mozanalysis_metrics))
        ):
            err_msg = (
                "It appears some configuration for specified "
                f"metrics is missing: {metrics_missing_configuration}"
            )
            raise MetricsConfigurationException(err_msg)

    return None


@attr.s(auto_attribs=True)
class ExternalOutcome:
    """Represents an external outcome snippet."""

    slug: str
    spec: OutcomeSpec
    platform: str
    commit_hash: Optional[str]

    def validate(self) -> None:
        if self.platform not in PLATFORM_CONFIGS:
            raise ValueError(f"Platform '{self.platform}' is unsupported.")
        app_id = PLATFORM_CONFIGS[self.platform].app_id
        dummy_experiment = jetstream.experimenter.Experiment(
            experimenter_slug="dummy-experiment",
            normandy_slug="dummy_experiment",
            type="v6",
            status="Live",
            branches=[],
            end_date=None,
            reference_branch="control",
            is_high_population=False,
            start_date=dt.datetime.now(UTC),
            proposed_enrollment=14,
            app_id=app_id,
            app_name=self.platform,  # seems to be unused
        )
        spec = AnalysisSpec.default_for_experiment(dummy_experiment)
        spec.merge_outcome(self.spec)
        conf = spec.resolve(dummy_experiment)
        Analysis("no project", "no dataset", conf).validate()


def entity_from_path(path: Path) -> Union[ExternalConfig, ExternalOutcome]:
    is_outcome = path.parent.parent.name == OUTCOMES_DIR
    slug = path.stem

    validate_config_settings(path)

    config_dict = toml.loads(path.read_text())

    if is_outcome:
        platform = path.parent.name
        spec = OutcomeSpec.from_dict(config_dict)
        return ExternalOutcome(slug=slug, spec=spec, platform=platform, commit_hash=None)
    return ExternalConfig(
        slug=slug,
        spec=AnalysisSpec.from_dict(config_dict),
        last_modified=dt.datetime.fromtimestamp(path.stat().st_mtime, UTC),
    )


@attr.s(auto_attribs=True)
class ExternalConfigCollection:
    """
    Collection of experiment-specific configurations pulled in
    from an external GitHub repository.
    """

    configs: List[ExternalConfig] = attr.Factory(list)
    outcomes: List[ExternalOutcome] = attr.Factory(list)

    JETSTREAM_CONFIG_URL = "https://github.com/mozilla/jetstream-config"

    @classmethod
    def from_github_repo(cls) -> "ExternalConfigCollection":
        """Pull in external config files."""
        # download files to tmp directory
        with TemporaryDirectory() as tmp_dir:
            repo = Repo.clone_from(cls.JETSTREAM_CONFIG_URL, tmp_dir)

            external_configs = []

            config_files_to_skip = (
                "bug-1695015-pref-new-tab-modernized-ux-region-1-release-86-88.toml",
                "bug-1671484-pref-validation-of-relpreload-performance-impact-release-82-83.toml",
                "bug-1726656-pref-tab-unloading-nightly-93-94.toml",
            )  # TODO: temporary, remove once resolved issue with this specific config

            for config_file in tmp_dir.glob("*.toml"):

                last_modified = next(repo.iter_commits("main", paths=config_file)).committed_date

                if (
                    str(config_file).split("/")[-1] not in config_files_to_skip
                ):  # TODO: temporary, remove once resolved issue with this specific config
                    validate_config_settings(config_file)

                external_configs.append(
                    ExternalConfig(
                        config_file.stem,
                        AnalysisSpec.from_dict(toml.load(config_file)),
                        UTC.localize(dt.datetime.utcfromtimestamp(last_modified)),
                    )
                )

            outcomes = []

            for outcome_file in tmp_dir.glob(f"**/{OUTCOMES_DIR}/*/*.toml"):
                commit_hash = next(repo.iter_commits("main", paths=outcome_file)).hexsha

                outcomes.append(
                    ExternalOutcome(
                        slug=outcome_file.stem,
                        spec=OutcomeSpec.from_dict(toml.load(outcome_file)),
                        platform=outcome_file.parent.name,
                        commit_hash=commit_hash,
                    )
                )

        return cls(external_configs, outcomes)

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
            rf"""
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
            seen = False
            table_prefix = bq_normalize_name(config.slug)
            for row in result:
                if not row.table_name.startswith(table_prefix):
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
