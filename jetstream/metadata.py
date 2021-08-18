import datetime as dt
import json
import logging
from typing import Callable, Dict, List, Optional

import attr
import cattr
import google.cloud.storage as storage

from jetstream import bq_normalize_name, outcomes
from jetstream.config import AnalysisConfiguration
from jetstream.external_config import ExternalConfigCollection
from jetstream.statistics import StatisticResult

logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True)
class MetricsMetadata:
    friendly_name: str
    description: str
    bigger_is_better: bool
    analysis_bases: List[str]


@attr.s(auto_attribs=True)
class OutcomeMetadata:
    slug: str
    friendly_name: str
    description: str
    metrics: List[str]
    default_metrics: List[str]
    commit_hash: Optional[str]


@attr.s(auto_attribs=True)
class ExternalConfigMetadata:
    reference_branch: Optional[str]
    end_date: Optional[dt.datetime]
    start_date: Optional[dt.datetime]
    enrollment_period: Optional[int]
    skip: Optional[bool]
    url: str


@attr.s(auto_attribs=True)
class ExperimentMetadata:
    metrics: Dict[str, MetricsMetadata]
    outcomes: Dict[str, OutcomeMetadata]
    external_config: Optional[ExternalConfigMetadata]
    schema_version: int = StatisticResult.SCHEMA_VERSION

    @classmethod
    def from_config(cls, config: AnalysisConfiguration) -> "ExperimentMetadata":
        all_metrics = [
            summary.metric for period, summaries in config.metrics.items() for summary in summaries
        ]

        metrics_metadata = {
            metric.name: MetricsMetadata(
                friendly_name=metric.friendly_name or "",
                description=metric.description or "",
                bigger_is_better=metric.bigger_is_better,
                analysis_bases=[a.value for a in metric.analysis_bases],
            )
            for metric in all_metrics
        }

        all_outcomes = outcomes.OutcomesResolver.data

        outcomes_metadata = {
            external_outcome.slug: OutcomeMetadata(
                slug=external_outcome.slug,
                friendly_name=external_outcome.spec.friendly_name,
                description=external_outcome.spec.description,
                metrics=[m for m, _ in external_outcome.spec.metrics.items()],
                default_metrics=[m.name for m in external_outcome.spec.default_metrics]
                if external_outcome.spec.default_metrics
                else [],
                commit_hash=external_outcome.commit_hash,
            )
            for experiment_outcome in config.experiment.outcomes
            for _, external_outcome in all_outcomes.items()
            if external_outcome.slug == experiment_outcome
        }

        # determine parameters that have been overridden by external config in jetstream-config
        external_config = None
        if config.experiment.has_external_config_overrides():
            external_config = ExternalConfigMetadata(
                reference_branch=config.experiment.reference_branch
                if config.experiment.reference_branch
                != config.experiment.experimenter_experiment.reference_branch
                else None,
                end_date=config.experiment.end_date
                if config.experiment.end_date != config.experiment.experimenter_experiment.end_date
                else None,
                start_date=config.experiment.start_date
                if config.experiment.start_date
                != config.experiment.experimenter_experiment.start_date
                else None,
                enrollment_period=config.experiment.proposed_enrollment
                if config.experiment.proposed_enrollment
                != config.experiment.experimenter_experiment.proposed_enrollment
                else None,
                skip=config.experiment.skip,
                url=ExternalConfigCollection.JETSTREAM_CONFIG_URL
                + "/blob/main/"
                + config.experiment.normandy_slug
                + ".toml",
            )

        return cls(
            metrics=metrics_metadata,
            outcomes=outcomes_metadata,
            external_config=external_config,
        )


def export_metadata(config: AnalysisConfiguration, bucket_name: str, project_id: str):
    """Export experiment metadata to GCS."""
    if config.experiment.normandy_slug is None:
        return

    metadata = ExperimentMetadata.from_config(config)

    storage_client = storage.Client(project_id)
    bucket = storage_client.get_bucket(bucket_name)
    target_file = f"metadata_{bq_normalize_name(config.experiment.normandy_slug)}"
    target_path = "metadata"
    blob = bucket.blob(f"{target_path}/{target_file}.json")

    logger.info(f"Uploading {target_file} to {bucket_name}/{target_path}.")

    converter = cattr.Converter()
    _datetime_to_json: Callable[[dt.datetime], str] = lambda dt: dt.strftime("%Y-%m-%d")
    converter.register_unstructure_hook(dt.datetime, _datetime_to_json)

    blob.upload_from_string(
        data=json.dumps(converter.unstructure(metadata), sort_keys=True, indent=4),
        content_type="application/json",
    )
