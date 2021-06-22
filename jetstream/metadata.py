import json
import logging
from typing import Dict, List, Optional

import attr
import cattr
import google.cloud.storage as storage

from jetstream import bq_normalize_name, outcomes
from jetstream.config import AnalysisConfiguration
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
    commit_hash: Optional[str]


@attr.s(auto_attribs=True)
class ExperimentMetadata:
    metrics: Dict[str, MetricsMetadata]
    outcomes: Dict[str, OutcomeMetadata]
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
                commit_hash=external_outcome.commit_hash,
            )
            for experiment_outcome in config.experiment.outcomes
            for _, external_outcome in all_outcomes.items()
            if external_outcome.slug == experiment_outcome
        }

        return cls(metrics=metrics_metadata, outcomes=outcomes_metadata)


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
    blob.upload_from_string(
        data=json.dumps(converter.unstructure(metadata), sort_keys=True, indent=4),
        content_type="application/json",
    )
