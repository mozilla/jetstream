import json
import logging
from typing import Dict, List, Optional, Union

import attr
import cattr
import google.cloud.storage as storage

from jetstream import STATISTICS_SCHEMA_VERSION, bq_normalize_name, outcomes
from jetstream.config import AnalysisConfiguration
from jetstream.statistics import StatisticResult

logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True)
class MetricsMetadata:
    friendly_name: Optional[str]
    description: Optional[str]
    bigger_is_better: bool
    analysis_basis: Union[str, List[str]]


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
        all_metrics_distinct = list(
            set(all_metrics)
        )  # some metrics are used in multiple analysis periods

        metrics_metadata = {
            metric.name: MetricsMetadata(
                friendly_name=metric.friendly_name,
                description=metric.description,
                bigger_is_better=metric.bigger_is_better,
                analysis_basis=[a.value for a in metric.analysis_basis]
                if isinstance(metric.analysis_basis, list)
                else metric.analysis_basis.value,
            )
            for metric in all_metrics_distinct
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
