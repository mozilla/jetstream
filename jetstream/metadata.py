from google.cloud import storage
from typing import Dict, List

import attr
import cattr
import logging

from jetstream import bq_normalize_name
from jetstream.config import AnalysisConfiguration

logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True)
class MetricsMetadata:
    friendly_name: str
    description: str
    bigger_is_better: bool


@attr.s(auto_attribs=True)
class ExperimentMetadata:
    metrics: Dict[str, MetricsMetadata]
    probesets: Dict[str, List[str]]

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
            )
            for metric in all_metrics_distinct
        }

        probesets_metadata = {
            probe_set.name: [probes.name for probes in probe_set.probes]
            for probe_set in config.experiment.probe_sets
        }

        return cls(metrics=metrics_metadata, probesets=probesets_metadata)


def export_metadata(config: AnalysisConfiguration, bucket_name: str):
    """Export experiment metadata to GCS."""
    if config.experiment.normandy_slug is None:
        return

    metadata = ExperimentMetadata.from_config(config)

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    target_file = f"metadata_{bq_normalize_name(config.experiment.normandy_slug)}"
    target_path = "metadata"
    blob = bucket.blob(f"{target_path}/{target_file}.json")

    logger.info(f"Uploading {target_file} to {bucket_name}/{target_path}.")

    converter = cattr.Converter()
    blob.upload_from_string(data=converter.unstructure(metadata), content_type="application/json")
