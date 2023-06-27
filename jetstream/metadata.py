import datetime as dt
import logging

import google.cloud.storage as storage
from metric_config_parser.analysis import AnalysisConfiguration
from mozilla_nimbus_schemas.jetstream import ExternalConfig as ExternalConfigMetadata
from mozilla_nimbus_schemas.jetstream import Metadata
from mozilla_nimbus_schemas.jetstream import Metric as MetricsMetadata
from mozilla_nimbus_schemas.jetstream import Outcome as OutcomeMetadata

from jetstream import bq_normalize_name
from jetstream.config import METRIC_HUB_REPO, ConfigLoader

logger = logging.getLogger(__name__)


class ExperimentMetadata(Metadata):
    class Config:
        json_encoders = {
            dt.date: lambda d: d.strftime("%Y-%m-%d"),
            dt.datetime: lambda dt: str(dt),
        }

    @classmethod
    def from_config(
        cls,
        config: AnalysisConfiguration,
        analysis_start_time: dt.datetime = None,
        config_loader=ConfigLoader,
    ) -> "ExperimentMetadata":
        all_metrics = [
            summary.metric for _, summaries in config.metrics.items() for summary in summaries
        ]

        metrics_metadata = {
            metric.name: MetricsMetadata(
                friendly_name=metric.friendly_name or metric.name.replace("_", " ").title(),
                description=metric.description or "",
                bigger_is_better=metric.bigger_is_better,
                analysis_bases=[a.value for a in metric.analysis_bases],
            )
            for metric in all_metrics
        }

        outcomes = [
            config_loader.get_outcome(experiment_outcome, config.experiment.app_name)
            for experiment_outcome in config.experiment.outcomes
        ]

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
            for external_outcome in outcomes
            if external_outcome is not None
        }

        # determine parameters that have been overridden by external config in jetstream-config
        external_config = None
        if config.experiment.has_external_config_overrides():
            external_config = ExternalConfigMetadata(
                reference_branch=config.experiment.reference_branch
                if config.experiment.reference_branch
                != config.experiment.experiment.reference_branch
                else None,
                end_date=config.experiment.end_date.date()
                if config.experiment.end_date is not None
                and config.experiment.end_date != config.experiment.experiment.end_date
                else None,
                start_date=config.experiment.start_date.date()
                if config.experiment.start_date is not None
                and config.experiment.start_date != config.experiment.experiment.start_date
                else None,
                enrollment_period=config.experiment.enrollment_period
                if config.experiment.enrollment_period
                != config.experiment.experiment.proposed_enrollment
                else None,
                skip=config.experiment.skip,
                url=METRIC_HUB_REPO
                + "/blob/main/jetstream/"
                + config.experiment.normandy_slug
                + ".toml",
            )

        return cls(
            metrics=metrics_metadata,
            outcomes=outcomes_metadata,
            external_config=external_config,
            analysis_start_time=analysis_start_time,
        )


def export_metadata(
    config: AnalysisConfiguration,
    bucket_name: str,
    project_id: str,
    analysis_start_time: dt.datetime = None,
):
    """Export experiment metadata to GCS."""
    if config.experiment.normandy_slug is None:
        return

    # do not export metadata for confidential experiments
    if config.experiment.is_private:
        return

    metadata = ExperimentMetadata.from_config(config, analysis_start_time)

    storage_client = storage.Client(project_id)
    bucket = storage_client.get_bucket(bucket_name)
    target_file = f"metadata_{bq_normalize_name(config.experiment.normandy_slug)}"
    target_path = "metadata"
    blob = bucket.blob(f"{target_path}/{target_file}.json")

    logger.info(f"Uploading {target_file} to {bucket_name}/{target_path}.")

    blob.upload_from_string(
        data=metadata.json(),
        content_type="application/json",
    )
