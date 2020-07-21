from datetime import datetime, timedelta
import logging

import click
from pathlib import Path
import pytz
import sys
import toml

from . import experimenter
from .config import AnalysisSpec
from .experimenter import ExperimentCollection
from .export_json import export_statistics_tables
from .analysis import Analysis
from .external_config import ExternalConfigCollection


DEFAULT_METRICS_CONFIG = Path(__file__).parent / "config" / "default_metrics.toml"
CFR_METRICS_CONFIG = Path(__file__).parent / "config" / "cfr_metrics.toml"


@click.group()
def cli():
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(asctime)s:%(name)s:%(message)s",
    )


def inclusive_date_range(start_date, end_date):
    """Generator for a range of dates, includes end_date."""
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


def default_spec_for_experiment(experiment: experimenter.Experiment) -> AnalysisSpec:
    default_metrics = AnalysisSpec.from_dict(toml.load(DEFAULT_METRICS_CONFIG))

    if experiment.type == "message":
        # CFR experiment
        cfr_metrics = AnalysisSpec.from_dict(toml.load(CFR_METRICS_CONFIG))
        default_metrics.merge(cfr_metrics)
        return default_metrics

    return default_metrics


class ClickDate(click.ParamType):
    name = "date"

    def convert(self, value, param, ctx):
        if isinstance(value, datetime):
            return value
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=pytz.utc)


project_id_option = click.option(
    "--project_id", "--project-id", default="moz-fx-data-experiments", help="Project to write to"
)
dataset_id_option = click.option(
    "--dataset_id", "--dataset-id", default="mozanalysis", help="Dataset to write to"
)
dry_run_option = click.option(
    "--dry_run/--no_dry_run", help="Don't publish any changes to BigQuery"
)

experiment_slug_option = click.option(
    "--experiment_slug",
    "--experiment-slug",
    help="Experimenter or Normandy slug of the experiment to (re)run analysis for",
)

secret_config_file_option = click.option(
    "--i-solemnly-swear-i-am-up-to-no-good", "config_file", type=click.File("rt"), hidden=True
)

bucket_option = click.option("--bucket", default="mozanalysis", help="GCS bucket to write to")


@cli.command()
@project_id_option
@dataset_id_option
@click.option(
    "--date",
    type=ClickDate(),
    help="Date for which experiments should be analyzed",
    metavar="YYYY-MM-DD",
    required=True,
)
@experiment_slug_option
@dry_run_option
@secret_config_file_option
def run(project_id, dataset_id, date, experiment_slug, dry_run, config_file):
    """Fetches experiments from Experimenter and runs analysis on active experiments."""
    # fetch experiments that are still active
    collection = ExperimentCollection.from_experimenter()

    active_experiments = collection.end_on_or_after(date).of_type(("pref", "addon", "message"))

    if experiment_slug is not None:
        # run analysis for specific experiment
        active_experiments = active_experiments.with_slug(experiment_slug)

    # get experiment-specific external configs
    external_configs = ExternalConfigCollection.from_github_repo()

    # calculate metrics for experiments and write to BigQuery
    for experiment in active_experiments.experiments:
        spec = default_spec_for_experiment(experiment)

        if config_file:
            # secret CLI configs overwrite external configs
            custom_spec = AnalysisSpec.from_dict(toml.load(config_file))
            spec.merge(custom_spec)
        else:
            external_experiment_config = external_configs.spec_for_experiment(experiment.slug)

            if external_experiment_config:
                spec.merge(external_experiment_config)

        config = spec.resolve(experiment)
        Analysis(project_id, dataset_id, config).run(date, dry_run=dry_run)


def rerun(project_id, dataset_id, experiment_slug, dry_run, config_file):
    """Rerun all available analyses for a specific experiment."""
    collection = ExperimentCollection.from_experimenter()

    experiments = collection.with_slug(experiment_slug)

    if experiment_slug is None or len(experiments.experiments) == 0:
        click.echo(f"No experiment with slug {experiment_slug} found.", err=True)
        sys.exit(1)

    experiment = experiments.experiments[0]
    end_date = min(
        experiment.end_date,
        datetime.combine(
            datetime.now(tz=pytz.utc).date() - timedelta(days=1),
            datetime.min.time(),
            tzinfo=pytz.utc,
        ),
    )
    spec = default_spec_for_experiment(experiment)
    if config_file:
        custom_spec = AnalysisSpec.from_dict(toml.load(config_file))
        spec.merge(custom_spec)
    else:
        # get experiment-specific external configs
        external_configs = ExternalConfigCollection.from_github_repo()
        external_experiment_config = external_configs.spec_for_experiment(experiment.slug)

        if external_experiment_config:
            spec.merge(external_experiment_config)

    config = spec.resolve(experiment)

    for date in inclusive_date_range(experiment.start_date, end_date):
        logging.info(f"*** {date}")
        Analysis(project_id, dataset_id, config).run(date, dry_run=dry_run)


@cli.command("rerun")
@experiment_slug_option
@project_id_option
@dataset_id_option
@dry_run_option
@secret_config_file_option
def rerun_cmd(project_id, dataset_id, experiment_slug, dry_run, config_file):
    """CLI command for re-running analyses."""
    rerun(project_id, dataset_id, experiment_slug, dry_run, config_file)


@cli.command()
@project_id_option
@dataset_id_option
@bucket_option
def export_statistics_to_json(project_id, dataset_id, bucket):
    """Export all tables as JSON to a GCS bucket."""
    export_statistics_tables(project_id, dataset_id, bucket)


@cli.command("rerun_config_changed")
@project_id_option
@dataset_id_option
def rerun_config_changed(project_id, dataset_id):
    """Rerun all available analyses for experiments with new or updated config files."""
    # get experiment-specific external configs
    external_configs = ExternalConfigCollection.from_github_repo()

    updated_external_configs = external_configs.updated_configs(project_id, dataset_id)
    for external_config in updated_external_configs:
        rerun(project_id, dataset_id, external_config.experimenter_slug, dry_run=False)
