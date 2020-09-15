from datetime import datetime, timedelta

import click
import os
import logging
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
from .logging.bigquery_log_handler import BigQueryLogHandler
from .bigquery_client import BigQueryClient
from . import bq_normalize_name

DEFAULT_METRICS_CONFIG = Path(__file__).parent / "config" / "default_metrics.toml"
CFR_METRICS_CONFIG = Path(__file__).parent / "config" / "cfr_metrics.toml"


def setup_logger(
    log_project_id, log_dataset_id, log_table_id, log_to_bigquery, client=None, capacity=50
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(asctime)s:%(name)s:%(message)s",
    )
    logger = logging.getLogger()

    if log_to_bigquery:
        bigquery_handler = BigQueryLogHandler(
            log_project_id, log_dataset_id, log_table_id, client, capacity
        )
        bigquery_handler.setLevel(logging.WARNING)
        logger.addHandler(bigquery_handler)


logger = logging.getLogger(__name__)


@click.group()
@click.option(
    "--log_project_id",
    "--log-project-id",
    default="moz-fx-data-experiments",
    help="GCP project to write logs to",
)
@click.option(
    "--log_dataset_id",
    "--log-dataset-id",
    default="monitoring",
    help="Dataset to write logs to",
)
@click.option("--log_table_id", "--log-table-id", default="logs", help="Table to write logs to")
@click.option("--log_to_bigquery", "--log-to-bigquery", is_flag=True, default=False)
def cli(log_project_id, log_dataset_id, log_table_id, log_to_bigquery):
    setup_logger(log_project_id, log_dataset_id, log_table_id, log_to_bigquery)


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
@secret_config_file_option
def run(project_id, dataset_id, date, experiment_slug, config_file):
    """Fetches experiments from Experimenter and runs analysis on active experiments."""
    # fetch experiments that are still active
    collection = ExperimentCollection.from_experimenter()

    active_experiments = collection.end_on_or_after(date).of_type(
        ("pref", "addon", "message", "v4")
    )

    if experiment_slug is not None:
        # run analysis for specific experiment
        active_experiments = active_experiments.with_slug(experiment_slug)

    # get experiment-specific external configs
    external_configs = ExternalConfigCollection.from_github_repo()

    # calculate metrics for experiments and write to BigQuery
    for experiment in active_experiments.experiments:
        try:
            spec = default_spec_for_experiment(experiment)

            if config_file:
                # secret CLI configs overwrite external configs
                custom_spec = AnalysisSpec.from_dict(toml.load(config_file))
                spec.merge(custom_spec)
            else:
                external_experiment_config = external_configs.spec_for_experiment(
                    experiment.normandy_slug
                )

                if external_experiment_config:
                    spec.merge(external_experiment_config)

            config = spec.resolve(experiment)

            Analysis(project_id, dataset_id, config).run(date)
        except Exception as e:
            logger.exception(str(e), exc_info=e, extra={"experiment": experiment.normandy_slug})


@cli.command("rerun")
@experiment_slug_option
@project_id_option
@dataset_id_option
@secret_config_file_option
def rerun(project_id, dataset_id, experiment_slug, config_file):
    """Rerun all available analyses for a specific experiment."""
    collection = ExperimentCollection.from_experimenter()

    try:
        experiments = collection.with_slug(experiment_slug)

        if experiment_slug is None or len(experiments.experiments) == 0:
            raise Exception(f"No experiment with slug {experiment_slug} found.")

        experiment = experiments.experiments[0]

        if experiment.end_date is None:
            raise Exception(f"End date is missing for experiment {experiment_slug}")

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
            external_experiment_config = external_configs.spec_for_experiment(
                experiment.normandy_slug
            )

            if external_experiment_config:
                spec.merge(external_experiment_config)

        config = spec.resolve(experiment)

        # delete all tables previously created when this experiment was analysed
        client = BigQueryClient(project_id, dataset_id)
        normalized_slug = bq_normalize_name(experiment.normandy_slug)
        table_name_re = f"(statistics_)?{normalized_slug}.*"
        client.delete_tables_matching_regex(table_name_re)

        for date in inclusive_date_range(experiment.start_date, end_date):
            logger.info(f"*** {date}")
            Analysis(project_id, dataset_id, config).run(date)
    except Exception as e:
        logger.exception(str(e), exc_info=e, extra={"experiment": experiment_slug})


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
@click.pass_context
def rerun_config_changed(ctx, project_id, dataset_id):
    """Rerun all available analyses for experiments with new or updated config files."""
    # get experiment-specific external configs
    external_configs = ExternalConfigCollection.from_github_repo()

    updated_external_configs = external_configs.updated_configs(project_id, dataset_id)
    for external_config in updated_external_configs:
        ctx.invoke(
            rerun,
            project_id=project_id,
            dataset_id=dataset_id,
            experiment_slug=external_config.slug,
        )


@cli.command("validate_config")
@click.argument("path", type=click.Path(exists=True), nargs=-1)
def validate_config(path):
    """Validate config files."""
    config_files = [p for p in path if os.path.isfile(p)]

    collection = ExperimentCollection.from_experimenter()

    for file in config_files:
        click.echo(f"Validate {file}", err=False)

        custom_spec = AnalysisSpec.from_dict(toml.load(file))

        # check if there is an experiment with a matching slug in Experimenter
        slug = os.path.splitext(os.path.basename(file))[0]
        if (experiments := collection.with_slug(slug).experiments) == []:
            click.echo(f"No experiment with slug {slug} in Experimenter.", err=True)
            sys.exit(1)

        spec = default_spec_for_experiment(experiments[0])
        spec.merge(custom_spec)
        conf = spec.resolve(experiments[0])
        Analysis("no project", "no dataset", conf).validate()

        click.echo(f"Config file at {file} is valid.", err=False)
