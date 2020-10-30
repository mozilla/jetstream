from datetime import datetime, timedelta

import attr
import base64
import cattr
import click
import json
import os
from pathlib import Path
import logging
import sys
from typing import Callable, Iterable, Mapping, Optional, Protocol, TextIO, Tuple, Type, Union

import attr
import click
import pytz
import toml
from typing import Optional

from .argo import submit_workflow
from .config import AnalysisSpec
from .experimenter import ExperimentCollection, Experiment
from .export_json import export_statistics_tables
from .analysis import Analysis
from .external_config import ExternalConfigCollection
from .logging.bigquery_log_handler import BigQueryLogHandler
from .bigquery_client import BigQueryClient
from .util import inclusive_date_range
from . import bq_normalize_name, AnalysisPeriod


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


RECOGNIZED_EXPERIMENT_TYPES = ("pref", "addon", "message", "v6")


@attr.s
class AllType:
    """Sentinel value for AnalysisExecutor"""


All = AllType()


@attr.s(auto_attribs=True)
class AnalysisExecutor:
    project_id: str
    dataset_id: str
    date: Union[datetime, AllType]
    experiment_slugs: Union[Iterable[str], AllType]
    configuration_map: Mapping[str, TextIO] = attr.ib(factory=dict)
    analysis_class: Type = Analysis

    @staticmethod
    def _today() -> datetime:
        return datetime.combine(
            datetime.now(tz=pytz.utc).date() - timedelta(days=1),
            datetime.min.time(),
            tzinfo=pytz.utc,
        )

    def execute(
        self,
        *,
        experiment_getter: Callable[
            [], ExperimentCollection
        ] = ExperimentCollection.from_experimenter,
        config_getter: Callable[
            [], ExternalConfigCollection
        ] = ExternalConfigCollection.from_github_repo,
        today: Optional[datetime] = None,
    ) -> bool:
        experiments = experiment_getter()
        external_configs = None

        if isinstance(self.experiment_slugs, AllType):
            if isinstance(self.date, AllType):
                raise ValueError("Declining to re-run all experiments for all time.")
            run_experiments = [
                e.normandy_slug
                for e in (
                    experiments.end_on_or_after(self.date)
                    .of_type(RECOGNIZED_EXPERIMENT_TYPES)
                    .experiments
                )
                if e.normandy_slug is not None
            ]
        else:
            run_experiments = list(self.experiment_slugs)

        failed = False

        for slug in run_experiments:
            experiment = experiments.with_slug(slug).experiments[0]
            spec = AnalysisSpec.default_for_experiment(experiment)
            if slug in self.configuration_map:
                config_dict = toml.load(self.configuration_map[slug])
                spec.merge(AnalysisSpec.from_dict(config_dict))
            else:
                external_configs = external_configs or config_getter()
                if external_spec := external_configs.spec_for_experiment(slug):
                    spec.merge(external_spec)

            if self.date == All:
                today = today or self._today()
                end_date = min(
                    experiment.end_date or today,
                    today,
                )
                run_dates = inclusive_date_range(experiment.start_date, end_date)
            else:
                run_dates = [self.date]

            for run_date in run_dates:
                try:
                    experiment = experiments.with_slug(slug).experiments[0]
                    config = spec.resolve(experiment)
                    self.analysis_class(self.project_id, self.dataset_id, config).run(run_date)
                except Exception as e:
                    failed = True
                    logger.exception(str(e), exc_info=e, extra={"experiment": slug})
        return not failed


@attr.s(auto_attribs=True)
class ArgoExecutor:
    project_id: str
    zone: str
    cluster_id: str
    monitor_status: bool
    date: Union[datetime, AllType]
    experiment_slugs: Union[Iterable[str], AllType]

    WORKLFOW_DIR = Path(__file__).parent / "workflows"
    RUN_WORKFLOW = WORKLFOW_DIR / "run.yaml"

    def execute(self) -> bool:
        # todo: handle specific experiment_slug
        # todo: handle date all
        return submit_workflow(
            self.project_id,
            self.zone,
            self.cluster_id,
            RUN_WORKFLOW,
            {"date": self.date.strftime("%Y-%m-%d")},
            monitor_status=self.monitor_status,
        )


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


class ClickDate(click.ParamType):
    name = "date"

    def convert(self, value, param, ctx):
        if isinstance(value, datetime):
            return value
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=pytz.utc)


project_id_option = click.option(
    "--project_id",
    "--project-id",
    default="moz-fx-data-experiments",
    help="Project to write to",
    required=True,
)
dataset_id_option = click.option(
    "--dataset_id", "--dataset-id", default="mozanalysis", help="Dataset to write to", required=True
)
zone_option = click.option(
    "--zone", default="us-central1-a", help="Kubernetes cluster zone", required=True
)
cluster_id_option = click.option(
    "--cluster_id",
    "--cluster-id",
    default="jetstream",
    help="Kubernetes cluster name",
    required=True,
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

argo_option = click.option(
    "--argo", is_flag=True, default=False, help="Run on Kubernetes with Argo"
)

monitor_status_option = click.option(
    "--monitor_status",
    "--monitor-status",
    default=True,
    help="Monitor the status of the Argo workflow",
)


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
@argo_option
@zone_option
@cluster_id_option
@monitor_status_option
def run(
    project_id,
    dataset_id,
    date,
    experiment_slug,
    config_file,
    argo,
    zone,
    cluster_id,
    monitor_status,
):
    """
    Runs analysis on active experiments for the provided date.

    This command is invoked by Airflow. All errors are written to the console and
    BigQuery.
    """
    if argo:
        success = ArgoExecutor(
            project_id=project_id,
            dataset_id=dataset_id,
            date=date,
            zone=zone,
            cluster_id=cluster_id,
            monitor_status=monitor_status,
            experiment_slugs=[experiment_slug] if experiment_slug else All,
        ).execute()
    else:
        # run locally
        success = AnalysisExecutor(
            project_id=project_id,
            dataset_id=dataset_id,
            date=date,
            experiment_slugs=[experiment_slug] if experiment_slug else All,
            configuration_map={experiment_slug: config_file}
            if experiment_slug and config_file
            else {},
        ).execute()

    sys.exit(0 if success else 1)


@cli.command("rerun")
@experiment_slug_option
@project_id_option
@dataset_id_option
@secret_config_file_option
@argo_option
@zone_option
@cluster_id_option
@monitor_status_option
def rerun(
    project_id, dataset_id, experiment_slug, config_file, argo, zone, cluster_id, monitor_status
):
    """
    Rerun all available analyses for a specific experiment.

    This command is invoked after adding new custom configs via jetstream-config.
    If exceptions are thrown during a re-run, Jetstream will return with an error code.
    jetstream-config launches Jetstream on a separate Kubernetes cluster which needs to
    report back to CircleCI whether or not the run was successful.
    """
    if argo:
        success = ArgoExecutor(
            project_id=project_id,
            dataset_id=dataset_id,
            date=All,
            zone=zone,
            cluster_id=cluster_id,
            monitor_status=monitor_status,
            experiment_slugs=[experiment_slug] if experiment_slug else All,
        ).execute()
    else:
        success = AnalysisExecutor(
            project_id=project_id,
            dataset_id=dataset_id,
            date=All,
            experiment_slugs=[experiment_slug],
            configuration_map={experiment_slug: config_file} if config_file else {},
        ).execute()

    BigQueryClient(project_id, dataset_id).touch_tables(experiment_slug)

    sys.exit(0 if success else 1)


@cli.command()
@project_id_option
@dataset_id_option
@bucket_option
def export_statistics_to_json(project_id, dataset_id, bucket):
    """Export all tables as JSON to a GCS bucket."""
    export_statistics_tables(project_id, dataset_id, bucket)


@cli.command()
@project_id_option
@dataset_id_option
@click.pass_context
@argo_option
@zone_option
@cluster_id_option
@monitor_status_option
def rerun_config_changed(ctx, project_id, dataset_id, argo, zone, cluster_id, monitor_status):
    """Rerun all available analyses for experiments with new or updated config files."""
    # get experiment-specific external configs
    external_configs = ExternalConfigCollection.from_github_repo()
    updated_external_configs = external_configs.updated_configs(project_id, dataset_id)

    if argo:
        success = ArgoExecutor(
            project_id=project_id,
            dataset_id=dataset_id,
            date=All,
            zone=zone,
            cluster_id=cluster_id,
            monitor_status=monitor_status,
            experiment_slugs=[config.slug for config in updated_external_configs],
        ).execute()
    else:
        success = AnalysisExecutor(
            project_id=project_id,
            dataset_id=dataset_id,
            date=All,
            experiment_slugs=[config.slug for config in updated_external_configs],
        ).execute()

    client = BigQueryClient(project_id, dataset_id)
    for config in updated_external_configs:
        client.touch_tables(config.slug)

    sys.exit(0 if success else 1)


@cli.command()
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

        spec = AnalysisSpec.default_for_experiment(experiments[0])
        spec.merge(custom_spec)
        conf = spec.resolve(experiments[0])
        Analysis("no project", "no dataset", conf).validate()

        click.echo(f"Config file at {file} is valid.", err=False)


@attr.s(auto_attribs=True)
class AnalysisRunConfig:
    """Run config used to define what should be executed in a experiment analysis step."""

    date: datetime
    slug: str


@cli.command()
@project_id_option
@dataset_id_option
@click.option(
    "--start_date",
    "--start-date",
    type=ClickDate(),
    help="Start date for which experiments should be analyzed. "
    "If not set, analyze for period experiments were active.",
    metavar="YYYY-MM-DD",
)
@click.option(
    "--end_date",
    "--end-date",
    type=ClickDate(),
    help="End date for which experiments should be analyzed. "
    "If not set, analyze for period experiments were active.",
    metavar="YYYY-MM-DD",
)
@click.option(
    "--config_changed",
    "--config_changed",
    is_flag=True,
    help="Only get experiments whose config has changed since the last run.",
    default=False,
)
@experiment_slug_option
@click.pass_context
def get_active_experiments(
    ctx, project_id, dataset_id, start_date, end_date, config_changed, experiment_slug
):
    """
    Get active experiments and their run dates and print them out as JSON. Argo will
    use the JSON as input for running experiment analyses.
    """
    # fetch experiments that are still active
    collection = ExperimentCollection.from_experimenter()

    if experiment_slug:
        collection = collection.with_slug(experiment_slug)

    active_experiments = collection.of_type(("pref", "addon", "message", "v4"))

    # get experiment-specific external configs
    external_configs = ExternalConfigCollection.from_github_repo()

    if config_changed:
        updated_external_configs = external_configs.updated_configs(project_id, dataset_id)
        active_experiments = [
            experiment
            for external_config in updated_external_configs
            for experiment in collection.with_slug(external_config.slug)
        ]

    experiment_runs_configs = []

    for experiment in active_experiments.experiments:
        if experiment.start_date is None or experiment.end_date is None:
            click.echo(
                f"Start date or end date undefined for experiment {experiment.normandy_slug}.",
                err=True,
            )
            continue

        start = max(start_date, experiment.start_date) if start_date else experiment.start_date
        end = min(end_date, experiment.end_date) if end_date else experiment.end_date
        end = min(
            end,
            datetime.combine(
                datetime.now(tz=pytz.utc).date() - timedelta(days=1),
                datetime.min.time(),
                tzinfo=pytz.utc,
            ),
        )

        for date in inclusive_date_range(start, end):
            experiment_runs_configs.append(AnalysisRunConfig(date, experiment.normandy_slug))

    # Write config to stdout which will be read by argo and then used to spawn analysis jobs
    converter = cattr.Converter()
    converter.register_unstructure_hook(datetime, lambda d: str(d.date()))
    json.dump(converter.unstructure(experiment_runs_configs), sys.stdout)
