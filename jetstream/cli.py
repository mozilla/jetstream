from datetime import datetime, timedelta
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


WORKLFOW_DIR = Path(__file__).parent / "workflows"
RERUN_WORKFLOW = WORKLFOW_DIR / "rerun.yaml"
RUN_WORKFLOW = WORKLFOW_DIR / "run.yaml"
RERUN_CONFIG_CHANGED_WORKFLOW = WORKLFOW_DIR / "rerun_config_changed.yaml"


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


class ExecutorStrategy(Protocol):
    project_id: str
    dataset_id: str

    def __init__(self, project_id: str, dataset_id: str, *args, **kwargs) -> None:
        ...

    def execute(self, worklist: Iterable[Tuple[str, AnalysisSpec, datetime]]) -> bool:
        ...


@attr.s(auto_attribs=True)
class SerialExecutorStrategy:
    project_id: str
    dataset_id: str
    analysis_class: Type = Analysis
    experiment_getter: Callable[[], ExperimentCollection] = ExperimentCollection.from_experimenter

    def execute(self, worklist):
        failed = False
        experiments = self.experiment_getter()
        for slug, spec, date in worklist:
            try:
                experiment = experiments.with_slug(slug).experiments[0]
                config = spec.resolve(experiment)
                self.analysis_class(self.project_id, self.dataset_id, config).run(date)
            except Exception as e:
                failed = True
                logger.exception(str(e), exc_info=e, extra={"experiment": slug})
        return not failed


@attr.s(auto_attribs=True)
class AnalysisExecutor:
    project_id: str
    dataset_id: str
    date: Union[datetime, AllType]
    experiment_slugs: Union[Iterable[str], AllType]
    configuration_map: Mapping[str, TextIO] = attr.ib(factory=dict)

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
        strategy: Callable[[str, str], ExecutorStrategy] = SerialExecutorStrategy,
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

        worklist = []

        for slug in run_experiments:
            experiment = experiments.with_slug(slug).experiments[0]
            spec = default_spec_for_experiment(experiment)
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
                worklist.append((slug, spec, run_date))

        return strategy(self.project_id, self.dataset_id).execute(worklist)


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


def _update_tables_last_modified(project_id, dataset_id, normandy_slug):
    """Update the last_updated label of the tables matching the slug."""
    client = BigQueryClient(project_id, dataset_id)
    normalized_slug = bq_normalize_name(normandy_slug)
    analysis_periods = "|".join([p.value for p in AnalysisPeriod])
    table_name_re = f"^(statistics_)?{normalized_slug}_({analysis_periods})_.*$"
    tables = client.tables_matching_regex(table_name_re)

    for table in tables:
        client.add_labels_to_table({"last_updated": client.current_timestamp_label()})


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
def run(project_id, dataset_id, date, experiment_slug, config_file, argo, zone, cluster_id):
    """
    Runs analysis on active experiments for the provided date.

    This command is invoked by Airflow. All errors are written to the console and
    BigQuery. Runs will always return a success code, even if exceptions were
    thrown during some experiment analyses. This ensures that the Airflow task will
    not retry the task and run all of the analyses again.
    """
    AnalysisExecutor(
        project_id=project_id,
        dataset_id=dataset_id,
        date=date,
        experiment_slugs=[experiment_slug] if experiment_slug else All,
        configuration_map={experiment_slug: config_file} if experiment_slug and config_file else {},
    ).execute()

    # todo: uncomment for Argo
    # sys.exit(0 if success else 1)


@cli.command("rerun")
@experiment_slug_option
@project_id_option
@dataset_id_option
@secret_config_file_option
@argo_option
@zone_option
@cluster_id_option
def rerun(project_id, dataset_id, experiment_slug, config_file, argo, zone, cluster_id):
    """
    Rerun all available analyses for a specific experiment.

    This command is invoked after adding new custom configs via jetstream-config.
    If exceptions are thrown during a re-run, Jetstream will return with an error code.
    jetstream-config launches Jetstream on a separate Kubernetes cluster which needs to
    report back to CircleCI whether or not the run was successful.
    """
    if experiment_slug is None:
        click.echo("experiment_slug is required.", err=True)
        sys.exit(1)

    if argo:
        result = submit_workflow(
            project_id,
            zone,
            cluster_id,
            RERUN_WORKFLOW,
            {"experiment_slug": experiment_slug},
            monitor_status=True,
        )
        _update_tables_last_modified(project_id, dataset_id, experiment_slug)
        return result

    # run locally

    collection = ExperimentCollection.from_experimenter()

    try:
        experiments = collection.with_slug(experiment_slug)

        if len(experiments.experiments) == 0:
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
        spec = AnalysisSpec.default_for_experiment(experiment)
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

        for date in inclusive_date_range(experiment.start_date, end_date):
            logger.info(f"*** {date}")
            Analysis(project_id, dataset_id, config).run(date)

        _update_tables_last_modified(project_id, dataset_id, experiment_slug)
    except Exception as e:
        logger.exception(str(e), exc_info=e, extra={"experiment": experiment_slug})


@cli.command()
@project_id_option
@dataset_id_option
@bucket_option
@argo_option
@zone_option
@cluster_id_option
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
def rerun_config_changed(ctx, project_id, dataset_id, argo, zone, cluster_id):
    """Rerun all available analyses for experiments with new or updated config files."""
    # get experiment-specific external configs
    external_configs = ExternalConfigCollection.from_github_repo()

    updated_external_configs = external_configs.updated_configs(project_id, dataset_id)

    if argo:
        result = submit_workflow(
            project_id,
            zone,
            cluster_id,
            RERUN_CONFIG_CHANGED_WORKFLOW,
            {},
            monitor_status=True,
        )

        for external_config in updated_external_configs:
            _update_tables_last_modified(project_id, dataset_id, external_config.slug)

        return result

    # run locally

    # get experiment-specific external configs
    external_configs = ExternalConfigCollection.from_github_repo()
    updated_external_configs = external_configs.updated_configs(project_id, dataset_id)

    AnalysisExecutor(
        project_id=project_id,
        dataset_id=dataset_id,
        date=All,
        experiment_slugs=[config.slug for config in updated_external_configs],
    ).execute()

    client = BigQueryClient(project_id, dataset_id)
    for config in updated_external_configs:
        client.touch_tables(config.slug)

    # sys.exit(0 if success else 1)


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

        spec = AnalysisSpec.default_for_experiment(experiments[0])
        spec.merge(custom_spec)
        conf = spec.resolve(experiments[0])
        Analysis("no project", "no dataset", conf).validate()

        click.echo(f"Config file at {file} is valid.", err=False)


def _base64_encode(d):
    """Base64 encodes an object."""
    converter = cattr.Converter()
    converter.register_unstructure_hook(datetime, lambda d: str(d.date()))
    return base64.b64encode(json.dumps(converter.unstructure(d)).encode()).decode("utf-8")


def _base64_decode(d):
    """Decodes Base64 to a dict."""
    return json.loads(base64.b64decode(d.encode()).decode("utf-8"))


@attr.s(auto_attribs=True)
class AnalysisRunConfig:
    """Run config used to defined what should be executed in a experiment analysis step."""

    date: datetime
    experiment: Experiment
    external_config: Optional[AnalysisSpec]


@cli.command()
@project_id_option
@dataset_id_option
@click.option(
    "--start_date",
    type=ClickDate(),
    help="Start date for which experiments should be analyzed. "
    "If not set, analyze for period experiments were active.",
    metavar="YYYY-MM-DD",
)
@click.option(
    "--end_date",
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
    Get all active experiment, including their configurations as bse64 encoded JSON objects.

    We need to encode/decode the config dict to pass it between Argo workflow steps.
    The configs include some Jinja templates, Argo also uses Jinja templates for defining
    workflows, so when passing config dicts to a step in the workflow it tries to evaluate
    the template and will fail. To work around this, instead of passing the config dict as is,
    it is base64 encoded instead, and decoded within the step.
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

        external_experiment_config = external_configs.spec_for_experiment(experiment.normandy_slug)

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
            experiment_runs_configs.append(
                _base64_encode(AnalysisRunConfig(date, experiment, external_experiment_config))
            )

    # Write config to stdout which will be read by argo and then used to spawn analysis jobs
    json.dump(experiment_runs_configs, sys.stdout)


@cli.command()
@project_id_option
@dataset_id_option
@click.option(
    "--experiment_config",
    help="Experiment config as base64 encoded JSON",
    required=True,
)
def analyse_experiment(project_id, dataset_id, experiment_config):
    converter = cattr.Converter()
    converter.register_structure_hook(
        datetime,
        lambda num, _: pytz.utc.localize(datetime.strptime(num, "%Y-%m-%d")),
    )
    analysis_run_config = converter.structure(_base64_decode(experiment_config), AnalysisRunConfig)

    spec = AnalysisSpec.default_for_experiment(analysis_run_config.experiment)

    if analysis_run_config.external_config:
        spec.merge(analysis_run_config.external_config)

    config = spec.resolve(analysis_run_config.experiment)

    # calculate metrics for experiments and write to BigQuery
    try:
        Analysis(project_id, dataset_id, config).run(analysis_run_config.date)
    except Exception as e:
        logger.exception(
            str(e), exc_info=e, extra={"experiment": analysis_run_config.experiment.normandy_slug}
        )
