from datetime import datetime, timedelta
from typing import Callable, Class, Iterable, Mapping, TextIO, Tuple, Union

import attr
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
from . import bq_normalize_name, AnalysisPeriod

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


RECOGNIZED_EXPERIMENT_TYPES = ("pref", "addon", "message", "v6")


@attr.s
class AllType:
    """Sentinel value for AnalysisExecutor"""

    pass


All = AllType()


class ExecutorStrategy:
    def execute(worklist: Iterable[Tuple[str, AnalysisSpec, datetime]]) -> bool:
        ...


@attr.s
class SerialExecutorStrategy(ExecutorStrategy):
    analysis_class: Class = Analysis
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
        analysis_class: Class = Analysis,
        today: datetime = _today(),
        strategy: ExecutorStrategy = SerialExecutorStrategy(),
    ) -> bool:
        experiments = experiment_getter()
        external_configs = None

        if self.experiment_slugs == All:
            if self.date == All:
                raise ValueError("Declining to re-run all experiments for all time.")
            run_experiments = [
                e.slug
                for e in experiments.end_on_or_after(self.date).of_type(RECOGNIZED_EXPERIMENT_TYPES)
            ]
        else:
            run_experiments = self.experiment_slugs

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
                end_date = min(
                    experiment.end_date,
                    today,
                )
                run_dates = inclusive_date_range(experiment.start_date, end_date)
            else:
                run_dates = [self.date]

            for run_date in run_dates:
                worklist.append((slug, spec, run_date))

        return strategy.execute(worklist)


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


@cli.command("rerun")
@experiment_slug_option
@project_id_option
@dataset_id_option
@secret_config_file_option
def rerun(project_id, dataset_id, experiment_slug, config_file):
    """
    Rerun all available analyses for a specific experiment.

    This command is invoked after adding new custom configs via jetstream-config.
    If exceptions are thrown during a re-run, Jetstream will return with an error code.
    jetstream-config launches Jetstream on a separate Kubernetes cluster which needs to
    report back to CircleCI whether or not the run was successful.
    """
    AnalysisExecutor(
        project_id=project_id,
        dataset_id=dataset_id,
        date=All,
        experiment_slugs=[experiment_slug],
        configuration_map={experiment_slug: config_file} if config_file else {},
    ).execute()

    # todo: do something reasonable here
    # # delete all tables previously created when this experiment was analysed
    # client = BigQueryClient(project_id, dataset_id)
    # normalized_slug = bq_normalize_name(experiment.normandy_slug)
    # analysis_periods = "|".join([p.value for p in AnalysisPeriod])
    # table_name_re = f"^(statistics_)?{normalized_slug}_({analysis_periods})_.*$"
    # client.delete_tables_matching_regex(table_name_re)


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
