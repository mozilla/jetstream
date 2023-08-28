import logging
import os
import sys
from datetime import datetime, time, timedelta
from functools import partial
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    TextIO,
    Tuple,
    Type,
    Union,
)

import attr
import click
import pytz
import toml
from metric_config_parser.analysis import AnalysisConfiguration, AnalysisSpec
from metric_config_parser.config import (
    Config,
    DefaultConfig,
    DefinitionConfig,
    entity_from_path,
)
from metric_config_parser.data_source import DataSourceDefinition
from metric_config_parser.experiment import Branch, Experiment
from metric_config_parser.function import FunctionsSpec
from metric_config_parser.metric import AnalysisPeriod

from . import bq_normalize_name
from .analysis import Analysis
from .argo import submit_workflow
from .artifacts import ArtifactManager
from .bigquery_client import BigQueryClient
from .config import CONFIGS, METRIC_HUB_REPO, ConfigLoader, _ConfigLoader, validate
from .dryrun import DryRunFailedError
from .errors import ExplicitSkipException, ValidationException
from .experimenter import ExperimentCollection
from .export_json import export_experiment_logs, export_statistics_tables
from .logging import LOG_SOURCE, LogConfiguration
from .metadata import export_metadata
from .platform import PLATFORM_CONFIGS
from .preview import sampled_enrollment_query
from .util import inclusive_date_range

logger = logging.getLogger(__name__)


RECOGNIZED_EXPERIMENT_TYPES = ("pref", "addon", "message", "v6")
LOOKER_PREVIEW_URL = (
    "https://mozilla.cloud.looker.com/dashboards/experimentation::jetstream_preview"
)


@attr.s
class AllType:
    """Sentinel value for AnalysisExecutor"""


All = AllType()


class ExecutorStrategy(Protocol):
    """Abstract class determining how the analysis should be executed."""

    project_id: str

    def __init__(self, project_id: str, dataset_id: str, *args, **kwargs) -> None:
        ...

    def execute(
        self,
        worklist: Iterable[Tuple[AnalysisConfiguration, datetime]],
        configuration_map: Optional[Mapping[str, Union[TextIO, AnalysisSpec]]] = None,
    ) -> bool:
        ...


@attr.s(auto_attribs=True)
class ArgoExecutorStrategy:
    """Handler for executing experiment analyses on Argo."""

    project_id: str
    dataset_id: str
    zone: str
    cluster_id: str
    monitor_status: bool
    bucket: Optional[str] = None
    cluster_ip: Optional[str] = None
    cluster_cert: Optional[str] = None
    experiment_getter: Callable[[], ExperimentCollection] = ExperimentCollection.from_experimenter
    analysis_periods: List[AnalysisPeriod] = [
        AnalysisPeriod.DAY,
        AnalysisPeriod.WEEK,
        AnalysisPeriod.DAYS_28,
        AnalysisPeriod.OVERALL,
    ]
    image: str = "jetstream"
    image_version: Optional[str] = None

    WORKLFOW_DIR = Path(__file__).parent / "workflows"
    RUN_WORKFLOW = WORKLFOW_DIR / "run.yaml"

    def execute(
        self,
        worklist: Iterable[Tuple[AnalysisConfiguration, datetime]],
        configuration_map: Optional[Mapping[str, Union[TextIO, AnalysisSpec]]] = None,
    ):
        if configuration_map is not None:
            raise Exception("Custom configurations are not supported when running with Argo")

        experiments_config: Dict[str, List[str]] = {}
        for config, date in worklist:
            experiments_config.setdefault(config.experiment.normandy_slug, []).append(
                date.strftime("%Y-%m-%d")
            )

        # determine the docker image that was the most recent when enrollments ended for experiments
        artifact_manager = ArtifactManager(
            project=self.project_id, dataset=self.dataset_id, image=self.image
        )

        image_version = self.image_version
        if self.image_version == "latest":
            image_version = artifact_manager.latest_image()

        experiments_config_list = [
            {
                "slug": slug,
                "dates": dates,
                "image_hash": image_version
                if image_version
                else artifact_manager.image_for_slug(slug),
            }
            for slug, dates in experiments_config.items()
        ]
        analysis_period_default = (
            self.analysis_periods[0] if self.analysis_periods != [] else "days28"
        )

        # generate and submit the Argo workflow to the cluster
        return submit_workflow(
            project_id=self.project_id,
            zone=self.zone,
            cluster_id=self.cluster_id,
            workflow_file=self.RUN_WORKFLOW,
            parameters={
                "experiments": experiments_config_list,
                "project_id": self.project_id,
                "dataset_id": self.dataset_id,
                "bucket": self.bucket,
                "analysis_periods_day": "day"
                if AnalysisPeriod.DAY in self.analysis_periods
                else analysis_period_default.value,
                "analysis_periods_week": "week"
                if AnalysisPeriod.WEEK in self.analysis_periods
                else analysis_period_default.value,
                "analysis_periods_days28": "days28"
                if AnalysisPeriod.DAYS_28 in self.analysis_periods
                else analysis_period_default.value,
                "analysis_periods_overall": "overall"
                if AnalysisPeriod.OVERALL in self.analysis_periods
                else analysis_period_default.value,
                "image": self.image,
            },
            monitor_status=self.monitor_status,
            cluster_ip=self.cluster_ip,
            cluster_cert=self.cluster_cert,
        )


@attr.s(auto_attribs=True)
class SerialExecutorStrategy:
    """Handler for executing experiment analyses serially."""

    project_id: str
    dataset_id: str
    bucket: Optional[str] = None
    log_config: Optional[LogConfiguration] = None
    analysis_class: Type = Analysis
    experiment_getter: Callable[[], ExperimentCollection] = ExperimentCollection.from_experimenter
    config_getter: _ConfigLoader = ConfigLoader
    analysis_periods: List[AnalysisPeriod] = [
        AnalysisPeriod.DAY,
        AnalysisPeriod.WEEK,
        AnalysisPeriod.DAYS_28,
        AnalysisPeriod.OVERALL,
    ]
    sql_output_dir: Optional[str] = None

    def execute(
        self,
        worklist: Iterable[Tuple[AnalysisConfiguration, datetime]],
        configuration_map: Optional[Mapping[str, Union[TextIO, AnalysisSpec]]] = None,
    ):
        failed = False
        for config, date in worklist:
            try:
                if config.experiment.is_private or config.experiment.dataset_id is not None:
                    # private experiments must override the dataset and set it to a private dataset
                    dataset_id = config.experiment.dataset_id
                else:
                    dataset_id = self.dataset_id

                # run the analysis
                analysis = self.analysis_class(
                    self.project_id,
                    dataset_id,
                    config,
                    self.log_config,
                    None,
                    self.analysis_periods,
                    self.sql_output_dir,
                )
                analysis.run(date)

                # export metadata to GCS
                if self.bucket:
                    export_metadata(config, self.bucket, self.project_id, analysis.start_time)
            except ValidationException as e:
                # log custom Jetstream exceptions but let the workflow succeed;
                # this prevents Argo from retrying the analysis unnecessarily
                # when it is already clear that it won't succeed
                logger.exception(
                    str(e), exc_info=e, extra={"experiment": config.experiment.normandy_slug}
                )
            except Exception as e:
                failed = True
                logger.exception(
                    str(e), exc_info=e, extra={"experiment": config.experiment.normandy_slug}
                )
            finally:
                # export experiment log from BigQuery to GCS
                if self.log_config is None:
                    log_project = self.project_id
                    log_dataset = self.dataset_id
                    log_table = "logs"
                else:
                    log_project = self.log_config.log_project_id or self.project_id
                    log_dataset = self.log_config.log_dataset_id or self.dataset_id
                    log_table = self.log_config.log_table_id or "logs"

                if self.bucket:
                    export_experiment_logs(
                        self.project_id,
                        self.bucket,
                        config.experiment.normandy_slug,
                        log_project,
                        log_dataset,
                        log_table,
                        analysis.start_time,
                        config.experiment.enrollment_end_date,
                        self.log_config,
                    )
        return not failed


@attr.s(auto_attribs=True)
class AnalysisExecutor:
    """Executes the analyses for the specified experiments."""

    project_id: str
    dataset_id: str
    bucket: str
    date: Union[datetime, AllType]
    experiment_slugs: Union[Iterable[str], AllType]
    configuration_map: Optional[Mapping[str, Union[TextIO, AnalysisSpec]]] = attr.ib(None)
    recreate_enrollments: bool = False
    sql_output_dir: Optional[str] = None
    log_config: Optional[LogConfiguration] = None

    @staticmethod
    def _today() -> datetime:
        return datetime.combine(
            datetime.now(tz=pytz.utc).date() - timedelta(days=1),
            datetime.min.time(),
            tzinfo=pytz.utc,
        )

    def execute(
        self,
        strategy: ExecutorStrategy,
        *,
        experiment_getter: Callable[
            [], ExperimentCollection
        ] = ExperimentCollection.from_experimenter,
        config_getter: _ConfigLoader = ConfigLoader,
        today: Optional[datetime] = None,
    ) -> bool:
        """Execute analyses."""
        run_configs = self._experiment_configs_to_analyse(experiment_getter, config_getter)
        worklist = []

        for config in run_configs:
            if self.date == All:
                today = today or self._today()
                end_date = today
                if config.experiment.end_date:
                    end_date = config.experiment.end_date + timedelta(days=1)

                end_date = min(end_date, today)
                run_dates = inclusive_date_range(config.experiment.start_date, end_date)
            else:
                run_dates = [self.date]

            for run_date in run_dates:
                assert config.experiment.normandy_slug
                worklist.append((config, run_date))

            if self.recreate_enrollments:
                self._delete_enrollment_table(config)

        return strategy.execute(worklist, self.configuration_map)

    def _delete_enrollment_table(self, config: AnalysisConfiguration) -> None:
        """Deletes all enrollment table associated with the experiment."""
        print(f"Delete enrollment table for {config.experiment.normandy_slug}")

        if config.experiment.is_private or config.experiment.dataset_id is not None:
            # private experiments must override the dataset and set it to a private dataset
            dataset_id = config.experiment.dataset_id
        else:
            dataset_id = self.dataset_id

        client = BigQueryClient(project=self.project_id, dataset=dataset_id)
        normalized_slug = bq_normalize_name(config.experiment.normandy_slug)
        enrollments_table = f"{self.project_id}.{dataset_id}.enrollments_{normalized_slug}"
        client.delete_table(enrollments_table)

    def _experiments_to_configs(
        self,
        experiments: List[Experiment],
        config_getter: _ConfigLoader = ConfigLoader,
    ) -> List[AnalysisConfiguration]:
        """Convert mozanalysis experiments to analysis configs."""
        configs = []
        client = BigQueryClient(self.project_id, self.dataset_id)

        def _load_experiment_config(experiment_config):
            # get first updated timestamp for experiment
            first_updated = client.experiment_table_first_updated(experiment_config.normandy_slug)

            # get the configs that were the most recent when the experiment was last updated
            config_collection = config_getter.configs.as_of(first_updated)
            spec = AnalysisSpec.default_for_experiment(experiment_config, config_collection)

            if self.configuration_map and experiment_config.normandy_slug in self.configuration_map:
                if isinstance(
                    self.configuration_map[experiment_config.normandy_slug], AnalysisSpec
                ):
                    spec.merge(self.configuration_map[experiment_config.normandy_slug])
                else:
                    config_dict = toml.load(self.configuration_map[experiment_config.normandy_slug])
                    spec.merge(AnalysisSpec.from_dict(config_dict))
            else:
                if external_spec := config_collection.spec_for_experiment(
                    experiment_config.normandy_slug
                ):
                    spec.merge(external_spec)

            return spec.resolve(experiment_config, config_collection)

        with ThreadPool() as pool:
            configs = pool.map(_load_experiment_config, experiments)

        return configs

    def _experiment_configs_to_analyse(
        self,
        experiment_getter: Callable[
            [], ExperimentCollection
        ] = ExperimentCollection.from_experimenter,
        config_getter: _ConfigLoader = ConfigLoader,
    ) -> List[AnalysisConfiguration]:
        """Fetch configs of experiments that are to be analysed."""
        experiments = experiment_getter()
        run_configs = []

        if isinstance(self.experiment_slugs, AllType):
            if isinstance(self.date, AllType):
                raise ValueError("Declining to re-run all experiments for all time.")

            # only consider experiments that ended within the last 90 days or are live
            ended_threshold = self.date - timedelta(days=90)
            launched_experiments = [
                e
                for e in experiments.ended_after_or_live(ended_threshold)
                .of_type(RECOGNIZED_EXPERIMENT_TYPES)
                .experiments
                if not e.is_rollout
            ]

            launched_configs = self._experiments_to_configs(launched_experiments, config_getter)

            for config in launched_configs:
                if config.experiment.normandy_slug is not None:
                    # get end_date from external config
                    if config.experiment.end_date is None or (
                        config.experiment.end_date and config.experiment.end_date >= self.date
                    ):
                        run_configs.append(config)
        else:
            existing_experiments = []

            for slug in self.experiment_slugs:
                if e := experiments.with_slug(slug).experiments:
                    existing_experiments.append(e[0])
                else:
                    logger.warning(
                        f"Slug {slug} provided but not found in Experimenter; skipping.",
                        extra={"experiment": slug},
                    )

            run_configs = self._experiments_to_configs(existing_experiments, config_getter)

        # filter out experiments that are always getting skipped
        non_skipped_configs = []
        for c in run_configs:
            if not c.experiment.skip:
                non_skipped_configs.append(c)
            else:
                logger.warning(
                    f"Skipping {c.experiment.normandy_slug}; skip=true in config.",
                    extra={"experiment": c.experiment.normandy_slug},
                )

        return non_skipped_configs

    def ensure_enrollments(
        self,
        config_getter: _ConfigLoader = ConfigLoader,
        experiment_getter: Callable[
            [], ExperimentCollection
        ] = ExperimentCollection.from_experimenter,
    ) -> None:
        """Ensure that enrollment tables for experiment are up-to-date or re-create."""
        run_configs = self._experiment_configs_to_analyse(experiment_getter, config_getter)
        for config in run_configs:
            try:
                analysis = Analysis(
                    self.project_id,
                    self.dataset_id,
                    config,
                    sql_output_dir=self.sql_output_dir,
                    log_config=self.log_config,
                )

                if isinstance(self.date, AllType):
                    today = self._today()
                    end_date = today
                    if config.experiment.end_date:
                        end_date = config.experiment.end_date

                    end_date = min(end_date, today)
                else:
                    end_date = self.date

                if self.recreate_enrollments:
                    self._delete_enrollment_table(config)

                # make sure enrollment is actually ended (and enrollment is not manually overridden)
                if not (
                    (
                        hasattr(config.experiment, "is_enrollment_paused")
                        and config.experiment.is_enrollment_paused is False
                    )
                    and (
                        config.experiment.proposed_enrollment
                        == config.experiment.experiment.proposed_enrollment
                        and config.experiment.enrollment_end_date
                        == config.experiment.experiment.enrollment_end_date
                        and config.experiment.experiment_spec.enrollment_period is None
                    )
                ):
                    analysis.ensure_enrollments(end_date)
            except Exception as e:
                logger.exception(
                    str(e), exc_info=e, extra={"experiment": config.experiment.normandy_slug}
                )
                raise e


log_project_id_option = click.option(
    "--log_project_id",
    "--log-project-id",
    default="moz-fx-data-experiments",
    help="GCP project to write logs to",
)
log_dataset_id_option = click.option(
    "--log_dataset_id",
    "--log-dataset-id",
    default="monitoring",
    help="Dataset to write logs to",
)
log_table_id_option = click.option(
    "--log_table_id", "--log-table-id", default="logs", help="Table to write logs to"
)
log_source = click.option(
    "--log-source",
    "--log_source",
    default=LOG_SOURCE.JETSTREAM,
    type=LOG_SOURCE,
    help="Source column for logs",
)


@click.group()
@log_project_id_option
@log_dataset_id_option
@log_table_id_option
@click.option(
    "--task_profiling_log_table_id",
    "--task-profiling-log-table-id",
    default="task_profiling_logs",
    help="Table to write task profiling logs to",
)
@click.option(
    "--task_monitoring_log_table_id",
    "--task-monitoring-log-table-id",
    default="task_monitoring_logs",
    help="Table to write task monitoring logs to",
)
@click.option("--log_to_bigquery", "--log-to-bigquery", is_flag=True, default=False)
@log_source
@click.pass_context
def cli(
    ctx,
    log_project_id,
    log_dataset_id,
    log_table_id,
    task_profiling_log_table_id,
    task_monitoring_log_table_id,
    log_to_bigquery,
    log_source,
):
    log_config = LogConfiguration(
        log_project_id,
        log_dataset_id,
        log_table_id,
        task_profiling_log_table_id,
        task_monitoring_log_table_id,
        log_to_bigquery,
        log_source,
    )
    log_config.setup_logger()
    ctx.ensure_object(dict)
    ctx.obj["log_config"] = log_config


class ClickDate(click.ParamType):
    name = "date"

    def convert(self, value, param, ctx):
        if isinstance(value, datetime):
            return value
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=pytz.utc)


class ClickNullableString(click.ParamType):
    name = "nullable_string"

    def convert(self, value, param, ctx):
        if not isinstance(value, str):
            value = str(value)

        if value.lower() == "none" or value.lower() == "null":
            return None

        return value


def project_id_option(default="moz-fx-data-experiments"):
    return click.option(
        "--project_id",
        "--project-id",
        default=default,
        help="Project to write to",
    )


def dataset_id_option(default="mozanalysis"):
    return click.option(
        "--dataset_id", "--dataset-id", default=default, help="Dataset to write to", required=True
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
    multiple=True,
    help="Experimenter or Normandy slug of the experiment to (re)run analysis for",
)

config_file_option = click.option(
    "--i-solemnly-swear-i-am-up-to-no-good",
    "--config_file",
    "--config-file",
    "config_file",
    type=click.File("rt"),
)

bucket_option = click.option(
    "--bucket",
    default="mozanalysis",
    help="GCS bucket to write to",
    required=False,
    type=ClickNullableString(),
)

argo_option = click.option(
    "--argo", is_flag=True, default=False, help="Run on Kubernetes with Argo"
)

return_status_option = click.option(
    "--return_status",
    "--return-status",
    is_flag=True,
    default=False,
    help="Return success/failed status code",
)

monitor_status_option = click.option(
    "--monitor_status",
    "--monitor-status",
    default=True,
    help="Monitor the status of the Argo workflow",
)

cluster_ip_option = click.option(
    "--cluster_ip",
    "--cluster-ip",
    help="Kubernetes cluster IP address",
)

cluster_cert_option = click.option(
    "--cluster_cert",
    "--cluster-cert",
    help="Kubernetes cluster certificate used for authenticating to the cluster",
)

recreate_enrollments_option = click.option(
    "--recreate_enrollments",
    "--recreate-enrollments",
    help="Recreate the enrollments tables",
    is_flag=True,
    default=False,
)

date_option = click.option(
    "--date",
    type=ClickDate(),
    help="Date for which experiments should be analyzed",
    metavar="YYYY-MM-DD",
    required=True,
)
config_repos_option = click.option(
    "--config_repos",
    "--config-repos",
    help="URLs to public repos with configs",
    multiple=True,
    default=[METRIC_HUB_REPO, CONFIGS],
)
private_config_repos_option = click.option(
    "--private_config_repos",
    "--private-config-repos",
    help="URLs to private repos with configs",
    multiple=True,
)

image_option = click.option(
    "--image",
    help="Name of the docker image to use in Argo.",
    default="jetstream",
)

image_version_option = click.option(
    "--image_version",
    "--image-version",
    help="Hash of the image to use in Argo, or 'latest'",
    required=False,
)


def analysis_periods_option(
    default=[
        AnalysisPeriod.DAY,
        AnalysisPeriod.WEEK,
        AnalysisPeriod.DAYS_28,
        AnalysisPeriod.OVERALL,
    ]
):
    return click.option(
        "--analysis_periods",
        "--analysis-periods",
        help="Analysis periods to run analysis for.",
        multiple=True,
        type=AnalysisPeriod,
        default=default,
    )


sql_output_dir_option = click.option(
    "--sql-output-dir",
    "--sql_output_dir",
    type=click.Path(exists=False),
    help="Write generated SQL to given directory",
    required=False,
    show_default=True,
    default=None,
    metavar="OUTDIR",
)


@cli.command()
@project_id_option()
@dataset_id_option()
@date_option
@experiment_slug_option
@bucket_option
@config_file_option
@recreate_enrollments_option
@config_repos_option
@private_config_repos_option
@analysis_periods_option()
@sql_output_dir_option
@click.pass_context
def run(
    ctx,
    project_id,
    dataset_id,
    date,
    experiment_slug,
    bucket,
    config_file,
    recreate_enrollments,
    config_repos,
    private_config_repos,
    analysis_periods,
    sql_output_dir,
):
    """Runs analysis for the provided date."""
    if len(experiment_slug) > 1 and config_file:
        raise ValueError(
            "Cannot process multiple experiments with custom configs. "
            "Trigger separate runs for experiments with custom configs"
        )

    analysis_executor = AnalysisExecutor(
        project_id=project_id,
        dataset_id=dataset_id,
        bucket=bucket,
        date=date,
        experiment_slugs=experiment_slug if experiment_slug else All,
        configuration_map={experiment_slug[0]: config_file}
        if experiment_slug and config_file
        else {},
        recreate_enrollments=recreate_enrollments,
        sql_output_dir=sql_output_dir,
    )

    success = analysis_executor.execute(
        strategy=SerialExecutorStrategy(
            project_id,
            dataset_id,
            bucket,
            ctx.obj["log_config"],
            analysis_periods=analysis_periods,
            sql_output_dir=sql_output_dir,
        ),
        config_getter=ConfigLoader.with_configs_from(config_repos).with_configs_from(
            private_config_repos, is_private=True
        ),
    )

    sys.exit(0 if success else 1)


@cli.command()
@project_id_option()
@dataset_id_option()
@click.option(
    "--date",
    type=ClickDate(),
    help="Date for which experiments should be analyzed",
    metavar="YYYY-MM-DD",
    required=True,
)
@experiment_slug_option
@bucket_option
@zone_option
@cluster_id_option
@monitor_status_option
@cluster_ip_option
@cluster_cert_option
@recreate_enrollments_option
@config_repos_option
@private_config_repos_option
@image_option
@image_version_option
@analysis_periods_option()
def run_argo(
    project_id,
    dataset_id,
    date,
    experiment_slug,
    bucket,
    zone,
    cluster_id,
    monitor_status,
    cluster_ip,
    cluster_cert,
    recreate_enrollments,
    config_repos,
    private_config_repos,
    analysis_periods,
    image,
    image_version,
):
    """Runs analysis for the provided date using Argo."""
    strategy = ArgoExecutorStrategy(
        project_id=project_id,
        dataset_id=dataset_id,
        bucket=bucket,
        zone=zone,
        cluster_id=cluster_id,
        monitor_status=monitor_status,
        cluster_ip=cluster_ip,
        cluster_cert=cluster_cert,
        analysis_periods=analysis_periods,
        image=image,
        image_version=image_version,
    )

    AnalysisExecutor(
        project_id=project_id,
        dataset_id=dataset_id,
        bucket=bucket,
        date=date,
        experiment_slugs=experiment_slug if experiment_slug else All,
        recreate_enrollments=recreate_enrollments,
    ).execute(
        strategy=strategy,
        config_getter=ConfigLoader.with_configs_from(config_repos).with_configs_from(
            private_config_repos, is_private=True
        ),
    )


@cli.command("rerun")
@experiment_slug_option
@project_id_option()
@dataset_id_option()
@bucket_option
@config_file_option
@argo_option
@zone_option
@cluster_id_option
@monitor_status_option
@cluster_ip_option
@cluster_cert_option
@return_status_option
@recreate_enrollments_option
@config_repos_option
@private_config_repos_option
@image_option
@image_version_option
@analysis_periods_option()
@click.pass_context
def rerun(
    ctx,
    project_id,
    dataset_id,
    experiment_slug,
    bucket,
    config_file,
    argo,
    zone,
    cluster_id,
    monitor_status,
    cluster_ip,
    cluster_cert,
    return_status,
    recreate_enrollments,
    config_repos,
    private_config_repos,
    analysis_periods,
    image,
    image_version,
):
    """Rerun all available analyses for a specific experiment."""
    if len(experiment_slug) > 1 and config_file:
        raise ValueError(
            "Cannot rerun multiple experiments with custom configs. "
            "Trigger separate reruns for experiments with custom configs"
        )

    # update table timestamps which indicate whether an experiment needs to be rerun
    client = BigQueryClient(project_id, dataset_id)
    for slug in experiment_slug:
        client.touch_tables(slug)
        # delete existing tables
        client.delete_experiment_tables(slug, analysis_periods, recreate_enrollments)

    strategy = SerialExecutorStrategy(
        project_id, dataset_id, bucket, ctx.obj["log_config"], analysis_periods=analysis_periods
    )

    if argo:
        strategy = ArgoExecutorStrategy(
            project_id=project_id,
            dataset_id=dataset_id,
            bucket=bucket,
            zone=zone,
            cluster_id=cluster_id,
            monitor_status=monitor_status,
            cluster_ip=cluster_ip,
            cluster_cert=cluster_cert,
            analysis_periods=analysis_periods,
            image=image,
            image_version=image_version,
        )

    success = AnalysisExecutor(
        project_id=project_id,
        dataset_id=dataset_id,
        bucket=bucket,
        date=All,
        experiment_slugs=experiment_slug,
        configuration_map={experiment_slug[0]: config_file} if config_file else None,
        recreate_enrollments=recreate_enrollments,
    ).execute(
        strategy=strategy,
        config_getter=ConfigLoader.with_configs_from(config_repos).with_configs_from(
            private_config_repos, is_private=True
        ),
    )

    if return_status:
        sys.exit(not success)


@cli.command()
@experiment_slug_option
@project_id_option()
@dataset_id_option()
@config_repos_option
@private_config_repos_option
@click.option(
    "--parallelism",
    "-p",
    help="Number of parallel threads",
    default=8,
)
def rerun_skip(
    experiment_slug, project_id, dataset_id, config_repos, private_config_repos, parallelism
):
    """Skip rerun for experiments and mark them as up to date."""
    if not experiment_slug:
        # get experiment-specific external configs
        ConfigLoader.with_configs_from(config_repos).with_configs_from(
            private_config_repos, is_private=True
        )
        updated_configs = ConfigLoader.updated_configs(project_id, dataset_id)
        experiments_with_updated_defaults = ConfigLoader.updated_defaults(project_id, dataset_id)
        experiment_slug = set(
            experiments_with_updated_defaults + [conf.slug for conf in updated_configs]
        )

    client = BigQueryClient(project_id, dataset_id)
    logger.info(f"Skip reruns for {experiment_slug}")
    with Pool(parallelism) as pool:
        pool.map(client.touch_tables, experiment_slug)


@cli.command()
@project_id_option()
@dataset_id_option()
@bucket_option
@experiment_slug_option
def export_statistics_to_json(project_id, dataset_id, bucket, experiment_slug):
    """Export all tables as JSON to a GCS bucket."""
    if bucket is None:
        logger.warn("No bucket specified. Analysis results won't be exported to GCS.")
    else:
        for slug in experiment_slug:
            export_statistics_tables(project_id, dataset_id, bucket, slug)


@cli.command()
@log_project_id_option
@log_dataset_id_option
@log_table_id_option
@bucket_option
@experiment_slug_option
@project_id_option()
@date_option
@click.pass_context
def export_experiment_logs_to_json(
    ctx, log_project_id, log_dataset_id, log_table_id, bucket, experiment_slug, project_id, date
):
    """Export all error logs for this experiment as JSON to a GCS bucket."""
    if bucket is None:
        logger.warn("No bucket specified. Logs results won't be exported to GCS.")
    else:
        for slug in experiment_slug:
            export_experiment_logs(
                project_id,
                bucket,
                slug,
                log_project_id,
                log_dataset_id,
                log_table_id,
                date,
                None,
                ctx.obj["log_config"],
            )


@cli.command()
@project_id_option()
@dataset_id_option()
@bucket_option
@argo_option
@zone_option
@cluster_id_option
@monitor_status_option
@cluster_ip_option
@cluster_cert_option
@return_status_option
@recreate_enrollments_option
@config_repos_option
@private_config_repos_option
@image_option
@image_version_option
@analysis_periods_option()
@click.pass_context
def rerun_config_changed(
    ctx,
    project_id,
    dataset_id,
    bucket,
    argo,
    zone,
    cluster_id,
    monitor_status,
    cluster_ip,
    cluster_cert,
    return_status,
    recreate_enrollments,
    config_repos,
    private_config_repos,
    analysis_periods,
    image,
    image_version,
):
    """Rerun all available analyses for experiments with new or updated config files."""

    strategy = SerialExecutorStrategy(
        project_id, dataset_id, bucket, ctx.obj["log_config"], analysis_periods=analysis_periods
    )

    # get experiment-specific external configs
    ConfigLoader.with_configs_from(config_repos).with_configs_from(
        private_config_repos, is_private=True
    )
    updated_configs = ConfigLoader.updated_configs(project_id, dataset_id)
    experiments_with_updated_defaults = ConfigLoader.updated_defaults(project_id, dataset_id)
    experiment_slugs = set(
        experiments_with_updated_defaults + [conf.slug for conf in updated_configs]
    )

    # update the table timestamps which indicate whether a experiment needs to be rerun
    client = BigQueryClient(project_id, dataset_id)
    for slug in experiment_slugs:
        client.touch_tables(slug)
        # delete existing tables
        client.delete_experiment_tables(slug, analysis_periods, recreate_enrollments)

    if argo:
        strategy = ArgoExecutorStrategy(
            project_id=project_id,
            dataset_id=dataset_id,
            bucket=bucket,
            zone=zone,
            cluster_id=cluster_id,
            monitor_status=monitor_status,
            cluster_ip=cluster_ip,
            cluster_cert=cluster_cert,
            analysis_periods=analysis_periods,
            image=image,
            image_version=image_version,
        )

    success = AnalysisExecutor(
        project_id=project_id,
        dataset_id=dataset_id,
        bucket=bucket,
        date=All,
        experiment_slugs=experiment_slugs,
        recreate_enrollments=recreate_enrollments,
    ).execute(
        strategy=strategy,
        config_getter=ConfigLoader,
    )

    if return_status:
        sys.exit(0 if success else 1)


@cli.command("validate_config")
@click.argument("path", type=click.Path(exists=True), nargs=-1)
@config_repos_option
@private_config_repos_option
@click.option(
    "--is_private",
    "--is-private",
    help="Treat the configs as private configs",
    is_flag=True,
    default=False,
)
def validate_config(path: Iterable[os.PathLike], config_repos, private_config_repos, is_private):
    """Validate config files."""
    dirty = False
    collection = ExperimentCollection.from_experimenter()

    for config_file in path:
        config_file = Path(config_file)
        if not config_file.is_file():
            continue
        if ".example" in config_file.suffixes:
            print(f"Skipping example config {config_file}")
            continue

        print(f"Evaluating {config_file}...")

        if "functions.toml" == config_file.name:
            FunctionsSpec.from_dict(toml.load(config_file))
            print(f"{config_file} OK")
            continue
        entity = entity_from_path(config_file, is_private)
        call = partial(
            validate,
            config=entity,
            config_getter=ConfigLoader.with_configs_from(config_repos).with_configs_from(
                private_config_repos, is_private=True
            ),
        )
        if (
            isinstance(entity, Config)
            and not isinstance(entity, DefaultConfig)
            and not isinstance(entity, DefinitionConfig)
        ):
            if (experiments := collection.with_slug(entity.slug).experiments) == []:
                print(f"No experiment with slug {entity.slug} in Experimenter.")
                dirty = True
                continue
            call = partial(
                validate,
                config=entity,
                config_getter=ConfigLoader.with_configs_from(config_repos).with_configs_from(
                    private_config_repos, is_private=True
                ),
                experiment=experiments[0],
            )
        try:
            call()
        except DryRunFailedError as e:
            print("Error evaluating SQL:")
            for i, line in enumerate(e.sql.split("\n")):
                print(f"{i+1: 4d} {line.rstrip()}")
            print("")
            print(str(e))
            dirty = True
        except ExplicitSkipException:
            print("Found an explicit skip directive; will ignore this experiment.")
    sys.exit(1 if dirty else 0)


@cli.command()
@project_id_option()
@dataset_id_option()
@bucket_option
@experiment_slug_option
@config_file_option
@recreate_enrollments_option
@config_repos_option
@private_config_repos_option
def ensure_enrollments(
    project_id,
    dataset_id,
    bucket,
    experiment_slug,
    config_file,
    recreate_enrollments,
    config_repos,
    private_config_repos,
):
    """Ensure that enrollment tables for experiment are up-to-date or re-create."""
    if len(experiment_slug) > 1 and config_file:
        raise ValueError(
            "Cannot process multiple experiments with custom configs. "
            "Trigger separate runs for experiments with custom configs"
        )

    AnalysisExecutor(
        project_id=project_id,
        dataset_id=dataset_id,
        bucket=bucket,
        date=AnalysisExecutor._today(),
        experiment_slugs=experiment_slug if experiment_slug else All,
        configuration_map={experiment_slug[0]: config_file} if config_file else None,
        recreate_enrollments=recreate_enrollments,
    ).ensure_enrollments(
        config_getter=ConfigLoader.with_configs_from(config_repos).with_configs_from(
            private_config_repos, is_private=True
        ),
    )


@cli.command()
@project_id_option(default="mozdata")
@dataset_id_option(default="tmp")
@click.option(
    "--start_date",
    "--start-date",
    type=ClickDate(),
    help="Date for which project should be started to get analyzed. Default: current date - 3 days",
    metavar="YYYY-MM-DD",
    required=False,
)
@click.option(
    "--end_date",
    "--end-date",
    type=ClickDate(),
    help="Date for which project should be stop to get analyzed. Default: current date",
    metavar="YYYY-MM-DD",
    required=False,
)
@click.option(
    "--num-days",
    "--num-days",
    type=int,
    help="Number of days for which the project be analyzed. Default: 3",
    default=3,
    required=False,
)
@experiment_slug_option
@config_file_option
@config_repos_option
@private_config_repos_option
@analysis_periods_option(
    [
        AnalysisPeriod.DAY,
    ]
)
@sql_output_dir_option
@click.option(
    "--platform",
    type=str,
    help="Platform/app to run analysis for.",
    required=True,
    default="firefox_desktop",
)
@click.option(
    "--generate-population",
    "--generate_population",
    is_flag=True,
    default=False,
    help="Generate a random population sample based on the provided population size. "
    + "Useful if enrollment hasn't happened yet",
)
@click.option(
    "--population-sample-size",
    "--population_sample_size",
    type=int,
    required=False,
    default=1,
    help="Generated population sample size. "
    + "Only used when `--generate-population` is specified. "
    + "Use floats to specify population sizes in percent, e.g 0.01 == 1% of clients",
)
@click.option(
    "--enrollment_period",
    "--enrollment-period",
    type=int,
    required=False,
    default=3,
    help="Numer of days used as enrollment period when generating population.",
)
def preview(
    project_id,
    dataset_id,
    start_date,
    end_date,
    num_days,
    experiment_slug,
    config_file,
    config_repos,
    private_config_repos,
    analysis_periods,
    sql_output_dir,
    platform,
    generate_population,
    population_sample_size,
    enrollment_period,
):
    """Create a preview for a specific experiment based on a subset of data."""
    if not experiment_slug and not config_file:
        raise ValueError(
            "One of `--experiment-slug` or `--config-file` is required for generating a preview."
        )

    if start_date is None and end_date is None:
        yesterday_midnight = datetime.combine(datetime.today() - timedelta(days=1), time.min)
        end_date = yesterday_midnight
        start_date = end_date - timedelta(days=num_days)
    elif start_date is None:
        start_date = end_date - timedelta(days=num_days)
    else:
        end_date = start_date + timedelta(days=num_days)

    # At least one of `--slug` and `--config-file` is required. If slug is not
    # given, find it from the config file.
    if not experiment_slug:
        external_config = entity_from_path(Path(config_file))
        experiment_slug = [external_config.slug]

    for slug in experiment_slug:
        collection = ExperimentCollection.from_experimenter(with_draft_experiments=True)
        experimenter_experiments = collection.with_slug(slug)
        if experimenter_experiments.experiments == [] and not config_file:
            click.echo(
                f"Experiment {slug} doesn't exist in Experimenter and no config file specified."
            )
            continue

        click.echo(f"Generate preview for {slug}")
        table = bq_normalize_name(slug)

        # delete previously created preview tables if exist
        client = BigQueryClient(project_id, dataset_id)
        client.delete_experiment_tables(slug, analysis_periods, delete_enrollments=True)

        config_getter = ConfigLoader.with_configs_from(config_repos).with_configs_from(
            private_config_repos, is_private=True
        )

        experiment: Optional[Experiment] = None
        if experimenter_experiments.experiments != []:
            experiment = experimenter_experiments.experiments[0]

        # set dummy experiment values and adjust dates
        experiment = Experiment(
            experimenter_slug=experiment.experimenter_slug if experiment else slug,
            normandy_slug=experiment.normandy_slug if experiment else slug,
            type=experiment.type if experiment else "v6",
            status="Live",
            start_date=start_date - timedelta(days=3),  # subtract enrollment days
            end_date=end_date,
            proposed_enrollment=enrollment_period,
            branches=experiment.branches if experiment else [Branch(slug="control", ratio=1)],
            reference_branch=experiment.reference_branch if experiment else "control",
            is_high_population=False,
            app_name=platform,
            app_id=PLATFORM_CONFIGS[platform].app_id,
            outcomes=experiment.outcomes if experiment else [],
            enrollment_end_date=None,
        )

        spec = AnalysisSpec.default_for_experiment(experiment, config_getter.configs)
        if external_spec := config_getter.spec_for_experiment(experiment.normandy_slug):
            spec.merge(external_spec)

        config = spec.resolve(experiment=experiment, configs=config_getter.configs)

        # generated sampled enrollment query
        if generate_population:
            spec.experiment.enrollment_query = sampled_enrollment_query(
                start_date, config, population_sample_size
            )

            # update dates
            spec.experiment.start_date = (start_date - timedelta(days=3)).strftime("%Y-%m-%d")
            spec.experiment.end_date = end_date.strftime("%Y-%m-%d")
            spec.experiment.enrollment_period = enrollment_period

            # set experiments_column_type to none for all data sources used in the experiment;
            # this needs to be done otherwise all clients will be filtered out since none
            # are enrolled in the experiment
            for _, summaries in config.metrics.items():
                for summary in summaries:
                    ds = summary.metric.data_source

                    if ds.name in spec.data_sources.definitions:
                        spec.data_sources.definitions[ds.name].experiments_column_type = "none"
                    else:
                        spec.data_sources.definitions[ds.name] = DataSourceDefinition(
                            name=ds.name,
                            from_expression=ds.from_expression,
                            client_id_column=ds.client_id_column,
                            submission_date_column=ds.submission_date_column,
                            default_dataset=ds.default_dataset,
                            build_id_column=ds.build_id_column,
                            friendly_name=ds.friendly_name,
                            description=ds.description,
                            experiments_column_type="none",
                        )

        # log to a table in the temporary dataset, will be displayed on the Looker dashboard
        log_config = LogConfiguration(
            log_project_id=project_id,
            log_dataset_id=dataset_id,
            log_table_id=f"logs_{table}",
            log_to_bigquery=True,
            task_profiling_log_table_id=None,
            task_monitoring_log_table_id=None,
            log_level=logging.INFO,
            capacity=5,
            log_source=LOG_SOURCE.PREVIEW,
        )
        client.delete_table(f"{project_id}.{dataset_id}.logs_{table}")

        # recreate enrollments
        AnalysisExecutor(
            project_id=project_id,
            dataset_id=dataset_id,
            bucket=None,
            date=start_date,
            experiment_slugs=[slug] if slug else All,
            configuration_map={slug: spec},
            recreate_enrollments=True,
            log_config=log_config,
        ).ensure_enrollments(
            config_getter=ConfigLoader.with_configs_from(config_repos).with_configs_from(
                private_config_repos, is_private=True
            ),
            experiment_getter=lambda: ExperimentCollection(experiments=[experiment]),
        )

        # run preview analysis
        for date in [
            start_date + timedelta(days=d) for d in range(0, (end_date - start_date).days + 1)
        ]:
            click.echo(f"Generate preview for {date}")
            analysis_executor = AnalysisExecutor(
                project_id=project_id,
                dataset_id=dataset_id,
                bucket=None,
                date=date,
                experiment_slugs=[slug] if slug else All,
                configuration_map={slug: spec},
                recreate_enrollments=False,
                sql_output_dir=sql_output_dir,
            )

            # add experiment to API results if it's not already available in Experimenter
            analysis_executor.execute(
                strategy=SerialExecutorStrategy(
                    project_id,
                    dataset_id,
                    None,
                    log_config,
                    analysis_periods=analysis_periods,
                    sql_output_dir=sql_output_dir,
                    experiment_getter=lambda: ExperimentCollection(experiments=[experiment]),
                ),
                config_getter=ConfigLoader.with_configs_from(config_repos).with_configs_from(
                    private_config_repos, is_private=True
                ),
                experiment_getter=lambda: ExperimentCollection(experiments=[experiment]),
            )

        click.echo(
            "A preview is available at: "
            + f"{LOOKER_PREVIEW_URL}?Project='{project_id}'&Dataset='{dataset_id}'&Slug='{table}'"
        )
