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
from .analysis import Analysis


DEFAULT_METRICS_CONFIG = Path(__file__).parent / "default_metrics.toml"


@click.group()
def cli():
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(asctime)s:%(name)s:%(message)s",
    )


def format_date(date):
    """Returns the current date with UTC timezone and time set to 00:00:00."""
    return datetime.combine(date, datetime.min.time()).replace(tzinfo=pytz.utc)


def inclusive_date_range(start_date, end_date):
    """Generator for a range of dates, includes end_date."""
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


def default_spec_for_experiment(experiment: experimenter.Experiment) -> AnalysisSpec:
    return AnalysisSpec.from_dict(toml.load(DEFAULT_METRICS_CONFIG))


class ClickDate(click.ParamType):
    name = "date"

    def convert(self, value, param, ctx):
        if isinstance(value, datetime):
            return value
        return datetime.strptime(value, "%Y-%m-%d")


@cli.command()
@click.option(
    "--project_id", "--project-id", default="moz-fx-data-experiments", help="Project to write to"
)
@click.option("--dataset_id", "--dataset-id", default="mozanalysis", help="Dataset to write to")
@click.option(
    "--start_date",
    "--start-date",
    type=ClickDate(),
    help="First date for which data should be analyzed",
    metavar="YYYY-MM-DD",
)
@click.option(
    "--end_date",
    "--end-date",
    type=ClickDate(),
    help="Last date for which data should be analyzed",
    metavar="YYYY-MM-DD",
)
@click.option(
    "--experiment_slug",
    "--experiment-slug",
    help="Experimenter or Normandy slug of the experiment to rerun analysis for",
)
@click.option("--dry_run/--no_dry_run", help="Don't publish any changes to BigQuery")
def run(project_id, dataset_id, start_date, end_date, experiment_slug, dry_run):
    """Fetches experiments from Experimenter and runs analysis on active experiments."""
    # fetch experiments that are still active
    collection = ExperimentCollection.from_experimenter()

    if start_date is None:
        start_date = format_date(datetime.today())
    else:
        start_date = format_date(start_date)

    if end_date is None:
        end_date = format_date(datetime.today())
    else:
        end_date = format_date(end_date)

    active_experiments = collection.end_on_or_after(start_date).of_type(("pref", "addon"))

    if experiment_slug is not None:
        # run analysis for specific experiment
        active_experiments = active_experiments.with_slug(experiment_slug)

    # calculate metrics for experiments and write to BigQuery
    for experiment in active_experiments.experiments:
        spec = default_spec_for_experiment(experiment)
        config = spec.resolve(experiment)

        for date in inclusive_date_range(start_date, end_date):
            Analysis(project_id, dataset_id, config).run(date, dry_run=dry_run)


@cli.command()
@click.option(
    "--experiment_slug",
    "--experiment-slug",
    help="Experimenter or Normandy slug of the experiment to rerun analysis for",
    required=True,
)
@click.option(
    "--project_id", "--project-id", default="moz-fx-data-experiments", help="Project to write to"
)
@click.option("--dataset_id", "--dataset-id", default="mozanalysis", help="Dataset to write to")
@click.option("--dry_run/--no_dry_run", help="Don't publish any changes to BigQuery")
@click.pass_context
def rerun(ctx, project_id, dataset_id, experiment_slug, dry_run):
    """Rerun previous analyses for a specific experiment."""
    collection = ExperimentCollection.from_experimenter()

    experiments = collection.with_slug(experiment_slug)

    if len(experiments.experiments) == 0:
        click.echo(f"No experiment with slug {experiment_slug} found.", err=True)
        sys.exit(1)

    experiment = experiments.experiments[0]
    ctx.invoke(
        run,
        project_id=project_id,
        dataset_id=dataset_id,
        start_date=experiment.start_date,
        end_date=None,
        experiment_slug=experiment_slug,
        dry_run=dry_run,
    )
