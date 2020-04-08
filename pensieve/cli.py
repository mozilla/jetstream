from datetime import datetime
import logging

import click
import pytz

from .experimenter import ExperimentCollection
from .analysis import Analysis


@click.group()
def cli():
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(asctime)s:%(name)s:%(message)s",
    )


def format_date(date):
    """Returns the current date with UTC timezone and time set to 00:00:00."""
    return datetime.combine(date, datetime.min.time()).replace(tzinfo=pytz.utc)


class ClickDate(click.ParamType):
    name = "date"

    def convert(self, value, param, ctx):
        if isinstance(value, datetime):
            return value
        return datetime.strptime(value, "%Y-%m-%d")


@cli.command()
@click.option("--project_id", default="moz-fx-data-experiments", help="Project to write to")
@click.option("--dataset_id", default="mozanalysis", help="Dataset to write to")
@click.option(
    "--date",
    type=ClickDate(),
    help="Last date for which data should be analyzed",
    metavar="YYYY-MM-DD",
)
@click.option("--dry_run/--no_dry_run", help="Don't publish any changes to BigQuery")
def run(project_id, dataset_id, date, dry_run):
    """Fetches experiments from Experimenter and runs analysis on active experiments."""
    # fetch experiments that are still active
    collection = ExperimentCollection.from_experimenter()
    if date is None:
        date = format_date(datetime.today())
    else:
        date = format_date(date)

    active_experiments = collection.end_on_or_after(date).of_type(("pref", "addon"))

    # calculate metrics for experiments and write to BigQuery
    for experiment in active_experiments.experiments:
        Analysis(project_id, dataset_id, experiment).run(date, dry_run=dry_run)
