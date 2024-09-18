import datetime as dt

import pytest
import pytz
from metric_config_parser.experiment import Branch, Experiment

from jetstream.experimenter import ExperimentCollection


@pytest.fixture
def experiment_collection():
    try:
        collection = ExperimentCollection.from_experimenter()
        return collection
    except Exception as e:
        pytest.fail(f"Failed to fetch experiment collection: {e!s}")


def test_from_experimenter(experiment_collection):
    for experiment in experiment_collection.experiments:
        assert isinstance(experiment, Experiment)
        for branch in experiment.branches:
            assert isinstance(branch, Branch)


def test_of_type(experiment_collection):
    v6_experiments = experiment_collection.of_type("v6")
    for experiment in v6_experiments.experiments:
        assert experiment.type == "v6"

    v1_experiments = experiment_collection.of_type("v1")
    for experiment in v1_experiments.experiments:
        assert experiment.type == "v1"


def test_with_slug(experiment_collection):
    if experiment_collection and experiment_collection.experiments[0].experimenter_slug:
        example_slug = experiment_collection.experiments[0].experimenter_slug
        experiments_with_slug = experiment_collection.with_slug(example_slug)
        for experiment in experiments_with_slug.experiments:
            assert experiment.experimenter_slug == example_slug
    else:
        assert experiment_collection.experiments[0].experimenter_slug is None


def test_started_since(experiment_collection):
    since_date = dt.datetime(2020, 1, 1, tzinfo=pytz.utc)
    started_experiments = experiment_collection.started_since(since_date)
    for experiment in started_experiments.experiments:
        assert experiment.start_date >= since_date


def test_ended_after_or_live(experiment_collection):
    after_date = dt.datetime(2020, 1, 1, tzinfo=pytz.utc)
    ended_or_live_experiments = experiment_collection.ended_after_or_live(after_date)
    for experiment in ended_or_live_experiments.experiments:
        assert (
            experiment.end_date is None
            or experiment.end_date >= after_date
            or experiment.status == "Live"
        )
