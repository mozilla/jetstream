import pytest
from metric_config_parser.experiment import Experiment

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
