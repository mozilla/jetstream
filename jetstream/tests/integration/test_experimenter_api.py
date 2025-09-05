import pytest
from metric_config_parser.experiment import Experiment

from jetstream.experimenter import ExperimentCollection


@pytest.fixture
def experiment_collection(request):
    try:
        collection = ExperimentCollection.from_experimenter(slug=request.param)
        return collection
    except Exception as e:
        pytest.fail(f"Failed to fetch experiment collection: {e!s}")


@pytest.mark.parametrize("experiment_collection", [None], indirect=["experiment_collection"])
def test_from_experimenter(experiment_collection):
    assert len(experiment_collection.experiments) > 1
    for experiment in experiment_collection.experiments:
        assert isinstance(experiment, Experiment)


@pytest.mark.parametrize(
    "experiment_collection", ["automated-segments-test-1"], indirect=["experiment_collection"]
)
def test_from_experimenter_with_slug(experiment_collection):
    experiments = experiment_collection.experiments
    assert len(experiments) == 1
    assert isinstance(experiments[0], Experiment)
