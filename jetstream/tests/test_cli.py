import attr
import datetime as dt

from pytz import UTC

from jetstream import cli, experimenter, external_config
from click.testing import CliRunner
from datetime import date, datetime
import pytest
from unittest.mock import MagicMock

from jetstream import cli
from jetstream.experimenter import Experiment
from jetstream.analysis import Analysis
from jetstream.config import AnalysisSpec


class TestCli:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_inclusive_date_range(self):
        start_date = dt.date(2020, 5, 1)
        end_date = dt.date(2020, 5, 1)
        date_range = list(cli.inclusive_date_range(start_date, end_date))
        assert len(date_range) == 1
        assert date_range[0] == dt.date(2020, 5, 1)

        start_date = dt.date(2020, 5, 1)
        end_date = dt.date(2020, 5, 5)
        date_range = list(cli.inclusive_date_range(start_date, end_date))
        assert len(date_range) == 5
        assert date_range[0] == dt.date(2020, 5, 1)
        assert date_range[4] == dt.date(2020, 5, 5)


@attr.s(auto_attribs=True)
class DummyExecutorStrategy:
    project_id: str
    dataset_id: str
    return_value: bool = True

    def execute(self, worklist):
        self.worklist = worklist
        return self.return_value


class TestAnalysisExecutor:
    def test_trivial_case(self):
        analysis_class_mock = MagicMock(spec=Analysis)
        executor = cli.AnalysisExecutor(
            project_id="project",
            dataset_id="dataset",
            date=dt.datetime(2020, 10, 28, tzinfo=UTC),
            experiment_slugs=[],
            analysis_class=analysis_class_mock,
        )

        success = executor.execute(
            experiment_getter=experimenter.ExperimentCollection,
            config_getter=external_config.ExternalConfigCollection,
        )
        assert success
        assert analysis_class_mock.called is False

    def test_single_date(self):
        analysis_class_mock_instance = MagicMock()
        analysis_class_mock = MagicMock(spec=Analysis, return_value=analysis_class_mock_instance)
        executor = cli.AnalysisExecutor(
            project_id="project",
            dataset_id="dataset",
            date=dt.datetime(2020, 10, 28, tzinfo=UTC),
            experiment_slugs=["my_cool_experiment"],
            analysis_class=analysis_class_mock,
        )
        experiments = experimenter.ExperimentCollection(
            [
                experimenter.Experiment(
                    experimenter_slug=None,
                    normandy_slug="my_cool_experiment",
                    type="v6",
                    status="Live",
                    branches=[
                        experimenter.Branch(slug="treatment", ratio=1),
                        experimenter.Branch(slug="control", ratio=1),
                    ],
                    probe_sets=[],
                    start_date=dt.datetime(2020, 1, 1, tzinfo=UTC),
                    end_date=dt.datetime(2020, 12, 31, tzinfo=UTC),
                    proposed_enrollment=None,
                    reference_branch="control",
                    is_high_population=False,
                ),
                experimenter.Experiment(
                    experimenter_slug=None,
                    normandy_slug="distracting_experiment",
                    type="v6",
                    status="Live",
                    branches=[
                        experimenter.Branch(slug="treatment", ratio=1),
                        experimenter.Branch(slug="control", ratio=1),
                    ],
                    probe_sets=[],
                    start_date=dt.datetime(2020, 1, 1, tzinfo=UTC),
                    end_date=dt.datetime(2020, 12, 31, tzinfo=UTC),
                    proposed_enrollment=None,
                    reference_branch="control",
                    is_high_population=False,
                ),
            ]
        )
        success = executor.execute(
            experiment_getter=lambda: experiments,
            config_getter=external_config.ExternalConfigCollection,
        )

        spec = AnalysisSpec.default_for_experiment(experiments.experiments[0])
        config = spec.resolve(experiments.experiments[0])

        assert success
        analysis_class_mock.assert_called_once()
        analysis_class_mock.assert_called_with("project", "dataset", config)
        analysis_class_mock_instance.run.assert_called_with(dt.datetime(2020, 10, 28, tzinfo=UTC))

    def test_any_date(self):
        analysis_class_mock_instance = MagicMock()
        analysis_class_mock = MagicMock(spec=Analysis, return_value=analysis_class_mock_instance)
        executor = cli.AnalysisExecutor(
            project_id="project",
            dataset_id="dataset",
            date=cli.All,
            experiment_slugs=["my_cool_experiment"],
            analysis_class=analysis_class_mock,
        )
        experiments = experimenter.ExperimentCollection(
            [
                experimenter.Experiment(
                    experimenter_slug=None,
                    normandy_slug="my_cool_experiment",
                    type="v6",
                    status="Live",
                    branches=[
                        experimenter.Branch(slug="treatment", ratio=1),
                        experimenter.Branch(slug="control", ratio=1),
                    ],
                    probe_sets=[],
                    start_date=dt.datetime(2020, 1, 1, tzinfo=UTC),
                    end_date=dt.datetime(2021, 2, 1, tzinfo=UTC),
                    proposed_enrollment=None,
                    reference_branch="control",
                    is_high_population=False,
                ),
                experimenter.Experiment(
                    experimenter_slug=None,
                    normandy_slug="distracting_experiment",
                    type="v6",
                    status="Live",
                    branches=[
                        experimenter.Branch(slug="treatment", ratio=1),
                        experimenter.Branch(slug="control", ratio=1),
                    ],
                    probe_sets=[],
                    start_date=dt.datetime(2020, 1, 1, tzinfo=UTC),
                    end_date=dt.datetime(2020, 12, 31, tzinfo=UTC),
                    proposed_enrollment=None,
                    reference_branch="control",
                    is_high_population=False,
                ),
            ]
        )
        success = executor.execute(
            experiment_getter=lambda: experiments,
            config_getter=external_config.ExternalConfigCollection,
            today=dt.datetime(2020, 12, 31, tzinfo=UTC),
        )
        assert success
        assert analysis_class_mock.call_count == 366
        # todo
        # print([w.experiment.normandy_slug for w in analysis_class_mock.call_args[0]])
        # assert date_range[0] == date(2020, 5, 1)
        # assert date_range[4] == date(2020, 5, 5)
