import attr
import datetime as dt

import pytest
from pytz import UTC
from unittest.mock import Mock

from jetstream import cli, experimenter, external_config


@pytest.fixture
def cli_experiments():
    return experimenter.ExperimentCollection(
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
        executor = cli.AnalysisExecutor(
            project_id="project",
            dataset_id="dataset",
            date=dt.datetime(2020, 10, 28, tzinfo=UTC),
            experiment_slugs=[],
        )
        strategy = DummyExecutorStrategy("project", "dataset")
        success = executor.execute(
            experiment_getter=experimenter.ExperimentCollection,
            config_getter=external_config.ExternalConfigCollection,
            strategy=lambda *args, **kwargs: strategy,
        )
        assert success
        assert strategy.worklist == []

    def test_single_date(self, cli_experiments):
        executor = cli.AnalysisExecutor(
            project_id="project",
            dataset_id="dataset",
            date=dt.datetime(2020, 10, 28, tzinfo=UTC),
            experiment_slugs=["my_cool_experiment"],
        )
        strategy = DummyExecutorStrategy("project", "dataset")
        success = executor.execute(
            experiment_getter=lambda: cli_experiments,
            config_getter=external_config.ExternalConfigCollection,
            strategy=lambda *args, **kwargs: strategy,
        )
        assert success
        assert len(strategy.worklist) == 1
        assert strategy.worklist[0][0] == "my_cool_experiment"
        assert strategy.worklist[0][2] == dt.datetime(2020, 10, 28, tzinfo=UTC)

    def test_all_single_date(self, cli_experiments):
        executor = cli.AnalysisExecutor(
            project_id="project",
            dataset_id="dataset",
            date=dt.datetime(2020, 10, 28, tzinfo=UTC),
            experiment_slugs=cli.All,
        )
        strategy = DummyExecutorStrategy("project", "dataset")
        success = executor.execute(
            experiment_getter=lambda: cli_experiments,
            config_getter=external_config.ExternalConfigCollection,
            strategy=lambda *args, **kwargs: strategy,
        )
        assert success
        assert len(strategy.worklist) == 2
        assert {slug for slug, _, _ in strategy.worklist} == {
            x.normandy_slug for x in cli_experiments.experiments
        }

    def test_any_date(self, cli_experiments):
        executor = cli.AnalysisExecutor(
            project_id="project",
            dataset_id="dataset",
            date=cli.All,
            experiment_slugs=["my_cool_experiment"],
        )
        strategy = DummyExecutorStrategy("project", "dataset")
        success = executor.execute(
            experiment_getter=lambda: cli_experiments,
            config_getter=external_config.ExternalConfigCollection,
            strategy=lambda *args, **kwargs: strategy,
            today=dt.datetime(2020, 12, 31, tzinfo=UTC),
        )
        assert success
        assert len(strategy.worklist) == 366
        assert set(w[0] for w in strategy.worklist) == {"my_cool_experiment"}

    def test_bartleby(self, cli_experiments):
        "'I would prefer not to.' - Bartleby"
        executor = cli.AnalysisExecutor(
            project_id="project",
            dataset_id="dataset",
            date=cli.All,
            experiment_slugs=cli.All,
        )
        strategy = DummyExecutorStrategy("project", "dataset")
        with pytest.raises(ValueError):
            executor.execute(
                experiment_getter=lambda: cli_experiments,
                config_getter=external_config.ExternalConfigCollection,
                strategy=lambda *args, **kwargs: strategy,
                today=dt.datetime(2020, 12, 31, tzinfo=UTC),
            )


class TestSerialExecutorStrategy:
    def test_trivial_workflow(self, cli_experiments):
        fake_analysis = Mock()
        strategy = cli.SerialExecutorStrategy(
            "spam", "eggs", fake_analysis, lambda: cli_experiments
        )
        strategy.execute([])
        fake_analysis().run.assert_not_called()

    def test_simple_workflow(self, cli_experiments):
        fake_analysis = Mock()
        experiment = cli_experiments.experiments[0]
        spec = cli.default_spec_for_experiment(experiment)
        strategy = cli.SerialExecutorStrategy(
            "spam", "eggs", fake_analysis, lambda: cli_experiments
        )
        run_date = dt.datetime(2020, 10, 31, tzinfo=UTC)
        strategy.execute([(experiment.normandy_slug, spec, run_date)])
        fake_analysis.assert_called_once_with("spam", "eggs", spec.resolve(experiment))
        fake_analysis().run.assert_called_once_with(run_date)
