from click.testing import CliRunner
import datetime as dt
from textwrap import dedent
from unittest.mock import Mock

import attr
import os
import pytest
from pytz import UTC
from unittest import mock

from jetstream import cli, experimenter, external_config
from jetstream.config import AnalysisSpec


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
                start_date=dt.datetime(2020, 1, 1, tzinfo=UTC),
                end_date=dt.datetime(2021, 2, 1, tzinfo=UTC),
                proposed_enrollment=None,
                reference_branch="control",
                is_high_population=False,
                app_name="firefox_desktop",
                app_id="firefox-desktop",
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
                start_date=dt.datetime(2020, 1, 1, tzinfo=UTC),
                end_date=dt.datetime(2020, 12, 31, tzinfo=UTC),
                proposed_enrollment=None,
                reference_branch="control",
                is_high_population=False,
                app_name="firefox_desktop",
                app_id="firefox-desktop",
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

    def test_validate_example_config(self, runner):
        with runner.isolated_filesystem():
            conf = dedent(
                """
                [experiment]
                start_date = "2020-12-31"
                end_date = "2021-02-01"
                """
            )

            with open("example_config.toml.example", "w") as config:
                config.write(conf)

            result = runner.invoke(cli.validate_config, ["example_config.toml.example"])

            assert "Skipping example config" in result.output
            assert result.exit_code == 0

    def test_validate_example_outcome_config(self, runner):
        with runner.isolated_filesystem():
            conf = dedent(
                """
                friendly_name = "outcome"
                """
            )

            os.makedirs("outcomes/fenix")
            with open("outcomes/fenix/example_config.toml.example", "w") as config:
                config.write(conf)

            result = runner.invoke(
                cli.validate_config, ["outcomes/fenix/example_config.toml.example"]
            )

            assert "Skipping example config" in result.output
            assert result.exit_code == 0


@attr.s(auto_attribs=True)
class DummyExecutorStrategy:
    project_id: str
    dataset_id: str
    return_value: bool = True

    def execute(self, worklist, configuration_map={}):
        self.worklist = worklist
        return self.return_value


class TestAnalysisExecutor:
    def test_trivial_case(self):
        executor = cli.AnalysisExecutor(
            project_id="project",
            dataset_id="dataset",
            bucket="bucket",
            date=dt.datetime(2020, 10, 28, tzinfo=UTC),
            experiment_slugs=[],
        )
        strategy = DummyExecutorStrategy("project", "dataset")
        success = executor.execute(
            experiment_getter=experimenter.ExperimentCollection,
            config_getter=external_config.ExternalConfigCollection,
            strategy=strategy,
        )
        assert success
        assert strategy.worklist == []

    def test_single_date(self, cli_experiments):
        executor = cli.AnalysisExecutor(
            project_id="project",
            dataset_id="dataset",
            bucket="bucket",
            date=dt.datetime(2020, 10, 28, tzinfo=UTC),
            experiment_slugs=["my_cool_experiment"],
        )
        strategy = DummyExecutorStrategy("project", "dataset")
        success = executor.execute(
            experiment_getter=lambda: cli_experiments,
            config_getter=external_config.ExternalConfigCollection,
            strategy=strategy,
        )
        assert success
        assert len(strategy.worklist) == 1
        assert strategy.worklist[0][0] == "my_cool_experiment"
        assert strategy.worklist[0][1] == dt.datetime(2020, 10, 28, tzinfo=UTC)

    def test_all_single_date(self, cli_experiments):
        executor = cli.AnalysisExecutor(
            project_id="project",
            dataset_id="dataset",
            bucket="bucket",
            date=dt.datetime(2020, 10, 28, tzinfo=UTC),
            experiment_slugs=cli.All,
        )
        strategy = DummyExecutorStrategy("project", "dataset")
        success = executor.execute(
            experiment_getter=lambda: cli_experiments,
            config_getter=external_config.ExternalConfigCollection,
            strategy=strategy,
        )
        assert success
        assert len(strategy.worklist) == 2
        assert {slug for slug, _ in strategy.worklist} == {
            x.normandy_slug for x in cli_experiments.experiments
        }

    def test_any_date(self, cli_experiments):
        executor = cli.AnalysisExecutor(
            project_id="project",
            dataset_id="dataset",
            bucket="bucket",
            date=cli.All,
            experiment_slugs=["my_cool_experiment"],
        )
        strategy = DummyExecutorStrategy("project", "dataset")
        success = executor.execute(
            experiment_getter=lambda: cli_experiments,
            config_getter=external_config.ExternalConfigCollection,
            strategy=strategy,
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
            bucket="bucket",
            date=cli.All,
            experiment_slugs=cli.All,
        )
        strategy = DummyExecutorStrategy("project", "dataset")
        with pytest.raises(ValueError):
            executor.execute(
                experiment_getter=lambda: cli_experiments,
                config_getter=external_config.ExternalConfigCollection,
                strategy=strategy,
                today=dt.datetime(2020, 12, 31, tzinfo=UTC),
            )

    def test_bogus_experiment(self, cli_experiments):
        executor = cli.AnalysisExecutor(
            project_id="project",
            dataset_id="dataset",
            bucket="bucket",
            date=cli.All,
            experiment_slugs=["bogus_experiment", "my_cool_experiment"],
        )
        strategy = DummyExecutorStrategy("project", "dataset")
        executor.execute(
            experiment_getter=lambda: cli_experiments,
            config_getter=external_config.ExternalConfigCollection,
            strategy=strategy,
            today=dt.datetime(2020, 12, 31, tzinfo=UTC),
        )
        assert set(w[0] for w in strategy.worklist) == {"my_cool_experiment"}

    def test_experiments_to_analyze(self, cli_experiments):
        executor = cli.AnalysisExecutor(
            project_id="project",
            dataset_id="dataset",
            bucket="bucket",
            date=cli.All,
            experiment_slugs=["bogus_experiment", "my_cool_experiment"],
        )
        result = executor._experiments_to_analyse(lambda: cli_experiments)
        assert set(e.normandy_slug for e in result) == {"my_cool_experiment"}

    def test_experiments_to_analyze_all(self, cli_experiments):
        executor = cli.AnalysisExecutor(
            project_id="project",
            dataset_id="dataset",
            bucket="bucket",
            date=cli.All,
            experiment_slugs=cli.All,
        )

        with pytest.raises(ValueError):
            executor._experiments_to_analyse(lambda: cli_experiments)

    def test_experiments_to_analyze_specific_date(self, cli_experiments):
        executor = cli.AnalysisExecutor(
            project_id="project",
            dataset_id="dataset",
            bucket="bucket",
            date=dt.datetime(2020, 10, 31, tzinfo=UTC),
            experiment_slugs=cli.All,
        )

        result = executor._experiments_to_analyse(lambda: cli_experiments)
        assert len(result) == 2

    def test_ensure_enrollments(self, cli_experiments, monkeypatch):
        executor = cli.AnalysisExecutor(
            project_id="project",
            dataset_id="dataset",
            bucket="bucket",
            date=cli.All,
            experiment_slugs=["my_cool_experiment"],
        )

        Analysis = Mock()
        monkeypatch.setattr("jetstream.cli.Analysis", Analysis)

        executor.ensure_enrollments(
            recreate_enrollments=False, experiment_getter=lambda: cli_experiments
        )

        assert Analysis.ensure_enrollments.called_once()


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
        spec = AnalysisSpec.default_for_experiment(experiment)
        strategy = cli.SerialExecutorStrategy(
            "spam", "eggs", "bucket", False, fake_analysis, lambda: cli_experiments
        )
        run_date = dt.datetime(2020, 10, 31, tzinfo=UTC)
        strategy.execute([(experiment.normandy_slug, run_date)])
        fake_analysis.assert_called_once_with("spam", "eggs", spec.resolve(experiment))
        fake_analysis().run.assert_called_once_with(run_date)


class TestArgoExecutorStrategy:
    def test_simple_workflow(self, cli_experiments):
        experiment = cli_experiments.experiments[0]

        with mock.patch("jetstream.cli.submit_workflow") as submit_workflow_mock:
            strategy = cli.ArgoExecutorStrategy(
                "spam",
                "eggs",
                "bucket",
                "zone",
                "cluster_id",
                False,
                False,
                None,
                None,
                lambda: cli_experiments,
            )
            run_date = dt.datetime(2020, 10, 31, tzinfo=UTC)
            strategy.execute([(experiment.normandy_slug, run_date)])

            submit_workflow_mock.assert_called_once_with(
                project_id="spam",
                zone="zone",
                cluster_id="cluster_id",
                workflow_file=strategy.RUN_WORKFLOW,
                parameters={
                    "experiments": [{"slug": "my_cool_experiment", "dates": ["2020-10-31"]}],
                    "project_id": "spam",
                    "dataset_id": "eggs",
                    "bucket": "bucket",
                    "recreate_enrollments": False,
                },
                monitor_status=False,
                cluster_ip=None,
                cluster_cert=None,
            )
