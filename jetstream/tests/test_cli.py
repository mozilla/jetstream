import datetime as dt
import os
from textwrap import dedent
from unittest import mock
from unittest.mock import Mock

import attr
import pytest
import toml
from click.testing import CliRunner
from metric_config_parser.analysis import AnalysisSpec
from metric_config_parser.config import Config, ConfigCollection
from metric_config_parser.experiment import Branch, Experiment
from metric_config_parser.metric import AnalysisPeriod
from pytz import UTC

from jetstream import cli, experimenter
from jetstream.config import ConfigLoader, _ConfigLoader


@pytest.fixture(name="cli_experiments")
def cli_experiment_fixture():
    return cli_experiments()


def cli_experiments():
    return experimenter.ExperimentCollection(
        [
            Experiment(
                experimenter_slug=None,
                normandy_slug="my_cool_experiment",
                type="v6",
                status="Live",
                branches=[
                    Branch(slug="treatment", ratio=1),
                    Branch(slug="control", ratio=1),
                ],
                start_date=dt.datetime(2020, 1, 1, tzinfo=UTC),
                end_date=dt.datetime(2021, 2, 1, tzinfo=UTC),
                proposed_enrollment=None,
                reference_branch="control",
                is_high_population=False,
                app_name="firefox_desktop",
                app_id="firefox-desktop",
            ),
            Experiment(
                experimenter_slug=None,
                normandy_slug="distracting_experiment",
                type="v6",
                status="Live",
                branches=[
                    Branch(slug="treatment", ratio=1),
                    Branch(slug="control", ratio=1),
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

    def test_validate_example_config(self, runner, monkeypatch):
        monkeypatch.setattr("jetstream.cli.ExperimentCollection.from_experimenter", cli_experiments)
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

    def test_validate_private_example_config(self, runner, monkeypatch):
        monkeypatch.setattr("jetstream.cli.ExperimentCollection.from_experimenter", cli_experiments)
        with runner.isolated_filesystem():
            conf = dedent(
                """
                [experiment]
                start_date = "2020-12-31"
                end_date = "2021-02-01"
                proposed_enrollment = 7
                enrollment_period = 7
                dataset_id = "test"
                """
            )

            with open("my_cool_experiment.toml", "w") as config:
                config.write(conf)

            result = runner.invoke(cli.validate_config, ["my_cool_experiment.toml", "--is_private"])

            assert result.exit_code == 0

    def test_validate_private_example_config_no_dataset(self, runner, monkeypatch):
        monkeypatch.setattr("jetstream.cli.ExperimentCollection.from_experimenter", cli_experiments)
        with runner.isolated_filesystem():
            conf = dedent(
                """
                [experiment]
                start_date = "2020-12-31"
                end_date = "2021-02-01"
                proposed_enrollment = 7
                enrollment_period = 7
                """
            )

            with open("my_cool_experiment.toml", "w") as config:
                config.write(conf)

            result = runner.invoke(cli.validate_config, ["my_cool_experiment.toml", "--is_private"])
            assert result.exit_code == 1
            assert "dataset_id needs to be explicitly set for private experiments" in str(result)

    def test_validate_example_outcome_config(self, runner, monkeypatch):
        monkeypatch.setattr("jetstream.cli.ExperimentCollection.from_experimenter", cli_experiments)
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
            config_getter=ConfigLoader,
            strategy=strategy,
        )
        assert success
        assert strategy.worklist == []

    def test_single_date(self, monkeypatch):
        executor = cli.AnalysisExecutor(
            project_id="project",
            dataset_id="dataset",
            bucket="bucket",
            date=dt.datetime(2020, 10, 28, tzinfo=UTC),
            experiment_slugs=["my_cool_experiment"],
        )

        bigquery_mock_client = Mock()
        monkeypatch.setattr("jetstream.cli.BigQueryClient", bigquery_mock_client)

        strategy = DummyExecutorStrategy("project", "dataset")
        success = executor.execute(
            experiment_getter=cli_experiments,
            config_getter=ConfigLoader,
            strategy=strategy,
        )
        assert success
        assert len(strategy.worklist) == 1
        assert strategy.worklist[0][0].experiment.normandy_slug == "my_cool_experiment"
        assert strategy.worklist[0][1] == dt.datetime(2020, 10, 28, tzinfo=UTC)
        assert bigquery_mock_client.called is False

    def test_recreate_enrollments(self, monkeypatch):
        executor = cli.AnalysisExecutor(
            project_id="project",
            dataset_id="dataset",
            bucket="bucket",
            date=dt.datetime(2020, 10, 28, tzinfo=UTC),
            experiment_slugs=["my_cool_experiment"],
            recreate_enrollments=True,
        )

        bigquery_mock_client = Mock()
        monkeypatch.setattr("jetstream.cli.BigQueryClient", bigquery_mock_client)
        strategy = DummyExecutorStrategy("project", "dataset")
        success = executor.execute(
            experiment_getter=cli_experiments,
            config_getter=ConfigLoader,
            strategy=strategy,
        )
        assert success
        assert len(strategy.worklist) == 1
        assert strategy.worklist[0][0].experiment.normandy_slug == "my_cool_experiment"
        assert strategy.worklist[0][1] == dt.datetime(2020, 10, 28, tzinfo=UTC)
        assert bigquery_mock_client.delete_table.called_once_with(
            "project.dataset.enrollments_my_cool_experiment"
        )

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
            config_getter=ConfigLoader,
            strategy=strategy,
        )
        assert success
        assert len(strategy.worklist) == 2
        assert {c.experiment.normandy_slug for c, _ in strategy.worklist} == {
            x.normandy_slug for x in cli_experiments.experiments
        }

    def test_any_date(self):
        executor = cli.AnalysisExecutor(
            project_id="project",
            dataset_id="dataset",
            bucket="bucket",
            date=cli.All,
            experiment_slugs=["my_cool_experiment"],
        )
        strategy = DummyExecutorStrategy("project", "dataset")
        success = executor.execute(
            experiment_getter=cli_experiments,
            config_getter=ConfigLoader,
            strategy=strategy,
            today=dt.datetime(2020, 12, 31, tzinfo=UTC),
        )
        assert success
        assert len(strategy.worklist) == 366
        assert set(w[0].experiment.normandy_slug for w in strategy.worklist) == {
            "my_cool_experiment"
        }

    def test_post_facto_rerun_includes_overall_date(self):
        executor = cli.AnalysisExecutor(
            project_id="project",
            dataset_id="dataset",
            bucket="bucket",
            date=cli.All,
            experiment_slugs=["my_cool_experiment"],
        )
        strategy = DummyExecutorStrategy("project", "dataset")
        success = executor.execute(
            experiment_getter=cli_experiments,
            config_getter=ConfigLoader,
            strategy=strategy,
            today=dt.datetime(2022, 12, 31, tzinfo=UTC),
        )
        assert success
        assert max(w[1] for w in strategy.worklist).date() == dt.date(2021, 2, 2)

    def test_bartleby(self):
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
                experiment_getter=cli_experiments,
                config_getter=ConfigLoader,
                strategy=strategy,
                today=dt.datetime(2020, 12, 31, tzinfo=UTC),
            )

    def test_bogus_experiment(self):
        executor = cli.AnalysisExecutor(
            project_id="project",
            dataset_id="dataset",
            bucket="bucket",
            date=cli.All,
            experiment_slugs=["bogus_experiment", "my_cool_experiment"],
        )
        strategy = DummyExecutorStrategy("project", "dataset")
        executor.execute(
            experiment_getter=cli_experiments,
            config_getter=ConfigLoader,
            strategy=strategy,
            today=dt.datetime(2020, 12, 31, tzinfo=UTC),
        )
        assert set(w[0].experiment.normandy_slug for w in strategy.worklist) == {
            "my_cool_experiment"
        }

    def test_experiments_to_analyze(self):
        executor = cli.AnalysisExecutor(
            project_id="project",
            dataset_id="dataset",
            bucket="bucket",
            date=cli.All,
            experiment_slugs=["bogus_experiment", "my_cool_experiment"],
        )
        result = executor._experiment_configs_to_analyse(cli_experiments, ConfigLoader)
        assert set(e.experiment.normandy_slug for e in result) == {"my_cool_experiment"}

    def test_experiments_to_analyze_end_date_override(self):
        executor = cli.AnalysisExecutor(
            project_id="project",
            dataset_id="dataset",
            bucket="bucket",
            date=dt.datetime(2021, 2, 15, tzinfo=UTC),
            experiment_slugs=cli.All,
        )
        result = executor._experiment_configs_to_analyse(cli_experiments, ConfigLoader)
        assert result == []

        conf = dedent(
            """
            [experiment]
            end_date = 2021-03-01
            """
        )

        external_configs = ConfigCollection(
            [
                Config(
                    slug="my_cool_experiment",
                    spec=AnalysisSpec.from_dict(toml.loads(conf)),
                    last_modified=dt.datetime(2021, 2, 15, tzinfo=UTC),
                )
            ]
        )

        config_loader = _ConfigLoader()
        config_loader.config_collection = external_configs
        result = executor._experiment_configs_to_analyse(cli_experiments, config_loader)
        assert set(e.experiment.normandy_slug for e in result) == {"my_cool_experiment"}

    def test_experiments_to_analyze_all(self):
        executor = cli.AnalysisExecutor(
            project_id="project",
            dataset_id="dataset",
            bucket="bucket",
            date=cli.All,
            experiment_slugs=cli.All,
        )

        with pytest.raises(ValueError):
            executor._experiment_configs_to_analyse(cli_experiments)

    def test_experiments_to_analyze_specific_date(self):
        executor = cli.AnalysisExecutor(
            project_id="project",
            dataset_id="dataset",
            bucket="bucket",
            date=dt.datetime(2020, 10, 31, tzinfo=UTC),
            experiment_slugs=cli.All,
        )

        result = executor._experiment_configs_to_analyse(cli_experiments, ConfigLoader)
        assert len(result) == 2

    def test_ensure_enrollments(self, monkeypatch):
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
            experiment_getter=cli_experiments,
            config_getter=ConfigLoader,
        )

        assert Analysis.ensure_enrollments.called_once()


class TestSerialExecutorStrategy:
    def test_trivial_workflow(self, monkeypatch):
        monkeypatch.setattr("jetstream.cli.export_metadata", Mock())
        monkeypatch.setattr("jetstream.cli.export_experiment_logs", Mock())
        fake_analysis = Mock()
        strategy = cli.SerialExecutorStrategy(
            project_id="spam",
            dataset_id="eggs",
            bucket="bucket",
            analysis_class=fake_analysis,
            experiment_getter=cli_experiments,
            config_getter=ConfigLoader,
        )
        strategy.execute([])
        fake_analysis().run.assert_not_called()

    def test_simple_workflow(self, cli_experiments, monkeypatch):
        monkeypatch.setattr("jetstream.cli.export_metadata", Mock())
        monkeypatch.setattr("jetstream.cli.export_experiment_logs", Mock())
        fake_analysis = Mock()
        experiment = cli_experiments.experiments[0]
        spec = AnalysisSpec.default_for_experiment(experiment, ConfigLoader.configs)
        strategy = cli.SerialExecutorStrategy(
            project_id="spam",
            dataset_id="eggs",
            bucket="bucket",
            analysis_class=fake_analysis,
            experiment_getter=lambda: cli_experiments,
            config_getter=ConfigLoader,
        )
        config = spec.resolve(experiment, ConfigLoader.configs)
        run_date = dt.datetime(2020, 10, 31, tzinfo=UTC)
        strategy.execute([(config, run_date)])
        fake_analysis.assert_called_once_with(
            "spam",
            "eggs",
            config,
            None,
            None,
            [
                AnalysisPeriod.DAY,
                AnalysisPeriod.WEEK,
                AnalysisPeriod.DAYS_28,
                AnalysisPeriod.OVERALL,
            ],
            None,
        )
        fake_analysis().run.assert_called_once_with(run_date)


class TestArgoExecutorStrategy:
    def test_simple_workflow(self, cli_experiments):
        experiment = cli_experiments.experiments[0]
        spec = AnalysisSpec.default_for_experiment(experiment, ConfigLoader.configs)
        config = spec.resolve(experiment, ConfigLoader.configs)

        with mock.patch("jetstream.cli.submit_workflow") as submit_workflow_mock:
            strategy = cli.ArgoExecutorStrategy(
                project_id="spam",
                dataset_id="eggs",
                bucket="bucket",
                zone="zone",
                cluster_id="cluster_id",
                monitor_status=False,
                cluster_ip=None,
                cluster_cert=None,
                experiment_getter=lambda: cli_experiments,
                analysis_periods=[
                    AnalysisPeriod.DAY,
                    AnalysisPeriod.WEEK,
                    AnalysisPeriod.DAYS_28,
                    AnalysisPeriod.OVERALL,
                ],
            )
            run_date = dt.datetime(2020, 10, 31, tzinfo=UTC)
            strategy.execute([(config, run_date)])

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
                    "analysis_periods_day": "day",
                    "analysis_periods_week": "week",
                    "analysis_periods_days28": "days28",
                    "analysis_periods_overall": "overall",
                },
                monitor_status=False,
                cluster_ip=None,
                cluster_cert=None,
            )
