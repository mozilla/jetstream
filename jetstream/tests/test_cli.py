from click.testing import CliRunner
from datetime import date, datetime
import pytest

from jetstream import cli
from jetstream.experimenter import Experiment


class TestCli:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_inclusive_date_range(self):
        start_date = date(2020, 5, 1)
        end_date = date(2020, 5, 1)
        date_range = list(cli.inclusive_date_range(start_date, end_date))
        assert len(date_range) == 1
        assert date_range[0] == date(2020, 5, 1)

        start_date = date(2020, 5, 1)
        end_date = date(2020, 5, 5)
        date_range = list(cli.inclusive_date_range(start_date, end_date))
        assert len(date_range) == 5
        assert date_range[0] == date(2020, 5, 1)
        assert date_range[4] == date(2020, 5, 5)

    def test_get_active_experiments(self, runner):
        result = runner.invoke(cli.get_active_experiments, ["--start_date=2020-01-01"])
        assert result.exit_code == 0
        assert result.output != ""

    def test_analyse_experiment(self, runner):
        result = runner.invoke(cli.analyse_experiment, [""])
        assert result.exit_code == 2

        epxeriment = Experiment(
            experimenter_slug=None,
            normandy_slug="test",
            type="v6",
            status="Live",
            features=[],
            branches=[],
            start_date=datetime(2019, 12, 1),
            end_date=datetime(2020, 12, 1),
            proposed_enrollment=14,
            reference_branch=None,
            is_high_population=False,
        )
        experiment_config = cli._base64_encode(
            cli.AnalysisRunConfig(datetime(2020, 1, 1), epxeriment, None)
        )
        result = runner.invoke(cli.analyse_experiment, [f"--experiment_config={experiment_config}"])
        assert result.exit_code == 0
