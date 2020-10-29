import attr
import datetime as dt

from pytz import UTC

from jetstream import cli, experimenter, external_config


class TestCli:
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
class DummyExecutorStrategy(cli.ExecutorStrategy):
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
        strategy = DummyExecutorStrategy
        success = executor.execute(
            experiment_getter=experimenter.ExperimentCollection,
            config_getter=external_config.ExternalConfigCollection,
            strategy=strategy,
        )
        assert success
        assert strategy.worklist == []
