import datetime as dt
from unittest import mock
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from metric_config_parser.experiment import Branch, Experiment
from metric_config_parser.metric import AnalysisPeriod
from pytz import UTC

from jetstream import cli
from jetstream.config import ConfigLoader


class TestCliIntegration:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_experiments_to_configs_as_of(self):
        with mock.patch("jetstream.cli.BigQueryClient") as fixture:
            bigquery_mock_client = MagicMock()
            bigquery_mock_client.experiment_table_first_updated.return_value = dt.datetime(
                2023, 5, 28, tzinfo=UTC
            )  # 0f92ef5
            fixture.return_value = bigquery_mock_client

            executor = cli.AnalysisExecutor(
                project_id="project",
                dataset_id="dataset",
                bucket="bucket",
                date=dt.datetime(2020, 10, 31, tzinfo=UTC),
                experiment_slugs=cli.All,
            )

            config = executor._experiments_to_configs(
                experiments=[
                    Experiment(
                        experimenter_slug="1-click-pin-experiment",
                        normandy_slug="1-click-pin-experiment",
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
                        outcomes=["networking"],
                    )
                ],
                config_getter=ConfigLoader,
            )[0]

            found_dns_lookup = False
            for overall_metric in config.metrics[AnalysisPeriod.OVERALL]:
                if overall_metric.metric.name == "time_to_response_start_ms":
                    assert False

                if overall_metric.metric.name == "dns_lookup_time":
                    found_dns_lookup = True

            assert found_dns_lookup
