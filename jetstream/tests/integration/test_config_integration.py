import datetime
import re
from pathlib import Path
from textwrap import dedent

import pytest
import pytz
import toml
from metric_config_parser.analysis import AnalysisSpec
from metric_config_parser.config import Config, ConfigCollection, DefaultConfig, Outcome
from metric_config_parser.experiment import Channel, Experiment
from metric_config_parser.metric import AnalysisPeriod
from metric_config_parser.outcome import OutcomeSpec

from jetstream.config import ConfigLoader, validate
from jetstream.dryrun import DryRunFailedError
from jetstream.statistics import LinearModelMean, Summary

TEST_DIR = Path(__file__).parent.parent


class TestConfigIntegration:
    config_str = dedent(
        """
        [metrics]
        weekly = ["view_about_logins"]

        [metrics.view_about_logins.statistics.bootstrap_mean]
        """
    )
    spec = AnalysisSpec.from_dict(toml.loads(config_str))

    def test_old_config(self, client, project_id, temporary_dataset):
        config = Config(
            slug="new_table",
            spec=self.spec,
            last_modified=pytz.UTC.localize(
                datetime.datetime.utcnow() - datetime.timedelta(days=1)
            ),
        )

        # table created after config loaded
        client.client.create_table(f"{temporary_dataset}.statistics_new_table_day1")
        client.add_metadata_to_table(
            "statistics_new_table_day1",
            {"last_updated": client._current_timestamp_label()},
        )
        config_collection = ConfigLoader
        config_collection.configs.configs = [config]
        updated_configs = config_collection.updated_configs(project_id, temporary_dataset)

        assert len(updated_configs) == 0

    def test_updated_config(self, client, temporary_dataset, project_id):
        config = Config(
            slug="old_table",
            spec=self.spec,
            last_modified=pytz.UTC.localize(
                datetime.datetime.utcnow() + datetime.timedelta(days=1)
            ),
        )

        client.client.create_table(f"{temporary_dataset}.old_table_day1")
        client.add_metadata_to_table(
            "old_table_day1",
            {"last_updated": client._current_timestamp_label()},
        )
        client.client.create_table(f"{temporary_dataset}.old_table_day2")
        client.add_metadata_to_table(
            "old_table_day2",
            {"last_updated": client._current_timestamp_label()},
        )

        config_collection = ConfigLoader
        config_collection.configs.configs = [config]
        updated_configs = config_collection.updated_configs(project_id, temporary_dataset)

        assert len(updated_configs) == 1
        assert updated_configs[0].slug == config.slug

    def test_updated_config_while_analysis_active(self, client, temporary_dataset, project_id):
        client.client.create_table(f"{temporary_dataset}.active_table_day0")
        client.add_metadata_to_table(
            "active_table_day0",
            {"last_updated": client._current_timestamp_label()},
        )
        client.client.create_table(f"{temporary_dataset}.active_table_day1")
        client.add_metadata_to_table(
            "active_table_day1",
            {"last_updated": client._current_timestamp_label()},
        )

        config = Config(
            slug="active_table",
            spec=self.spec,
            last_modified=pytz.UTC.localize(datetime.datetime.utcnow()),
        )

        client.client.create_table(f"{temporary_dataset}.active_table_day2")
        client.add_metadata_to_table(
            "active_table_day2",
            {"last_updated": client._current_timestamp_label()},
        )
        client.client.create_table(f"{temporary_dataset}.active_table_weekly")
        client.add_metadata_to_table(
            "active_table_weekly",
            {"last_updated": client._current_timestamp_label()},
        )

        config_collection = ConfigLoader
        config_collection.configs.configs = [config]
        updated_configs = config_collection.updated_configs(project_id, temporary_dataset)

        assert len(updated_configs) == 1
        assert updated_configs[0].slug == config.slug

    def test_new_config_without_a_table_is_marked_changed(
        self, client, temporary_dataset, project_id
    ):
        config = Config(
            slug="my_cool_experiment",
            spec=self.spec,
            last_modified=pytz.UTC.localize(datetime.datetime.utcnow()),
        )
        config_collection = ConfigLoader
        config_collection.configs.configs = [config]
        updated_configs = config_collection.updated_configs(project_id, temporary_dataset)
        assert [updated.slug for updated in updated_configs] == ["my_cool_experiment"]

    def test_valid_config_validates(self, experiments):
        extern = Config(
            slug="cool_experiment",
            spec=self.spec,
            last_modified=datetime.datetime.now(),
        )
        extern.validate(ConfigLoader.configs, experiments[0])

    def test_busted_config_fails(self, experiments):
        config = dedent(
            """\
            [metrics]
            weekly = ["bogus_metric"]

            [metrics.bogus_metric]
            select_expression = "SUM(fake_column)"
            data_source = "clients_daily"
            statistics = { bootstrap_mean = {} }
            """
        )
        spec = AnalysisSpec.from_dict(toml.loads(config))
        extern = Config(
            slug="bad_experiment",
            spec=spec,
            last_modified=datetime.datetime.now(),
        )
        with pytest.raises(DryRunFailedError):
            validate(extern, experiments[0])

    def test_valid_outcome_validates(self):
        config = dedent(
            """\
            friendly_name = "Fred"
            description = "Just your average paleolithic dad."

            [metrics.rocks_mined]
            select_expression = "COALESCE(SUM(pings_aggregated_by_this_row), 0)"
            data_source = "clients_daily"
            statistics = { bootstrap_mean = {} }
            friendly_name = "Rocks mined"
            description = "Number of rocks mined at the quarry"
            """
        )
        spec = OutcomeSpec.from_dict(toml.loads(config))
        extern = Outcome(
            slug="good_outcome",
            spec=spec,
            platform="firefox_desktop",
            commit_hash="0000000",
        )

        validate(extern)

    def test_busted_outcome_fails(self):
        config = dedent(
            """\
            friendly_name = "Fred"
            description = "Just your average paleolithic dad."

            [metrics.rocks_mined]
            select_expression = "COALESCE(SUM(fake_column_whoop_whoop), 0)"
            data_source = "clients_daily"
            statistics = { bootstrap_mean = {} }
            friendly_name = "Rocks mined"
            description = "Number of rocks mined at the quarry"
            """
        )
        spec = OutcomeSpec.from_dict(toml.loads(config))
        extern = Outcome(
            slug="bogus_outcome",
            spec=spec,
            platform="firefox_desktop",
            commit_hash="0000000",
        )
        with pytest.raises(DryRunFailedError):
            validate(extern)

    def test_valid_default_config_validates(self):
        extern = DefaultConfig(
            slug="firefox_desktop",
            spec=self.spec,
            last_modified=datetime.datetime.now(),
        )
        validate(extern)

    def test_busted_default_config_fails(self):
        config = dedent(
            """\
            [metrics]
            weekly = ["bogus_metric"]

            [metrics.bogus_metric]
            select_expression = "SUM(fake_column)"
            data_source = "clients_daily"
            statistics = { bootstrap_mean = {} }
            """
        )
        spec = AnalysisSpec.from_dict(toml.loads(config))
        extern = DefaultConfig(
            slug="firefox_desktop",
            spec=spec,
            last_modified=datetime.datetime.now(),
        )
        with pytest.raises(DryRunFailedError):
            validate(extern)

    @pytest.mark.parametrize(
        "period",
        [AnalysisPeriod.OVERALL, AnalysisPeriod.DAY, AnalysisPeriod.DAYS_28, AnalysisPeriod.WEEK],
    )
    def test_linear_models_covariate_parsing(self, period: AnalysisPeriod):
        config = dedent(
            """\
            [metrics]
            weekly = ["bogus_metric"]

            [metrics.bogus_metric]
            select_expression = "SUM(fake_column)"
            data_source = "source_name"

            [metrics.bogus_metric.statistics.linear_model_mean]
            [metrics.bogus_metric.statistics.linear_model_mean.covariate_adjustment]
            metric = "bogus_metric"
            period = "preenrollment_week"

            [data_sources]
            [data_sources.source_name]
            from_expression = "project.dataset.table"
            friendly_name = "Source"
            description = "Source"
            """
        )

        spec = AnalysisSpec.from_dict(toml.loads(config))

        dummy_experiment = Experiment(
            experimenter_slug="dummy-experiment",
            normandy_slug="dummy_experiment",
            type="v6",
            status="Live",
            branches=[],
            end_date=None,
            reference_branch="control",
            is_high_population=False,
            start_date=datetime.datetime.now(pytz.UTC),
            proposed_enrollment=14,
            app_name="desktop",
            channel=Channel.NIGHTLY,
        )

        external_configs = ConfigCollection(
            [
                Config(
                    slug="dummy-experiment",
                    spec=spec,
                    last_modified=datetime.datetime(2021, 2, 15, tzinfo=pytz.UTC),
                )
            ]
        )

        analysis_configuration = spec.resolve(dummy_experiment, external_configs)

        summary = analysis_configuration.metrics[AnalysisPeriod.WEEK][0]
        assert summary.metric.name == "bogus_metric"

        statistic = summary.statistic
        assert statistic.name == "linear_model_mean"

        covariate_params = statistic.params.get("covariate_adjustment")
        assert covariate_params["metric"] == "bogus_metric"
        assert AnalysisPeriod(covariate_params["period"]) == AnalysisPeriod.PREENROLLMENT_WEEK

        jetstream_statistic = Summary.from_config(summary, 7, period).statistic

        assert isinstance(jetstream_statistic, LinearModelMean)  # make mypy happy

        assert jetstream_statistic.covariate_adjustment == {
            "metric": "bogus_metric",
            "period": "preenrollment_week",
        }

        assert jetstream_statistic.period == period

    @pytest.mark.parametrize(
        "period",
        [AnalysisPeriod.OVERALL, AnalysisPeriod.DAY, AnalysisPeriod.DAYS_28, AnalysisPeriod.WEEK],
    )
    def test_linear_models_covariate_parsing_bad_period(self, period: AnalysisPeriod):
        config = dedent(
            """\
            [metrics]
            weekly = ["bogus_metric"]

            [metrics.bogus_metric]
            select_expression = "SUM(fake_column)"
            data_source = "source_name"

            [metrics.bogus_metric.statistics.linear_model_mean]
            [metrics.bogus_metric.statistics.linear_model_mean.covariate_adjustment]
            period = "overall"

            [data_sources]
            [data_sources.source_name]
            from_expression = "project.dataset.table"
            friendly_name = "Source"
            description = "Source"
            """
        )

        spec = AnalysisSpec.from_dict(toml.loads(config))

        dummy_experiment = Experiment(
            experimenter_slug="dummy-experiment",
            normandy_slug="dummy_experiment",
            type="v6",
            status="Live",
            branches=[],
            end_date=None,
            reference_branch="control",
            is_high_population=False,
            start_date=datetime.datetime.now(pytz.UTC),
            proposed_enrollment=14,
            app_name="desktop",
            channel=Channel.NIGHTLY,
        )

        external_configs = ConfigCollection(
            [
                Config(
                    slug="dummy-experiment",
                    spec=spec,
                    last_modified=datetime.datetime(2021, 2, 15, tzinfo=pytz.UTC),
                )
            ]
        )

        analysis_configuration = spec.resolve(dummy_experiment, external_configs)

        summary = analysis_configuration.metrics[AnalysisPeriod.WEEK][0]

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Covariate adjustment must be done using a pre-treatment analysis period (one of: ['preenrollment_week', 'preenrollment_days28'])"  # noqa: E501
            ),
        ):
            Summary.from_config(summary, 7, period)
