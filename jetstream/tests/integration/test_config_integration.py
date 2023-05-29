import datetime
from pathlib import Path
from textwrap import dedent

import pytest
import pytz
import toml
from metric_config_parser.analysis import AnalysisSpec
from metric_config_parser.config import Config, DefaultConfig, Outcome
from metric_config_parser.outcome import OutcomeSpec

from jetstream.config import ConfigLoader, validate
from jetstream.dryrun import DryRunFailedError

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
        client.add_labels_to_table(
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
        client.add_labels_to_table(
            "old_table_day1",
            {"last_updated": client._current_timestamp_label()},
        )
        client.client.create_table(f"{temporary_dataset}.old_table_day2")
        client.add_labels_to_table(
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
        client.add_labels_to_table(
            "active_table_day0",
            {"last_updated": client._current_timestamp_label()},
        )
        client.client.create_table(f"{temporary_dataset}.active_table_day1")
        client.add_labels_to_table(
            "active_table_day1",
            {"last_updated": client._current_timestamp_label()},
        )

        config = Config(
            slug="active_table",
            spec=self.spec,
            last_modified=pytz.UTC.localize(datetime.datetime.utcnow()),
        )

        client.client.create_table(f"{temporary_dataset}.active_table_day2")
        client.add_labels_to_table(
            "active_table_day2",
            {"last_updated": client._current_timestamp_label()},
        )
        client.client.create_table(f"{temporary_dataset}.active_table_weekly")
        client.add_labels_to_table(
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
