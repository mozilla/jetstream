import datetime
from pathlib import Path
from textwrap import dedent

import pytz
import toml

from jetstream.config import AnalysisSpec
from jetstream.external_config import ExternalConfig, ExternalConfigCollection

TEST_DIR = Path(__file__).parent.parent


class TestExternalConfigIntegration:
    config_str = dedent(
        """
        [metrics]
        weekly = ["view_about_logins"]

        [metrics.view_about_logins.statistics.bootstrap_mean]
        """
    )
    spec = AnalysisSpec.from_dict(toml.loads(config_str))

    def test_new_config(self, client, project_id, temporary_dataset):
        config = ExternalConfig(
            slug="new_experiment",
            spec=self.spec,
            last_modified=datetime.datetime.utcnow(),
        )
        config_collection = ExternalConfigCollection([config])
        updated_configs = config_collection.updated_configs(project_id, temporary_dataset)

        assert len(updated_configs) == 0

    def test_old_config(self, client, project_id, temporary_dataset):
        config = ExternalConfig(
            slug="new_table",
            spec=self.spec,
            last_modified=pytz.UTC.localize(
                datetime.datetime.utcnow() - datetime.timedelta(days=1)
            ),
        )

        # table created after config loaded
        client.client.create_table(f"{temporary_dataset}.new_table_day1")
        client.add_labels_to_table(
            "new_table_day1",
            {"last_updated": client._current_timestamp_label()},
        )
        config_collection = ExternalConfigCollection([config])
        updated_configs = config_collection.updated_configs(project_id, temporary_dataset)

        assert len(updated_configs) == 0

    def test_updated_config(self, client, temporary_dataset, project_id):
        config = ExternalConfig(
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

        config_collection = ExternalConfigCollection([config])
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

        config = ExternalConfig(
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

        config_collection = ExternalConfigCollection([config])
        updated_configs = config_collection.updated_configs(project_id, temporary_dataset)

        assert len(updated_configs) == 1
        assert updated_configs[0].slug == config.slug

    def test_new_config_without_a_table_is_marked_changed(
        self, client, temporary_dataset, project_id
    ):
        config = ExternalConfig(
            slug="my_cool_experiment",
            spec=self.spec,
            last_modified=pytz.UTC.localize(datetime.datetime.utcnow()),
        )
        config_collection = ExternalConfigCollection([config])
        updated_configs = config_collection.updated_configs(project_id, temporary_dataset)
        assert [updated.slug for updated in updated_configs] == ["my_cool_experiment"]
