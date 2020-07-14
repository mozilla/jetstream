from google.cloud import bigquery
from pathlib import Path
import datetime
import pytest
import pytz
import random
import string
from textwrap import dedent
import toml

from pensieve.external_config import ExternalConfig, ExternalConfigCollection
from pensieve.config import AnalysisSpec

TEST_DIR = Path(__file__).parent.parent


class TestExternalConfigIntegration:
    project_id = "pensieve-integration-test"

    # generate a random test dataset to avoid conflicts when running tests in parallel
    test_dataset = "test_" + "".join(random.choice(string.ascii_lowercase) for i in range(10))
    # contains the tables filled with test data required to run metrics analysis
    static_dataset = "test_data"

    config_str = dedent(
        """
        [metrics]
        weekly = ["view_about_logins"]

        [metrics.view_about_logins.statistics.bootstrap_mean]
        """
    )
    spec = AnalysisSpec.from_dict(toml.loads(config_str))

    @pytest.fixture(scope="class")
    def client(self):
        self._client = getattr(self, "_client", None) or bigquery.client.Client(self.project_id)
        return self._client

    @pytest.fixture(autouse=True)
    def setup(self, client):
        # remove all tables previously created
        client.delete_dataset(self.test_dataset, delete_contents=True, not_found_ok=True)
        client.create_dataset(self.test_dataset)

    def test_new_config(self, client):
        config = ExternalConfig(
            experimenter_slug="new_experiment",
            spec=self.spec,
            last_modified=datetime.datetime.utcnow(),
        )
        config_collection = ExternalConfigCollection([config])
        updated_configs = config_collection.updated_configs(self.project_id, self.test_dataset)

        assert len(updated_configs) == 0

    def test_old_config(self, client):
        config = ExternalConfig(
            experimenter_slug="new_table",
            spec=self.spec,
            last_modified=pytz.UTC.localize(
                datetime.datetime.utcnow() - datetime.timedelta(days=1)
            ),
        )

        # table created after config loaded
        client.create_table(f"{self.test_dataset}.new_table_day1")
        config_collection = ExternalConfigCollection([config])
        updated_configs = config_collection.updated_configs(self.project_id, self.test_dataset)

        assert len(updated_configs) == 0

    def test_updated_config(self, client):
        config = ExternalConfig(
            experimenter_slug="old_table",
            spec=self.spec,
            last_modified=pytz.UTC.localize(
                datetime.datetime.utcnow() + datetime.timedelta(days=1)
            ),
        )

        client.create_table(f"{self.test_dataset}.old_table_day1")
        client.create_table(f"{self.test_dataset}.old_table_day2")

        config_collection = ExternalConfigCollection([config])
        updated_configs = config_collection.updated_configs(self.project_id, self.test_dataset)

        print(updated_configs)

        assert len(updated_configs) == 1
        assert updated_configs[0].experimenter_slug == config.experimenter_slug

    def test_updated_config_while_analysis_active(self, client):
        client.create_table(f"{self.test_dataset}.active_table_day0")
        client.create_table(f"{self.test_dataset}.active_table_day1")

        config = ExternalConfig(
            experimenter_slug="active_table",
            spec=self.spec,
            last_modified=pytz.UTC.localize(datetime.datetime.utcnow()),
        )

        client.create_table(f"{self.test_dataset}.active_table_day2")
        client.create_table(f"{self.test_dataset}.active_table_weekly")

        config_collection = ExternalConfigCollection([config])
        updated_configs = config_collection.updated_configs(self.project_id, self.test_dataset)

        assert len(updated_configs) == 1
        assert updated_configs[0].experimenter_slug == config.experimenter_slug
