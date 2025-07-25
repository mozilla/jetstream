import datetime
from textwrap import dedent

import pytest
import toml
from metric_config_parser.analysis import AnalysisSpec
from metric_config_parser.config import Config, DefaultConfig, Outcome
from metric_config_parser.outcome import OutcomeSpec
from mozanalysis.experiment import EnrollmentsQueryType

from jetstream.config import ConfigLoader, _ConfigLoader, validate
from jetstream.dryrun import DryRunFailedError
from jetstream.platform import (
    Platform,
    PlatformConfigurationException,
    _generate_platform_config,
)


class TestConfig:
    config_str = dedent(
        """
        [metrics]
        weekly = ["view_about_logins"]

        [metrics.view_about_logins.statistics.bootstrap_mean]
        """
    )
    spec = AnalysisSpec.from_dict(toml.loads(config_str))

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


class TestConfigLoader:
    """Test cases for _ConfigLoader"""

    def test_load_configs(self):
        configs_collection = ConfigLoader
        assert configs_collection.configs is not None
        assert len(configs_collection.configs.configs) > 0

    def test_configs_from(self):
        loader = _ConfigLoader()
        configs_collection = loader.with_configs_from(
            ["https://github.com/mozilla/metric-hub/tree/main/jetstream"]
        )
        assert configs_collection.configs is not None
        assert len(configs_collection.configs.configs) == len(loader.configs.configs)

    def test_configs_from_null(self):
        loader = _ConfigLoader()
        base_collection = loader.with_configs_from(
            ["https://github.com/mozilla/metric-hub/tree/main/jetstream"]
        )
        new_collection = base_collection.with_configs_from(None)
        assert new_collection == base_collection

    def test_configs_from_empty(self):
        loader = _ConfigLoader()
        base_collection = loader.with_configs_from(
            ["https://github.com/mozilla/metric-hub/tree/main/jetstream"]
        )
        new_collection = base_collection.with_configs_from(())
        assert new_collection == base_collection

    def test_spec_for_experiment(self):
        experiment = ConfigLoader.configs.configs[0].slug
        assert ConfigLoader.spec_for_experiment(experiment) is not None

    def test_spec_for_nonexisting_experiment(self):
        assert ConfigLoader.spec_for_experiment("non_exisiting") is None

    def test_get_outcome(self):
        outcome = ConfigLoader.configs.outcomes[0]
        assert ConfigLoader.get_outcome(outcome.slug, outcome.platform) is not None

    def test_get_nonexisting_outcome(self):
        assert ConfigLoader.get_outcome("non_existing", "foo") is None

    def test_get_data_source(self):
        config_definition = next(
            (
                config
                for config in ConfigLoader.configs.definitions
                if config.slug == "firefox_desktop"
            ),
            None,
        )
        metric = list(config_definition.spec.metrics.definitions.values())[3]
        platform = config_definition.platform
        assert ConfigLoader.get_data_source(metric.data_source.name, platform) is not None

    def test_get_nonexisting_data_source(self):
        with pytest.raises(
            Exception, match="Could not find definition for data source non_existing"
        ):
            ConfigLoader.get_data_source("non_existing", "foo")


class TestGeneratePlatformConfig:
    """
    Test cases for checking that platform configuration objects are generated correctly
    """

    config_file = "default_metrics.toml"

    @pytest.mark.parametrize(
        ("test_input", "expected"),
        [
            (
                {
                    "platform": {
                        "firefox_desktop": {
                            "enrollments_query_type": "normandy",
                            "app_id": "firefox-desktop",
                        }
                    }
                },
                {
                    "firefox_desktop": Platform(
                        enrollments_query_type=EnrollmentsQueryType.NORMANDY,
                        app_id="firefox-desktop",
                        app_name="firefox_desktop",
                    )
                },
            ),
            (
                {
                    "platform": {
                        "firefox_desktop": {
                            "app_id": "firefox-desktop",
                        },
                        "desktop": {
                            "enrollments_query_type": "normandy",
                            "app_id": "EDI",
                        },
                        "monitor_cirrus": {
                            "enrollments_query_type": "cirrus",
                            "app_id": "monitor.cirrus",
                        },
                    }
                },
                {
                    "firefox_desktop": Platform(
                        enrollments_query_type=EnrollmentsQueryType.GLEAN_EVENT,
                        app_id="firefox-desktop",
                        app_name="firefox_desktop",
                    ),
                    "desktop": Platform(
                        enrollments_query_type=EnrollmentsQueryType.NORMANDY,
                        app_id="EDI",
                        app_name="desktop",
                    ),
                    "monitor_cirrus": Platform(
                        enrollments_query_type=EnrollmentsQueryType.CIRRUS,
                        app_id="monitor.cirrus",
                        app_name="monitor_cirrus",
                    ),
                },
            ),
        ],
    )
    def test_generate_platform_config(self, test_input, expected):
        actual = _generate_platform_config(test_input)

        for platform_config in actual.values():
            assert isinstance(platform_config, Platform)

        assert actual == expected

    @pytest.mark.parametrize(
        "test_input",
        [
            {
                "platform": {
                    "firefox_desktop": {
                        "enrollments_query_type": "glean-event",
                    },
                }
            },
            {
                "platform": {
                    "firefox_desktop": {
                        "enrollments_query_type": "N7",
                        "app_id": "firefox-desktop",
                    },
                }
            },
        ],
    )
    def test_generate_platform_config_invalid_config(self, test_input):
        with pytest.raises(PlatformConfigurationException):
            _generate_platform_config(test_input)
