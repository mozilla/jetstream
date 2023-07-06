import pytest

from jetstream.config import ConfigLoader, _ConfigLoader
from jetstream.platform import (
    Platform,
    PlatformConfigurationException,
    _generate_platform_config,
)


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
        with pytest.raises(Exception):
            ConfigLoader.get_data_source("non_existing", "foo") is None


class TestGeneratePlatformConfig:
    """
    Test cases for checking that platform configuration objects are generated correctly
    """

    config_file = "default_metrics.toml"

    @pytest.mark.parametrize(
        "test_input,expected",
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
                        enrollments_query_type="normandy",
                        app_id="firefox-desktop",
                        app_name="firefox_desktop",
                    )
                },
            ),
            (
                {
                    "platform": {
                        "firefox_desktop": {
                            "enrollments_query_type": "normandy",
                            "app_id": "firefox-desktop",
                        }
                    },
                },
                {
                    "firefox_desktop": Platform(
                        enrollments_query_type="normandy",
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
                    }
                },
                {
                    "firefox_desktop": Platform(
                        enrollments_query_type="glean-event",
                        app_id="firefox-desktop",
                        app_name="firefox_desktop",
                    ),
                    "desktop": Platform(
                        enrollments_query_type="normandy",
                        app_id="EDI",
                        app_name="desktop",
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
            {
                "platform": {
                    "firefox_desktop": {
                        "enrollments_query_type": "N7",
                        "app_id": "firefox-desktop",
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
