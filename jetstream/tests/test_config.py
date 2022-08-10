import pytest

from jetstream.platform import (
    Platform,
    PlatformConfigurationException,
    _generate_platform_config,
)


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
