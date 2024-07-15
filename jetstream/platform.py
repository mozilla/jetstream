from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import attr
import toml
from metric_config_parser.analysis import AnalysisSpec
from mozanalysis.experiment import EnrollmentsQueryType

from jetstream.config import ConfigLoader

platform_config = toml.load(Path(__file__).parent.parent / "platform_config.toml")


class PlatformConfigurationException(Exception):
    """
    Custom exception type for Jetstream platform configuration related issues.
    """

    pass


@attr.s(auto_attribs=True)
class Platform:
    """
    Platform configuration object. Contains all required settings for jetstream.
    More info about Jetstream configuration: https://experimenter.info/jetstream/configuration

    :param enrollments_query_type:
    :type enrollments_query_type: mozanalysis.experiment.EnrollmentsQueryType
    :param app_id:
    :type app_id: str

    :returns: returns an instance of the object with all configuration settings as attributes
    :rtype: Platform
    """

    def _check_value_not_null(self, attribute, value):
        if not value and str(value).lower() == "none":
            raise PlatformConfigurationException(
                f"'{attribute.name}' attribute requires a value, please double check \
                    platform configuration file. Value provided: {value}"
            )

    def validate_enrollments_query_type(self, attribute, value):
        self._check_value_not_null(attribute, value)

        if value not in iter(EnrollmentsQueryType):
            raise PlatformConfigurationException(
                f"Invalid value provided for {attribute.name}, value provided: {value}. \
                    Valid options are: {iter(EnrollmentsQueryType)}"
            )

        return value

    enrollments_query_type: EnrollmentsQueryType = attr.ib(
        validator=validate_enrollments_query_type
    )
    app_id: str = attr.ib(validator=_check_value_not_null)
    app_name: str = attr.ib(validator=_check_value_not_null)

    def resolve_config(self) -> AnalysisSpec:
        config = ConfigLoader.configs.get_platform_defaults(self.app_name)

        if config is None:
            raise PlatformConfigurationException(
                f"No default config for platform {self.app_name} in jetstream-config."
            )

        return config


def _generate_platform_config(config: MutableMapping[str, Any]) -> dict[str, Platform]:
    """
    Takes platform configuration and generates platform object map
    """

    processed_config = {}

    for platform, platform_config in config["platform"].items():
        processed_config[platform] = {
            "enrollments_query_type": platform_config.get(
                "enrollments_query_type", EnrollmentsQueryType.GLEAN_EVENT
            ),
            "app_id": platform_config.get("app_id"),
            "app_name": platform,
        }

    return {
        platform: Platform(**platform_config)
        for platform, platform_config in processed_config.items()
    }


PLATFORM_CONFIGS = _generate_platform_config(platform_config)
