import importlib
from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict, MutableMapping, Optional

import attr
import mozanalysis
import mozanalysis.experiment
import mozanalysis.exposure
import mozanalysis.segments

if TYPE_CHECKING:
    from jetstream.config import AnalysisSpec


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

    :param enrollments_query_type: "glean-event" or "normandy"
    :type enrollments_query_type: str
    :param app_id:
    :type app_id: str
    :param metrics_module: (Optional) name of metrics module to use
    :type metrics_module: Optional[ModuleType]
    :param segments_module: (Optional) name of segments module to use \
    :type segments_module: Optional[ModuleType]

    :returns: returns an instance of the object with all configuration settings as attributes
    :rtype: Platform
    """

    VALID_MODULES_METRICS = [
        f"mozanalysis.{metric}"
        for metric in filter(lambda module: "metrics." in module, mozanalysis.__all__)
    ]

    VALID_MODULES_SEGMENTS = [
        f"mozanalysis.{segment}"
        for segment in filter(lambda module: "segments." in module, mozanalysis.__all__)
    ]

    def _check_value_not_null(self, attribute, value):
        if not value and str(value).lower() == "none":
            raise PlatformConfigurationException(
                "'%s' attribute requires a value, please double check \
                    platform configuration file. Value provided: %s"
                % (attribute.name, str(value))
            )

    def validate_enrollments_query_type(self, attribute, value):
        self._check_value_not_null(attribute, value)

        valid_entrollments_query_types = (
            "glean-event",
            "normandy",
        )

        if value not in valid_entrollments_query_types:
            raise PlatformConfigurationException(
                "Invalid value provided for %s, value provided: %s. Valid options are: %s"
                % (
                    attribute.name,
                    value,
                    valid_entrollments_query_types,
                )
            )

        return value

    def validate_metrics_module(self, attribute, value):
        if not value or str(value).lower() == "none":
            return

        if value.__name__ not in self.VALID_MODULES_METRICS:
            raise PlatformConfigurationException(
                "Invalid module provided for %s, module provided: %s. Valid modules are: %s"
                % (
                    attribute.name,
                    value,
                    self.VALID_MODULES_METRICS,
                )
            )

        return value

    def validate_segments_module(self, attribute, value):
        if not value or str(value).lower() == "none":
            return

        if value.__name__ not in self.VALID_MODULES_SEGMENTS:
            raise PlatformConfigurationException(
                "Invalid module provided for %s, module provided: %s. Valid modules are: %s"
                % (
                    attribute.name,
                    value,
                    self.VALID_MODULES_SEGMENTS,
                )
            )

        return value

    enrollments_query_type: str = attr.ib(validator=validate_enrollments_query_type)
    app_id: str = attr.ib(validator=_check_value_not_null)
    app_name: str = attr.ib(validator=_check_value_not_null)
    metrics_module: Optional[ModuleType] = attr.ib(default=None, validator=validate_metrics_module)
    segments_module: Optional[ModuleType] = attr.ib(
        default=None, validator=validate_segments_module
    )

    def resolve_config(self) -> "AnalysisSpec":
        from . import default_config

        config = default_config.DefaultConfigsResolver.resolve(self.app_name)

        if config is None:
            raise PlatformConfigurationException(
                f"No default config for platform {self.app_name} in jetstream-config."
            )

        return config.spec


def _generate_platform_config(config: MutableMapping[str, Any]) -> Dict[str, Platform]:
    """
    Takes platform configuration and generates platform object map
    """

    processed_config = dict()

    for platform, platform_config in config["platform"].items():
        metrics_module = platform_config.get("metrics_module", platform)
        segments_module = platform_config.get("segments_module", platform)

        try:
            processed_config[platform] = {
                "metrics_module": importlib.import_module(f"mozanalysis.metrics.{metrics_module}")
                if metrics_module and metrics_module.lower() != "none"
                else None,
                "segments_module": importlib.import_module(
                    f"mozanalysis.segments.{segments_module}"
                )
                if segments_module and segments_module.lower() != "none"
                else None,
                "enrollments_query_type": platform_config.get(
                    "enrollments_query_type", "glean-event"
                ),
                "app_id": platform_config.get("app_id"),
                "app_name": platform,
            }
        except ModuleNotFoundError as _err:
            raise PlatformConfigurationException(
                f"{_err}\nIf metrics or segments module does not exist,"
                'please set the value to "None" inside platform_config.toml'
            )

    return {
        platform: Platform(**platform_config)
        for platform, platform_config in processed_config.items()
    }
