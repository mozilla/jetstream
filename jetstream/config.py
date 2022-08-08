"""
Parses configuration specifications into concrete objects.

Users should write something like:
    my_config = (
        config.AnalysisSpec
        .from_dict(toml.load(my_config_file))
        .resolve(an_experimenter_object)
    )
to obtain a concrete AnalysisConfiguration object.

Spec objects are direct representations of the configuration and contain unresolved references
to metrics and data sources.

Calling .resolve(config_spec) on a Spec object produces a concrete resolved Configuration class.

Definition and Reference classes are also direct representations of the configuration,
which produce concrete mozanalysis classes when resolved.
"""

import copy
import datetime as dt
from collections import defaultdict
from inspect import isabstract
from pathlib import Path
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Type,
    TypeVar,
    Union,
)

import attr
import cattr
import jinja2
import mozanalysis
import mozanalysis.experiment
import mozanalysis.exposure
import mozanalysis.segments
import pytz
import toml
from jinja2 import StrictUndefined

from jetstream.errors import InvalidConfigurationException, NoStartDateException
from jetstream.exposure_signal import AnalysisWindow, ExposureSignal, WindowLimit
from jetstream.metric import Metric
from jetstream.platform import Platform, _generate_platform_config
from jetstream.pre_treatment import PreTreatment
from jetstream.statistics import Statistic, Summary

from jetstream_config_parser.config import ConfigCollection

from . import AnalysisPeriod

if TYPE_CHECKING:
    import jetstream.experimenter

    from .external_config import ExternalConfigCollection

platform_config = toml.load(Path(__file__).parent.parent / "platform_config.toml")
PLATFORM_CONFIGS = _generate_platform_config(platform_config)

_converter = cattr.Converter()


class _ConfigLoader:
    """
    Loads config files from an external repository.

    Config objects are converted into jetstream native types.
    """

    config_collection: Optional[ConfigCollection] = None

    @property
    def configs(self) -> ConfigCollection:
        configs = getattr(self, "_configs", None)
        if configs:
            return configs

        if self.config_collection is None:
            self.config_collection = ConfigCollection.from_github_repo()
        self._configs = self.config_collection
        return self._configs

    def get_metric(self, metric_slug: str, app_name: str):
        from mozanalysis.metrics import Metric

        metric_definition = self.configs.get_metric_definition(metric_slug, app_name)
        if metric_definition is None:
            raise Exception(f"Could not find definition for metric {metric_slug}")

        return Metric(
            name=metric_definition.name,
            select_expr=self.configs.get_env()
            .from_string(metric_definition.select_expression)
            .render(),
            friendly_name=metric_definition.friendly_name,
            description=metric_definition.friendly_name,
            data_source=self.get_data_source(
                metric_definition.data_source.name, app_name
            ),
            bigger_is_better=metric_definition.bigger_is_better,
        )

    def get_data_source(self, data_source_slug: str, app_name: str):
        from mozanalysis.metrics import DataSource

        data_source_definition = self.configs.get_data_source_definition(
            data_source_slug, app_name
        )
        if data_source_definition is None:
            raise Exception(
                f"Could not find definition for data source {data_source_slug}"
            )

        return DataSource(
            name=data_source_definition.name,
            from_expr=data_source_definition.from_expression,
            client_id_column=data_source_definition.client_id_column,
            submission_date_column=data_source_definition.submission_date_column,
            experiments_column_type=None
            if data_source_definition.experiments_column_type == "none"
            else data_source_definition.experiments_column_type,
            default_dataset=data_source_definition.default_dataset,
        )

    def get_segment(self, segment_slug: str, app_name: str):
        from mozanalysis.segments import Segment

        segment_definition = self.configs.get_segment_definition(segment_slug, app_name)
        if segment_definition is None:
            raise Exception(f"Could not find definition for segment {segment_slug}")

        return Segment(
            name=segment_definition.name,
            data_source=self.get_segment_data_source(
                segment_definition.data_source.name, app_name
            ),
            select_expr=segment_definition.select_expression,
            friendly_name=segment_definition.friendly_name,
            description=segment_definition.description,
        )

    def get_segment_data_source(self, data_source_slug: str, app_name: str):
        from mozanalysis.segments import SegmentDataSource

        data_source_definition = self.configs.get_segment_data_source_definition(
            data_source_slug, app_name
        )
        if data_source_definition is None:
            raise Exception(
                f"Could not find definition for segment data source {data_source_slug}"
            )

        return SegmentDataSource(
            name=data_source_definition.name,
            from_expr=data_source_definition.from_expression,
            window_start=data_source_definition.window_start,
            window_end=data_source_definition.window_end,
            client_id_column=data_source_definition.client_id_column,
            submission_date_column=data_source_definition.submission_date_column,
            default_dataset=data_source_definition.default_dataset,
        )


ConfigLoader = _ConfigLoader()
