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
from inspect import isabstract
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Type, TypeVar

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

from jetstream.errors import NoStartDateException
from jetstream.exposure_signal import AnalysisWindow, ExposureSignal, WindowLimit
from jetstream.metric import Metric
from jetstream.platform import Platform, _generate_platform_config
from jetstream.pre_treatment import PreTreatment
from jetstream.statistics import Statistic, Summary

from . import AnalysisPeriod

if TYPE_CHECKING:
    import jetstream.experimenter

    from .external_config import ExternalConfigCollection

platform_config = toml.load(Path(__file__).parent.parent / "platform_config.toml")
PLATFORM_CONFIGS = _generate_platform_config(platform_config)

_converter = cattr.Converter()


def _populate_environment() -> jinja2.Environment:
    """Create a Jinja2 environment that understands the SQL agg_* helpers in mozanalysis.metrics.

    Just a wrapper to avoid leaking temporary variables to the module scope."""
    env = jinja2.Environment(autoescape=False, undefined=StrictUndefined)
    for name in dir(mozanalysis.metrics):
        if not name.startswith("agg_"):
            continue
        obj = getattr(mozanalysis.metrics, name)
        if not callable(obj):
            continue
        env.globals[name] = obj
    return env


_metrics_environment = _populate_environment()


T = TypeVar("T")


def _lookup_name(
    name: str,
    klass: Type[T],
    spec: "AnalysisSpec",
    module: Optional[ModuleType],
    definitions: dict,
    resolve_extras: Optional[Mapping[str, Any]] = None,
) -> T:
    needle = None
    if module and hasattr(module, name):
        needle = getattr(module, name)
    if name in definitions:
        needle = definitions[name].resolve(spec, **(resolve_extras or {}))
    if isinstance(needle, klass):
        return needle
    raise ValueError(f"Could not locate {klass.__name__} {name}")


@attr.s(auto_attribs=True)
class MetricReference:
    name: str

    def resolve(self, spec: "AnalysisSpec", experiment: "ExperimentConfiguration") -> List[Summary]:
        if self.name in spec.metrics.definitions:
            return spec.metrics.definitions[self.name].resolve(spec, experiment)
        if hasattr(experiment.platform.metrics_module, self.name):
            raise ValueError(f"Please define a statistical treatment for the metric {self.name}")
        raise ValueError(f"Could not locate metric {self.name}")


# These are bare strings in the configuration file.
_converter.register_structure_hook(MetricReference, lambda obj, _type: MetricReference(name=obj))


@attr.s(auto_attribs=True)
class DataSourceReference:
    name: str

    def resolve(
        self, spec: "AnalysisSpec", experiment: "ExperimentConfiguration"
    ) -> mozanalysis.metrics.DataSource:
        search = experiment.platform.metrics_module
        return _lookup_name(
            name=self.name,
            klass=mozanalysis.metrics.DataSource,
            spec=spec,
            module=search,
            definitions=spec.data_sources.definitions,
        )


_converter.register_structure_hook(
    DataSourceReference, lambda obj, _type: DataSourceReference(name=obj)
)


@attr.s(auto_attribs=True)
class SegmentReference:
    name: str

    def resolve(
        self, spec: "AnalysisSpec", experiment: "ExperimentConfiguration"
    ) -> mozanalysis.segments.Segment:
        search = experiment.platform.segments_module
        return _lookup_name(
            name=self.name,
            klass=mozanalysis.segments.Segment,
            spec=spec,
            module=search,
            definitions=spec.segments.definitions,
            resolve_extras={"experiment": experiment},
        )


_converter.register_structure_hook(SegmentReference, lambda obj, _type: SegmentReference(name=obj))


@attr.s(auto_attribs=True)
class PreTreatmentReference:
    name: str
    args: Dict[str, Any]

    def resolve(self, spec: "AnalysisSpec") -> PreTreatment:
        for pre_treatment in PreTreatment.__subclasses__():
            if isabstract(pre_treatment):
                continue
            if pre_treatment.name() == self.name:
                return pre_treatment.from_dict(self.args)  # type: ignore

        raise ValueError(f"Could not find pre-treatment {self.name}.")


@attr.s(auto_attribs=True)
class ExperimentConfiguration:
    """Represents the configuration of an experiment for analysis."""

    experiment_spec: "ExperimentSpec"
    experimenter_experiment: "jetstream.experimenter.Experiment"
    segments: List[mozanalysis.segments.Segment]
    exposure_signal: Optional[ExposureSignal] = None

    def __attrs_post_init__(self):
        # Catch any exceptions at instantiation
        self._enrollment_query = self.enrollment_query

    @property
    def enrollment_query(self) -> Optional[str]:
        if self.experiment_spec.enrollment_query is None:
            return None

        if cached := getattr(self, "_enrollment_query", None):
            return cached

        class ExperimentProxy:
            @property
            def enrollment_query(proxy):
                raise ValueError()

            def __getattr__(proxy, name):
                return getattr(self, name)

        env = jinja2.Environment(autoescape=False, undefined=StrictUndefined)
        return env.from_string(self.experiment_spec.enrollment_query).render(
            experiment=ExperimentProxy()
        )

    @property
    def proposed_enrollment(self) -> int:
        return (
            self.experiment_spec.enrollment_period
            or self.experimenter_experiment.proposed_enrollment
            or 0
        )

    @property
    def reference_branch(self) -> Optional[str]:
        return (
            self.experiment_spec.reference_branch or self.experimenter_experiment.reference_branch
        )

    @property
    def start_date(self) -> Optional[dt.datetime]:
        return (
            ExperimentSpec.parse_date(self.experiment_spec.start_date)
            or self.experimenter_experiment.start_date
        )

    @property
    def end_date(self) -> Optional[dt.datetime]:
        return (
            ExperimentSpec.parse_date(self.experiment_spec.end_date)
            or self.experimenter_experiment.end_date
        )

    @property
    def status(self) -> Optional[str]:
        """Assert the experiment is Complete if an end date is provided.

        Functionally, this lets the Overall metrics run on the specified date.
        """
        return "Complete" if self.experiment_spec.end_date else self.experimenter_experiment.status

    # Helpers for configuration templates
    @property
    def start_date_str(self) -> str:
        if not self.start_date:
            raise NoStartDateException(self.normandy_slug)
        return self.start_date.strftime("%Y-%m-%d")

    @property
    def last_enrollment_date_str(self) -> str:
        if not self.start_date:
            raise NoStartDateException(self.normandy_slug)
        return (self.start_date + dt.timedelta(days=self.proposed_enrollment)).strftime("%Y-%m-%d")

    @property
    def platform(self) -> Platform:
        try:
            return PLATFORM_CONFIGS[self.app_name]
        except KeyError:
            raise ValueError(f"Unknown platform {self.app_name}")

    @property
    def skip(self) -> bool:
        return self.experiment_spec.skip

    def has_external_config_overrides(self) -> bool:
        """Check whether the external config overrides Experimenter configuration."""
        return (
            self.reference_branch != self.experimenter_experiment.reference_branch
            or self.start_date != self.experimenter_experiment.start_date
            or self.end_date != self.experimenter_experiment.end_date
            or self.proposed_enrollment != self.experimenter_experiment.proposed_enrollment
        )

    # see https://stackoverflow.com/questions/50888391/pickle-of-object-with-getattr-method-in-
    # python-returns-typeerror-object-no
    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)

    def __getattr__(self, name: str) -> Any:
        if "experimenter_experiment" not in vars(self):
            raise AttributeError
        return getattr(self.experimenter_experiment, name)


def _validate_yyyy_mm_dd(instance: Any, attribute: Any, value: Any) -> None:
    instance.parse_date(value)


def structure_window_limit(value: Any, _klass: Type) -> WindowLimit:
    try:
        return AnalysisWindow(value)
    except Exception:
        return int(value)


_converter.register_structure_hook(WindowLimit, structure_window_limit)


@attr.s(auto_attribs=True)
class ExposureSignalDefinition:
    """Describes the interface for defining an exposure signal in configuration."""

    name: str
    data_source: DataSourceReference
    select_expression: str
    friendly_name: str
    description: str
    window_start: WindowLimit = None
    window_end: WindowLimit = None

    def resolve(self, spec: "AnalysisSpec", experiment: ExperimentConfiguration) -> ExposureSignal:
        return ExposureSignal(
            name=self.name,
            data_source=self.data_source.resolve(spec, experiment=experiment),
            select_expression=self.select_expression,
            friendly_name=self.friendly_name,
            description=self.description,
            window_start=self.window_start,
            window_end=self.window_end,
        )


@attr.s(auto_attribs=True, kw_only=True)
class ExperimentSpec:
    """Describes the interface for overriding experiment details."""

    enrollment_query: Optional[str] = None
    enrollment_period: Optional[int] = None
    reference_branch: Optional[str] = None
    start_date: Optional[str] = attr.ib(default=None, validator=_validate_yyyy_mm_dd)
    end_date: Optional[str] = attr.ib(default=None, validator=_validate_yyyy_mm_dd)
    segments: List[SegmentReference] = attr.Factory(list)
    skip: bool = False
    exposure_signal: Optional[ExposureSignalDefinition] = None

    @staticmethod
    def parse_date(yyyy_mm_dd: Optional[str]) -> Optional[dt.datetime]:
        if not yyyy_mm_dd:
            return None
        return dt.datetime.strptime(yyyy_mm_dd, "%Y-%m-%d").replace(tzinfo=pytz.utc)

    def resolve(
        self,
        spec: "AnalysisSpec",
        experimenter: "jetstream.experimenter.Experiment",
    ) -> ExperimentConfiguration:
        experiment = ExperimentConfiguration(self, experimenter, [])
        # Segment data sources may need to know the enrollment dates of the experiment,
        # so we'll forward the Experiment we know about so far.
        experiment.segments = [ref.resolve(spec, experiment) for ref in self.segments]

        if self.exposure_signal:
            experiment.exposure_signal = self.exposure_signal.resolve(spec, experiment=experiment)

        return experiment

    def merge(self, other: "ExperimentSpec") -> None:
        for key in attr.fields_dict(type(self)):
            setattr(self, key, getattr(other, key) or getattr(self, key))


@attr.s(auto_attribs=True)
class MetricDefinition:
    """Describes the interface for defining a metric in configuration.

    The `select_expression` of the metric may use Jinja2 template syntax to refer to the
    aggregation helper functions defined in `mozanalysis.metrics`, like
        '{{agg_any("payload.processes.scalars.some_boolean_thing")}}'
    """

    name: str  # implicit in configuration
    statistics: Dict[str, Dict[str, Any]]
    select_expression: Optional[str] = None
    data_source: Optional[DataSourceReference] = None
    friendly_name: Optional[str] = None
    description: Optional[str] = None
    bigger_is_better: bool = True
    analysis_bases: Optional[List[mozanalysis.experiment.AnalysisBasis]] = None

    def resolve(self, spec: "AnalysisSpec", experiment: ExperimentConfiguration) -> List[Summary]:
        if self.select_expression is None or self.data_source is None:
            # checks if a metric from mozanalysis was referenced
            search = experiment.platform.metrics_module
            mozanalysis_metric = _lookup_name(
                name=self.name,
                klass=mozanalysis.metrics.Metric,
                spec=spec,
                module=search,
                definitions={},
            )
            metric = Metric.from_mozanalysis_metric(
                mozanalysis_metric=mozanalysis_metric,
                analysis_bases=self.analysis_bases
                or [mozanalysis.experiment.AnalysisBasis.ENROLLMENTS],
            )
        else:
            select_expression = _metrics_environment.from_string(self.select_expression).render()

            metric = Metric(
                name=self.name,
                data_source=self.data_source.resolve(spec, experiment),
                select_expression=select_expression,
                friendly_name=self.friendly_name,
                description=self.description,
                bigger_is_better=self.bigger_is_better,
                analysis_bases=self.analysis_bases
                or [mozanalysis.experiment.AnalysisBasis.ENROLLMENTS],
            )

        metrics_with_treatments = []

        for statistic_name, params in self.statistics.items():
            for statistic in Statistic.__subclasses__():
                if statistic.name() == statistic_name:
                    break
            else:
                raise ValueError(f"Statistic {statistic_name} does not exist.")

            stats_params = copy.deepcopy(params)
            pre_treatments = []
            for pt in stats_params.pop("pre_treatments", []):
                if isinstance(pt, str):
                    ref = PreTreatmentReference(pt, {})
                else:
                    name = pt.pop("name")
                    ref = PreTreatmentReference(name, pt)
                pre_treatments.append(ref.resolve(spec))

            metrics_with_treatments.append(
                Summary(
                    metric=metric,
                    statistic=statistic.from_dict(stats_params),
                    pre_treatments=pre_treatments,
                )
            )

        if len(metrics_with_treatments) == 0:
            raise ValueError(f"Metric {self.name} has no statistical treatment defined.")

        return metrics_with_treatments


MetricsConfigurationType = Dict[AnalysisPeriod, List[Summary]]


@attr.s(auto_attribs=True)
class MetricsSpec:
    """Describes the interface for the metrics section in configuration."""

    daily: List[MetricReference] = attr.Factory(list)
    weekly: List[MetricReference] = attr.Factory(list)
    days28: List[MetricReference] = attr.Factory(list)
    overall: List[MetricReference] = attr.Factory(list)

    definitions: Dict[str, MetricDefinition] = attr.Factory(dict)

    @classmethod
    def from_dict(cls, d: dict) -> "MetricsSpec":
        params: Dict[str, Any] = {}
        known_keys = {f.name for f in attr.fields(cls)}
        for k in known_keys:
            if k == "days28":
                v = d.get("28_day", [])
            else:
                v = d.get(k, [])
            if not isinstance(v, list):
                raise ValueError(f"metrics.{k} should be a list of metrics")
            params[k] = [MetricReference(m) for m in v]

        params["definitions"] = {
            k: _converter.structure(
                {"name": k, **dict((kk.lower(), vv) for kk, vv in v.items())}, MetricDefinition
            )
            for k, v in d.items()
            if k not in known_keys and k != "28_day"
        }
        return cls(**params)

    def resolve(
        self, spec: "AnalysisSpec", experiment: ExperimentConfiguration
    ) -> MetricsConfigurationType:
        result = {}
        for period in AnalysisPeriod:
            # these summaries might contain duplicates
            summaries = [
                summary
                for ref in getattr(self, period.table_suffix)
                for summary in ref.resolve(spec, experiment)
            ]
            unique_summaries = []
            seen_summaries = set()

            # summaries needs to be reversed to make sure merged configs overwrite existing ones
            summaries.reverse()
            for summary in summaries:
                if (summary.metric.name, summary.statistic.name) not in seen_summaries:
                    seen_summaries.add((summary.metric.name, summary.statistic.name))
                    unique_summaries.append(summary)

            result[period] = unique_summaries

        return result

    def merge(self, other: "MetricsSpec"):
        """
        Merges another metrics spec into the current one.

        The `other` MetricsSpec overwrites existing metrics.
        """
        self.daily += other.daily
        self.weekly += other.weekly
        self.days28 += other.days28
        self.overall += other.overall
        self.definitions.update(other.definitions)


_converter.register_structure_hook(MetricsSpec, lambda obj, _type: MetricsSpec.from_dict(obj))


@attr.s(auto_attribs=True)
class DataSourceDefinition:
    """Describes the interface for defining a data source in configuration."""

    name: str  # implicit in configuration
    from_expression: str
    experiments_column_type: Optional[str] = None
    client_id_column: Optional[str] = None
    submission_date_column: Optional[str] = None

    def resolve(self, spec: "AnalysisSpec") -> mozanalysis.metrics.DataSource:
        params: Dict[str, Any] = {"name": self.name, "from_expr": self.from_expression}
        # Allow mozanalysis to infer defaults for these values:
        for k in ("experiments_column_type", "client_id_column", "submission_date_column"):
            v = getattr(self, k)
            if v:
                params[k] = v
        # experiments_column_type is a little special, though!
        # `None` is a valid value, which means there isn't any `experiments` column in the
        # data source, so mozanalysis shouldn't try to use it.
        # But mozanalysis has a different default value for that param ("simple"), and
        # TOML can't represent an explicit null. So we'll look for the string "none" and
        # transform it to the value None.
        if (self.experiments_column_type or "").lower() == "none":
            params["experiments_column_type"] = None
        return mozanalysis.metrics.DataSource(**params)


@attr.s(auto_attribs=True)
class DataSourcesSpec:
    """Holds data source definitions.

    This doesn't have a resolve() method to produce a concrete DataSourcesConfiguration
    because it's just a container for the definitions, and we don't need it after the spec phase."""

    definitions: Dict[str, DataSourceDefinition] = attr.Factory(dict)

    @classmethod
    def from_dict(cls, d: dict) -> "DataSourcesSpec":
        definitions = {
            k: _converter.structure(
                {"name": k, **dict((kk.lower(), vv) for kk, vv in v.items())}, DataSourceDefinition
            )
            for k, v in d.items()
        }
        return cls(definitions)

    def merge(self, other: "DataSourcesSpec"):
        """
        Merge another datasource spec into the current one.
        The `other` DataSourcesSpec overwrites existing keys.
        """
        self.definitions.update(other.definitions)


_converter.register_structure_hook(
    DataSourcesSpec, lambda obj, _type: DataSourcesSpec.from_dict(obj)
)


@attr.s(auto_attribs=True)
class SegmentDataSourceDefinition:
    name: str
    from_expression: str
    window_start: int = 0
    window_end: int = 0
    client_id_column: Optional[str] = None
    submission_date_column: Optional[str] = None

    def resolve(
        self, spec: "AnalysisSpec", experiment: ExperimentConfiguration
    ) -> mozanalysis.segments.SegmentDataSource:
        env = jinja2.Environment(autoescape=False, undefined=StrictUndefined)
        from_expr = env.from_string(self.from_expression).render(experiment=experiment)
        kwargs = {
            "name": self.name,
            "from_expr": from_expr,
            "window_start": self.window_start,
            "window_end": self.window_end,
        }
        for k in ("client_id_column", "submission_date_column"):
            if v := getattr(self, k):
                kwargs[k] = v
        return mozanalysis.segments.SegmentDataSource(**kwargs)


@attr.s(auto_attribs=True)
class SegmentDataSourceReference:
    name: str

    def resolve(
        self, spec: "AnalysisSpec", experiment: ExperimentConfiguration
    ) -> mozanalysis.segments.SegmentDataSource:
        return _lookup_name(
            name=self.name,
            klass=mozanalysis.segments.SegmentDataSource,
            spec=spec,
            module=experiment.platform.segments_module,
            definitions=spec.segments.data_sources,
            resolve_extras={"experiment": experiment},
        )


_converter.register_structure_hook(
    SegmentDataSourceReference, lambda obj, _type: SegmentDataSourceReference(name=obj)
)


@attr.s(auto_attribs=True)
class SegmentDefinition:
    name: str
    data_source: SegmentDataSourceReference
    select_expression: str
    friendly_name: Optional[str] = None
    description: Optional[str] = None

    def resolve(
        self, spec: "AnalysisSpec", experiment: ExperimentConfiguration
    ) -> mozanalysis.segments.Segment:
        data_source = self.data_source.resolve(spec, experiment)
        return mozanalysis.segments.Segment(
            name=self.name,
            data_source=data_source,
            select_expr=_metrics_environment.from_string(self.select_expression).render(),
            friendly_name=self.friendly_name,
            description=self.description,
        )


@attr.s(auto_attribs=True)
class SegmentsSpec:
    definitions: Dict[str, SegmentDefinition] = attr.Factory(dict)
    data_sources: Dict[str, SegmentDataSourceDefinition] = attr.Factory(dict)

    @classmethod
    def from_dict(cls, d: dict) -> "SegmentsSpec":
        data_sources = {
            k: _converter.structure(
                {"name": k, **dict((kk.lower(), vv) for kk, vv in v.items())},
                SegmentDataSourceDefinition,
            )
            for k, v in d.pop("data_sources", {}).items()
        }
        definitions = {
            k: _converter.structure(
                {"name": k, **dict((kk.lower(), vv) for kk, vv in v.items())}, SegmentDefinition
            )
            for k, v in d.items()
        }
        return cls(definitions, data_sources)

    def merge(self, other: "SegmentsSpec"):
        """
        Merge another segments spec into the current one.
        The `other` SegmentsSpec overwrites existing keys.
        """
        self.data_sources.update(other.data_sources)
        self.definitions.update(other.definitions)


_converter.register_structure_hook(SegmentsSpec, lambda obj, _type: SegmentsSpec.from_dict(obj))


@attr.s(auto_attribs=True)
class AnalysisConfiguration:
    """A fully concrete representation of the configuration for an experiment.

    Instead of instantiating this directly, consider using AnalysisSpec.resolve().
    """

    experiment: ExperimentConfiguration
    metrics: MetricsConfigurationType


@attr.s(auto_attribs=True)
class AnalysisSpec:
    """Represents a configuration file.

    The expected use is like:
        AnalysisSpec.from_dict(toml.load(my_configuration_file)).resolve(an_experimenter_object)
    which will produce a fully populated, concrete AnalysisConfiguration.
    """

    experiment: ExperimentSpec = attr.Factory(ExperimentSpec)
    metrics: MetricsSpec = attr.Factory(MetricsSpec)
    data_sources: DataSourcesSpec = attr.Factory(DataSourcesSpec)
    segments: SegmentsSpec = attr.Factory(SegmentsSpec)
    _resolved: bool = False

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AnalysisSpec":
        return _converter.structure(d, cls)

    @classmethod
    def default_for_experiment(
        cls, experiment: "jetstream.experimenter.Experiment"
    ) -> "AnalysisSpec":
        """Return the default spec based on the experiment type."""
        from . import default_config

        if platform := PLATFORM_CONFIGS.get(experiment.app_name):
            default_metrics = platform.resolve_config()
        else:
            default_metrics = cls()

        type_metrics = default_config.DefaultConfigsResolver.resolve(experiment.type)

        if type_metrics is not None:
            default_metrics.merge(type_metrics.spec)

        return default_metrics

    def resolve(
        self,
        experimenter: "jetstream.experimenter.Experiment",
        external_configs: Optional["ExternalConfigCollection"] = None,
    ) -> AnalysisConfiguration:
        from . import outcomes

        if self._resolved:
            raise Exception("Can't resolve an AnalysisSpec twice")
        self._resolved = True

        outcomes_resolver = outcomes.OutcomesResolver.with_external_configs(external_configs)

        for slug in experimenter.outcomes:
            outcome = outcomes_resolver.resolve(slug)

            if outcome.platform == experimenter.app_name:
                self.merge_outcome(outcome.spec)
            else:
                raise ValueError(
                    f"Outcome {slug} doesn't support the platform '{experimenter.app_name}'"
                )

        experiment = self.experiment.resolve(self, experimenter)
        metrics = self.metrics.resolve(self, experiment)
        return AnalysisConfiguration(experiment, metrics)

    def merge(self, other: "AnalysisSpec"):
        """Merges another analysis spec into the current one."""
        self.experiment.merge(other.experiment)
        self.metrics.merge(other.metrics)
        self.data_sources.merge(other.data_sources)
        self.segments.merge(other.segments)

    def merge_outcome(self, other: "OutcomeSpec"):
        """Merges an outcome snippet into the analysis spec."""
        metrics = [MetricReference(metric_name) for metric_name, _ in other.metrics.items()]

        # metrics defined in outcome snippets are only computed for
        # weekly and overall analysis windows
        self.metrics.merge(
            MetricsSpec(
                daily=[], weekly=metrics, days28=[], overall=metrics, definitions=other.metrics
            )
        )
        self.data_sources.merge(other.data_sources)


@attr.s(auto_attribs=True)
class OutcomeSpec:
    """Represents an outcome snippet."""

    friendly_name: str
    description: str
    metrics: Dict[str, MetricDefinition] = attr.Factory(dict)
    default_metrics: Optional[List[MetricReference]] = attr.ib(None)
    data_sources: DataSourcesSpec = attr.Factory(DataSourcesSpec)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "OutcomeSpec":
        params: Dict[str, Any] = {}
        params["friendly_name"] = d["friendly_name"]
        params["description"] = d["description"]
        params["data_sources"] = _converter.structure(d.get("data_sources", {}), DataSourcesSpec)
        params["metrics"] = {
            k: _converter.structure(
                {"name": k, **dict((kk.lower(), vv) for kk, vv in v.items())}, MetricDefinition
            )
            for k, v in d.get("metrics", {}).items()
        }
        params["default_metrics"] = [
            _converter.structure(m, MetricReference) for m in d.get("default_metrics", [])
        ]

        # check that default metrics are actually defined in outcome
        for default_metric in params["default_metrics"]:
            if default_metric.name not in params["metrics"].keys():
                raise ValueError(f"Default metric {default_metric} is not defined in outcome.")

        return cls(**params)
