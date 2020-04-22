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

from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional, Set, Type, TypeVar

import attr
import cattr
import jinja2
import mozanalysis.metrics
import mozanalysis.metrics.desktop

import pensieve.experimenter


# TODO: these should be objects that pair a metric with a default statistical treatment.
DEFAULT_METRICS = {
    "desktop": {
        "daily": {mozanalysis.metrics.desktop.unenroll},
        "weekly": {
            mozanalysis.metrics.desktop.active_hours,
            mozanalysis.metrics.desktop.uri_count,
            mozanalysis.metrics.desktop.ad_clicks,
            mozanalysis.metrics.desktop.search_count,
        },
        "overall": {
            mozanalysis.metrics.desktop.active_hours,
            mozanalysis.metrics.desktop.uri_count,
            mozanalysis.metrics.desktop.ad_clicks,
            mozanalysis.metrics.desktop.search_count,
        },
    }
}


_converter = cattr.Converter()


def _populate_environment() -> jinja2.Environment:
    """Create a Jinja2 environment that understands the SQL agg_* helpers in mozanalysis.metrics.

    Just a wrapper to avoid leaking temporary variables to the module scope."""
    env = jinja2.Environment(autoescape=False)
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
    name: str, klass: Type[T], spec: "AnalysisSpec", module: Optional[ModuleType], definitions: dict
) -> T:
    needle = None
    if module and hasattr(module, name):
        needle = getattr(module, name)
    if name in definitions:
        needle = definitions[name].resolve(spec)
    if isinstance(needle, klass):
        return needle
    raise ValueError(f"Could not locate {klass.__name__} {name}")


@attr.s(auto_attribs=True)
class MetricReference:
    name: str

    def resolve(self, spec: "AnalysisSpec") -> mozanalysis.metrics.Metric:
        search = mozanalysis.metrics.desktop
        return _lookup_name(
            name=self.name,
            klass=mozanalysis.metrics.Metric,
            spec=spec,
            module=search,
            definitions=spec.metrics.definitions,
        )


# These are bare strings in the configuration file.
_converter.register_structure_hook(MetricReference, lambda obj, _type: MetricReference(name=obj))


@attr.s(auto_attribs=True)
class DataSourceReference:
    name: str

    def resolve(self, spec: "AnalysisSpec") -> mozanalysis.metrics.DataSource:
        search = mozanalysis.metrics.desktop
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
class ExperimentConfiguration:
    """Represents the configuration of an experiment for analysis."""

    experiment_spec: "ExperimentSpec"
    experimenter_experiment: pensieve.experimenter.Experiment

    def __getattr__(self, name):
        if hasattr(self.experiment_spec, name):
            return getattr(self.experiment_spec, name)
        return getattr(self.experimenter_experiment, name)


@attr.s(auto_attribs=True)
class ExperimentSpec:
    """Describes the interface for overriding experiment details."""

    # TODO: Expand this list.
    enrollment_query: Optional[str] = None

    def resolve(self, experimenter: pensieve.experimenter.Experiment) -> ExperimentConfiguration:
        return ExperimentConfiguration(self, experimenter)


@attr.s(auto_attribs=True)
class MetricDefinition:
    """Describes the interface for defining a metric in configuration.

    The `select_expression` of the metric may use Jinja2 template syntax to refer to the
    aggregation helper functions defined in `mozanalysis.metrics`, like
        '{{agg_any("payload.processes.scalars.some_boolean_thing")}}'
    """

    name: str  # implicit in configuration
    select_expression: str
    data_source: DataSourceReference

    def resolve(self, spec: "AnalysisSpec") -> mozanalysis.metrics.Metric:
        select_expression = _metrics_environment.from_string(self.select_expression).render()

        return mozanalysis.metrics.Metric(
            name=self.name,
            data_source=self.data_source.resolve(spec),
            select_expr=select_expression,
        )


@attr.s(auto_attribs=True)
class MetricsConfiguration:
    """Contains the metrics configured for each analysis window of an experiment."""

    daily: Set[mozanalysis.metrics.Metric]
    weekly: Set[mozanalysis.metrics.Metric]
    overall: Set[mozanalysis.metrics.Metric]


@attr.s(auto_attribs=True)
class MetricsSpec:
    """Describes the interface for the metrics section in configuration."""

    daily: List[MetricReference] = attr.Factory(list)
    weekly: List[MetricReference] = attr.Factory(list)
    overall: List[MetricReference] = attr.Factory(list)

    definitions: Dict[str, MetricDefinition] = attr.Factory(dict)

    @classmethod
    def from_dict(cls, d: dict) -> "MetricsSpec":
        params: Dict[str, Any] = {}
        known_keys = ("daily", "weekly", "overall")
        for k in known_keys:
            v = d.get(k, [])
            if not isinstance(v, list):
                raise ValueError(f"metrics.{k} should be a list of metrics")
            params[k] = [MetricReference(m) for m in v]
        params["definitions"] = {
            k: _converter.structure({"name": k, **v}, MetricDefinition)
            for k, v in d.items()
            if k not in known_keys
        }
        return cls(**params)

    @staticmethod
    def _merge_metrics(
        user: Iterable[mozanalysis.metrics.Metric], default: Iterable[mozanalysis.metrics.Metric]
    ) -> Set[mozanalysis.metrics.Metric]:
        result = set(user)
        user_names = {m.name for m in user}
        for m in default:
            if m.name not in user_names:
                result.add(m)
        return result

    def resolve(self, spec: "AnalysisSpec") -> MetricsConfiguration:
        def merge(k: str):
            return self._merge_metrics(
                {ref.resolve(spec) for ref in getattr(self, k)}, DEFAULT_METRICS["desktop"][k]
            )

        return MetricsConfiguration(merge("daily"), merge("weekly"), merge("overall"))


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
        params = {"name": self.name, "from_expr": self.from_expression}
        # Allow mozanalysis to infer defaults for these values:
        for k in ("experiments_column_type", "client_id_column", "submission_date_column"):
            v = getattr(self, k)
            if v:
                params[k] = v
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
            k: _converter.structure({"name": k, **v}, DataSourceDefinition) for k, v in d.items()
        }
        return cls(definitions)


_converter.register_structure_hook(
    DataSourcesSpec, lambda obj, _type: DataSourcesSpec.from_dict(obj)
)


@attr.s(auto_attribs=True)
class AnalysisConfiguration:
    """A fully concrete representation of the configuration for an experiment.

    Instead of instantiating this directly, consider using AnalysisSpec.resolve().
    """

    experiment: ExperimentConfiguration
    metrics: MetricsConfiguration


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

    @classmethod
    def from_dict(cls, d: dict) -> "AnalysisSpec":
        return _converter.structure(d, cls)

    def resolve(self, experimenter: pensieve.experimenter.Experiment) -> AnalysisConfiguration:
        experiment = self.experiment.resolve(experimenter)
        metrics = self.metrics.resolve(self)
        return AnalysisConfiguration(experiment, metrics)
