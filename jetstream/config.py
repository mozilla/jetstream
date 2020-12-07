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
import mozanalysis.metrics
import mozanalysis.metrics.desktop
import mozanalysis.segments
import mozanalysis.segments.desktop
import pytz
import toml
from jinja2 import StrictUndefined

from jetstream.errors import NoStartDateException
from jetstream.pre_treatment import PreTreatment
from jetstream.statistics import Statistic, Summary

from . import AnalysisPeriod, probe_sets

if TYPE_CHECKING:
    import jetstream.experimenter

DEFAULT_METRICS_CONFIG = Path(__file__).parent / "config" / "default_metrics.toml"
CFR_METRICS_CONFIG = Path(__file__).parent / "config" / "cfr_metrics.toml"


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

    def resolve(self, spec: "AnalysisSpec") -> List[Summary]:
        if self.name in spec.metrics.definitions:
            return spec.metrics.definitions[self.name].resolve(spec)
        if hasattr(mozanalysis.metrics.desktop, self.name):
            raise ValueError(f"Please define a statistical treatment for the metric {self.name}")
        raise ValueError(f"Could not locate metric {self.name}")


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
class SegmentReference:
    name: str

    def resolve(
        self, spec: "AnalysisSpec", experiment: "ExperimentConfiguration"
    ) -> mozanalysis.segments.Segment:
        search = mozanalysis.segments.desktop
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
    probe_sets_resolver: probe_sets.ResolvesProbeSets
    segments: List[mozanalysis.segments.Segment]

    def __attrs_post_init__(self):
        # Catch any exceptions at instantiation
        self._enrollment_query = self.enrollment_query
        self._probe_sets = self.probe_sets

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
    def probe_sets(self) -> List[probe_sets.ProbeSet]:
        return [
            self.probe_sets_resolver.resolve(slug)
            for slug in self.experimenter_experiment.probe_sets
        ]

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


@attr.s(auto_attribs=True)
class ExperimentSpec:
    """Describes the interface for overriding experiment details."""

    # TODO: Expand this list.
    enrollment_query: Optional[str] = None
    enrollment_period: Optional[int] = None
    reference_branch: Optional[str] = None
    start_date: Optional[str] = attr.ib(default=None, validator=_validate_yyyy_mm_dd)
    end_date: Optional[str] = attr.ib(default=None, validator=_validate_yyyy_mm_dd)
    segments: List[SegmentReference] = attr.Factory(list)

    @staticmethod
    def parse_date(yyyy_mm_dd: Optional[str]) -> Optional[dt.datetime]:
        if not yyyy_mm_dd:
            return None
        return dt.datetime.strptime(yyyy_mm_dd, "%Y-%m-%d").replace(tzinfo=pytz.utc)

    def resolve(
        self,
        spec: "AnalysisSpec",
        experimenter: "jetstream.experimenter.Experiment",
        probe_sets_resolver: probe_sets.ResolvesProbeSets,
    ) -> ExperimentConfiguration:
        experiment = ExperimentConfiguration(self, experimenter, probe_sets_resolver, [])
        # Segment data sources may need to know the enrollment dates of the experiment,
        # so we'll forward the Experiment we know about so far.
        experiment.segments = [ref.resolve(spec, experiment) for ref in self.segments]
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

    def resolve(self, spec: "AnalysisSpec") -> List[Summary]:
        if self.select_expression is None or self.data_source is None:
            # checks if a metric from mozanalysis was referenced
            search = mozanalysis.metrics.desktop
            metric = _lookup_name(
                name=self.name,
                klass=mozanalysis.metrics.Metric,
                spec=spec,
                module=search,
                definitions={},
            )
        else:
            select_expression = _metrics_environment.from_string(self.select_expression).render()

            metric = mozanalysis.metrics.Metric(
                name=self.name,
                data_source=self.data_source.resolve(spec),
                select_expr=select_expression,
                friendly_name=self.friendly_name,
                description=self.description,
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
    overall: List[MetricReference] = attr.Factory(list)

    definitions: Dict[str, MetricDefinition] = attr.Factory(dict)

    @classmethod
    def from_dict(cls, d: dict) -> "MetricsSpec":
        params: Dict[str, Any] = {}
        known_keys = {f.name for f in attr.fields(cls)}
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

    def resolve(
        self, spec: "AnalysisSpec", experiment: ExperimentConfiguration
    ) -> MetricsConfigurationType:
        result = {}
        for period in AnalysisPeriod:
            # these summaries might contain duplicates
            summaries = []
            if period in (AnalysisPeriod.WEEK, AnalysisPeriod.OVERALL):
                for feature in experiment.probe_sets:
                    summaries.extend(feature.to_summaries())
            summaries.extend(
                [
                    summary
                    for ref in getattr(self, period.adjective)
                    for summary in ref.resolve(spec)
                ]
            )

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
            module=mozanalysis.segments.desktop,
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
            k: _converter.structure({"name": k, **v}, SegmentDataSourceDefinition)
            for k, v in d.pop("data_sources", {}).items()
        }
        definitions = {
            k: _converter.structure({"name": k, **v}, SegmentDefinition) for k, v in d.items()
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

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AnalysisSpec":
        return _converter.structure(d, cls)

    @classmethod
    def default_for_experiment(
        cls, experiment: "jetstream.experimenter.Experiment"
    ) -> "AnalysisSpec":
        """Return the default spec based on the experiment type."""
        default_metrics = cls.from_dict(toml.load(DEFAULT_METRICS_CONFIG))

        if experiment.type == "message":
            # CFR experiment
            cfr_metrics = cls.from_dict(toml.load(CFR_METRICS_CONFIG))
            default_metrics.merge(cfr_metrics)
            return default_metrics

        return default_metrics

    def resolve(self, experimenter: "jetstream.experimenter.Experiment") -> AnalysisConfiguration:
        experiment = self.experiment.resolve(self, experimenter, probe_sets.ProbeSetsResolver)
        metrics = self.metrics.resolve(self, experiment)
        return AnalysisConfiguration(experiment, metrics)

    def merge(self, other: "AnalysisSpec"):
        """Merges another analysis spec into the current one."""
        self.experiment.merge(other.experiment)
        self.metrics.merge(other.metrics)
        self.data_sources.merge(other.data_sources)
        self.segments.merge(other.segments)
