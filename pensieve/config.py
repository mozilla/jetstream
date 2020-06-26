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

from inspect import isabstract
from types import ModuleType
from typing import Any, Dict, List, Mapping, Optional, Type, TypeVar

import attr
import cattr
import jinja2
import mozanalysis.metrics
import mozanalysis.metrics.desktop
import pandas

from . import AnalysisPeriod
import pensieve.experimenter
from pensieve.statistics import Statistic, StatisticResultCollection
from pensieve.pre_treatment import PreTreatment


@attr.s(auto_attribs=True)
class Summary:
    """Represents a metric with a statistical treatment."""

    metric: mozanalysis.metrics.Metric
    statistic: Statistic
    pre_treatments: List[PreTreatment] = attr.Factory(list)

    def run(self, data: pandas.DataFrame) -> "StatisticResultCollection":
        """Apply the statistic transformation for data related to the specified metric."""
        for pre_treatment in self.pre_treatments:
            data = pre_treatment.apply(data, self.metric.name)

        return self.statistic.apply(data, self.metric.name)


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

    def resolve(
        self, spec: "AnalysisSpec", experimenter: pensieve.experimenter.Experiment
    ) -> List[Summary]:
        if self.name in spec.metrics.definitions:
            return spec.metrics.definitions[self.name].resolve(spec, experimenter)
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
class PreTreatmentReference:
    name: str

    def resolve(self, spec: "AnalysisSpec") -> PreTreatment:
        for pre_treatment in PreTreatment.__subclasses__():
            if isabstract(pre_treatment):
                continue
            if pre_treatment.name() == self.name:
                return pre_treatment()  # type: ignore

        raise ValueError(f"Could not find pre-treatment {self.name}.")


_converter.register_structure_hook(
    PreTreatmentReference, lambda obj, _type: PreTreatmentReference(name=obj)
)


@attr.s(auto_attribs=True)
class ExperimentConfiguration:
    """Represents the configuration of an experiment for analysis."""

    experiment_spec: "ExperimentSpec"
    experimenter_experiment: pensieve.experimenter.Experiment

    def __getattr__(self, name):
        equivalents = {
            # Experimenter name: config name
            "proposed_enrollment": "enrollment_period",
        }
        if name in equivalents:
            candidate_attr = getattr(self.experiment_spec, equivalents[name])
            if candidate_attr is not None:
                return candidate_attr
        if hasattr(self.experiment_spec, name):
            return getattr(self.experiment_spec, name)
        return getattr(self.experimenter_experiment, name)


@attr.s(auto_attribs=True)
class ExperimentSpec:
    """Describes the interface for overriding experiment details."""

    # TODO: Expand this list.
    enrollment_query: Optional[str] = None
    enrollment_period: Optional[int] = None

    def resolve(self, experimenter: pensieve.experimenter.Experiment) -> ExperimentConfiguration:
        return ExperimentConfiguration(self, experimenter)

    def merge(self, other: "ExperimentSpec") -> None:
        self.enrollment_query = other.enrollment_query or self.enrollment_query
        self.enrollment_period = other.enrollment_period or self.enrollment_period


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

    def resolve(
        self, spec: "AnalysisSpec", experimenter: pensieve.experimenter.Experiment
    ) -> List[Summary]:
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
            )

        metrics_with_treatments = []

        for statistic_name, params in self.statistics.items():
            for statistic in Statistic.__subclasses__():
                if statistic.name() == statistic_name:
                    break
            else:
                raise ValueError(f"Statistic {statistic_name} does not exist.")

            pre_treatments = []
            for pt in params.pop("pre_treatments", []):
                ref = PreTreatmentReference(pt)
                pre_treatments.append(ref.resolve(spec))

            if "ref_branch_label" not in params:
                for variant in experimenter.variants:
                    if variant.is_control:
                        params["ref_branch_label"] = variant.slug

            metrics_with_treatments.append(
                Summary(
                    metric=metric,
                    statistic=statistic.from_dict(params),
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
        self, spec: "AnalysisSpec", experimenter: pensieve.experimenter.Experiment
    ) -> MetricsConfigurationType:
        result = {}
        for period in AnalysisPeriod:
            # these summaries might contain duplicates
            summaries = [
                m
                for ref in getattr(self, period.adjective)
                for m in ref.resolve(spec, experimenter)
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

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AnalysisSpec":
        return _converter.structure(d, cls)

    def resolve(self, experimenter: pensieve.experimenter.Experiment) -> AnalysisConfiguration:
        experiment = self.experiment.resolve(experimenter)
        metrics = self.metrics.resolve(self, experimenter)
        return AnalysisConfiguration(experiment, metrics)

    def merge(self, other: "AnalysisSpec"):
        """Merges another analysis spec into the current one."""
        self.experiment.merge(other.experiment)
        self.metrics.merge(other.metrics)
        self.data_sources.merge(other.data_sources)
