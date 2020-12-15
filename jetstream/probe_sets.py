import re
from typing import Any, ClassVar, Dict, Iterable, List, Mapping, Optional, Protocol, Union

import attr
import cattr
import mozanalysis.metrics.desktop
from google.cloud import bigquery
from google.cloud.bigquery.schema import SchemaField
from mozanalysis.metrics import Metric
import requests

from . import statistics


class _ProbeLister:
    @property
    def schema(self) -> List[SchemaField]:
        if schema := getattr(self, "_schema", None):
            return schema
        client = bigquery.Client()
        self._schema = client.get_table("moz-fx-data-shared-prod.telemetry.main").schema
        return self._schema

    @staticmethod
    def _step(schema: List[SchemaField], keys: Iterable[str]) -> Dict[str, SchemaField]:
        for key in keys:
            d = {field.name: field for field in schema}
            schema = d[key].fields
        return {field.name: field for field in schema}

    def columns_for_scalar(self, scalar_name: str) -> List[str]:
        scalar_slug = re.sub("[^a-zA-Z0-9]+", "_", scalar_name)
        columns = []
        processes = self._step(self.schema, ("payload", "processes"))
        for process_name, field in processes.items():
            scalars = self._step(field.fields, ("scalars",))
            if scalar_slug in scalars:
                column_name = ".".join(("payload.processes", process_name, "scalars", scalar_slug))
                columns.append(column_name)
        return columns


ProbeLister = _ProbeLister()


@attr.s(auto_attribs=True, kw_only=True, slots=True, frozen=True)
class TelemetryScalarProbe:
    kind: ClassVar[str] = "scalar"
    name: str

    def to_summaries(self, feature_slug: str) -> List[statistics.Summary]:
        column_names = ProbeLister.columns_for_scalar(self.name)

        column_exprs = []
        for column_name in column_names:
            column_exprs.append(f"COALESCE({column_name}, 0)")

        ever_used = statistics.Summary(
            Metric(
                f"{feature_slug}_ever_used",
                mozanalysis.metrics.desktop.main,
                f"SUM({' + '.join(column_exprs)}) > 0",
            ),
            statistics.Binomial(),
        )

        sum_metric = Metric(
            f"{feature_slug}_sum",
            mozanalysis.metrics.desktop.main,
            f"SUM({' + '.join(column_exprs)})",
        )

        used_mean = statistics.Summary(sum_metric, statistics.BootstrapMean())
        used_deciles = statistics.Summary(sum_metric, statistics.Deciles())

        return [ever_used, used_mean, used_deciles]


@attr.s(auto_attribs=True, kw_only=True, slots=True, frozen=True)
class TelemetryEventProbe:
    kind: ClassVar[str] = "event"
    name: str
    event_category: str
    event_method: Optional[str] = None
    event_object: Optional[str] = None
    event_value: Optional[str] = None

    def to_summaries(self, feature_slug: str) -> List[statistics.Summary]:
        clauses = [f"event_category = '{self.event_category}'"]
        for k in ("method", "object", "value"):
            if v := getattr(self, f"event_{k}"):
                clauses.append(f"event_{k} = '{v}'")
        predicate = " AND ".join(clauses)

        ever_used = statistics.Summary(
            Metric(
                f"{feature_slug}_ever_used",
                mozanalysis.metrics.desktop.events,
                f"COALESCE(COUNTIF({predicate}), 0) > 0",
            ),
            statistics.Binomial(),
        )

        sum_metric = Metric(
            f"{feature_slug}_sum",
            mozanalysis.metrics.desktop.events,
            f"COALESCE(COUNTIF({predicate}), 0)",
        )

        used_mean = statistics.Summary(sum_metric, statistics.BootstrapMean())
        used_deciles = statistics.Summary(sum_metric, statistics.Deciles())

        return [ever_used, used_mean, used_deciles]


ProbeType = Union[TelemetryEventProbe, TelemetryScalarProbe]


@attr.s(auto_attribs=True, kw_only=True, slots=True, frozen=True)
class ProbeSet:
    slug: str
    name: str
    probes: List[ProbeType]

    def to_summaries(self) -> List[statistics.Summary]:
        summaries = []
        for p in self.probes:
            summaries.extend(p.to_summaries(self.slug))
        return summaries

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]):
        converter = cattr.Converter()

        def discriminate_telemetry(d, type):
            kind = d.pop("kind")
            for klass in type.__args__:
                if kind == klass.kind:
                    return converter.structure(d, klass)
            raise ValueError(f"Could not discriminate probe kind {kind}")

        converter.register_structure_hook(ProbeType, discriminate_telemetry)
        return converter.structure(d, cls)


class ResolvesProbeSets(Protocol):
    def resolve(self, slug: str) -> ProbeSet:
        ...


@attr.s(auto_attribs=True)
class _ProbeSetsResolver:
    """Consume probe_sets from the Experimenter probesets API."""

    EXPERIMENTER_API_URL_PROBESETS = "https://experimenter.services.mozilla.com/api/v6/probesets/"

    @property
    def data(self) -> Dict[str, ProbeSet]:
        if data := getattr(self, "_data", None):
            return data

        session = requests.Session()
        blob = session.get(self.EXPERIMENTER_API_URL_PROBESETS).json()
        probe_sets = {}

        for probe_set_blob in blob:
            probe_set = ProbeSet.from_dict(probe_set_blob)
            probe_sets[probe_set.slug] = probe_set

        self._data = probe_sets
        return self._data

    def resolve(self, slug: str) -> ProbeSet:
        return self.data[slug]


ProbeSetsResolver = _ProbeSetsResolver()
