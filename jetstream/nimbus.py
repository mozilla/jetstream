from pathlib import Path
import re
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Union,
)

import attr
import cattr
from github import Github
from github.ContentFile import ContentFile
from google.cloud import bigquery
from google.cloud.bigquery.schema import SchemaField
from mozanalysis.metrics import Metric
import mozanalysis.metrics.desktop
import toml

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
class FeatureScalarTelemetry:
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
class FeatureEventTelemetry:
    kind: ClassVar[str] = "event"
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


FeatureTelemetryType = Union[FeatureEventTelemetry, FeatureScalarTelemetry]


@attr.s(auto_attribs=True, kw_only=True, slots=True, frozen=True)
class Feature:
    slug: str
    name: str
    description: str
    telemetry: List[FeatureTelemetryType]

    def to_summaries(self) -> List[statistics.Summary]:
        summaries = []
        for t in self.telemetry:
            summaries.extend(t.to_summaries(self.slug))
        return summaries

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]):
        converter = cattr.Converter()

        def discriminate_telemetry(d, type):
            kind = d.pop("kind")
            for klass in type.__args__:
                if kind == klass.kind:
                    return klass(**d)
            raise ValueError(f"Could not discriminate telemetry kind {kind}")

        converter.register_structure_hook(FeatureTelemetryType, discriminate_telemetry)
        return converter.structure(d, cls)


class ResolvesFeatures(Protocol):
    def resolve(self, slug: str) -> Feature:
        ...


class _FeatureResolver:
    """Consume Features from the nimbus-shared repository.

    This document describes how data is represented in nimbus-shared:
    https://github.com/mozilla/nimbus-shared/blob/29526cb13c3b12ed6870ebd042261273a6e02785/docs/pages/dev/data.md

    The Feature data adopts the described convention of using a __nimbusMeta.toml
    file in the same path as our data to describe the type of the data, so we avoid
    consuming files with leading underscores.
    """

    FEATURE_DEFINITION_REPO = "mozilla/nimbus-shared"
    FEATURE_PATH = "data/features"

    @property
    def data(self) -> Dict[str, Feature]:
        if data := getattr(self, "_data", None):
            return data
        g = Github()
        repo = g.get_repo(self.FEATURE_DEFINITION_REPO)
        specs = repo.get_contents(self.FEATURE_PATH)

        if isinstance(specs, ContentFile):
            specs = [specs]

        data = {}

        for spec in specs:
            p = Path(spec.name)
            slug = p.stem
            if slug.startswith("_") or p.suffix != ".toml":
                continue
            contents = toml.loads(spec.decoded_content.decode("utf-8"))
            contents["slug"] = slug
            feature = Feature.from_dict(contents)
            data[slug] = feature

        self._data = data
        return self._data

    def resolve(self, slug: str) -> Feature:
        return self.data[slug]


FeatureResolver = _FeatureResolver()
