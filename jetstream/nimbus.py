from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Protocol, TYPE_CHECKING, Union

import attr
import cattr
from github import Github
from github.ContentFile import ContentFile
from mozanalysis.metrics import Metric
import mozanalysis.metrics.desktop
import toml

from . import statistics
from .statistics import Summary
from .probeinfo import DesktopProbeInfo

if TYPE_CHECKING:
    from .config import ExperimentConfiguration


@attr.s(auto_attribs=True, kw_only=True, slots=True, frozen=True)
class FeatureScalarTelemetry:
    kind: str = "scalar"
    name: str

    def to_summaries(
        self, feature_slug: str, experiment: "ExperimentConfiguration"
    ) -> List[Summary]:
        processes = DesktopProbeInfo.processes_for_scalar(self.name)
        scalar_slug = re.sub("[^a-zA-Z0-9]+", "_", self.name)
        columns = []
        for process in processes:
            process = "parent" if process == "main" else process
            columns.append(f"COALESCE(payload.processes.{process}.scalars.{scalar_slug}, 0)")

        kwargs: Dict[str, Any] = {}
        if experiment.reference_branch:
            kwargs["ref_branch_label"] = experiment.reference_branch

        ever_used = Summary(
            Metric(
                f"{feature_slug}_ever_used",
                mozanalysis.metrics.desktop.main,
                f"SUM({' + '.join(columns)}) > 0",
            ),
            statistics.Binomial(**kwargs),
        )

        sum_metric = Metric(
            f"{feature_slug}_sum", mozanalysis.metrics.desktop.main, f"SUM({' + '.join(columns)})"
        )

        used_mean = Summary(sum_metric, statistics.BootstrapMean(**kwargs))
        used_deciles = Summary(sum_metric, statistics.Deciles(**kwargs))

        return [ever_used, used_mean, used_deciles]


@attr.s(auto_attribs=True, kw_only=True, slots=True, frozen=True)
class FeatureEventTelemetry:
    kind: str = "event"
    event_category: str
    event_method: Optional[str] = None
    event_object: Optional[str] = None
    event_value: Optional[str] = None

    def to_summaries(
        self, feature_slug: str, experiment: "ExperimentConfiguration"
    ) -> List[Summary]:
        clauses = [f"event_category = '{self.event_category}'"]
        for k in ("method", "object", "value"):
            if v := getattr(self, f"event_{k}"):
                clauses.append(f"event_{k} = '{v}'")
        predicate = " AND ".join(clauses)

        kwargs: Dict[str, Any] = {}
        if experiment.reference_branch:
            kwargs["ref_branch_label"] = experiment.reference_branch

        ever_used = Summary(
            Metric(
                f"{feature_slug}_ever_used",
                mozanalysis.metrics.desktop.events,
                f"COALESCE(COUNTIF({predicate}), 0) > 0",
            ),
            statistics.Binomial(**kwargs),
        )

        sum_metric = Metric(
            f"{feature_slug}_sum",
            mozanalysis.metrics.desktop.events,
            f"COALESCE(COUNTIF({predicate}), 0)",
        )

        used_mean = Summary(sum_metric, statistics.BootstrapMean(**kwargs))
        used_deciles = Summary(sum_metric, statistics.Deciles(**kwargs))

        return [ever_used, used_mean, used_deciles]


@attr.s(auto_attribs=True, kw_only=True, slots=True, frozen=True)
class Feature:
    slug: str
    name: str
    description: str
    telemetry: List[Union[FeatureEventTelemetry, FeatureScalarTelemetry]]

    def to_summaries(self, experiment: "ExperimentConfiguration") -> List[Summary]:
        summaries = []
        for t in self.telemetry:
            summaries.extend(t.to_summaries(self.slug, experiment))
        return summaries

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]):
        return cattr.structure(d, cls)


class ResolvesFeatures(Protocol):
    def resolve(self, slug: str) -> Feature:
        ...


class _FeatureResolver:
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
