import datetime as dt
from typing import List, Iterable, Optional, Union

import attr
import cattr
import requests
import pytz


@attr.s(auto_attribs=True, kw_only=True, slots=True, frozen=True)
class Variant:
    is_control: bool
    slug: str
    ratio: int


def _coerce_none_to_zero(x: Optional[int]) -> int:
    return 0 if x is None else x


def _unix_millis_to_datetime(num: Optional[float]) -> Optional[dt.datetime]:
    if num is None:
        return None
    return dt.datetime.fromtimestamp(num / 1e3, pytz.utc)


@attr.s(auto_attribs=True, kw_only=True, slots=True, frozen=True)
class LegacyExperiment:
    """Experimenter v1 experiment."""

    slug: str  # experimenter slug
    type: str
    status: str
    start_date: Optional[dt.datetime]
    end_date: Optional[dt.datetime]
    proposed_enrollment: Optional[int] = attr.ib(converter=_coerce_none_to_zero)
    variants: List[Variant]
    normandy_slug: Optional[str] = None

    @classmethod
    def from_dict(cls, d) -> "LegacyExperiment":
        converter = cattr.Converter()
        converter.register_structure_hook(
            dt.datetime, lambda num, _: _unix_millis_to_datetime(num),
        )
        return converter.structure(d, cls)

    def to_experiment(self) -> "Experiment":
        """Convert to Experiment."""
        if not self.normandy_slug:
            raise ValueError(
                f"Cannot convert legacy experiment {self.slug}. Missing Normandy slug."
            )

        branches = [Branch(slug=variant.slug, ratio=variant.ratio) for variant in self.variants]
        control_slug = next(variant.slug for variant in self.variants if variant.is_control)

        return Experiment(
            slug=self.normandy_slug,
            active=self.status == "Live",
            start_date=self.start_date,
            end_Date=self.end_date,
            proposed_enrollment=self.proposed_enrollment,
            features=[],
            branches=branches,
            reference_branch=control_slug,
        )


@attr.s(auto_attribs=True, kw_only=True, slots=True, frozen=True)
class Experiment:
    """Represents a v4 experiment from Experimenter."""

    slug: str  # Normandy slug
    active: bool
    features: List[Feature]
    branches: List[Branch]
    start_date: dt.datetime
    end_date: Optional[dt.datetime]
    proposed_enrollment: int
    reference_branch: str

    @classmethod
    def from_dict(cls, d) -> "Experiment":
        converter = cattr.Converter()
        converter.register_structure_hook(
            dt.datetime, lambda num, _: _unix_millis_to_datetime(num),
        )
        return converter.structure(d, cls)


@attr.s(auto_attribs=True, kw_only=True, slots=True, frozen=True)
class Feature:
    slug: str
    telemetry: Union[FeatureEventTelemetry, FeatureScalarTelemetry]


@attr.s(auto_attribs=True, kw_only=True, slots=True, frozen=True)
class FeatureScalarTelemetry:
    kind: str = "scalar"
    name: str


@attr.s(auto_attribs=True, kw_only=True, slots=True, frozen=True)
class FeatureEventTelemetry:
    kind: str = "event"
    event_category: str
    event_method: Optional[str]
    event_object: Optional[str]
    event_value: Optional[str]


@attr.s(auto_attribs=True, kw_only=True, slots=True, frozen=True)
class Branch:
    slug: str
    ratio: int


@attr.s(auto_attribs=True)
class ExperimentCollection:
    experiments: List[Experiment] = attr.Factory(list)

    EXPERIMENTER_API_URL_V1 = "https://experimenter.services.mozilla.com/api/v1/experiments/"

    # for nimbus experiments
    EXPERIMENTER_API_URL_V4 = "https://experimenter.services.mozilla.com/api/v4/experiments/"

    @classmethod
    def from_experimenter(cls, session: requests.Session = None) -> "ExperimentCollection":
        session = session or requests.Session()
        experiments = session.get(cls.EXPERIMENTER_API_URL).json()
        return cls([Experiment.from_dict(experiment) for experiment in experiments])
        # todo get v4 experiments
        # todo convert experiments to a common structure or create a proxy that encapsulates them

    def of_type(self, type_or_types: Union[str, Iterable[str]]) -> "ExperimentCollection":
        if isinstance(type_or_types, str):
            type_or_types = (type_or_types,)
        cls = type(self)
        return cls([ex for ex in self.experiments if ex.type in type_or_types])

    def ever_launched(self) -> "ExperimentCollection":
        cls = type(self)
        return cls([ex for ex in self.experiments if ex.status in ("Complete", "Live")])

    def with_slug(self, slug: str) -> "ExperimentCollection":
        cls = type(self)
        return cls([ex for ex in self.experiments if ex.slug == slug or ex.normandy_slug == slug])

    def started_since(self, since: dt.datetime) -> "ExperimentCollection":
        """All experiments that ever launched after a given time.

        since should be a tz-aware datetime."""
        cls = type(self)
        return cls(
            [
                ex
                for ex in self.ever_launched().experiments
                if ex.start_date and ex.start_date >= since
            ]
        )

    def end_on_or_after(self, after: dt.datetime) -> "ExperimentCollection":
        """All experiments that ever launched that end on or after the specified time.

        after should be a tz-aware datetime."""
        cls = type(self)
        return cls(
            [ex for ex in self.ever_launched().experiments if ex.end_date and ex.end_date >= after]
        )
