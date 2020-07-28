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
class Feature:
    slug: str
    telemetry: Union[FeatureEventTelemetry, FeatureScalarTelemetry]


@attr.s(auto_attribs=True, kw_only=True, slots=True, frozen=True)
class Branch:
    slug: str
    ratio: int


@attr.s(auto_attribs=True, kw_only=True, slots=True, frozen=True)
class Experiment:
    """Common Experimenter experiment representation."""

    slug: str
    normandy_slug: Optional[str]
    type: Optional[str]
    status: Optional[str]
    active: bool
    features: List[Feature]
    branches: List[Branch]
    start_date: Optional[dt.datetime]
    end_date: Optional[dt.datetime]
    proposed_enrollment: int
    reference_branch: Optional[str]


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
        branches = [Branch(slug=variant.slug, ratio=variant.ratio) for variant in self.variants]
        control_slug = None

        control_slugs = [variant.slug for variant in self.variants if variant.is_control]
        if len(control_slugs) == 1:
            control_slug = control_slugs[0]

        return Experiment(
            normandy_slug=self.normandy_slug,
            slug=self.slug,
            type=self.type,
            status=self.status,
            active=self.status == "Live",
            start_date=self.start_date,
            end_date=self.end_date,
            proposed_enrollment=self.proposed_enrollment,
            features=[],
            branches=branches,
            reference_branch=control_slug,
        )


@attr.s(auto_attribs=True, kw_only=True, slots=True, frozen=True)
class NimbusExperiment:
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
    def from_dict(cls, d) -> "NimbusExperiment":
        converter = cattr.Converter()
        converter.register_structure_hook(
            dt.datetime, lambda num, _: _unix_millis_to_datetime(num),
        )
        return converter.structure(d, cls)

    def to_experiment(self) -> "Experiment":
        """Convert to Experiment."""
        return Experiment(
            normandy_slug=None,
            slug=self.slug,
            type="nimbus",
            status=None,
            active=self.active,
            start_date=self.start_date,
            end_date=self.end_date,
            proposed_enrollment=self.proposed_enrollment,
            features=self.features,
            branches=self.branches,
            reference_branch=self.reference_branch,
        )


@attr.s(auto_attribs=True)
class ExperimentCollection:
    experiments: List[Experiment] = attr.Factory(list)

    EXPERIMENTER_API_URL_V1 = "https://experimenter.services.mozilla.com/api/v1/experiments/"

    # for nimbus experiments
    EXPERIMENTER_API_URL_V4 = "https://experimenter.services.mozilla.com/api/v4/experiments/"

    @classmethod
    def from_experimenter(cls, session: requests.Session = None) -> "ExperimentCollection":
        session = session or requests.Session()
        legacy_experiments_json = session.get(cls.EXPERIMENTER_API_URL_V1).json()
        legacy_experiments = [
            LegacyExperiment.from_dict(experiment).to_experiment()
            for experiment in legacy_experiments_json
        ]

        nimbus_experiments_json = session.get(cls.EXPERIMENTER_API_URL_V4).json()
        nimbus_experiments = [
            NimbusExperiment.from_dict(experiment).to_experiment()
            for experiment in nimbus_experiments_json
        ]

        return cls(nimbus_experiments + legacy_experiments)

    def of_type(self, type_or_types: Union[str, Iterable[str]]) -> "ExperimentCollection":
        if isinstance(type_or_types, str):
            type_or_types = (type_or_types,)
        cls = type(self)
        return cls([ex for ex in self.experiments if ex.type in type_or_types])

    def ever_launched(self) -> "ExperimentCollection":
        cls = type(self)
        return cls(
            [
                ex
                for ex in self.experiments
                if ex.status in ("Complete", "Live") or ex.status is None
            ]
        )

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
