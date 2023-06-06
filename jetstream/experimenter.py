import datetime as dt
import logging
from typing import Iterable, List, Optional, Union

import attr
import cattr
import pytz
import requests
from metric_config_parser import experiment

from .util import retry_get

logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True, kw_only=True, slots=True, frozen=True)
class Variant:
    is_control: bool
    slug: str
    ratio: int


def _coerce_none_to_zero(x: Optional[int]) -> int:
    return 0 if x is None else x


@attr.s(auto_attribs=True, kw_only=True, slots=True, frozen=True)
class Outcome:
    slug: str


@attr.s(auto_attribs=True, kw_only=True, slots=True, frozen=True)
class ExperimentV1:
    """Experimenter v1 experiment."""

    slug: str  # experimenter slug
    type: str
    status: str
    start_date: Optional[dt.datetime]
    end_date: Optional[dt.datetime]
    proposed_enrollment: Optional[int] = attr.ib(converter=_coerce_none_to_zero)
    variants: List[Variant]
    normandy_slug: Optional[str] = None
    is_high_population: Optional[bool] = None
    outcomes: Optional[List[Outcome]] = None

    @staticmethod
    def _unix_millis_to_datetime(num: Optional[float]) -> Optional[dt.datetime]:
        if num is None:
            return None
        return dt.datetime.fromtimestamp(num / 1e3, pytz.utc)

    @classmethod
    def from_dict(cls, d) -> "ExperimentV1":
        converter = cattr.Converter()
        converter.register_structure_hook(
            dt.datetime,
            lambda num, _: cls._unix_millis_to_datetime(num),
        )
        return converter.structure(d, cls)

    def to_experiment(self) -> experiment.Experiment:
        """Convert to Experiment."""
        branches = [
            experiment.Branch(slug=variant.slug, ratio=variant.ratio) for variant in self.variants
        ]
        control_slug = None

        control_slugs = [variant.slug for variant in self.variants if variant.is_control]
        if len(control_slugs) == 1:
            control_slug = control_slugs[0]

        return experiment.Experiment(
            normandy_slug=self.normandy_slug,
            experimenter_slug=self.slug,
            type=self.type,
            status=self.status,
            start_date=self.start_date,
            end_date=self.end_date,
            proposed_enrollment=self.proposed_enrollment,
            branches=branches,
            reference_branch=control_slug,
            is_high_population=self.is_high_population or False,
            app_name="firefox_desktop",
            app_id="firefox-desktop",
            outcomes=[o.slug for o in self.outcomes] if self.outcomes else [],
        )


@attr.s(auto_attribs=True, kw_only=True, slots=True, frozen=True)
class ExperimentV6:
    """Represents a v6 experiment from Experimenter."""

    slug: str  # Normandy slug
    branches: List[experiment.Branch]
    startDate: Optional[dt.datetime]
    endDate: Optional[dt.datetime]
    proposedEnrollment: int
    referenceBranch: Optional[str]
    _appName: Optional[str] = None
    _appId: Optional[str] = None
    outcomes: Optional[List[Outcome]] = None
    enrollmentEndDate: Optional[dt.datetime] = None
    isEnrollmentPaused: Optional[bool] = None
    isRollout: bool = False

    @property
    def appName(self) -> str:
        return self._appName or "firefox_desktop"

    @property
    def appId(self) -> str:
        return self._appId or "firefox-desktop"

    @classmethod
    def from_dict(cls, d) -> "ExperimentV6":
        converter = cattr.Converter()
        converter.register_structure_hook(
            dt.datetime,
            lambda num, _: dt.datetime.strptime(num, "%Y-%m-%d"),
        )
        converter.register_structure_hook(
            cls,
            cattr.gen.make_dict_structure_fn(
                cls,
                converter,
                _appName=cattr.override(rename="appName"),
                _appId=cattr.override(rename="appId"),
            ),
        )
        return converter.structure(d, cls)

    def to_experiment(self) -> experiment.Experiment:
        """Convert to Experiment."""
        return experiment.Experiment(
            normandy_slug=self.slug,
            experimenter_slug=None,
            type="v6",
            status="Live"
            if (
                self.endDate
                and pytz.utc.localize(self.endDate) >= pytz.utc.localize(dt.datetime.now())
            )
            or self.endDate is None
            else "Complete",
            start_date=pytz.utc.localize(self.startDate) if self.startDate else None,
            end_date=pytz.utc.localize(self.endDate) if self.endDate else None,
            proposed_enrollment=self.proposedEnrollment,
            branches=self.branches,
            reference_branch=self.referenceBranch,
            is_high_population=False,
            app_name=self.appName,
            app_id=self.appId,
            outcomes=[o.slug for o in self.outcomes] if self.outcomes else [],
            enrollment_end_date=pytz.utc.localize(self.enrollmentEndDate)
            if self.enrollmentEndDate
            else None,
            is_enrollment_paused=bool(self.isEnrollmentPaused),
            is_rollout=self.isRollout,
        )


@attr.s(auto_attribs=True)
class ExperimentCollection:
    experiments: List[experiment.Experiment] = attr.Factory(list)

    MAX_RETRIES = 3
    EXPERIMENTER_API_URL_V1 = "https://experimenter.services.mozilla.com/api/v1/experiments/"

    # for nimbus experiments
    EXPERIMENTER_API_URL_V6 = "https://experimenter.services.mozilla.com/api/v6/experiments/"

    # experiments that are in draft state
    EXPERIMENTER_API_URL_V6_DRAFTS = (
        "https://experimenter.services.mozilla.com/api/v6/draft-experiments/"
    )

    # user agent sent to the Experimenter API
    USER_AGENT = "jetstream"

    @classmethod
    def from_experimenter(
        cls, session: requests.Session = None, with_draft_experiments=False
    ) -> "ExperimentCollection":
        session = session or requests.Session()
        legacy_experiments_json = retry_get(
            session, cls.EXPERIMENTER_API_URL_V1, cls.MAX_RETRIES, cls.USER_AGENT
        )
        legacy_experiments = []

        for legacy_experiment in legacy_experiments_json:
            if legacy_experiment["type"] != "rapid":
                try:
                    legacy_experiments.append(
                        ExperimentV1.from_dict(legacy_experiment).to_experiment()
                    )
                except Exception as e:
                    logger.exception(
                        str(e), exc_info=e, extra={"experiment": legacy_experiment["slug"]}
                    )

        nimbus_experiments_json = retry_get(
            session, cls.EXPERIMENTER_API_URL_V6, cls.MAX_RETRIES, cls.USER_AGENT
        )
        nimbus_experiments = []

        for nimbus_experiment in nimbus_experiments_json:
            try:
                nimbus_experiments.append(ExperimentV6.from_dict(nimbus_experiment).to_experiment())
            except Exception as e:
                logger.exception(
                    str(e), exc_info=e, extra={"experiment": nimbus_experiment["slug"]}
                )

        draft_experiments = []
        if with_draft_experiments:
            # draft experiments are mainly used to compute previews
            draft_experiments_json = retry_get(
                session, cls.EXPERIMENTER_API_URL_V6_DRAFTS, cls.MAX_RETRIES, cls.USER_AGENT
            )

            for draft_experiment in draft_experiments_json:
                try:
                    draft_experiments.append(
                        ExperimentV6.from_dict(draft_experiment).to_experiment()
                    )
                except Exception as e:
                    print(f"Error converting draft experiment {draft_experiment['slug']}")
                    print(str(e))

        return cls(nimbus_experiments + legacy_experiments + draft_experiments)

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
        return cls(
            [
                ex
                for ex in self.experiments
                if ex.experimenter_slug == slug or ex.normandy_slug == slug
            ]
        )

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

    def ended_after_or_live(self, after: dt.datetime) -> "ExperimentCollection":
        """All experiments that ended after a given time or that are still live."""

        cls = type(self)
        return cls(
            [
                ex
                for ex in self.ever_launched().experiments
                if (ex.end_date and ex.end_date >= after)
                or (ex.end_date is None and ex.status == "Live")
            ]
        )
