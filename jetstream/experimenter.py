import datetime as dt
import logging
from collections.abc import Iterable

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


def _coerce_none_to_zero(x: int | None) -> int:
    return 0 if x is None else x


@attr.s(auto_attribs=True, kw_only=True, slots=True, frozen=True)
class Outcome:
    slug: str


@attr.s(auto_attribs=True, kw_only=True, slots=True, frozen=True)
class Segment:
    slug: str


@attr.s(auto_attribs=True, kw_only=True, slots=True, frozen=True)
class NimbusExperiment:
    """Represents a Nimbus experiment from Experimenter (v8 API)."""

    slug: str  # Normandy slug
    branches: list[experiment.Branch]
    startDate: dt.datetime | None
    endDate: dt.datetime | None
    proposedEnrollment: int
    bucketConfig: experiment.BucketConfig
    referenceBranch: str | None
    _appName: str | None = None
    _appId: str | None = None
    outcomes: list[Outcome] | None = None
    segments: list[Segment] | None = None
    enrollmentEndDate: dt.datetime | None = None
    isEnrollmentPaused: bool | None = None
    isRollout: bool = False

    @property
    def appName(self) -> str:
        return self._appName or "firefox_desktop"

    @property
    def appId(self) -> str:
        return self._appId or "firefox-desktop"

    @classmethod
    def from_dict(cls, d) -> "NimbusExperiment":
        converter = cattr.Converter()
        converter.register_structure_hook(
            dt.datetime,
            lambda num, _: dt.datetime.strptime(num, "%Y-%m-%d"),
        )
        converter.register_structure_hook(
            experiment.BucketConfig,
            cattr.gen.make_dict_structure_fn(
                experiment.BucketConfig,
                converter,
                randomization_unit=cattr.override(rename="randomizationUnit"),
            ),
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
            type="v6",  # currently using v8 API, but attribute remains from v6 API
            status=(
                "Live"
                if (
                    self.endDate
                    and pytz.utc.localize(self.endDate) >= pytz.utc.localize(dt.datetime.now())
                )
                or self.endDate is None
                else "Complete"
            ),
            start_date=pytz.utc.localize(self.startDate) if self.startDate else None,
            end_date=pytz.utc.localize(self.endDate) if self.endDate else None,
            proposed_enrollment=self.proposedEnrollment,
            branches=self.branches,
            reference_branch=self.referenceBranch,
            is_high_population=False,
            app_name=self.appName,
            app_id=self.appId,
            outcomes=[o.slug for o in self.outcomes] if self.outcomes else [],
            segments=[s.slug for s in self.segments] if self.segments else [],
            enrollment_end_date=(
                pytz.utc.localize(self.enrollmentEndDate) if self.enrollmentEndDate else None
            ),
            is_enrollment_paused=bool(self.isEnrollmentPaused),
            is_rollout=self.isRollout,
            bucket_config=self.bucketConfig,
        )


@attr.s(auto_attribs=True)
class ExperimentCollection:
    experiments: list[experiment.Experiment] = attr.ib(default=attr.Factory(list))

    MAX_RETRIES = 3
    # for nimbus experiments
    EXPERIMENTER_API_URL_V8 = "https://experimenter.services.mozilla.com/api/v8/experiments/"

    # experiments that are in draft state
    EXPERIMENTER_API_URL_V8_DRAFTS = (
        "https://experimenter.services.mozilla.com/api/v8/draft-experiments/"
    )

    # user agent sent to the Experimenter API
    USER_AGENT = "jetstream"

    @classmethod
    def from_experimenter(
        cls, session: requests.Session = None, with_draft_experiments=False
    ) -> "ExperimentCollection":
        session = session or requests.Session()

        nimbus_experiments_json = retry_get(
            session, cls.EXPERIMENTER_API_URL_V8, cls.MAX_RETRIES, cls.USER_AGENT
        )
        nimbus_experiments = []

        for nimbus_experiment in nimbus_experiments_json:
            try:
                nimbus_experiments.append(
                    NimbusExperiment.from_dict(nimbus_experiment).to_experiment()
                )
            except Exception as e:
                logger.exception(
                    str(e), exc_info=e, extra={"experiment": nimbus_experiment["slug"]}
                )

        draft_experiments = []
        if with_draft_experiments:
            # draft experiments are mainly used to compute previews
            draft_experiments_json = retry_get(
                session, cls.EXPERIMENTER_API_URL_V8_DRAFTS, cls.MAX_RETRIES, cls.USER_AGENT
            )

            for draft_experiment in draft_experiments_json:
                try:
                    draft_experiments.append(
                        NimbusExperiment.from_dict(draft_experiment).to_experiment()
                    )
                except Exception as e:
                    print(f"Error converting draft experiment {draft_experiment['slug']}")
                    print(str(e))

        return cls(nimbus_experiments + draft_experiments)

    def of_type(self, type_or_types: str | Iterable[str]) -> "ExperimentCollection":
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
