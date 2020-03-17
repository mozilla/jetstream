import datetime as dt
from typing import List, Iterable, Optional, Union

import attr
import cattr
import requests
import pytz


@attr.s(auto_attribs=True)
class Variant:
    is_control: bool
    slug: bool
    ratio: int


@attr.s(auto_attribs=True)
class Experiment:
    slug: str  # experimenter slug
    type: str
    start_date: Optional[dt.datetime]
    end_date: Optional[dt.datetime]
    proposed_enrollment: Optional[int] = attr.ib(converter=lambda x: 0 if x is None else x)
    variants: List[Variant]
    normandy_slug: Optional[str] = None


@attr.s(auto_attribs=True)
class ExperimentCollection:
    experiments: List[Experiment] = attr.Factory(list)

    EXPERIMENTER_API_URL = "https://experimenter.services.mozilla.com/api/v1/experiments/"

    @staticmethod
    def _unix_millis_to_datetime(num: Optional[float]) -> dt.datetime:
        if num is None:
            return None
        return dt.datetime.fromtimestamp(num / 1e3, pytz.utc)

    @classmethod
    def from_experimenter(cls, session: requests.Session = None) -> "ExperimentCollection":
        session = session or requests.Session()
        experiments = session.get(cls.EXPERIMENTER_API_URL).json()

        converter = cattr.Converter()
        converter.register_structure_hook(
            dt.datetime, lambda num, _: cls._unix_millis_to_datetime(num),
        )
        return cls([converter.structure(experiment, Experiment) for experiment in experiments])

    def of_type(self, type_or_types: Union[str, Iterable[str]]) -> "ExperimentCollection":
        if isinstance(type_or_types, str):
            type_or_types = (type_or_types,)
        cls = type(self)
        return cls([ex for ex in self.experiments if ex.type in type_or_types])

    def started_since(self, since: dt.datetime) -> "ExperimentCollection":
        """since should be a tz-aware datetime in UTC."""
        cls = type(self)
        return cls([ex for ex in self.experiments if ex.start_date and ex.start_date >= since])

    def end_on_or_after(self, after: dt.datetime) -> "ExperimentCollection":
        """All experiments that end on or after the specified date."""
        cls = type(self)
        return cls([ex for ex in self.experiments if ex.end_date and ex.end_date >= after])
