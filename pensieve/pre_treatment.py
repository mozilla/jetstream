from abc import ABC, abstractmethod
import attr
from pandas import DataFrame
import re


@attr.s(auto_attribs=True)
class PreTreatment(ABC):
    """
    Represents an abstract pre-treatment step applied to data before
    calculating statistics.
    """

    @classmethod
    def name(cls):
        """Return snake-cased name of the statistic."""
        return re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()

    @abstractmethod
    def apply(self, df: DataFrame, col: str) -> DataFrame:
        """
        Applies the pre-treatment transformation to a DataFrame and returns
        the resulting DataFrame.
        """
        raise NotImplementedError


class RemoveNulls(PreTreatment):
    """Pre-treatment step to remove null rows and columns."""

    def apply(self, df: DataFrame, col: str):
        return df[col].dropna()
