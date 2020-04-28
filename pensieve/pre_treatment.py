from abc import ABC, abstractmethod
import attr
from pandas import DataFrame


@attr.s(auto_attribs=True)
class PreTreatment(ABC):
    """
    Represents an abstract pre-treatment step applied to data before
    calculating statistics.
    """

    @classmethod
    def name(cls):
        return __name__  # todo: snake case names?

    @abstractmethod
    def apply(self, df: DataFrame) -> DataFrame:
        """
        Applies the pre-treatment transformation to a DataFrame and returns
        the resulting DataFrame.
        """
        return NotImplemented


class RemoveNulls(PreTreatment):
    """Pre-treatment step to remove null rows and columns."""

    def apply(self, df: DataFrame):
        return df.dropna()
