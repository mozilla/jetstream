from abc import ABC, abstractmethod
import attr
import re
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame


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
    """Removes rows with null values."""

    def apply(self, df: DataFrame, col: str) -> DataFrame:
        return df.dropna(subset=[col])


class RemoveIndefinites(PreTreatment):
    """Removes null and infinite values."""

    def apply(self, df: DataFrame, col: str) -> DataFrame:
        with pd.option_context("mode.use_inf_as_na", True):
            return df.dropna(subset=[col])


@attr.s(auto_attribs=True)
class CensorHighestValues(PreTreatment):
    """Removes rows with the highest n% of values."""

    fraction: float = 1 - 1e-5

    def apply(self, df: DataFrame, col: str) -> DataFrame:
        mask = df[col] < df[col].quantile(self.fraction)
        return df.loc[mask, :]


@attr.s(auto_attribs=True)
class Log(PreTreatment):
    base: Optional[float] = 10.0

    def apply(self, df: DataFrame, col: str) -> DataFrame:
        # Silence divide-by-zero and domain warnings
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.log(df[col])
            if self.base:
                result /= np.log(self.base)
        return df.assign(**{col: result})


class ZeroFill(PreTreatment):
    def apply(self, df: DataFrame, col: str) -> DataFrame:
        return df.fillna(value={col: 0})
