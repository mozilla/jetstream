import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import attr
import numpy as np
import pandas as pd
from pandas import DataFrame


@attr.s(auto_attribs=True)
class PreTreatment(ABC):
    """
    Represents an abstract pre-treatment step applied to data before
    calculating statistics.
    """

    analysis_period_length: int = attr.ib(kw_only=True, default=1)

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

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create a class instance with the specified config parameters."""
        return cls(**config_dict)  # type: ignore


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
class CensorLowestValues(PreTreatment):
    """Removes rows with the lowest n% of values."""

    fraction: float = 1e-5

    def apply(self, df: DataFrame, col: str) -> DataFrame:
        mask = df[col] > df[col].quantile(self.fraction)
        return df.loc[mask, :]


@attr.s(auto_attribs=True)
class CensorValuesBelowThreshold(PreTreatment):
    """Removes rows with values below the provided threshold."""

    threshold: float

    def apply(self, df: DataFrame, col: str) -> DataFrame:
        mask = df[col] > self.threshold
        return df.loc[mask, :]


@attr.s(auto_attribs=True)
class CensorValuesAboveThreshold(PreTreatment):
    """Removes rows with values above the provided threshold."""

    threshold: float

    def apply(self, df: DataFrame, col: str) -> DataFrame:
        mask = df[col] < self.threshold
        return df.loc[mask, :]


@attr.s(auto_attribs=True)
class NormalizeOverAnalysisPeriod(PreTreatment):
    """Normalizes the row values over a given analysis period (number of days)."""

    analysis_period_length: int = 1

    def apply(self, df: DataFrame, col: str) -> DataFrame:
        df[col] = df[col] / self.analysis_period_length
        return df


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
