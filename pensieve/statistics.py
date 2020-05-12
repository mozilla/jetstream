from abc import ABC, abstractmethod
from decimal import Decimal
import re
from typing import Any, Dict, List, Optional, Tuple

import attr
import mozanalysis.bayesian_stats.bayesian_bootstrap as mabsbb
from pandas import DataFrame, Series


@attr.s(auto_attribs=True)
class StatisticResult:
    """
    Represents the resulting data after applying a statistic transformation
    to metric data.
    """

    metric: str
    statistic: str
    parameter: Optional[Decimal]
    label: str
    ci_width: Optional[float] = 0.0
    point: Optional[float] = 0.0
    lower: Optional[float] = 0.0
    upper: Optional[float] = 0.0


@attr.s(auto_attribs=True)
class StatisticResultCollection:
    """
    Represents a set of statistics result data.
    """

    data: List[StatisticResult] = []

    def to_dict(self):
        """Return statistic results as dict."""

        return attr.asdict(self)


@attr.s(auto_attribs=True)
class Statistic(ABC):
    """
    Abstract representation of a statistic.

    A statistic is a transformation that accepts a table of per-client aggregates and
    returns a table representing a summary of the aggregates with respect to the branches
    of the experiment.
    """

    ref_branch_label: str = "control"

    @classmethod
    def name(cls):
        """Return snake-cased name of the statistic."""
        return re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()

    def apply(self, df: DataFrame, metric: str) -> "StatisticResultCollection":
        """
        Run statistic on data provided by a DataFrame and return a collection
        of statistic results.
        """

        statistic_result_collection = StatisticResultCollection([])

        if metric in df:
            statistic_result_collection.data += self.transform(df, metric).data

        return statistic_result_collection

    @abstractmethod
    def transform(self, df: DataFrame, metric: str) -> "StatisticResultCollection":
        return NotImplemented

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create a class instance with the specified config parameters."""
        return cls(**config_dict)


def _extract_ci(
    series: Series, quantile: float, threshold: float = 1e-5
) -> Tuple[Optional[float], Optional[float]]:
    # floating point arithmetic was a mistake
    lower_index, upper_index = None, None
    low_quantile, high_quantile = quantile, 1 - quantile
    for i in series.index:
        try:
            f = float(i)
        except ValueError:
            continue
        if abs(f - low_quantile) < threshold:
            lower_index = i
        if abs(f - high_quantile) < threshold:
            upper_index = i
    return (
        series[lower_index] if lower_index else None,
        series[upper_index] if upper_index else None,
    )


@attr.s(auto_attribs=True)
class BootstrapMean(Statistic):
    num_samples: int = 1000
    confidence_interval: float = 0.95
    ref_branch_label: str = "control"

    def transform(self, df: DataFrame, metric: str) -> "StatisticResultCollection":
        stats_results = StatisticResultCollection([])

        critical_point = (1 - self.confidence_interval) / 2
        summary_quantiles = (critical_point, 1 - critical_point)

        ma_result = mabsbb.compare_branches(
            df,
            col_label=metric,
            ref_branch_label=self.ref_branch_label,
            num_samples=self.num_samples,
            individual_summary_quantiles=summary_quantiles,
        )

        for branch, branch_result in ma_result["individual"].items():
            lower, upper = _extract_ci(branch_result, critical_point)
            result = StatisticResult(
                metric=metric,
                statistic="mean",
                parameter=None,
                label=branch,
                ci_width=self.confidence_interval,
                point=branch_result["mean"],
                lower=lower,
                upper=upper,
            )
            stats_results.data.append(result)

        for branch, branch_result in ma_result["comparative"].items():
            lower_abs, upper_abs = _extract_ci(branch_result["abs_uplift"], critical_point)
            stats_results.data.append(
                StatisticResult(
                    metric=metric,
                    statistic="mean",
                    parameter=None,
                    label=f"{branch} - control",
                    ci_width=self.confidence_interval,
                    point=branch_result["abs_uplift"]["exp"],
                    lower=lower_abs,
                    upper=upper_abs,
                )
            )

            lower_rel, upper_rel = _extract_ci(branch_result["rel_uplift"], critical_point)
            stats_results.data.append(
                StatisticResult(
                    metric=metric,
                    statistic="mean",
                    parameter=None,
                    label=f"{branch}/control - 1",
                    ci_width=self.confidence_interval,
                    point=branch_result["rel_uplift"]["exp"],
                    lower=lower_rel,
                    upper=upper_rel,
                )
            )

        return stats_results
