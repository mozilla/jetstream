from abc import ABC, abstractmethod
from decimal import Decimal
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import attr
import mozanalysis.bayesian_stats.binary
import mozanalysis.bayesian_stats.bayesian_bootstrap
import mozanalysis.bayesian_stats.frequentist_bootstrap
from pandas import DataFrame, Series


def _maybe_decimal(value) -> Optional[Decimal]:
    if value is None:
        return None
    return Decimal(value)


@attr.s(auto_attribs=True)
class StatisticResult:
    """
    Represents the resulting data after applying a statistic transformation
    to metric data.
    """

    metric: str
    statistic: str
    parameter: Optional[Decimal] = attr.ib(converter=_maybe_decimal)
    branch: str
    comparison_to_control: Optional[str] = None
    ci_width: Optional[float] = 0.0
    point: Optional[float] = 0.0
    lower: Optional[float] = 0.0
    upper: Optional[float] = 0.0


@attr.s(auto_attribs=True)
class StatisticResultCollection:
    """
    Represents a set of statistics result data.
    """

    data: List[StatisticResult] = attr.Factory(list)

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
            branch_list = df.branch.to_numpy()
            if self.ref_branch_label not in branch_list:
                logging.warn(
                    f"Branch {self.ref_branch_label} not in {branch_list} for {self.name()}."
                )
            else:
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


def flatten_simple_compare_branches_result(
    ma_result: dict, metric_name: str, statistic_name: str, ci_width: float
) -> StatisticResultCollection:
    critical_point = (1 - ci_width) / 2
    statlist = []
    for branch, branch_result in ma_result["individual"].items():
        lower, upper = _extract_ci(branch_result, critical_point)
        statlist.append(
            StatisticResult(
                metric=metric_name,
                statistic=statistic_name,
                parameter=None,
                branch=branch,
                ci_width=ci_width,
                point=branch_result["mean"],
                lower=lower,
                upper=upper,
            )
        )

    for branch, branch_result in ma_result["comparative"].items():
        lower_abs, upper_abs = _extract_ci(branch_result["abs_uplift"], critical_point)
        statlist.append(
            StatisticResult(
                metric=metric_name,
                statistic=statistic_name,
                parameter=None,
                branch=branch,
                comparison_to_control="difference",
                ci_width=ci_width,
                point=branch_result["abs_uplift"]["exp"],
                lower=lower_abs,
                upper=upper_abs,
            )
        )

        lower_rel, upper_rel = _extract_ci(branch_result["rel_uplift"], critical_point)
        statlist.append(
            StatisticResult(
                metric=metric_name,
                statistic=statistic_name,
                parameter=None,
                branch=branch,
                comparison_to_control="relative_uplift",
                ci_width=ci_width,
                point=branch_result["rel_uplift"]["exp"],
                lower=lower_rel,
                upper=upper_rel,
            )
        )

    return StatisticResultCollection(statlist)


@attr.s(auto_attribs=True)
class BootstrapMean(Statistic):
    num_samples: int = 10000
    confidence_interval: float = 0.95
    ref_branch_label: str = "control"

    def transform(self, df: DataFrame, metric: str) -> StatisticResultCollection:
        critical_point = (1 - self.confidence_interval) / 2
        summary_quantiles = (critical_point, 1 - critical_point)

        ma_result = mozanalysis.bayesian_stats.bayesian_bootstrap.compare_branches(
            df,
            col_label=metric,
            ref_branch_label=self.ref_branch_label,
            num_samples=self.num_samples,
            individual_summary_quantiles=summary_quantiles,
        )

        return flatten_simple_compare_branches_result(
            ma_result, "mean", metric, self.confidence_interval
        )


@attr.s(auto_attribs=True)
class Binomial(Statistic):
    ref_branch_label: str = "control"
    confidence_interval: float = 0.95

    def transform(self, df: DataFrame, metric: str) -> StatisticResultCollection:
        critical_point = (1 - self.confidence_interval) / 2
        summary_quantiles = (critical_point, 1 - critical_point)

        ma_result = mozanalysis.bayesian_stats.binary.compare_branches(
            df,
            col_label=metric,
            ref_branch_label=self.ref_branch_label,
            individual_summary_quantiles=summary_quantiles,
            comparative_summary_quantiles=summary_quantiles,
        )

        return flatten_simple_compare_branches_result(
            ma_result, "binomial", metric, self.confidence_interval
        )
