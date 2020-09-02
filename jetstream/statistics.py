from abc import ABC, abstractmethod
from decimal import Decimal
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import attr
import cattr
from google.cloud import bigquery
import mozanalysis.bayesian_stats.binary
import mozanalysis.bayesian_stats.bayesian_bootstrap
import mozanalysis.frequentist_stats.bootstrap
import numpy as np
from pandas import DataFrame, Series
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF


from jetstream.logging import logger
from .pre_treatment import PreTreatment


_logger = logging.getLogger(__name__)


def _maybe_decimal(value) -> Optional[Decimal]:
    if value is None:
        return None
    return Decimal(value)


@attr.s(auto_attribs=True)
class Summary:
    """Represents a metric with a statistical treatment."""

    metric: mozanalysis.metrics.Metric
    statistic: "Statistic"
    pre_treatments: List[PreTreatment] = attr.Factory(list)

    def run(self, data: DataFrame, reference_branch: Optional[str]) -> "StatisticResultCollection":
        """Apply the statistic transformation for data related to the specified metric."""
        for pre_treatment in self.pre_treatments:
            data = pre_treatment.apply(data, self.metric.name)

        return self.statistic.apply(data, self.metric.name, reference_branch)


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
    comparison: Optional[str] = None
    comparison_to_branch: Optional[str] = None
    ci_width: Optional[float] = 0.0
    point: Optional[float] = 0.0
    lower: Optional[float] = 0.0
    upper: Optional[float] = 0.0
    segment: Optional[str] = None

    bq_schema = (
        bigquery.SchemaField("metric", "STRING"),
        bigquery.SchemaField("statistic", "STRING"),
        bigquery.SchemaField("parameter", "NUMERIC"),
        bigquery.SchemaField("branch", "STRING"),
        bigquery.SchemaField("comparison", "STRING"),
        bigquery.SchemaField("comparison_to_branch", "STRING"),
        bigquery.SchemaField("ci_width", "FLOAT64"),
        bigquery.SchemaField("point", "FLOAT64"),
        bigquery.SchemaField("lower", "FLOAT64"),
        bigquery.SchemaField("upper", "FLOAT64"),
        bigquery.SchemaField("segment", "STRING"),
    )


@attr.s(auto_attribs=True)
class StatisticResultCollection:
    """
    Represents a set of statistics result data.
    """

    data: List[StatisticResult] = attr.Factory(list)

    converter = cattr.Converter()
    converter.register_unstructure_hook(Decimal, lambda x: str(round(x, 6).normalize()))
    converter.register_unstructure_hook(float, lambda x: x if math.isfinite(x) else None)

    def to_dict(self) -> Dict[str, Any]:
        """Return statistic results as dict."""
        return self.converter.unstructure(self)

    def set_segment(self, segment: str) -> "StatisticResultCollection":
        """Sets the `segment` field in-place on all children."""
        for result in self.data:
            result.segment = segment
        return self


@attr.s(auto_attribs=True)
class Statistic(ABC):
    """
    Abstract representation of a statistic.

    A statistic is a transformation that accepts a table of per-client aggregates and
    returns a table representing a summary of the aggregates with respect to the branches
    of the experiment.
    """

    @classmethod
    def name(cls):
        """Return snake-cased name of the statistic."""
        # https://stackoverflow.com/a/1176023
        name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", cls.__name__)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()

    def apply(
        self, df: DataFrame, metric: str, reference_branch: Optional[str]
    ) -> "StatisticResultCollection":
        """
        Run statistic on data provided by a DataFrame and return a collection
        of statistic results.
        """

        statistic_result_collection = StatisticResultCollection([])

        if metric in df:
            branch_list = df.branch.unique()
            if reference_branch and reference_branch not in branch_list:
                logger.warn(f"Branch {reference_branch} not in {branch_list} for {self.name()}.")
            else:
                statistic_result_collection.data += self.transform(
                    df, metric, reference_branch
                ).data

        return statistic_result_collection

    @abstractmethod
    def transform(
        self, df: DataFrame, metric: str, reference_branch: str
    ) -> "StatisticResultCollection":
        return NotImplemented

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create a class instance with the specified config parameters."""
        return cls(**config_dict)  # type: ignore


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
    *,
    ma_result: dict,
    metric_name: str,
    statistic_name: str,
    reference_branch: str,
    ci_width: float,
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
                comparison="difference",
                comparison_to_branch=reference_branch,
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
                comparison="relative_uplift",
                comparison_to_branch=reference_branch,
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

    def transform(
        self, df: DataFrame, metric: str, reference_branch: str
    ) -> StatisticResultCollection:
        critical_point = (1 - self.confidence_interval) / 2
        summary_quantiles = (critical_point, 1 - critical_point)

        ma_result = mozanalysis.bayesian_stats.bayesian_bootstrap.compare_branches(
            df,
            col_label=metric,
            ref_branch_label=reference_branch,
            num_samples=self.num_samples,
            individual_summary_quantiles=summary_quantiles,
        )

        return flatten_simple_compare_branches_result(
            ma_result=ma_result,
            metric_name=metric,
            statistic_name="mean",
            reference_branch=reference_branch,
            ci_width=self.confidence_interval,
        )


@attr.s(auto_attribs=True)
class Binomial(Statistic):
    confidence_interval: float = 0.95

    def transform(
        self, df: DataFrame, metric: str, reference_branch: str
    ) -> StatisticResultCollection:
        critical_point = (1 - self.confidence_interval) / 2
        summary_quantiles = (critical_point, 1 - critical_point)

        ma_result = mozanalysis.bayesian_stats.binary.compare_branches(
            df,
            col_label=metric,
            ref_branch_label=reference_branch,
            individual_summary_quantiles=summary_quantiles,
            comparative_summary_quantiles=summary_quantiles,
        )

        return flatten_simple_compare_branches_result(
            ma_result=ma_result,
            metric_name=metric,
            statistic_name="binomial",
            reference_branch=reference_branch,
            ci_width=self.confidence_interval,
        )


@attr.s(auto_attribs=True)
class Deciles(Statistic):
    confidence_interval: float = 0.95
    num_samples: int = 10000

    @staticmethod
    def _decilize(arr):
        deciles = np.arange(1, 10) * 0.1
        arr_quantiles = np.quantile(arr, deciles)

        arr_dict = {
            f"{label:.1}": arr_quantile for label, arr_quantile in zip(deciles, arr_quantiles)
        }
        return arr_dict

    def transform(
        self, df: DataFrame, metric: str, reference_branch: str
    ) -> StatisticResultCollection:
        stats_results = StatisticResultCollection([])

        critical_point = (1 - self.confidence_interval) / 2
        summary_quantiles = (critical_point, 1 - critical_point)

        ma_result = mozanalysis.frequentist_stats.bootstrap.compare_branches(
            df,
            stat_fn=self._decilize,
            col_label=metric,
            ref_branch_label=reference_branch,
            num_samples=self.num_samples,
            individual_summary_quantiles=summary_quantiles,
            comparative_summary_quantiles=summary_quantiles,
        )

        for branch, branch_result in ma_result["individual"].items():
            for param, decile_result in branch_result.iterrows():
                lower, upper = _extract_ci(decile_result, critical_point)
                stats_results.data.append(
                    StatisticResult(
                        metric=metric,
                        statistic="deciles",
                        parameter=param,
                        branch=branch,
                        ci_width=self.confidence_interval,
                        point=decile_result["mean"],
                        lower=lower,
                        upper=upper,
                    )
                )

        for branch, branch_result in ma_result["comparative"].items():
            abs_uplift = branch_result["abs_uplift"]
            for param, decile_result in abs_uplift.iterrows():
                lower_abs, upper_abs = _extract_ci(decile_result, critical_point)
                stats_results.data.append(
                    StatisticResult(
                        metric=metric,
                        statistic="deciles",
                        parameter=param,
                        branch=branch,
                        comparison="difference",
                        comparison_to_branch=reference_branch,
                        ci_width=self.confidence_interval,
                        point=decile_result["exp"],
                        lower=lower_abs,
                        upper=upper_abs,
                    )
                )

            rel_uplift = branch_result["rel_uplift"]
            for param, decile_result in rel_uplift.iterrows():
                lower_rel, upper_rel = _extract_ci(decile_result, critical_point)
                stats_results.data.append(
                    StatisticResult(
                        metric=metric,
                        statistic="deciles",
                        parameter=param,
                        branch=branch,
                        comparison="relative_uplift",
                        comparison_to_branch=reference_branch,
                        ci_width=self.confidence_interval,
                        point=decile_result["exp"],
                        lower=lower_rel,
                        upper=upper_rel,
                    )
                )

        return stats_results


class Count(Statistic):
    def apply(self, df: DataFrame, metric: str, reference_branch: Optional[str]):
        return self.transform(df, metric, reference_branch or "control")

    def transform(
        self, df: DataFrame, metric: str, reference_branch: str
    ) -> StatisticResultCollection:
        results = []
        counts = df.groupby("branch").size()
        for branch, n in counts.items():
            results.append(
                StatisticResult(
                    metric="identity",
                    statistic="count",
                    parameter=None,
                    branch=branch,
                    comparison=None,
                    comparison_to_branch=None,
                    ci_width=None,
                    point=n,
                    lower=None,
                    upper=None,
                )
            )
        return StatisticResultCollection(results)


@attr.s(auto_attribs=True)
class KernelDensityEstimate(Statistic):
    bandwidth: str = "normal_reference"
    adjust: float = 1.0
    kernel: str = "gau"
    grid_size: int = 256

    def transform(
        self, df: DataFrame, metric: str, reference_branch: str
    ) -> StatisticResultCollection:
        results = []
        for branch, group in df.groupby("branch"):
            kde = sm.nonparametric.KDEUnivariate(group[metric])
            kde.fit(
                bw=self.bandwidth, adjust=self.adjust, kernel=self.kernel, gridsize=self.grid_size
            )
            for x, y in zip(kde.support, kde.density):
                results.append(
                    StatisticResult(
                        metric=metric,
                        statistic="kernel_density_estimate",
                        parameter=x,
                        branch=branch,
                        comparison=None,
                        comparison_to_branch=None,
                        ci_width=None,
                        point=y,
                        lower=None,
                        upper=None,
                    )
                )
        return StatisticResultCollection(results)


@attr.s(auto_attribs=True)
class EmpiricalCDF(Statistic):
    log_space: bool = False
    grid_size: int = 256

    def transform(
        self, df: DataFrame, metric: str, reference_branch: str
    ) -> StatisticResultCollection:
        results = []
        for branch, group in df.groupby("branch"):
            f = ECDF(group[metric])
            start, stop = group[metric].min(), group[metric].max()
            zero = None
            log_space = self.log_space
            if start < 0 and log_space:
                _logger.warning(
                    f"EmpiricalCDF: Refusing to create a geometric grid for metric {metric} "
                    f"in branch {branch}, which has negative values"
                )
                log_space = False
            if log_space:
                if start == 0:
                    start = group[metric].nsmallest(2)[1]
                    zero = f(0)
                grid = np.geomspace(start, stop, self.grid_size)
            else:
                grid = np.linspace(start, stop, self.grid_size)
            if zero is not None:
                results.append(
                    StatisticResult(
                        metric=metric,
                        statistic="empirical_cdf",
                        parameter=0,
                        branch=branch,
                        comparison=None,
                        comparison_to_branch=None,
                        ci_width=None,
                        point=zero,
                        lower=None,
                        upper=None,
                    )
                )
            cdf = f(grid)
            for x, y in zip(grid, cdf):
                results.append(
                    StatisticResult(
                        metric=metric,
                        statistic="empirical_cdf",
                        parameter=x,
                        branch=branch,
                        comparison=None,
                        comparison_to_branch=None,
                        ci_width=None,
                        point=y,
                        lower=None,
                        upper=None,
                    )
                )
        return StatisticResultCollection(results)
