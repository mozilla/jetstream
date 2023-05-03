import copy
import logging
import math
import numbers
import re
from abc import ABC, abstractmethod
from decimal import Decimal
from inspect import isabstract
from typing import Any, Callable, Dict, List, Optional, Tuple

import attr
import cattr
import mozanalysis.bayesian_stats.bayesian_bootstrap
import mozanalysis.bayesian_stats.binary
import mozanalysis.frequentist_stats.bootstrap
import mozanalysis.metrics
import numpy as np
import statsmodels.api as sm
from google.cloud import bigquery
from metric_config_parser import metric as parser_metric
from metric_config_parser.experiment import Experiment
from pandas import DataFrame, Series
from statsmodels.distributions.empirical_distribution import ECDF

from .errors import StatisticComputationException
from .metric import Metric
from .pre_treatment import PreTreatment

logger = logging.getLogger(__name__)


def _maybe_decimal(value) -> Optional[Decimal]:
    if value is None:
        return None
    return Decimal(value)


@attr.s(auto_attribs=True)
class Summary:
    """Represents a metric with a statistical treatment."""

    metric: Metric
    statistic: "Statistic"
    pre_treatments: List[PreTreatment] = attr.Factory(list)

    @classmethod
    def from_config(
        cls, summary_config: parser_metric.Summary, analysis_period_length: Optional[int]
    ) -> "Summary":
        """Create a Jetstream-native Summary representation."""
        metric = Metric.from_metric_config(summary_config.metric)

        found = False
        for statistic in Statistic.__subclasses__():
            if statistic.name() == summary_config.statistic.name:
                found = True
                break

        if not found:
            raise ValueError(f"Statistic '{summary_config.statistic.name}' does not exist.")

        stats_params = copy.deepcopy(summary_config.statistic.params)

        pre_treatments = []
        for pre_treatment_conf in summary_config.pre_treatments:
            found = False
            for pre_treatment in PreTreatment.__subclasses__():
                if isabstract(pre_treatment):
                    continue
                if pre_treatment.name() == pre_treatment_conf.name:
                    found = True
                    # inject analysis_period_length from experiment
                    pre_treatment.analysis_period_length = analysis_period_length or 1

                    pre_treatments.append(pre_treatment.from_dict(pre_treatment_conf.args))

            if not found:
                raise ValueError(f"Could not find pre-treatment {pre_treatment_conf.name}.")

        return cls(
            metric=metric,
            statistic=statistic.from_dict(stats_params),
            pre_treatments=pre_treatments,
        )

    def run(
        self,
        data: DataFrame,
        experiment: Experiment,
        analysis_basis: parser_metric.AnalysisBasis,
        segment: str,
    ) -> "StatisticResultCollection":
        """Apply the statistic transformation for data related to the specified metric."""
        for pre_treatment in self.pre_treatments:
            data = pre_treatment.apply(data, self.metric.name)

        return self.statistic.apply(data, self.metric.name, experiment, analysis_basis, segment)


@attr.s(auto_attribs=True, kw_only=True)
class StatisticResult:
    """
    Represents the resulting data after applying a statistic transformation
    to metric data.
    """

    SCHEMA_VERSION = 4

    metric: str
    statistic: str
    branch: str
    parameter: Optional[Decimal] = attr.ib(converter=_maybe_decimal, default=None)
    comparison: Optional[str] = None
    comparison_to_branch: Optional[str] = None
    ci_width: Optional[float] = None
    point: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    segment: Optional[str] = None
    analysis_basis: Optional[str] = None

    def __attrs_post_init__(self):
        for k in ("ci_width", "point", "lower", "upper"):
            v = getattr(self, k)
            if v is None:
                continue
            if not isinstance(v, numbers.Number):
                raise ValueError(f"Expected a number for {k}; got {repr(v)}")

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
        bigquery.SchemaField("analysis_basis", "STRING"),
    )


@attr.s(auto_attribs=True)
class StatisticResultCollection:
    """
    Represents a set of statistics result data.
    """

    data: List[StatisticResult] = attr.Factory(list)

    converter = cattr.BaseConverter()
    _normalize_decimal: Callable[[Decimal], str] = lambda x: str(round(x, 6).normalize())
    converter.register_unstructure_hook(Decimal, _normalize_decimal)
    _suppress_infinites: Callable[[float], Optional[float]] = (
        lambda x: x if math.isfinite(x) else None
    )
    converter.register_unstructure_hook(float, _suppress_infinites)

    def to_dict(self) -> Dict[str, Any]:
        """Return statistic results as dict."""
        return self.converter.unstructure(self)

    def set_segment(self, segment: str) -> "StatisticResultCollection":
        """Sets the `segment` field in-place on all children."""
        for result in self.data:
            result.segment = segment
        return self

    def set_analysis_basis(
        self, analysis_basis: parser_metric.AnalysisBasis
    ) -> "StatisticResultCollection":
        """Sets the `analysis_basis` field in-place on all children."""
        for result in self.data:
            result.analysis_basis = analysis_basis.value
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
        self,
        df: DataFrame,
        metric: str,
        experiment: Experiment,
        analysis_basis: parser_metric.AnalysisBasis,
        segment: str,
    ) -> "StatisticResultCollection":
        """
        Run statistic on data provided by a DataFrame and return a collection
        of statistic results.
        """

        statistic_result_collection = StatisticResultCollection([])

        if metric in df:
            branch_list = df.branch.unique()
            reference_branch = experiment.reference_branch
            if reference_branch and reference_branch not in branch_list:
                logger.warning(
                    f"Branch {reference_branch} not in {branch_list} for {self.name()}.",
                    extra={
                        "experiment": experiment.normandy_slug,
                        "metric": metric,
                        "statistic": self.name(),
                        "analysis_basis": analysis_basis.value,
                        "segment": segment,
                    },
                )
            else:
                if reference_branch is None:
                    ref_branch_list = branch_list
                else:
                    ref_branch_list = [reference_branch]

                for ref_branch in ref_branch_list:
                    try:
                        statistic_result_collection.data += self.transform(
                            df, metric, ref_branch, experiment, analysis_basis, segment
                        ).data
                    except Exception as e:
                        logger.exception(
                            f"Error while computing statistic {self.name()} "
                            + f"for metric {metric}: {e}",
                            exc_info=StatisticComputationException(
                                f"Error while computing statistic {self.name()} "
                                + f"for metric {metric}: {e}"
                            ),
                            extra={
                                "experiment": experiment.normandy_slug,
                                "metric": metric,
                                "statistic": self.name(),
                                "analysis_basis": analysis_basis.value,
                                "segment": segment,
                            },
                        )

                    df = df[df.branch != ref_branch]

        return statistic_result_collection

    @abstractmethod
    def transform(
        self,
        df: DataFrame,
        metric: str,
        reference_branch: str,
        experiment: Experiment,
        analysis_basis: parser_metric.AnalysisBasis,
        segment: str,
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
    drop_highest: float = 0.005
    confidence_interval: float = 0.95

    def transform(
        self,
        df: DataFrame,
        metric: str,
        reference_branch: str,
        experiment: Experiment,
        analysis_basis: parser_metric.AnalysisBasis,
        segment: str,
    ) -> StatisticResultCollection:
        critical_point = (1 - self.confidence_interval) / 2
        summary_quantiles = (critical_point, 1 - critical_point)

        ma_result = mozanalysis.bayesian_stats.bayesian_bootstrap.compare_branches(
            df,
            col_label=metric,
            ref_branch_label=reference_branch,
            num_samples=self.num_samples,
            individual_summary_quantiles=summary_quantiles,
            threshold_quantile=1 - self.drop_highest,
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
        self,
        df: DataFrame,
        metric: str,
        reference_branch: str,
        experiment: Experiment,
        analysis_basis: parser_metric.AnalysisBasis,
        segment: str,
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
        self,
        df: DataFrame,
        metric: str,
        reference_branch: str,
        experiment: Experiment,
        analysis_basis: parser_metric.AnalysisBasis,
        segment: str,
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
                        analysis_basis=analysis_basis,
                        segment=segment,
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
                        analysis_basis=analysis_basis,
                        segment=segment,
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
                        analysis_basis=analysis_basis,
                        segment=segment,
                    )
                )

        return stats_results


class Count(Statistic):
    def apply(
        self,
        df: DataFrame,
        metric: str,
        experiment: Experiment,
        analysis_basis: parser_metric.AnalysisBasis,
        segment: str,
    ):
        return self.transform(
            df,
            metric,
            experiment.reference_branch or "control",
            experiment.normandy_slug,
            analysis_basis,
            segment,
        )

    def transform(
        self,
        df: DataFrame,
        metric: str,
        reference_branch: str,
        experiment: Experiment,
        analysis_basis: parser_metric.AnalysisBasis,
        segment: str,
    ) -> StatisticResultCollection:
        results = []
        counts = df.groupby("branch").size()
        for branch, n in counts.items():
            results.append(
                StatisticResult(
                    metric=metric,
                    statistic="count",
                    parameter=None,
                    branch=branch,
                    comparison=None,
                    comparison_to_branch=None,
                    ci_width=None,
                    point=n,
                    lower=None,
                    upper=None,
                    analysis_basis=analysis_basis,
                    segment=segment,
                )
            )
        return StatisticResultCollection(results)


class Sum(Statistic):
    def apply(
        self,
        df: DataFrame,
        metric: str,
        experiment: Experiment,
        analysis_basis: parser_metric.AnalysisBasis,
        segment: str,
    ):
        return self.transform(
            df,
            metric,
            experiment.reference_branch or "control",
            experiment.normandy_slug,
            analysis_basis,
            segment,
        )

    def transform(
        self,
        df: DataFrame,
        metric: str,
        reference_branch: str,
        experiment: Experiment,
        analysis_basis: parser_metric.AnalysisBasis,
        segment: str,
    ) -> StatisticResultCollection:
        results = []
        sums = df.groupby("branch")[metric].sum()
        for branch, x in sums.items():
            results.append(
                StatisticResult(
                    metric=metric,
                    statistic="sum",
                    parameter=None,
                    branch=branch,
                    comparison=None,
                    comparison_to_branch=None,
                    ci_width=None,
                    point=float(x),  # Potential loss of precision here.
                    lower=None,
                    upper=None,
                    analysis_basis=analysis_basis,
                    segment=segment,
                )
            )
        return StatisticResultCollection(results)


@attr.s(auto_attribs=True)
class MakeGridResult:
    grid: np.ndarray
    geometric: bool
    message: Optional[str]


def _make_grid(values: Series, size: int, attempt_geometric: bool) -> MakeGridResult:
    start, stop = values.min(), values.max()
    message = None
    geometric = attempt_geometric
    if geometric and (start < 0 or stop <= 0):
        message = (
            "Refusing to create a geometric grid for a series with negative or all-zero values"
        )
        geometric = False
    if geometric and start == 0:
        start = values.drop_duplicates().nsmallest(2).iloc[1]
        assert start != 0
    f: Any = np.geomspace if geometric else np.linspace
    return MakeGridResult(
        grid=f(start, stop, size),
        geometric=geometric,
        message=message,
    )


@attr.s(auto_attribs=True)
class KernelDensityEstimate(Statistic):
    bandwidth: str = "normal_reference"
    adjust: float = 1.0
    kernel: str = "gau"
    grid_size: int = 256
    log_space: bool = False

    def transform(
        self,
        df: DataFrame,
        metric: str,
        reference_branch: str,
        experiment: Experiment,
        analysis_basis: parser_metric.AnalysisBasis,
        segment: str,
    ) -> StatisticResultCollection:
        results = []
        for branch, group in df.groupby("branch"):
            kde = sm.nonparametric.KDEUnivariate(group[metric])
            kde.fit(bw=self.bandwidth, adjust=self.adjust, kernel=self.kernel)
            grid = _make_grid(group[metric], self.grid_size, self.log_space)
            if grid.message:
                logger.warning(
                    f"KernelDensityEstimate for metric {metric}, branch {branch}: {grid.message}",
                    extra={
                        "experiment": experiment.normandy_slug,
                        "metric": metric,
                        "statistic": self.name(),
                        "analysis_basis": analysis_basis.value,
                        "segment": segment,
                    },
                )
            result = kde.evaluate(grid.grid)
            if group[metric].min() == 0 and grid.geometric:
                results.append(
                    StatisticResult(
                        metric=metric,
                        statistic="kernel_density_estimate",
                        parameter=0,
                        branch=branch,
                        comparison=None,
                        comparison_to_branch=None,
                        ci_width=None,
                        point=kde.evaluate(0)[0],
                        lower=None,
                        upper=None,
                        analysis_basis=analysis_basis,
                        segment=segment,
                    )
                )
            for x, y in zip(grid.grid, result):
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
                        analysis_basis=analysis_basis,
                        segment=segment,
                    )
                )
        return StatisticResultCollection(results)


@attr.s(auto_attribs=True)
class EmpiricalCDF(Statistic):
    log_space: bool = False
    grid_size: int = 256

    def transform(
        self,
        df: DataFrame,
        metric: str,
        reference_branch: str,
        experiment: Experiment,
        analysis_basis: parser_metric.AnalysisBasis,
        segment: str,
    ) -> StatisticResultCollection:
        results = []
        for branch, group in df.groupby("branch"):
            f = ECDF(group[metric])
            grid = _make_grid(group[metric], self.grid_size, self.log_space)
            if grid.message:
                logger.warning(
                    f"EmpiricalCDF for metric {metric}, branch {branch}: {grid.message}",
                    extra={
                        "experiment": experiment.normandy_slug,
                        "metric": metric,
                        "statistic": self.name(),
                        "analysis_basis": analysis_basis.value,
                        "segment": segment,
                    },
                )
            if group[metric].min() == 0 and grid.geometric:
                results.append(
                    StatisticResult(
                        metric=metric,
                        statistic="empirical_cdf",
                        parameter=0,
                        branch=branch,
                        comparison=None,
                        comparison_to_branch=None,
                        ci_width=None,
                        point=f(0),
                        lower=None,
                        upper=None,
                        analysis_basis=analysis_basis,
                        segment=segment,
                    )
                )
            cdf = f(grid.grid)
            for x, y in zip(grid.grid, cdf):
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
                        analysis_basis=analysis_basis,
                        segment=segment,
                    )
                )
        return StatisticResultCollection(results)
