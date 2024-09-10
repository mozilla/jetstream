import copy
import json
import logging
import math
import numbers
import re
from abc import ABC, abstractmethod
from decimal import Decimal
from inspect import isabstract
from typing import Any, ClassVar

import attr
import mozanalysis.bayesian_stats.bayesian_bootstrap
import mozanalysis.bayesian_stats.binary
import mozanalysis.frequentist_stats.bootstrap
import mozanalysis.frequentist_stats.linear_models
import mozanalysis.metrics
import numpy as np
from google.cloud import bigquery
from metric_config_parser import metric as parser_metric
from metric_config_parser.experiment import Experiment
from mozilla_nimbus_schemas.jetstream import AnalysisBasis
from mozilla_nimbus_schemas.jetstream import Statistic as StatisticSchema
from mozilla_nimbus_schemas.jetstream import Statistics as StatisticsSchema
from pandas import DataFrame, Series
from pydantic import ConfigDict, Field, field_validator
from sklearn.neighbors import KernelDensity
from statsmodels.distributions.empirical_distribution import ECDF

from .errors import StatisticComputationException
from .metric import Metric
from .pre_treatment import PreTreatment

logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True)
class Summary:
    """Represents a metric with a statistical treatment."""

    metric: Metric
    statistic: "Statistic"
    pre_treatments: list[PreTreatment] = attr.Factory(list)

    @classmethod
    def from_config(
        cls,
        summary_config: parser_metric.Summary,
        analysis_period_length: int | None,
        period: parser_metric.AnalysisPeriod,
    ) -> "Summary":
        """Create a Jetstream-native Summary representation."""
        metric = Metric.from_metric_config(summary_config.metric)

        found = False
        for statistic in set(Statistic.__subclasses__()).union(
            [
                subsubclass
                for subclass in Statistic.__subclasses__()
                for subsubclass in subclass.__subclasses__()
            ]
        ):
            if statistic.name() == summary_config.statistic.name:
                found = True
                break

        if not found:
            raise ValueError(f"Statistic '{summary_config.statistic.name}' does not exist.")

        stats_params = copy.deepcopy(summary_config.statistic.params)
        stats_params["period"] = period

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
        analysis_basis: AnalysisBasis,
        segment: str,
    ) -> "StatisticResultCollection":
        """Apply the statistic transformation for data related to the specified metric."""
        for pre_treatment in self.pre_treatments:
            data = pre_treatment.apply(data, self.metric.name)

            if self.metric.depends_on:
                for upstream_metric in self.metric.depends_on:
                    data = pre_treatment.apply(data, upstream_metric.metric.name)

        return self.statistic.apply(data, self.metric.name, experiment, analysis_basis, segment)


class StatisticResult(StatisticSchema):
    """
    Represents the resulting data after applying a statistic transformation
    to metric data.
    """

    # ClassVars are automatically ignored in instances
    SCHEMA_VERSION: ClassVar[int] = 4
    bq_schema: ClassVar[tuple] = (
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

    # override the behavior of window_index because this is not
    # a field in the bigquery schema, and so we need to exclude
    # it on the Jetstream side
    window_index: str = Field(default=None, exclude=True)

    model_config = ConfigDict(use_enum_values=True, coerce_numbers_to_str=True)

    @field_validator("ci_width", "point", "lower", "upper")
    @classmethod
    def check_number_fields(cls, v, field):
        if v is not None and not isinstance(v, numbers.Number):
            if math.isnan(v):
                return None
            raise ValueError(f"Expected a number for {field.name}; got {v!r}")
        return v

    @field_validator("parameter")
    @classmethod
    def normalize_decimal(cls, v):
        if isinstance(v, Decimal):
            return str(round(v, 6).normalize())
        if isinstance(v, float):
            return str(round(v, 6))
        return v

    @field_validator("*")
    @classmethod
    def suppress_infinites(cls, v):
        if not isinstance(v, float) or math.isfinite(v):
            return v
        return None


class StatisticResultCollection(StatisticsSchema):
    """
    Represents a set of statistics result data.
    """

    root: list[StatisticResult] = []

    def set_segment(self, segment: str) -> "StatisticResultCollection":
        """Sets the `segment` field in-place on all children."""
        for result in self.root:
            result.segment = segment
        return self

    def set_analysis_basis(self, analysis_basis: AnalysisBasis) -> "StatisticResultCollection":
        """Sets the `analysis_basis` field in-place on all children."""
        for result in self.root:
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

    period: parser_metric.AnalysisPeriod | None = attr.field(default=None)

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
        analysis_basis: AnalysisBasis,
        segment: str,
    ) -> "StatisticResultCollection":
        """
        Run statistic on data provided by a DataFrame and return a collection
        of statistic results.
        """

        # add results to a dict to ensure uniqueness
        # keyed by result metadata, so at most one
        # result per unique metadata
        results = {}

        if metric in df:
            branch_list = df.branch.unique()

            for ref_branch in branch_list:
                try:
                    for x in self.transform(
                        df,
                        metric,
                        ref_branch,
                        experiment,
                        analysis_basis,
                        segment,
                    ).root:
                        results[
                            (
                                x.metric,
                                x.statistic,
                                x.branch,
                                x.parameter,
                                x.comparison,
                                x.comparison_to_branch,
                                x.ci_width,
                                x.segment,
                                x.analysis_basis,
                            )
                        ] = json.dumps(x.model_dump_json(warnings=False))
                        # note for above: warnings=False is intended to suppress warnings
                        # of str coerced to numbers,  but may also suppress others

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

        # parse stringified json results as list of StatisticResults to be returned
        statistic_result_collection = StatisticResultCollection.model_validate(
            [StatisticResult.model_validate_json(json.loads(r)) for r in results.values()]
        )

        return statistic_result_collection

    @abstractmethod
    def transform(
        self,
        df: DataFrame,
        metric: str,
        reference_branch: str,
        experiment: Experiment,
        analysis_basis: AnalysisBasis,
        segment: str,
    ) -> "StatisticResultCollection":
        return NotImplemented

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]):
        """Create a class instance with the specified config parameters."""
        return cls(**config_dict)  # type: ignore


def _extract_ci(
    series: Series, quantile: float, threshold: float = 1e-5
) -> tuple[float | None, float | None]:
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

    return StatisticResultCollection.model_validate(statlist)


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
        analysis_basis: AnalysisBasis,
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
class LinearModelMean(Statistic):
    drop_highest: float = attr.field(default=0.005, validator=attr.validators.instance_of(float))
    # currently used keys are "metric" as the name of the metric
    # and "period" as the (preenrollment) period to pull from
    covariate_adjustment: dict[str, str] | None = attr.field(default=None)

    @covariate_adjustment.validator
    def check(self, attribute, value):
        if value is not None:
            covariate_period = parser_metric.AnalysisPeriod(value["period"])
            preenrollment_periods = [
                parser_metric.AnalysisPeriod.PREENROLLMENT_WEEK,
                parser_metric.AnalysisPeriod.PREENROLLMENT_DAYS_28,
            ]
            if covariate_period not in preenrollment_periods:
                raise ValueError(
                    "Covariate adjustment must be done using a pre-treatment analysis "
                    f"period (one of: {[p.value for p in preenrollment_periods]})"
                )

    def transform(
        self,
        df: DataFrame,
        metric: str,
        reference_branch: str,
        experiment: Experiment,
        analysis_basis: AnalysisBasis,
        segment: str,
    ) -> StatisticResultCollection:
        covariate_col_label = None
        if self.covariate_adjustment is not None:
            covariate_period = parser_metric.AnalysisPeriod(self.covariate_adjustment["period"])
            # we cannot apply covariate adjustment if the adjusting period is the current period
            # or if the current period is itself a pre-enrollment period (e.g., cannot adjust
            # preenrollment_week using preenrollment_days28)
            if (covariate_period != self.period) and self.period not in (
                parser_metric.AnalysisPeriod.PREENROLLMENT_WEEK,
                parser_metric.AnalysisPeriod.PREENROLLMENT_DAYS_28,
            ):
                covariate_col_label = f"{self.covariate_adjustment.get('metric', metric)}_pre"

        if covariate_col_label and covariate_col_label not in df.columns:
            logger.warning(
                f"Falling back to unadjusted inferences for {metric}",
                extra={
                    "experiment": experiment.normandy_slug,
                    "metric": metric,
                    "statistic": self.name(),
                    "analysis_basis": analysis_basis.value,
                    "segment": segment,
                },
            )
            covariate_col_label = None

        ma_result = mozanalysis.frequentist_stats.linear_models.compare_branches_lm(
            df,
            col_label=metric,
            ref_branch_label=reference_branch,
            covariate_col_label=covariate_col_label,
            threshold_quantile=1 - self.drop_highest,
            alphas=[0.05],
            interactive=False,
        )

        return flatten_simple_compare_branches_result(
            ma_result=ma_result,
            metric_name=metric,
            statistic_name="mean_lm",
            reference_branch=reference_branch,
            ci_width=0.95,
        )


@attr.s(auto_attribs=True)
class PerClientDAUImpact(BootstrapMean):
    drop_highest: float = 0.0

    def transform(
        self,
        df: DataFrame,
        metric: str,
        reference_branch: str,
        experiment: Experiment,
        analysis_basis: parser_metric.AnalysisBasis,
        segment: str,
    ) -> StatisticResultCollection:
        bootstrap_results = super().transform(
            df,
            metric,
            reference_branch,
            experiment,
            analysis_basis,
            segment,
        )

        # df contains the client metric data filtered to the current
        # segment/analysis basis, so we take its length to get # clients
        num_enrolled_clients = len(df)

        results = []
        for branch in experiment.branches:
            # for absolute differences we report the absolute difference in per-user
            # sum(DAU) scaled by total user enrollment. The interpretation here is the
            # DAU gain achieved if we deploy this branch to the same population as the
            # experiment
            branch_data_abs = [
                x
                for x in bootstrap_results.root
                if x.branch == branch.slug and x.comparison == "difference"
            ]
            for d in branch_data_abs:
                d.point = d.point * num_enrolled_clients
                d.upper = d.upper * num_enrolled_clients
                d.lower = d.lower * num_enrolled_clients
                d.statistic = "per_client_dau_impact"

                results.append(d)

            # for relative differences we simply report the relative difference in
            # per-user sum(DAU)
            branch_data_rel = [
                x
                for x in bootstrap_results.root
                if x.branch == branch.slug and x.comparison == "relative_uplift"
            ]
            for d in branch_data_rel:
                d.statistic = "per_client_dau_impact"

                results.append(d)

        return StatisticResultCollection.model_validate(results)


@attr.s(auto_attribs=True)
class Binomial(Statistic):
    confidence_interval: float = 0.95

    def transform(
        self,
        df: DataFrame,
        metric: str,
        reference_branch: str,
        experiment: Experiment,
        analysis_basis: AnalysisBasis,
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

    def transform(
        self,
        df: DataFrame,
        metric: str,
        reference_branch: str,
        experiment: Experiment,
        analysis_basis: AnalysisBasis,
        segment: str,
    ) -> StatisticResultCollection:
        stats_results = StatisticResultCollection.model_validate([])

        critical_point = (1 - self.confidence_interval) / 2
        summary_quantiles = (critical_point, 1 - critical_point)

        ma_result = mozanalysis.frequentist_stats.bootstrap.compare_branches_quantiles(
            df,
            col_label=metric,
            ref_branch_label=reference_branch,
            num_samples=self.num_samples,
            individual_summary_quantiles=summary_quantiles,
            comparative_summary_quantiles=summary_quantiles,
        )

        for branch, branch_result in ma_result["individual"].items():
            for param, decile_result in branch_result.iterrows():
                lower, upper = _extract_ci(decile_result, critical_point)
                stats_results.root.append(
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
                stats_results.root.append(
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
                stats_results.root.append(
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
        analysis_basis: AnalysisBasis,
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
        analysis_basis: AnalysisBasis,
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
        return StatisticResultCollection.model_validate(results)


class Sum(Statistic):
    def apply(
        self,
        df: DataFrame,
        metric: str,
        experiment: Experiment,
        analysis_basis: AnalysisBasis,
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
        analysis_basis: AnalysisBasis,
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
        return StatisticResultCollection.model_validate(results)


@attr.s(auto_attribs=True)
class MakeGridResult:
    grid: np.ndarray
    geometric: bool
    message: str | None


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
    bandwidth: str = "scott"
    adjust: float = 1.0
    kernel: str = "gaussian"
    grid_size: int = 256
    log_space: bool = False

    def transform(
        self,
        df: DataFrame,
        metric: str,
        reference_branch: str,
        experiment: Experiment,
        analysis_basis: AnalysisBasis,
        segment: str,
    ) -> StatisticResultCollection:
        if not np.isclose(self.adjust, 1.0):
            raise ValueError("KDE Adjust parameter no longer supported")
        results = []
        for branch, group in df.groupby("branch"):
            kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
            kde.fit(group[metric].values[:, np.newaxis])
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
            result = np.exp(kde.score_samples(grid.grid[:, np.newaxis]))
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
                        point=np.exp(kde.score_samples(np.asarray([0]).reshape(-1, 1))[0]),
                        lower=None,
                        upper=None,
                        analysis_basis=analysis_basis,
                        segment=segment,
                    )
                )
            for x, y in zip(grid.grid, result, strict=False):
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
        return StatisticResultCollection.model_validate(results)


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
        analysis_basis: AnalysisBasis,
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
            for x, y in zip(grid.grid, cdf, strict=False):
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
        return StatisticResultCollection.model_validate(results)


@attr.s(auto_attribs=True, kw_only=True)
class PopulationRatio(Statistic):
    numerator: str
    denominator: str
    confidence_interval: float = 0.95
    drop_highest: float = 0.005
    num_samples: int = 10000

    def transform(
        self,
        df: DataFrame,
        metric: str,
        reference_branch: str,
        experiment: Experiment,
        analysis_basis: AnalysisBasis,
        segment: str,
    ) -> StatisticResultCollection:
        critical_point = (1 - self.confidence_interval) / 2
        summary_quantiles = (critical_point, 1 - critical_point)

        ma_result = mozanalysis.frequentist_stats.bootstrap.compare_branches(
            df,
            col_label=[self.numerator, self.denominator],
            ref_branch_label=reference_branch,
            stat_fn=lambda data: {np.mean(data[:, 0]) / np.mean(data[:, 1])},
            num_samples=self.num_samples,
            individual_summary_quantiles=summary_quantiles,
            threshold_quantile=1 - self.drop_highest,
        )

        return flatten_simple_compare_branches_result(
            ma_result=ma_result,
            metric_name=metric,
            statistic_name="population_ratio",
            reference_branch=reference_branch,
            ci_width=self.confidence_interval,
        )
