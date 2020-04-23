import attr
import cattr
import mozanalysis.bayesian_stats.bayesian_bootstrap as mabsbb
from google.cloud import bigquery
from google.api_core.exceptions import NotFound
from typing import Callable, Any, Dict, List, Tuple, Optional
from pandas import DataFrame
import pandas
import numpy as np

from pensieve.pre_treatment import PreTreatment, RemoveNulls


@attr.s(auto_attribs=True)
class StatisticResult:
    """
    Represents the resulting data after applying a statistic transformation
    to metric data.
    """

    metric: str
    statistic: str
    parameter: float
    label: str
    ci_width: Optional[float] = 0.0
    point: Optional[float] = 0.0
    lower: Optional[float] = 0.0
    upper: Optional[float] = 0.0

    def with_ci(self, data: DataFrame, t: float, confidence_level: float) -> "StatisticResult":
        """Calculate the confidence interval and update result."""
        confidence_margin = 0.5 * (1.0 - confidence_level)
        confidence_high = (0.0 + confidence_margin) * 100
        confidence_low = (1.0 - confidence_margin) * 100
        self.lower = t - np.percentile(data - t, confidence_low)
        self.upper = t - np.percentile(data - t, confidence_high)
        self.ci_width = confidence_level
        self.point = t
        return self

    def with_point(self, point_value: float) -> "StatisticResult":
        """Set provided value as point value result for statistic."""
        self.point = point_value
        return self


@attr.s(auto_attribs=True)
class StatisticResultCollection:
    """
    Represents a set of statistics result data.
    """

    data: List[StatisticResult] = []

    def append(self, result: StatisticResult):
        self.data.append(result)

    def merge(self, result_collection: "StatisticResultCollection"):
        self.data = self.data + result_collection.data

    def save_to_bigquery(self, client, destination_table, append=True):
        """Stores the data to a BigQuery table with a defined schema."""

        job_config = bigquery.LoadJobConfig()
        job_config.schema = [
            bigquery.SchemaField("metric", "STRING"),
            bigquery.SchemaField("statistic", "STRING"),
            bigquery.SchemaField("parameter", "FLOAT64"),
            bigquery.SchemaField("label", "STRING"),
            bigquery.SchemaField("ci_width", "FLOAT64"),
            bigquery.SchemaField("point", "FLOAT64"),
            bigquery.SchemaField("lower", "FLOAT64"),
            bigquery.SchemaField("upper", "FLOAT64"),
        ]

        if append:
            job_config.write_disposition = bigquery.job.WriteDisposition.WRITE_APPEND
        else:
            job_config.write_disposition = bigquery.job.WriteDisposition.WRITE_TRUNCATE

        client.load_table_from_json(self.to_dict()["data"], destination_table)

    def to_dict(self):
        """Return statistic results as dict."""

        return attr.asdict(self)


@attr.s(auto_attribs=True)
class Statistic:
    """
    Abstract representation of a statistic.

    A statistic is a transformation that accepts a table of per-client aggregates and
    returns a table representing a summary of the aggregates with respect to the branches
    of the experiment.
    """

    metrics: List[str]
    pre_treatments: List[PreTreatment]

    @classmethod
    def name(cls):
        return __name__  # todo: snake case names?

    def apply(self, df: DataFrame) -> "StatisticResultCollection":
        """Run statistic on provided dataframe."""

        data = df
        for pre_treatment in self.pre_treatments:
            data = pre_treatment.apply(data)

        col = StatisticResultCollection([])

        for metric in self.metrics:
            if metric in df:
                col.merge(self.transformation(df, metric))

        return col

    def transformation(self, df: DataFrame, metric: str) -> "StatisticResultCollection":
        raise NotImplementedError("Statistic subclasses must override transformation()")

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):  # todo: plug in config file support
        """Create a class instance with the specified config parameters."""
        return cls(**config_dict)


@attr.s(auto_attribs=True)
class BootstrapOneBranch(Statistic):
    num_samples: int = 100
    summary_quantiles: Tuple[int] = (0.5)
    confidence_interval: float = 0.95
    pre_treatments: List[PreTreatment] = [RemoveNulls()]
    branches: List[str] = []

    def transformation(self, df: DataFrame, metric: str) -> "StatisticResultCollection":
        stats_results = StatisticResultCollection([])

        results_per_branch = df.groupby("branch")

        for branch in self.branches:
            branch_data = results_per_branch.get_group(branch)
            stats_result = mabsbb.bootstrap_one_branch(
                branch_data[metric],
                num_samples=self.num_samples,
                summary_quantiles=self.summary_quantiles,
            ).to_dict()

            for quantile in self.summary_quantiles:
                result = StatisticResult(
                    metric=metric, statistic="quantiles", parameter=quantile, label=branch
                ).with_ci(
                    branch_data[metric], stats_result[str(quantile)], self.confidence_interval
                )

                stats_results.append(result)

        return stats_results
