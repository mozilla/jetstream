from abc import ABC, abstractmethod
from decimal import Decimal
import re
from typing import Any, Dict, List, Optional

import attr
from google.cloud import bigquery
import mozanalysis.bayesian_stats.bayesian_bootstrap as mabsbb
from pandas import DataFrame


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

    def save_to_bigquery(self, client, destination_table, append=True):
        """Stores the data to a BigQuery table with a defined schema."""

        job_config = bigquery.LoadJobConfig()
        job_config.schema = [
            bigquery.SchemaField("metric", "STRING"),
            bigquery.SchemaField("statistic", "STRING"),
            bigquery.SchemaField("parameter", "NUMERIC"),
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

        client.load_table_from_json(
            self.to_dict()["data"], destination_table, job_config=job_config
        )

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


@attr.s(auto_attribs=True)
class BootstrapMean(Statistic):
    num_samples: int = 1000
    confidence_interval: float = 0.95
    ref_branch_label: str = "control"
    threshold_quantile = None

    def transform(self, df: DataFrame, metric: str) -> "StatisticResultCollection":
        stats_results = StatisticResultCollection([])

        critical_point = (1 - self.confidence_interval) / 2
        summary_quantiles = (critical_point, 1 - critical_point)

        ma_result = mabsbb.compare_branches(
            df,
            col_label=metric,
            ref_branch_label=self.ref_branch_label,
            num_samples=self.num_samples,
            threshold_quantile=self.threshold_quantile,
            individual_summary_quantiles=summary_quantiles,
        )

        ma_individual_result = ma_result["individual"]

        # floating point arithmetic was a mistake
        lower_index, upper_index = None, None
        for branch, branch_result in ma_individual_result.items():
            for i in branch_result.index:
                try:
                    f = float(i)
                except ValueError:
                    continue
                if abs(f - summary_quantiles[0]) < 0.0001:
                    lower_index = i
                if abs(f - summary_quantiles[1]) < 0.0001:
                    upper_index = i
            result = StatisticResult(
                metric=metric,
                statistic=self.name(),
                parameter=None,
                label=branch,
                ci_width=self.confidence_interval,
                point=branch_result["mean"],
                lower=branch_result[lower_index] if lower_index else None,
                upper=branch_result[upper_index] if upper_index else None,
            )
            stats_results.data.append(result)

        return stats_results
