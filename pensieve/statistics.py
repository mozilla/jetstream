import attr
import cattr
from decimal import Decimal
import mozanalysis.bayesian_stats.bayesian_bootstrap as mabsbb
from google.cloud import bigquery
from google.api_core.exceptions import NotFound
from typing import Callable, Any, Dict, List, Tuple, Optional
from pandas import DataFrame
import pandas
import numpy as np
import re

from pensieve.pre_treatment import PreTreatment, RemoveNulls


@attr.s(auto_attribs=True)
class StatisticResult:
    """
    Represents the resulting data after applying a statistic transformation
    to metric data.
    """

    metric: str
    statistic: str
    parameter: Decimal
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
        """Return snake-cased name of the statistic."""
        return re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()

    def apply(self, df: DataFrame) -> "StatisticResultCollection":
        """Run statistic on provided dataframe."""

        data = df
        for pre_treatment in self.pre_treatments:
            data = pre_treatment.apply(data)

        col = StatisticResultCollection([])

        for metric in self.metrics:
            if metric in df:
                col.data += self.transformation(df, metric).data

        return col

    def transformation(self, df: DataFrame, metric: str) -> "StatisticResultCollection":
        raise NotImplementedError("Statistic subclasses must override transformation()")

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):  # todo: plug in config file support
        """Create a class instance with the specified config parameters."""
        return cls(**config_dict)


@attr.s(auto_attribs=True)
class BootstrapQuantiles(Statistic):
    num_samples: int = 100
    summary_quantiles: Tuple[int] = (0.5, 0.75, 0.8, 0.9)
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
                # calculate confidence interval
                data = branch_data[metric]
                t = stats_result[str(quantile)]
                confidence_margin = 0.5 * (1.0 - self.confidence_interval)
                confidence_high = (0.0 + confidence_margin) * 100
                confidence_low = (1.0 - confidence_margin) * 100
                lower = t - np.percentile(data - t, confidence_low)
                upper = t - np.percentile(data - t, confidence_high)

                result = StatisticResult(
                    metric=metric,
                    statistic=self.name(),
                    parameter=quantile,
                    label=branch,
                    point=t,
                    ci_width=self.confidence_interval,
                    lower=lower,
                    upper=upper,
                )

                stats_results.data.append(result)

        return stats_results
