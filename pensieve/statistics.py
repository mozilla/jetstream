import attr
import cattr
import mozanalysis
from google.cloud import bigquery
from typing import Callable, Any, Dict, List, Tuple, Optional
from pandas import DataFrame
import pandas

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
    ci_width: Optional[float]
    point: Optional[float]
    lower: Optional[float]
    upper: Optional[float]

    def save_to_bigquery(self, client, destination_table, append=True):
        """Stores the data to a BigQuery table with a defined schema."""

        job_config = bigquery.LoadJobConfig()
        job_config.schema = [
            bigquery.SchemaField("metric", "STRING"),
            bigquery.SchemaField("statistic", "STRING"),
            bigquery.SchemaField("parameter", "FLOAT"),
            bigquery.SchemaField("label", "STRING"),
            bigquery.SchemaField("ci_width", "FLOAT"),
            bigquery.SchemaField("point", "FLOAT"),
            bigquery.SchemaField("lower", "FLOAT"),
            bigquery.SchemaField("upper", "FLOAT"),
        ]

        if append:
            job_config.write_disposition = bigquery.job.WriteDisposition.WRITE_APPEND
        else:
            job_config.write_disposition = bigquery.job.WriteDisposition.WRITE_TRUNCATE

        client.load_table_from_dataframe(self.data, destination_table, job_config=job_config)


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

    def apply(self, df: DataFrame) -> "StatisticResult":
        """Run statistic on provided dataframe."""

        data = df
        for pre_treatment in self.pre_treatments:
            data = pre_treatment.apply(data)

        results = [self.transformation(df, metric) for metric in self.metrics if metric in df]
        return StatisticResult(pandas.concat(results))

    def transformation(self, df: DataFrame, metric: str) -> "StatisticResult":
        raise NotImplementedError("Statistic subclasses must override transformation()")

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):  # todo: plug in config file support
        """Create a class instance with the specified config parameters."""
        return cls(**config_dict)


class BootstrapOneBranch(Statistic):
    num_samples: int = 100
    summary_quantiles: Tuple[int] = (0.5)
    pre_treatments = [RemoveNulls()]
    branches = []

    def transformation(self, df: DataFrame, metric: str):
        results_per_branch = df.groupby("branch")

        data_by_branch = [results_per_branch.get_group(branch) for branch in self.branches]

        results = [
            mozanalysis.bayesian_stats.bayesian_bootstrap(
                data[metric], self.num_samples, self.summary_quantiles
            )
            for data in data_by_branch
        ]

        print(results)

        # return StatisticResult(
        #     metric=metric,
        #     statistic=self.name(),
        #     parameter=0.0   # todo
        #     ci
        # )
