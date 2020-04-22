import attr
import cattr
import mozanalysis
from typing import Callable, Any, Dict, List, Tuple
from pandas import DataFrame
import pandas

from pensieve.pre_treatment import PreTreatment, RemoveNulls


@attr.s(auto_attribs=True)
class Statistic:
    """
    Abstract representation of a statistic.

    A statistic is a transformation that accepts a table of per-client aggregates and
    returns a table representing a summary of the aggregates with respect to the branches
    of the experiment.
    """

    name: str
    metrics: List[str]
    pre_treatments: List[PreTreatment]

    def apply(self, df: DataFrame):
        """Run statistic on provided dataframe."""

        data = df
        for pre_treatment in self.pre_treatments:
            data = pre_treatment.apply(data)

        results = [self.transformation(df, metric) for metric in self.metrics if metric in df]
        return pandas.concat(results)

    def transformation(self, df: DataFrame, metric: str):
        raise NotImplementedError("Statistic subclasses must override transformation()")

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):  # todo: plug in config file support
        """Create a class instance with the specified config parameters."""
        return cls(**config_dict)


class BootstrapOneBranch(Statistic):
    name = "bootstrap_one_branch"
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

        return pandas.concat(results)
