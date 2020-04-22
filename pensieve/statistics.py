import attr
import cattr
import mozanalysis
from typing import Callable, Any, Dict, List, Tuple
from pandas import DataFrame

from pensieve.pre_treatment import PreTreatment


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

        return self.transformation(df)

    def transformation(self, df: DataFrame):
        raise NotImplementedError("Statistic subclasses must override transformation()")

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        """Create a class instance with the specified config parameters."""
        return cls(**config_dict)


class BootstrapOneBranch(Statistic):
    name = "bootstrap_one_branch"
    num_samples: int = 100
    summary_quantiles: Tuple[int] = (0.5)

    def transformation(self, df: DataFrame):
        return mozanalysis.bayesian_stats.bayesian_bootstrap(
            df, self.num_samples, self.summary_quantiles
        )
