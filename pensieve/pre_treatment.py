import attr
import cattr
from pandas import DataFrame


@attr.s(auto_attribs=True)
class PreTreatment:
    """Represents an abstract pre-treatment step applied to data before calculating statistics."""

    name: str

    def apply(self, df: DataFrame):
        """Applies the pre-treatment transformation to a DataFrame."""
        return self.transformation(df)

    def transformation(self, df: DataFrame):
        raise NotImplementedError("PreTreatment subclasses must override transformation()")


class RemoveNulls(PreTreatment):
    """Pre-treatment step to remove null rows and columns."""

    name = "remove_nulls"

    def transformation(self, df: DataFrame):
        return df.dropna()
