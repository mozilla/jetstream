import numpy as np
import pandas as pd
import pytest

from pensieve import pre_treatment


@pytest.fixture
def example_data():
    return pd.DataFrame([{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 7}])


class TestPreTreatment:
    def test_remove_nulls(self, example_data):
        example_data.iloc[1, 1] = np.nan  # will be removed
        example_data.iloc[2, 1] = np.inf  # will not be removed
        pt = pre_treatment.RemoveNulls()
        ex1 = pt.apply(example_data, "a")
        pd.testing.assert_frame_equal(example_data, ex1)
        ex2 = pt.apply(example_data, "b")
        assert ex2.shape == (2, 2)

    def test_remove_indefinites(self, example_data):
        example_data.iloc[1, 1] = np.inf
        pt = pre_treatment.RemoveIndefinites()
        ex1 = pt.apply(example_data, "a")
        pd.testing.assert_frame_equal(example_data, ex1)
        ex2 = pt.apply(example_data, "b")
        assert ex2.shape == (2, 2)

    def test_censor_highest_values(self):
        df = pd.DataFrame({"value": range(10000), "label": ["a"] * 10000})
        pt = pre_treatment.CensorHighestValues(0.9)
        assert df.shape == (10000, 2)
        ex1 = pt.apply(df, "value")
        assert ex1.shape == (9000, 2)
