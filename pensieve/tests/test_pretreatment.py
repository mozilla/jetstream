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

    def test_log_transform(self):
        df = pd.DataFrame({"value": range(-1, 11)})
        pt = pre_treatment.Log(10)
        assert df.shape == (12, 1)
        ex1 = pt.apply(df, "value")
        assert ex1.shape == (12, 1)
        assert np.isnan(ex1["value"]).sum() == 1
        assert np.isinf(ex1["value"]).sum() == 1
        assert ex1["value"].iloc[-1] == 1

    def test_zero_fill(self, example_data):
        pt = pre_treatment.ZeroFill()
        pd.testing.assert_frame_equal(example_data, pt.apply(example_data, "a"))
        example_data.loc[1, "a"] = np.nan
        example_data.loc[1, "b"] = np.nan
        ex1 = pt.apply(example_data, "a")
        assert ex1.loc[1, "a"] == 0
        assert np.isnan(ex1.loc[1, "b"])
