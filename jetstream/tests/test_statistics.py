import json
from pathlib import Path

import jsonschema
import numpy as np
import pandas as pd
import pytest
from mozanalysis.bayesian_stats.bayesian_bootstrap import get_bootstrap_samples
from mozilla_nimbus_schemas.jetstream import AnalysisBasis

from jetstream.statistics import (
    Binomial,
    BootstrapMean,
    Count,
    EmpiricalCDF,
    KernelDensityEstimate,
    PopulationRatio,
    StatisticResult,
    Sum,
    _make_grid,
)


@pytest.fixture()
def wine():
    return pd.read_csv(Path(__file__).parent / "data/wine.data").rename(
        columns={"cultivar": "branch"}
    )


class TestStatistics:
    def test_bootstrap_means(self):
        stat = BootstrapMean(num_samples=10)
        test_data = pd.DataFrame(
            {"branch": ["treatment"] * 10 + ["control"] * 10, "value": list(range(20))}
        )
        results = stat.transform(
            test_data, "value", "control", None, AnalysisBasis.ENROLLMENTS, "all"
        ).__root__

        branch_results = [r for r in results if r.comparison is None]
        treatment_result = [r for r in branch_results if r.branch == "treatment"][0]
        control_result = [r for r in branch_results if r.branch == "control"][0]
        assert treatment_result.point < control_result.point
        assert treatment_result.lower and treatment_result.upper

    def test_binomial(self):
        stat = Binomial()
        test_data = pd.DataFrame(
            {
                "branch": ["treatment"] * 10 + ["control"] * 10,
                "value": [False] * 7 + [True] * 3 + [False] * 5 + [True] * 5,
            }
        )
        results = stat.transform(
            test_data, "value", "control", None, AnalysisBasis.ENROLLMENTS, "all"
        ).__root__
        branch_results = [r for r in results if r.comparison is None]
        treatment_result = [r for r in branch_results if r.branch == "treatment"][0]
        control_result = [r for r in branch_results if r.branch == "control"][0]
        assert treatment_result.point < control_result.point
        assert treatment_result.point - 0.7 < 1e-5

        difference = [r for r in results if r.comparison == "difference"][0]
        assert difference.point - 0.2 < 1e-5
        assert difference.lower and difference.upper

    def test_count(self):
        stat = Count()
        test_data = pd.DataFrame(
            {"branch": ["treatment"] * 20 + ["control"] * 10, "value": list(range(30))}
        )
        results = stat.transform(
            test_data, "identity", "control", None, AnalysisBasis.ENROLLMENTS, "all"
        ).__root__
        assert all(r.metric == "identity" for r in results)
        assert [r.point for r in results if r.branch == "treatment"] == [20]
        assert [r.point for r in results if r.branch == "control"] == [10]

    def test_sum_int(self):
        stat = Sum()
        treatment_values = [0] * 5 + [-1] * 5 + [1] * 10
        control_values = [0] * 5 + [1] * 5
        test_data = pd.DataFrame(
            {
                "branch": ["treatment"] * 20 + ["control"] * 10,
                "value": treatment_values + control_values,
            }
        )
        results = stat.transform(
            test_data, "value", "control", None, AnalysisBasis.ENROLLMENTS, "all"
        ).__root__
        assert all(r.metric == "value" for r in results)
        assert [r.point for r in results if r.branch == "treatment"] == [5]
        assert [r.point for r in results if r.branch == "control"] == [5]

    def test_sum_bool(self):
        stat = Sum()
        treatment_values = [False] * 5 + [True] * 15
        control_values = [False] * 5 + [True] * 5
        test_data = pd.DataFrame(
            {
                "branch": ["treatment"] * 20 + ["control"] * 10,
                "value": treatment_values + control_values,
            }
        )
        results = stat.transform(
            test_data, "value", "control", None, AnalysisBasis.ENROLLMENTS, "all"
        ).__root__
        assert all(r.metric == "value" for r in results)
        assert [r.point for r in results if r.branch == "treatment"] == [15]
        assert [r.point for r in results if r.branch == "control"] == [5]

    def test_binomial_no_reference_branch(self, experiments):
        stat = Binomial()
        test_data = pd.DataFrame(
            {
                "branch": ["treatment"] * 10 + ["control"] * 10 + ["foo"] * 10,
                "value": [False] * 7
                + [True] * 3
                + [False] * 5
                + [True] * 5
                + [False] * 5
                + [True] * 5,
            }
        )
        results = stat.apply(
            test_data, "value", experiments[1], AnalysisBasis.ENROLLMENTS, "all"
        ).__root__

        branch_results = [r for r in results if r.comparison is None]
        treatment_result = [r for r in branch_results if r.branch == "treatment"][0]
        control_result = [r for r in branch_results if r.branch == "control"][0]
        assert treatment_result.point < control_result.point
        assert treatment_result.point - 0.7 < 1e-5

        difference = [r for r in results if r.comparison == "difference"][0]
        assert difference.point - 0.2 < 1e-5
        assert difference.lower and difference.upper

        comparison_branches = [(r.comparison_to_branch, r.branch, r.comparison) for r in results]
        assert (None, "control", None) in comparison_branches
        assert (None, "foo", None) in comparison_branches
        assert (None, "treatment", None) in comparison_branches
        assert ("treatment", "control", "difference") in comparison_branches
        assert ("treatment", "control", "relative_uplift") in comparison_branches
        assert ("control", "foo", "difference") in comparison_branches
        assert ("control", "foo", "relative_uplift") in comparison_branches

    @pytest.mark.parametrize("geometric", [True, False])
    def test_make_grid_makes_a_grid(self, wine, geometric):
        result = _make_grid(wine["ash"], 256, geometric)
        assert result.grid.shape == (256,)
        assert result.geometric is geometric
        assert result.message is None

    def test_make_grid_handles_negatives(self, wine):
        ash = wine["ash"].copy()
        ash.iloc[0] = -1
        result = _make_grid(ash, 256, True)
        assert result.geometric is False
        assert result.grid.shape == (256,)
        assert result.message is not None

        result = _make_grid(ash, 256, False)
        assert result.geometric is False
        assert result.grid.shape == (256,)
        assert result.message is None
        assert result.grid.max() == ash.max()
        assert result.grid.min() == -1

    def test_make_grid_handles_zeros(self, wine):
        ash = wine["ash"].copy()
        ash.iloc[0] = 0
        result = _make_grid(ash, 256, True)
        assert result.geometric is True
        assert result.grid.shape == (256,)
        assert result.message is None
        assert result.grid.min() > 0

        result = _make_grid(ash, 256, False)
        assert result.geometric is False
        assert result.grid.shape == (256,)
        assert result.message is None
        assert result.grid.max() == ash.max()
        assert result.grid.min() == 0

    def test_kde(self, wine):
        stat = KernelDensityEstimate()
        results = stat.transform(wine, "ash", "*", None, AnalysisBasis.ENROLLMENTS, "all").__root__
        assert len(results) > 0

    def test_kde_with_geom_zero(self, wine):
        wine = wine.copy()
        wine.loc[0, "ash"] = 0
        stat = KernelDensityEstimate(log_space=True)
        results = stat.transform(wine, "ash", "*", None, AnalysisBasis.ENROLLMENTS, "all").__root__
        for r in results:
            assert isinstance(r.point, float)
        df = pd.DataFrame([r.dict() for r in results])
        assert float(df["parameter"].min()) == 0.0

    def test_ecdf(self, wine, experiments):
        stat = EmpiricalCDF()
        results = stat.transform(
            wine, "ash", "*", experiments[0], AnalysisBasis.ENROLLMENTS, "all"
        ).__root__
        assert len(results) > 0

        logstat = EmpiricalCDF(log_space=True)
        results = logstat.transform(
            wine, "ash", "*", experiments[0], AnalysisBasis.ENROLLMENTS, "all"
        ).__root__
        assert len(results) > 0

        wine["ash"] = -wine["ash"]
        results = logstat.transform(
            wine, "ash", "*", experiments[0], AnalysisBasis.ENROLLMENTS, "all"
        ).__root__
        assert len(results) > 0

        assert stat.name() == "empirical_cdf"

    def test_statistic_result_rejects_invalid_types(self):
        args = {"metric": "foo", "statistic": "bar", "branch": "baz"}
        StatisticResult(**args)
        with pytest.raises(ValueError):
            StatisticResult(point=[3], **args)

    def test_type_conversions(self):
        df = pd.array([1, 2, np.nan], dtype="Int64")
        with pytest.raises(TypeError):
            np.isnan(np.array(df)).any()

        d = np.array(df.to_numpy(dtype="float", na_value=np.nan))
        assert np.isnan(d).any()

    def test_mozanalysis_nan(self):
        df = pd.array([1, 2, np.nan], dtype="Int64")

        with pytest.raises(ValueError):
            get_bootstrap_samples(df)

    def test_population_ratio(self):
        stat = PopulationRatio(num_samples=10, numerator="ad_click", denominator="sap")
        test_data = pd.DataFrame(
            {
                "branch": ["treatment"] * 10 + ["control"] * 10,
                "ad_click": [x for x in range(10, 0, -1)] * 2,
                "sap": [10 * x for x in range(10, 0, -1)] * 2,
                "ad_ratio": np.nan,
            }
        )
        results = stat.transform(
            test_data, "ad_ratio", "control", None, AnalysisBasis.ENROLLMENTS, "all"
        ).__root__

        branch_results = [r for r in results if r.comparison is None]
        treatment_result = [r for r in branch_results if r.branch == "treatment"][0]
        control_result = [r for r in branch_results if r.branch == "control"][0]
        assert treatment_result.point == pytest.approx(control_result.point, rel=1e-5)
        assert treatment_result.point == pytest.approx(0.1, rel=1e-5)
        assert control_result.point == pytest.approx(0.1, rel=1e-5)

    def test_population_ratio_non_existing_metrics(self):
        stat = PopulationRatio(num_samples=10, numerator="non_existing", denominator="non_existing")
        test_data = pd.DataFrame(
            {"branch": ["treatment"] * 10 + ["control"] * 10, "ad_ratio": np.nan}
        )

        with pytest.raises(Exception):
            stat.transform(test_data, "ad_ratio", "control", None, AnalysisBasis.ENROLLMENTS, "all")


class TestStatisticExport:
    def test_data_schema(self):
        schema = json.loads((Path(__file__).parent / "data/Statistics_v1.0.json").read_text())

        jsonschema.validate([], schema)

        statistics_export_data = [
            {
                "metric": "tagged_sap_searches",
                "statistic": "deciles",
                "parameter": "0.8",
                "branch": "treatment-a",
                "comparison": "relative_uplift",
                "comparison_to_branch": "control",
                "ci_width": 0.95,
                "point": 0,
                "lower": 0,
                "upper": 0,
                "segment": "all",
                "analysis_basis": "enrollments",
                "window_index": "3",
            },
            {
                "metric": "tagged_sap_searches",
                "statistic": "deciles",
                "branch": "treatment-a",
                "comparison": "relative_uplift",
                "comparison_to_branch": "control",
                "ci_width": 0.95,
                "point": 0,
                "lower": 0,
                "upper": 0,
                "segment": "all",
                "analysis_basis": "exposures",
                "window_index": "3",
            },
            {
                "metric": "tagged_sap_searches",
                "statistic": "deciles",
                "parameter": 0.8,
                "branch": "treatment-a",
                "ci_width": 0.95,
                "point": 0,
                "lower": 0,
                "upper": 0,
                "segment": "all",
                "analysis_basis": "enrollments",
                "window_index": "3",
            },
        ]
        jsonschema.validate(statistics_export_data, schema)
