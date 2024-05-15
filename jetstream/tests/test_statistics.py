import datetime as dt
import json
from pathlib import Path

import jsonschema
import numpy as np
import pandas as pd
import pytest
from metric_config_parser.experiment import Branch, BucketConfig, Experiment
from mozanalysis.bayesian_stats.bayesian_bootstrap import get_bootstrap_samples
from mozilla_nimbus_schemas.jetstream import AnalysisBasis

from jetstream.statistics import (
    Binomial,
    BootstrapMean,
    Count,
    Deciles,
    EmpiricalCDF,
    KernelDensityEstimate,
    LinearModelMean,
    PerClientDAUImpact,
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

    def test_linear_model_mean(self):
        stat = LinearModelMean()
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

    def test_linear_model_mean_covariate(self):
        stat = LinearModelMean()
        np.random.seed(42)
        control_mean, treatment_effect = 2, 1
        rel_diff = treatment_effect / control_mean
        y_c = np.random.normal(loc=control_mean, scale=1, size=200)
        te = np.random.normal(loc=treatment_effect, scale=1, size=200)
        y_t = y_c + te
        test_data = pd.DataFrame(
            {
                "branch": ["treatment"] * 100 + ["control"] * 100,
                "value": np.concatenate([y_t[:100], y_c[100:]]),
                "value_pre": y_c + np.random.normal(scale=1, size=200),
            }
        )

        results = stat.transform(
            test_data, "value", "control", None, AnalysisBasis.ENROLLMENTS, "all"
        ).__root__

        branch_results = [r for r in results if r.comparison is None]
        treatment_result = [r for r in branch_results if r.branch == "treatment"][0]
        control_result = [r for r in branch_results if r.branch == "control"][0]
        assert treatment_result.point > control_result.point
        assert treatment_result.lower and treatment_result.upper

        rel_results = [r for r in results if r.comparison == "relative_uplift"][0]
        results_unadj = stat.transform(
            test_data.drop(columns=["value_pre"]),
            "value",
            "control",
            None,
            AnalysisBasis.ENROLLMENTS,
            "all",
        ).__root__
        rel_results_unadj = [r for r in results_unadj if r.comparison == "relative_uplift"][0]
        # test that point estimate after adjustment is closer to truth
        assert np.abs(rel_results.point - rel_diff) < np.abs(rel_results_unadj.point - rel_diff)
        # test that confidence intervals are tighter
        assert rel_results.lower > rel_results_unadj.lower
        assert rel_results.upper < rel_results_unadj.upper

    def test_per_client_dau_impact(self):
        stat = PerClientDAUImpact()
        test_data = pd.DataFrame(
            {
                "branch": ["control"] * 10 + ["treatment"] * 10,
                "value": [x / 20 for x in range(20)],
            }
        )
        experiment = Experiment(
            experimenter_slug="test_slug",
            type="pref",
            status="Live",
            start_date=dt.datetime.now(),
            end_date=None,
            proposed_enrollment=7,
            branches=[
                Branch(slug="control", ratio=1),
                Branch(slug="treatment", ratio=1),
            ],
            normandy_slug="normandy-test-slug",
            reference_branch=None,
            is_high_population=False,
            app_name="firefox_desktop",
            app_id="firefox-desktop",
            bucket_config=BucketConfig(
                randomization_unit="test-unit", namespace="test", start=50, count=100
            ),
        )
        result = stat.transform(
            test_data, "value", "control", experiment, AnalysisBasis.ENROLLMENTS, "all"
        ).__root__

        difference = [r for r in result if r.comparison == "difference"][0]
        # analytically, we should see a point estimate of 10, with 95% CI of (7.155,12.844)
        # at these small sample sizes, mozanalysis's bootstrap can be quite variable
        # so use a large tolerance
        assert np.abs(difference.point - 10) < 1.0
        assert np.abs(difference.lower - 7.155) < 1.0
        assert np.abs(difference.upper - 12.844) < 1.0

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

    def test_binomial_pairwise_branch_comparisons(self, experiments):
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

        # there should only be 15 results (would be 21 without removing dupes)
        assert len(results) == 15

        comparison_branches = set((r.comparison_to_branch, r.branch, r.comparison) for r in results)
        all_comparisons = [
            (None, "control", None),
            (None, "foo", None),
            (None, "treatment", None),
            ("treatment", "control", "difference"),
            ("treatment", "control", "relative_uplift"),
            ("treatment", "foo", "difference"),
            ("treatment", "foo", "relative_uplift"),
            ("foo", "control", "difference"),
            ("foo", "control", "relative_uplift"),
            ("foo", "treatment", "difference"),
            ("foo", "treatment", "relative_uplift"),
            ("control", "treatment", "difference"),
            ("control", "treatment", "relative_uplift"),
            ("control", "foo", "difference"),
            ("control", "foo", "relative_uplift"),
        ]
        assert sorted(all_comparisons, key=lambda c: (str(c[0]), str(c[1]), str(c[2]))) == sorted(
            comparison_branches, key=lambda c: (str(c[0]), str(c[1]), str(c[2]))
        )

    def test_pairwise_branch_comparison_row_counts(self, experiments):
        # tests if each statistic outputs the correct number of rows
        # under pairwise comparisons. Several have 3 branches * 5 results = 15 rows
        # (one individual, and 4 comparative: an absolute & relative comparison
        # for each of the other 2 branches), but some have more
        expectations = {
            Binomial: 3 * 5,
            BootstrapMean: 3 * 5,
            Deciles: 3 * 5 * 9,  # 9 measurements, one for each decile
            EmpiricalCDF: 3 * 256,  # no comparative, default grid size of 256
            KernelDensityEstimate: 3 * 256,  # no comparative, default grid size of 256
            Sum: 3,  # no comparative
        }
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
        for stat_class, expected_count in expectations.items():
            stat = stat_class()
            results = stat.apply(
                test_data, "value", experiments[1], AnalysisBasis.ENROLLMENTS, "all"
            ).__root__
            assert len(results) == expected_count

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
        results = sorted(
            stat.transform(wine, "ash", "*", None, AnalysisBasis.ENROLLMENTS, "all").__root__,
            key=lambda res: (res.branch, res.parameter),
        )

        assert len(results) > 0

        assert results[0].parameter == "2.04"
        assert results[0].point == pytest.approx(0.5, abs=0.1)

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
