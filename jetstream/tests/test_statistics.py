import pandas as pd

from jetstream.statistics import BootstrapMean, Binomial, Count


class TestStatistics:
    def test_bootstrap_means(self):
        stat = BootstrapMean(num_samples=10)
        test_data = pd.DataFrame(
            {"branch": ["treatment"] * 10 + ["control"] * 10, "value": list(range(20))}
        )
        result = stat.transform(test_data, "value", "control")

        branch_results = [r for r in result.data if r.comparison is None]
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
        result = stat.transform(test_data, "value", "control")
        branch_results = [r for r in result.data if r.comparison is None]
        treatment_result = [r for r in branch_results if r.branch == "treatment"][0]
        control_result = [r for r in branch_results if r.branch == "control"][0]
        assert treatment_result.point < control_result.point
        assert treatment_result.point - 0.7 < 1e-5

        difference = [r for r in result.data if r.comparison == "difference"][0]
        assert difference.point - 0.2 < 1e-5
        assert difference.lower and difference.upper

    def test_count(self):
        stat = Count()
        test_data = pd.DataFrame(
            {"branch": ["treatment"] * 20 + ["control"] * 10, "value": list(range(30))}
        )
        result = stat.transform(test_data, "asdfasdf", "control").data
        assert [r.point for r in result if r.branch == "treatment"] == [20]
        assert [r.point for r in result if r.branch == "control"] == [10]

    def test_binomial_no_reference_branch(self):
        stat = Binomial(ref_branch_label=None)
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
        result = stat.transform(test_data, "value")

        branch_results = [r for r in result.data if r.comparison is None]
        treatment_result = [r for r in branch_results if r.branch == "treatment"][0]
        control_result = [r for r in branch_results if r.branch == "control"][0]
        assert treatment_result.point < control_result.point
        assert treatment_result.point - 0.7 < 1e-5

        difference = [r for r in result.data if r.comparison == "difference"][0]
        assert difference.point - 0.2 < 1e-5
        assert difference.lower and difference.upper

        comparison_branches = [r.comparison_to_branch for r in result.data]
        assert "foo" in comparison_branches
        assert "control" in comparison_branches
        assert "treatment" in comparison_branches
