from textwrap import dedent

import toml
import pytest

from pensieve import AnalysisPeriod, config
from pensieve.statistics import BootstrapMean


class TestAnalysisSpec:
    def test_trivial_configuration(self, experiments):
        spec = config.AnalysisSpec.from_dict({})
        assert isinstance(spec, config.AnalysisSpec)
        cfg = spec.resolve(experiments[0])
        assert isinstance(cfg, config.AnalysisConfiguration)
        # There should be some default metrics
        assert len(cfg.metrics[AnalysisPeriod.WEEK])

    def test_scalar_metrics_throws_exception(self):
        config_str = dedent(
            """
            [metrics]
            weekly = "my_cool_metric"
            """
        )
        with pytest.raises(ValueError):
            config.AnalysisSpec.from_dict(toml.loads(config_str))

    def test_template_expansion(self, experiments):
        config_str = dedent(
            """
            [metrics]
            weekly = ["my_cool_metric"]
            [metrics.my_cool_metric]
            data_source = "main"
            select_expression = "{{agg_histogram_mean('payload.content.my_cool_histogram')}}"

            [metrics.my_cool_metric.statistics.bootstrap_mean]
            """
        )
        spec = config.AnalysisSpec.from_dict(toml.loads(config_str))
        cfg = spec.resolve(experiments[0])
        metric = [m for m in cfg.metrics[AnalysisPeriod.WEEK] if m.metric.name == "my_cool_metric"][
            0
        ].metric
        assert "agg_histogram_mean" not in metric.select_expr
        assert "json_extract_histogram" in metric.select_expr

    def test_recognizes_metrics(self, experiments):
        config_str = dedent(
            """
            [metrics]
            weekly = ["view_about_logins"]
            """
        )
        spec = config.AnalysisSpec.from_dict(toml.loads(config_str))
        cfg = spec.resolve(experiments[0])
        assert (
            len(
                [
                    m
                    for m in cfg.metrics[AnalysisPeriod.WEEK]
                    if m.metric.name == "view_about_logins"
                ]
            )
            == 1
        )

    def test_duplicate_metrics_are_okay(self, experiments):
        config_str = dedent(
            """
            [metrics]
            weekly = ["unenroll", "unenroll", "active_hours"]
            """
        )
        spec = config.AnalysisSpec.from_dict(toml.loads(config_str))
        cfg = spec.resolve(experiments[0])
        assert (
            len([m for m in cfg.metrics[AnalysisPeriod.WEEK] if m.metric.name == "unenroll"]) == 1
        )

    def test_data_source_definition(self, experiments):
        config_str = dedent(
            """
            [metrics]
            weekly = ["spam"]
            [metrics.spam]
            data_source = "eggs"
            select_expression = "1"
            [metrics.spam.statistics.bootstrap_mean]

            [data_sources.eggs]
            from_expression = "england.camelot"
            client_id_column = "client_info.client_id"
            """
        )
        spec = config.AnalysisSpec.from_dict(toml.loads(config_str))
        cfg = spec.resolve(experiments[0])
        spam = [m for m in cfg.metrics[AnalysisPeriod.WEEK] if m.metric.name == "spam"][0].metric
        assert spam.data_source.name == "eggs"
        assert "camelot" in spam.data_source.from_expr
        assert "client_info" in spam.data_source.client_id_column

    def test_definitions_override_other_metrics(self, experiments):
        """Test that config definitions override mozanalysis definitions.
        Users can specify a metric with the same name as a metric built into mozanalysis.
        The user's metric from the config file should win.
        """
        config_str = dedent(
            """
            [metrics]
            weekly = ["active_hours"]
            """
        )
        spec = config.AnalysisSpec.from_dict(toml.loads(config_str))
        cfg = spec.resolve(experiments[0])
        stock = [m for m in cfg.metrics[AnalysisPeriod.WEEK] if m.metric.name == "active_hours"][
            0
        ].metric

        config_str = dedent(
            """
            [metrics]
            weekly = ["active_hours"]
            [metrics.active_hours]
            select_expression = "spam"
            data_source = "main"

            [metrics.active_hours.statistics.bootstrap_mean]
            """
        )
        spec = config.AnalysisSpec.from_dict(toml.loads(config_str))
        cfg = spec.resolve(experiments[0])
        custom = [m for m in cfg.metrics[AnalysisPeriod.WEEK] if m.metric.name == "active_hours"][
            0
        ].metric

        assert stock != custom
        assert custom.select_expr == "spam"
        assert stock.select_expr != custom.select_expr

    def test_unknown_statistic_failure(self, experiments):
        config_str = dedent(
            """
            [metrics]
            weekly = ["spam"]

            [metrics.spam]
            data_source = "main"
            select_expression = "1"

            [metrics.spam.statistics.unknown_stat]
            """
        )

        with pytest.raises(ValueError) as e:
            spec = config.AnalysisSpec.from_dict(toml.loads(config_str))
            spec.resolve(experiments[0])

        assert "Statistic unknown_stat does not exist" in str(e)

    def test_overwrite_statistic(self, experiments):
        config_str = dedent(
            """
            [metrics]
            weekly = ["spam"]

            [metrics.spam]
            data_source = "main"
            select_expression = "1"

            [metrics.spam.statistics.bootstrap_mean]
            num_samples = 10
            """
        )

        spec = config.AnalysisSpec.from_dict(toml.loads(config_str))
        cfg = spec.resolve(experiments[0])
        bootstrap_mean = [m for m in cfg.metrics[AnalysisPeriod.WEEK] if m.metric.name == "spam"][
            0
        ].treatment
        bootstrap_mean.__class__ = BootstrapMean

        assert bootstrap_mean.num_samples == 10


# todo: test for configuring pre-treatments
