from textwrap import dedent

import toml
import pytest

from pensieve import AnalysisPeriod, config


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
            weekly = [{metric = "my_cool_metric", treatment = "bootstrap_mean"}]

            [metrics.my_cool_metric]
            data_source = "main"
            select_expression = "{{agg_histogram_mean('payload.content.my_cool_histogram')}}"
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
            weekly = [{metric = "view_about_logins", treatment = "bootstrap_mean"}]
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
            weekly = [
                {metric = "unenroll", treatment = "bootstrap_mean"},
                {metric = "unenroll", treatment = "bootstrap_mean"},
                {metric = "active_hours", treatment = "bootstrap_mean"}]
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
            weekly = [{metric = "spam", treatment = "bootstrap_mean"}]

            [metrics.spam]
            data_source = "eggs"
            select_expression = "1"

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
            weekly = [{metric = "active_hours", treatment = "bootstrap_mean"}]
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
            weekly = [{metric = "active_hours", treatment = "bootstrap_mean"}]

            [metrics.active_hours]
            select_expression = "spam"
            data_source = "main"
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

    def test_lookup_name_failure(self):
        class MyCoolClass:
            pass

        with pytest.raises(ValueError) as e:
            config._lookup_name(
                name="spam",
                klass=MyCoolClass,
                spec=config.AnalysisSpec(),
                module=None,
                definitions={},
            )

        assert "Could not locate MyCoolClass spam" in str(e.value)
