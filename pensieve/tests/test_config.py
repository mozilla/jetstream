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
            weekly = ["my_cool_metric"]

            [metrics.my_cool_metric]
            data_source = "main"
            select_expression = "{{agg_histogram_mean('payload.content.my_cool_histogram')}}"
            """
        )
        spec = config.AnalysisSpec.from_dict(toml.loads(config_str))
        cfg = spec.resolve(experiments[0])
        metric = [m for m in cfg.metrics[AnalysisPeriod.WEEK] if m.name == "my_cool_metric"][0]
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
            len([m for m in cfg.metrics[AnalysisPeriod.WEEK] if m.name == "view_about_logins"]) == 1
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
        assert len([m for m in cfg.metrics[AnalysisPeriod.WEEK] if m.name == "unenroll"]) == 1

    def test_data_source_definition(self, experiments):
        config_str = dedent(
            """
            [metrics]
            weekly = ["spam"]

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
        spam = [m for m in cfg.metrics[AnalysisPeriod.WEEK] if m.name == "spam"][0]
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
        stock = [m for m in cfg.metrics[AnalysisPeriod.WEEK] if m.name == "active_hours"][0]

        config_str = dedent(
            """
            [metrics]
            weekly = ["active_hours"]

            [metrics.active_hours]
            select_expression = "spam"
            data_source = "main"
            """
        )
        spec = config.AnalysisSpec.from_dict(toml.loads(config_str))
        cfg = spec.resolve(experiments[0])
        custom = [m for m in cfg.metrics[AnalysisPeriod.WEEK] if m.name == "active_hours"][0]

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
