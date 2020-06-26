from textwrap import dedent

import toml
import pytest

from pensieve import AnalysisPeriod, config
from pensieve.cli import DEFAULT_METRICS_CONFIG
from pensieve.statistics import BootstrapMean
from pensieve.pre_treatment import RemoveNulls


class TestAnalysisSpec:
    default_metrics_config = DEFAULT_METRICS_CONFIG

    def test_trivial_configuration(self, experiments):
        spec = config.AnalysisSpec.from_dict({})
        assert isinstance(spec, config.AnalysisSpec)
        cfg = spec.resolve(experiments[0])
        assert isinstance(cfg, config.AnalysisConfiguration)

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

            [metrics.view_about_logins.statistics.bootstrap_mean]
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

            [metrics.unenroll.statistics.binomial]
            [metrics.active_hours.statistics.bootstrap_mean]
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
        default_spec = config.AnalysisSpec.from_dict(toml.load(self.default_metrics_config))
        spec = config.AnalysisSpec.from_dict(toml.loads(config_str))
        spec.merge(default_spec)
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
        ].statistic
        bootstrap_mean.__class__ = BootstrapMean

        assert bootstrap_mean.num_samples == 10

    def test_overwrite_default_statistic(self, experiments):
        config_str = dedent(
            """
            [metrics]
            weekly = ["ad_clicks"]

            [metrics.ad_clicks.statistics.bootstrap_mean]
            num_samples = 10
            """
        )

        default_spec = config.AnalysisSpec.from_dict(toml.load(self.default_metrics_config))
        spec = config.AnalysisSpec.from_dict(toml.loads(config_str))
        default_spec.merge(spec)
        cfg = default_spec.resolve(experiments[0])
        bootstrap_mean = [
            m for m in cfg.metrics[AnalysisPeriod.WEEK] if m.metric.name == "ad_clicks"
        ][0].statistic
        bootstrap_mean.__class__ = BootstrapMean

        assert bootstrap_mean.num_samples == 10

    def test_pre_treatment(self, experiments):
        config_str = dedent(
            """
            [metrics]
            weekly = ["spam"]

            [metrics.spam]
            data_source = "main"
            select_expression = "1"

            [metrics.spam.statistics.bootstrap_mean]
            num_samples = 10
            pre_treatments = ["remove_nulls"]
            """
        )

        spec = config.AnalysisSpec.from_dict(toml.loads(config_str))
        cfg = spec.resolve(experiments[0])
        pre_treatments = [m for m in cfg.metrics[AnalysisPeriod.WEEK] if m.metric.name == "spam"][
            0
        ].pre_treatments

        assert len(pre_treatments) == 1
        assert pre_treatments[0].__class__ == RemoveNulls

    def test_invalid_pre_treatment(self, experiments):
        config_str = dedent(
            """
            [metrics]
            weekly = ["spam"]

            [metrics.spam]
            data_source = "main"
            select_expression = "1"

            [metrics.spam.statistics.bootstrap_mean]
            num_samples = 10
            pre_treatments = ["not_existing"]
            """
        )

        with pytest.raises(ValueError) as e:
            spec = config.AnalysisSpec.from_dict(toml.loads(config_str))
            spec.resolve(experiments[0])

        assert "Could not find pre-treatment not_existing." in str(e)

    def test_merge_configs(self, experiments):
        orig_conf = dedent(
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

        custom_conf = dedent(
            """
            [metrics]
            weekly = ["foo"]

            [metrics.foo]
            data_source = "main"
            select_expression = "2"

            [metrics.foo.statistics.bootstrap_mean]
            num_samples = 100
            """
        )

        spec = config.AnalysisSpec.from_dict(toml.loads(orig_conf))
        spec.merge(config.AnalysisSpec.from_dict(toml.loads(custom_conf)))
        cfg = spec.resolve(experiments[0])

        assert len(cfg.metrics[AnalysisPeriod.WEEK]) == 2
        assert len([m for m in cfg.metrics[AnalysisPeriod.WEEK] if m.metric.name == "spam"]) == 1
        assert len([m for m in cfg.metrics[AnalysisPeriod.WEEK] if m.metric.name == "foo"]) == 1

    def test_merge_configs_override_metric(self, experiments):
        orig_conf = dedent(
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

        custom_conf = dedent(
            """
            [metrics]
            weekly = ["spam"]

            [metrics.spam]
            data_source = "events"
            select_expression = "2"

            [metrics.spam.statistics.bootstrap_mean]
            num_samples = 100
            """
        )

        spec = config.AnalysisSpec.from_dict(toml.loads(orig_conf))
        spec.merge(config.AnalysisSpec.from_dict(toml.loads(custom_conf)))
        cfg = spec.resolve(experiments[0])

        spam = [m for m in cfg.metrics[AnalysisPeriod.WEEK] if m.metric.name == "spam"][0]

        assert len(cfg.metrics[AnalysisPeriod.WEEK]) == 1
        assert spam.metric.data_source.name == "events"
        assert spam.metric.select_expr == "2"
        assert spam.statistic.name() == "bootstrap_mean"
        assert spam.statistic.num_samples == 100
