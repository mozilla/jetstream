import datetime as dt

import pytest
from pytz import UTC

import jetstream.analysis
import jetstream.config
import jetstream.experimenter


class TestDefaultConfigs:
    @pytest.mark.parametrize("platform_name,platform", jetstream.config.PLATFORM_CONFIGS.items())
    def test_default_configs(self, platform_name, platform):
        experiment = jetstream.experimenter.Experiment(
            experimenter_slug="dummy-experiment",
            normandy_slug="dummy_experiment",
            type="v6",
            status="Live",
            branches=[],
            end_date=None,
            reference_branch="control",
            is_high_population=False,
            start_date=dt.datetime.now(UTC),
            proposed_enrollment=14,
            app_id=platform.app_id,
            app_name=platform_name,
        )

        spec = jetstream.config.AnalysisSpec.default_for_experiment(experiment)
        config = spec.resolve(experiment)
        jetstream.analysis.Analysis("no project", "no dataset", config).validate()
