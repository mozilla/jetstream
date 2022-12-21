from textwrap import dedent

import mozanalysis
import pytest
import toml
from metric_config_parser.analysis import AnalysisSpec
from metric_config_parser.exposure_signal import (
    ExposureSignal as MetricConfigParserExposureSignal,
)

from jetstream.config import ConfigLoader
from jetstream.exposure_signal import ExposureSignal


class TestExposureSignal:
    def test_create_exposure_signal(self):
        exposure_signal = ExposureSignal(
            name="ad_exposure",
            data_source=ConfigLoader.get_data_source("search_clients_daily", "firefox_desktop"),
            select_expression="ad_click > 0",
            friendly_name="Ad exposure",
            description="Clients have clicked on ad",
            window_start=12,
        )

        assert exposure_signal.window_end is None
        assert exposure_signal.window_start == 12

    def test_invalid_window(self):
        with pytest.raises(Exception):
            ExposureSignal(
                name="ad_exposure",
                data_source=ConfigLoader.get_data_source("search_clients_daily", "firefox_desktop"),
                select_expression="ad_click > 0",
                friendly_name="Ad exposure",
                description="Clients have clicked on ad",
                window_start="invalid",
            )

    def test_window_limit_to_int(self):
        time_limits = mozanalysis.experiment.TimeLimits(
            first_enrollment_date="2019-01-05",
            last_enrollment_date="2019-01-06",
            analysis_windows=(mozanalysis.experiment.AnalysisWindow(2, 4),),
            first_date_data_required="2019-01-07",
            last_date_data_required="2019-01-10",
        )

        exposure_signal = ExposureSignal(
            name="ad_exposure",
            data_source=ConfigLoader.get_data_source("search_clients_daily", "firefox_desktop"),
            select_expression="ad_click > 0",
            friendly_name="Ad exposure",
            description="Clients have clicked on ad",
        )

        assert exposure_signal._window_limit_to_int(None, time_limits) is None
        assert exposure_signal._window_limit_to_int(11, time_limits) == 11
        assert exposure_signal._window_limit_to_int("enrollment_start", time_limits) == 0
        assert exposure_signal._window_limit_to_int("enrollment_end", time_limits) == 1
        assert exposure_signal._window_limit_to_int("analysis_window_end", time_limits) == 5
        assert exposure_signal._window_limit_to_int("analysis_window_start", time_limits) == 3

        with pytest.raises(Exception):
            exposure_signal._window_limit_to_int("invalid", time_limits)

    def test_exposure_signal_to_mozanalysis(self):
        time_limits = mozanalysis.experiment.TimeLimits(
            first_enrollment_date="2019-01-05",
            last_enrollment_date="2019-01-06",
            analysis_windows=(mozanalysis.experiment.AnalysisWindow(2, 4),),
            first_date_data_required="2019-01-07",
            last_date_data_required="2019-01-10",
        )

        exposure_signal = ExposureSignal(
            name="ad_exposure",
            data_source=ConfigLoader.get_data_source("search_clients_daily", "firefox_desktop"),
            select_expression="ad_click > 0",
            friendly_name="Ad exposure",
            description="Clients have clicked on ad",
        )

        assert isinstance(
            exposure_signal.to_mozanalysis_exposure_signal(time_limits),
            mozanalysis.exposure.ExposureSignal,
        )
        assert exposure_signal.to_mozanalysis_exposure_signal(time_limits).build_query(time_limits)

    def test_exposure_signal_from_config(self, experiments):
        """Custom exposure signals support `window_start` and `window_end`."""
        time_limits = mozanalysis.experiment.TimeLimits(
            first_enrollment_date="2019-01-05",
            last_enrollment_date="2019-01-06",
            analysis_windows=(mozanalysis.experiment.AnalysisWindow(2, 4),),
            first_date_data_required="2019-01-07",
            last_date_data_required="2019-01-10",
        )

        conf = dedent(
            """
            [data_sources.background_update_events]
            from_expression = '''(
                SELECT
                    DATE(events.submission_timestamp) AS submission_date,
                    events.metrics.uuid.background_update_client_id AS client_id,
                    events.* EXCEPT (events),
                    event.timestamp AS event_timestamp,
                    event.category AS event_category,
                    event.name AS event_name,
                    event.extra AS event_extra,
                FROM
                    `mozdata.firefox_desktop_background_update.background_update` events,
                UNNEST(events.events) AS event
            )'''
            experiments_column_type="native"
            friendly_name = "Background Update Events"
            description = "Background Update Events"

            [experiment.exposure_signal]
            name = "nimbus_exposure"
            friendly_name = "Nimbus Exposure"
            description = "Notification count per analysis window"
            select_expression = '''
                event_category = 'nimbus_events'
                AND event_name = 'exposure'
                AND mozfun.map.get_key(event_extra, 'experiment') = 'SLUG'
            '''
            data_source = "background_update_events"
            window_start = "enrollment_start"
            window_end = "analysis_window_end"
            """
        )

        spec = AnalysisSpec.from_dict(toml.loads(conf))
        configured = spec.resolve(experiments[0], ConfigLoader.configs)

        mcp_exposure_signal = configured.experiment.exposure_signal
        assert isinstance(mcp_exposure_signal, MetricConfigParserExposureSignal)

        jetstream_exposure_signal = ExposureSignal.from_exposure_signal_config(mcp_exposure_signal)
        assert jetstream_exposure_signal.to_mozanalysis_exposure_signal(time_limits).build_query(
            time_limits
        )
