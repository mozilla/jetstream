import mozanalysis
import pytest

from jetstream.exposure_signal import ExposureSignal


class TestExposureSignal:
    def test_create_exposure_signal(self):
        exposure_signal = ExposureSignal(
            name="ad_exposure",
            data_source=mozanalysis.metrics.desktop.search_clients_daily,
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
                data_source=mozanalysis.metrics.desktop.search_clients_daily,
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
            data_source=mozanalysis.metrics.desktop.search_clients_daily,
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
            data_source=mozanalysis.metrics.desktop.search_clients_daily,
            select_expression="ad_click > 0",
            friendly_name="Ad exposure",
            description="Clients have clicked on ad",
        )

        assert isinstance(
            exposure_signal.to_mozanalysis_exposure_signal(time_limits),
            mozanalysis.exposure.ExposureSignal,
        )
        assert exposure_signal.to_mozanalysis_exposure_signal(time_limits).build_query(time_limits)
