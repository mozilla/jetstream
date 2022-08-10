import enum
from datetime import datetime
from typing import Union

import attr
import mozanalysis.experiment
import mozanalysis.metrics
from jetstream_config_parser import exposure_signal
from mozanalysis import exposure


class AnalysisWindow(enum.Enum):
    """
    Predefined timelimits that can be used for defining when exposures
    should be computed.
    """

    ANALYSIS_WINDOW_START = "analysis_window_start"
    ANALYSIS_WINDOW_END = "analysis_window_end"
    ENROLLMENT_START = "enrollment_start"
    ENROLLMENT_END = "enrollment_end"


WindowLimit = Union[int, AnalysisWindow, None]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ExposureSignal(exposure_signal.ExposureSignal):
    """
    Jetstream exposure signal representation.

    Jetstream exposure signals are supersets of mozanalysis exposure signals
    with some additional metdata required for analysis.
    """

    def to_mozanalysis_exposure_signal(
        self, time_limits: mozanalysis.experiment.TimeLimits
    ) -> exposure.ExposureSignal:
        """Converts the Jetstream `ExposureSignal` to the corresponding mozanalysis instance."""
        window_start = self._window_limit_to_int(self.window_start, time_limits)
        window_end = self._window_limit_to_int(self.window_end, time_limits)

        return exposure.ExposureSignal(
            name=self.name,
            data_source=mozanalysis.metrics.DataSource(
                name=self.data_source.name,
                from_expr=self.data_source.from_expression,
                experiments_column_type=self.data_source.experiments_column_type,
                client_id_column=self.data_source.client_id_column,
                submission_date_column=self.data_source.submission_date_column,
                default_dataset=self.data_source.default_dataset,
            ),
            select_expr=self.select_expression,
            friendly_name=self.friendly_name,
            description=self.description,
            window_start=window_start,
            window_end=window_end,
        )

    def _window_limit_to_int(
        self, window_limit: WindowLimit, time_limits: mozanalysis.experiment.TimeLimits
    ):
        """
        Convert the `WindowLimit` to an integer value representing the window.

        These window values are representing the number of days before or after the
        first enrollment date.
        """
        last_enrollment_date = datetime.strptime(time_limits.last_enrollment_date, "%Y-%m-%d")
        first_enrollment_date = datetime.strptime(time_limits.first_enrollment_date, "%Y-%m-%d")
        num_dates_enrollment = abs(last_enrollment_date - first_enrollment_date).days

        try:
            limit = AnalysisWindow(window_limit)
            if limit == AnalysisWindow.ENROLLMENT_START:
                return 0
            elif limit == AnalysisWindow.ENROLLMENT_END:
                return num_dates_enrollment
            elif limit == AnalysisWindow.ANALYSIS_WINDOW_START:
                return time_limits.analysis_windows[0].start + num_dates_enrollment
            elif limit == AnalysisWindow.ANALYSIS_WINDOW_END:
                return time_limits.analysis_windows[0].end + num_dates_enrollment
        except Exception:
            if not isinstance(window_limit, int) and window_limit is not None:
                raise ValueError(f"Invalid window limit: {window_limit}")
            return window_limit
