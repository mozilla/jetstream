from typing import Optional, Union

import attr
import enum
import mozanalysis.experiment
import mozanalysis.metrics


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


class ExposureSignal:
    """
    Jetstream exposure signal representation.

    Jetstream exposure signals are supersets of mozanalysis exposure signals
    with some additional metdata required for analysis.
    """

    name: str
    data_source: mozanalysis.metrics.DataSource
    select_expression: str
    friendly_name: str
    description: str
    window_start: WindowLimit = None
    window_end: WindowLimit = None

    def to_mozanalysis_exposure_signal(
        self, time_limits: mozanalysis.experiment.TimeLimits
    ) -> mozanalysis.exposure.ExposureSignal:
        window_start = self._window_limit_to_int(self.window_start, time_limits)
        window_end = self._window_limit_to_int(self.window_end, time_limits)

        return mozanalysis.exposure.ExposureSignal(
            name=self.name,
            data_source=self.data_source,
            select_expr=self.select_expression,
            friendly_name=self.friendly_name,
            description=self.description,
            window_start=window_start,
            window_end=window_end,
        )

    def _window_limit_to_int(
        self, window_limit: WindowLimit, time_limits: mozanalysis.experiment.TimeLimits
    ):
        if window_limit == AnalysisWindow.ENROLLMENT_START:
            return 0
        elif window_limit == AnalysisWindow.ENROLLMENT_END:
            return abs(time_limits.last_enrollment_date - time_limits.first_enrollment_date).days
        elif window_limit == AnalysisWindow.ANALYSIS_WINDOW_START:
            return time_limits.analysis_windows[0].start
        elif window_limit == AnalysisWindow.ANALYSIS_WINDOW_END:
            return time_limits.analysis_windows[0].end
        else:
            return window_limit
