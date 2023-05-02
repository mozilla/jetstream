class ValidationException(Exception):
    """Exception thrown when an experiment is invalid."""

    def __init__(self, message):
        super().__init__(message)


class NoSlugException(ValidationException):
    def __init__(self, message="Experiment has no slug."):
        super().__init__(message)


class NoEnrollmentPeriodException(ValidationException):
    def __init__(self, normandy_slug, message="Experiment has no enrollment period."):
        super().__init__(f"{normandy_slug} -> {message}")


class EnrollmentNotCompleteException(ValidationException):
    def __init__(self, normandy_slug, message="Experiment has not finished enrollment."):
        super().__init__(f"{normandy_slug} -> {message}")


class NoStartDateException(ValidationException):
    def __init__(self, normandy_slug, message="Experiment has no start date."):
        super().__init__(f"{normandy_slug} -> {message}")


class EndedException(ValidationException):
    def __init__(self, normandy_slug, message="Experiment has already ended."):
        super().__init__(f"{normandy_slug} -> {message}")


class EnrollmentLongerThanAnalysisException(ValidationException):
    def __init__(self, normandy_slug, message="Enrollment period is longer than analysis dates."):
        super().__init__(f"{normandy_slug} -> {message}")


class HighPopulationException(ValidationException):
    def __init__(self, normandy_slug, message="Experiment has high population."):
        super().__init__(f"{normandy_slug} -> {message}")


class ExplicitSkipException(ValidationException):
    def __init__(self, normandy_slug, message="Experiment is configured with skip=true."):
        super().__init__(f"{normandy_slug} -> {message}")


class RolloutSkipException(ValidationException):
    def __init__(self, normandy_slug, message="Experiment is a rollout and will not be analyzed."):
        super().__init__(f"{normandy_slug} -> {message}")


class InvalidConfigurationException(Exception):
    """Exception thrown when experiment configuration is invalid."""

    def __init__(self, message):
        super().__init__(message)


class StatisticComputationException(Exception):
    """Exception thrown when statistic of a metric could not get computed."""

    def __init__(self, message):
        super().__init__(message)


class UnexpectedKeyConfigurationException(InvalidConfigurationException):
    pass


class SegmentsConfigurationException(InvalidConfigurationException):
    pass


class MetricsConfigurationException(InvalidConfigurationException):
    pass
