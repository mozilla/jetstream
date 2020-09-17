class NoSlugException(Exception):
    def __init__(self, message="Experiment has no slug"):
        super().__init__(message)


class NoEnrollmentPeriodException(Exception):
    def __init__(self, normandy_slug, message="Experiment has no enrollment period"):
        super().__init__(f"{normandy_slug} -> {message}")


class NoStartDateException(Exception):
    def __init__(self, normandy_slug, message="Experiment has no start date."):
        super().__init__(f"{normandy_slug} -> {message}")


class EndedException(Exception):
    def __init__(self, normandy_slug, message="Experiment has already ended."):
        super().__init__(f"{normandy_slug} -> {message}")


class EnrollmentLongerThanAnalysisException(Exception):
    def __init__(self, normandy_slug, message="Enrollment period is longer than analysis dates."):
        super().__init__(f"{normandy_slug} -> {message}")


class HighPopulationException(Exception):
    def __init__(self, normandy_slug, message="Experiment has high population."):
        super().__init__(f"{normandy_slug} -> {message}")
