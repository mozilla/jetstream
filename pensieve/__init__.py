import enum


class AnalysisPeriod(enum.Enum):
    DAY = "day"
    WEEK = "week"
    OVERALL = "overall"
    ENROLLMENT = "enrollment"  # The day of enrollment; useful for some messaging experiments

    @property
    def adjective(self) -> str:
        d = {"day": "daily", "week": "weekly"}
        return d.get(self.value, self.value)
