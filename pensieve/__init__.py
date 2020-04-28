import enum


class AnalysisPeriod(enum.Enum):
    DAY = "day"
    WEEK = "week"
    OVERALL = "overall"

    @property
    def adjective(self) -> str:
        d = {"day": "daily", "week": "weekly", "overall": "overall"}
        return d[self.value]
