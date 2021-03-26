import enum
import re


class AnalysisPeriod(enum.Enum):
    DAY = "day"
    WEEK = "week"
    DAYS_28 = "days_28"
    OVERALL = "overall"

    @property
    def adjective(self) -> str:
        d = {"day": "daily", "week": "weekly", "days_28": "days_28", "overall": "overall"}
        return d[self.value]


def bq_normalize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)
