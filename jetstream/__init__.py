import enum
import re


class AnalysisPeriod(enum.Enum):
    DAY = "day"
    WEEK = "week"
    OVERALL = "overall"

    @property
    def adjective(self) -> str:
        d = {"day": "daily", "week": "weekly", "overall": "overall"}
        return d[self.value]


def bq_normalize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)
