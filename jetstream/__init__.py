import enum
import re


class AnalysisPeriod(enum.Enum):
    DAY = "day"
    WEEK = "week"
    DAYS_28 = "days28"
    OVERALL = "overall"

    @property
    def mozanalysis_label(self) -> str:
        d = {"day": "daily", "week": "weekly", "days28": "28_day", "overall": "overall"}
        return d[self.value]

    @property
    def table_suffix(self) -> str:
        d = {"day": "daily", "week": "weekly", "days28": "days28", "overall": "overall"}
        return d[self.value]


def bq_normalize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)
