from datetime import date
from pensieve import cli


class TestCli:
    def test_inclusive_date_range(self):
        start_date = date(2020, 5, 1)
        end_date = date(2020, 5, 1)
        date_range = list(cli.inclusive_date_range(start_date, end_date))
        assert len(date_range) == 1
        assert date_range[0] == date(2020, 5, 1)

        start_date = date(2020, 5, 1)
        end_date = date(2020, 5, 5)
        date_range = list(cli.inclusive_date_range(start_date, end_date))
        assert len(date_range) == 5
        assert date_range[0] == date(2020, 5, 1)
        assert date_range[4] == date(2020, 5, 5)
