import datetime as dt
from datetime import timedelta
import pytz
import pytest

from pensieve.analysis import Analysis
from pensieve.experimenter import Experiment


@pytest.fixture
def experiments():
    return [
        Experiment(
            "test_slug",
            dt.datetime(2019, 12, 1, tzinfo=pytz.utc),
            dt.datetime(2020, 3, 1, tzinfo=pytz.utc),
            [],
            "normandy-test-slug",
        ),
        Experiment(
            "test_slug",
            dt.datetime(2019, 12, 1, tzinfo=pytz.utc),
            dt.datetime(2020, 3, 1, tzinfo=pytz.utc),
            [],
            None,
        ),
    ]


def test_should_analyse_experiment(experiments):
    analysis = Analysis("test", "test")
    date = dt.datetime(2019, 12, 1, tzinfo=pytz.utc) + timedelta(analysis.ANALYSIS_PERIOD)
    assert analysis._should_analyse_experiment(experiments[0], date) is True

    date = dt.datetime(2019, 12, 1, tzinfo=pytz.utc) + timedelta(0)
    assert analysis._should_analyse_experiment(experiments[0], date) is False

    date = dt.datetime(2019, 12, 1, tzinfo=pytz.utc) + timedelta(2)
    assert 2 != analysis.ANALYSIS_PERIOD
    assert analysis._should_analyse_experiment(experiments[0], date) is False


def test_sanitize_table_name_for_bq(experiments):
    analysis = Analysis("test", "test")
    assert (
        analysis._sanitize_table_name_for_bq(experiments[0].normandy_slug) == "normandy_test_slug"
    )
