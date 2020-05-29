import datetime as dt

import pytest
import pytz

from pensieve.experimenter import Experiment


def pytest_addoption(parser):
    parser.addoption(
        "--integration", action="store_true", help="Run integration tests",
    )


@pytest.fixture
def experiments():
    return [
        Experiment(
            slug="test_slug",
            type="pref",
            status="Complete",
            start_date=dt.datetime(2019, 12, 1, tzinfo=pytz.utc),
            end_date=dt.datetime(2020, 3, 1, tzinfo=pytz.utc),
            proposed_enrollment=7,
            variants=[],
            normandy_slug="normandy-test-slug",
        ),
        Experiment(
            slug="test_slug",
            type="addon",
            status="Complete",
            start_date=dt.datetime(2019, 12, 1, tzinfo=pytz.utc),
            end_date=dt.datetime(2020, 3, 1, tzinfo=pytz.utc),
            proposed_enrollment=0,
            variants=[],
            normandy_slug=None,
        ),
    ]
