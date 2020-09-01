import datetime as dt

import pytest
import pytz

from jetstream.experimenter import Experiment, Branch


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        help="Run integration tests",
    )


@pytest.fixture
def experiments():
    return [
        Experiment(
            experimenter_slug="test_slug",
            type="pref",
            status="Complete",
            active=False,
            start_date=dt.datetime(2019, 12, 1, tzinfo=pytz.utc),
            end_date=dt.datetime(2020, 3, 1, tzinfo=pytz.utc),
            proposed_enrollment=7,
            branches=[Branch(slug="a", ratio=1), Branch(slug="b", ratio=1)],
            normandy_slug="normandy-test-slug",
            features=[],
            reference_branch="b",
        ),
        Experiment(
            experimenter_slug="test_slug",
            type="addon",
            status="Complete",
            active=False,
            start_date=dt.datetime(2019, 12, 1, tzinfo=pytz.utc),
            end_date=dt.datetime(2020, 3, 1, tzinfo=pytz.utc),
            proposed_enrollment=0,
            branches=[],
            features=[],
            normandy_slug=None,
            reference_branch=None,
        ),
        Experiment(
            experimenter_slug="test_slug",
            type="pref",
            status="Live",
            active=True,
            start_date=dt.datetime(2019, 12, 1, tzinfo=pytz.utc),
            end_date=dt.datetime(2020, 3, 1, tzinfo=pytz.utc),
            proposed_enrollment=7,
            branches=[],
            features=[],
            normandy_slug="normandy-test-slug",
            reference_branch=None,
        ),
    ]
