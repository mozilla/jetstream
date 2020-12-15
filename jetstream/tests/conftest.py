import datetime as dt

import pytest
import pytz

from jetstream.experimenter import Branch, Experiment


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
            start_date=dt.datetime(2019, 12, 1, tzinfo=pytz.utc),
            end_date=dt.datetime(2020, 3, 1, tzinfo=pytz.utc),
            proposed_enrollment=7,
            branches=[Branch(slug="a", ratio=1), Branch(slug="b", ratio=1)],
            probe_sets=[],
            normandy_slug="normandy-test-slug",
            reference_branch="b",
            is_high_population=False,
        ),
        Experiment(
            experimenter_slug="test_slug",
            type="addon",
            status="Complete",
            start_date=dt.datetime(2019, 12, 1, tzinfo=pytz.utc),
            end_date=dt.datetime(2020, 3, 1, tzinfo=pytz.utc),
            proposed_enrollment=0,
            branches=[],
            probe_sets=[],
            normandy_slug=None,
            reference_branch=None,
            is_high_population=False,
        ),
        Experiment(
            experimenter_slug="test_slug",
            type="pref",
            status="Live",
            start_date=dt.datetime(2019, 12, 1, tzinfo=pytz.utc),
            end_date=dt.datetime(2020, 3, 1, tzinfo=pytz.utc),
            proposed_enrollment=7,
            branches=[],
            probe_sets=[],
            normandy_slug="normandy-test-slug",
            reference_branch=None,
            is_high_population=False,
        ),
        Experiment(
            experimenter_slug="test_slug",
            type="pref",
            status="Live",
            start_date=dt.datetime(2019, 12, 1, tzinfo=pytz.utc),
            end_date=dt.datetime(2020, 3, 1, tzinfo=pytz.utc),
            proposed_enrollment=7,
            branches=[],
            probe_sets=[],
            normandy_slug="normandy-test-slug",
            reference_branch=None,
            is_high_population=True,
        ),
        Experiment(
            experimenter_slug="test_slug",
            type="pref",
            status="Complete",
            start_date=dt.datetime(2019, 12, 1, tzinfo=pytz.utc),
            end_date=dt.datetime(2020, 3, 1, tzinfo=pytz.utc),
            proposed_enrollment=7,
            branches=[Branch(slug="a", ratio=1), Branch(slug="b", ratio=1)],
            probe_sets=["pinned_tabs"],
            normandy_slug="normandy-test-slug",
            reference_branch="b",
            is_high_population=False,
        ),
    ]
