import datetime as dt
from unittest import mock
from unittest.mock import MagicMock

import pytest
import pytz
from google.cloud import artifactregistry
from google.protobuf import timestamp_pb2
from metric_config_parser.experiment import Branch, Experiment
from pytz import UTC


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
            normandy_slug="normandy-test-slug",
            reference_branch="b",
            is_high_population=False,
            app_name="firefox_desktop",
            app_id="firefox-desktop",
        ),
        Experiment(
            experimenter_slug="test_slug",
            type="addon",
            status="Complete",
            start_date=dt.datetime(2019, 12, 1, tzinfo=pytz.utc),
            end_date=dt.datetime(2020, 3, 1, tzinfo=pytz.utc),
            proposed_enrollment=0,
            branches=[],
            normandy_slug=None,
            reference_branch=None,
            is_high_population=False,
            app_name="firefox_desktop",
            app_id="firefox-desktop",
        ),
        Experiment(
            experimenter_slug="test_slug",
            type="pref",
            status="Live",
            start_date=dt.datetime(2019, 12, 1, tzinfo=pytz.utc),
            end_date=dt.datetime(2020, 3, 1, tzinfo=pytz.utc),
            proposed_enrollment=7,
            branches=[],
            normandy_slug="normandy-test-slug",
            reference_branch=None,
            is_high_population=False,
            app_name="firefox_desktop",
            app_id="firefox-desktop",
        ),
        Experiment(
            experimenter_slug="test_slug",
            type="pref",
            status="Live",
            start_date=dt.datetime(2019, 12, 1, tzinfo=pytz.utc),
            end_date=dt.datetime(2020, 3, 1, tzinfo=pytz.utc),
            proposed_enrollment=7,
            branches=[],
            normandy_slug="normandy-test-slug",
            reference_branch=None,
            is_high_population=True,
            app_name="firefox_desktop",
            app_id="firefox-desktop",
        ),
        Experiment(
            experimenter_slug="test_slug",
            type="pref",
            status="Complete",
            start_date=dt.datetime(2019, 12, 1, tzinfo=pytz.utc),
            end_date=dt.datetime(2020, 3, 1, tzinfo=pytz.utc),
            proposed_enrollment=7,
            branches=[Branch(slug="a", ratio=1), Branch(slug="b", ratio=1)],
            normandy_slug="normandy-test-slug",
            reference_branch="b",
            is_high_population=False,
            app_name="firefox_desktop",
            app_id="firefox-desktop",
        ),
        Experiment(
            experimenter_slug="test_slug",
            type="pref",
            status="Live",
            start_date=dt.datetime(2019, 12, 1, tzinfo=pytz.utc),
            end_date=dt.datetime(2020, 3, 1, tzinfo=pytz.utc),
            proposed_enrollment=7,
            branches=[],
            normandy_slug="normandy-test-slug",
            reference_branch=None,
            is_high_population=True,
            outcomes=["performance", "tastiness"],
            app_name="firefox_desktop",
            app_id="firefox-desktop",
        ),
        Experiment(
            experimenter_slug="test_slug",
            type="pref",
            status="Live",
            start_date=dt.datetime(2019, 12, 1, tzinfo=pytz.utc),
            end_date=dt.datetime(2020, 3, 1, tzinfo=pytz.utc),
            proposed_enrollment=7,
            branches=[],
            normandy_slug="normandy-test-slug",
            reference_branch=None,
            is_high_population=True,
            outcomes=["parameterized"],
            app_name="firefox_desktop",
            app_id="firefox-desktop",
        ),
        Experiment(
            experimenter_slug="test_slug",
            type="pref",
            status="Live",
            start_date=dt.datetime(2019, 12, 1, tzinfo=pytz.utc),
            end_date=dt.datetime(2020, 3, 1, tzinfo=pytz.utc),
            proposed_enrollment=7,
            branches=[],
            normandy_slug="normandy-test-slug",
            reference_branch=None,
            is_high_population=True,
            outcomes=["parameterised_distinct_by_branch_config"],
            app_name="firefox_desktop",
            app_id="firefox-desktop",
        ),
        Experiment(
            experimenter_slug=None,
            type="pref",
            status="Live",
            start_date=dt.datetime(2019, 12, 1, tzinfo=pytz.utc),
            end_date=None,
            proposed_enrollment=7,
            branches=[],
            normandy_slug="normandy-test-slug",
            reference_branch=None,
            is_high_population=False,
            app_name="firefox_desktop",
            app_id="firefox-desktop",
            enrollment_end_date=dt.datetime(2019, 12, 8, tzinfo=pytz.utc),
            is_enrollment_paused=False,
        ),
        Experiment(
            experimenter_slug=None,
            type="pref",
            status="Live",
            start_date=dt.datetime(2019, 12, 1, tzinfo=pytz.utc),
            end_date=None,
            proposed_enrollment=7,
            branches=[],
            normandy_slug="normandy-test-slug",
            reference_branch=None,
            is_high_population=False,
            app_name="firefox_desktop",
            app_id="firefox-desktop",
            enrollment_end_date=dt.datetime(2019, 12, 8, tzinfo=pytz.utc),
            is_enrollment_paused=True,
        ),
    ]


@pytest.fixture
def fenix_experiments():
    return [
        Experiment(
            experimenter_slug="my_fenix_experiment",
            normandy_slug="my_fenix_experiment",
            type="v6",
            status="Live",
            branches=[Branch(slug="foo", ratio=1), Branch(slug="bar", ratio=1)],
            start_date=dt.datetime(2020, 1, 1, tzinfo=pytz.UTC),
            end_date=dt.datetime(2020, 10, 10, tzinfo=pytz.UTC),
            proposed_enrollment=7,
            reference_branch="foo",
            app_name="fenix",
            app_id="org.mozilla.firefox",
            is_high_population=False,
        ),
        Experiment(
            experimenter_slug="my_fenix_nightly_experiment",
            normandy_slug="my_fenix_nightly_experiment",
            type="v6",
            status="Live",
            branches=[Branch(slug="foo", ratio=1), Branch(slug="bar", ratio=1)],
            start_date=dt.datetime(2020, 1, 1, tzinfo=pytz.UTC),
            end_date=dt.datetime(2020, 10, 10, tzinfo=pytz.UTC),
            proposed_enrollment=7,
            reference_branch="foo",
            app_name="fenix",
            app_id="org.mozilla.fenix",
            is_high_population=False,
        ),
    ]


@pytest.fixture
def firefox_ios_experiments():
    return [
        Experiment(
            experimenter_slug="my_ios_experiment",
            normandy_slug="my_ios_experiment",
            type="v6",
            status="Live",
            branches=[Branch(slug="foo", ratio=1), Branch(slug="bar", ratio=1)],
            start_date=dt.datetime(2020, 1, 1, tzinfo=pytz.UTC),
            end_date=dt.datetime(2020, 10, 10, tzinfo=pytz.UTC),
            proposed_enrollment=7,
            reference_branch="foo",
            app_name="firefox_ios",
            app_id="org.mozilla.ios.FirefoxBeta",
            is_high_population=False,
        )
    ]


@pytest.fixture
def klar_android_experiments():
    return [
        Experiment(
            experimenter_slug="my_klar_experiment",
            normandy_slug="my_klar_experiment",
            type="v6",
            status="Live",
            branches=[Branch(slug="foo", ratio=1), Branch(slug="bar", ratio=1)],
            start_date=dt.datetime(2020, 1, 1, tzinfo=pytz.UTC),
            end_date=dt.datetime(2020, 10, 10, tzinfo=pytz.UTC),
            proposed_enrollment=7,
            reference_branch="foo",
            app_name="klar_android",
            app_id="org.mozilla.klar",
            is_high_population=False,
        )
    ]


@pytest.fixture
def focus_android_experiments():
    return [
        Experiment(
            experimenter_slug="my_focus_experiment",
            normandy_slug="my_focus_experiment",
            type="v6",
            status="Live",
            branches=[Branch(slug="foo", ratio=1), Branch(slug="bar", ratio=1)],
            start_date=dt.datetime(2020, 1, 1, tzinfo=pytz.UTC),
            end_date=dt.datetime(2020, 10, 10, tzinfo=pytz.UTC),
            proposed_enrollment=7,
            reference_branch="foo",
            app_name="focus_android",
            app_id="org.mozilla.focus",
            is_high_population=False,
        )
    ]


@pytest.fixture
def docker_images():
    return [
        artifactregistry.DockerImage(
            name="projects/moz-fx-data-experiments/locations/us/repositories/"
            + "gcr.io/dockerImages/jetstream@sha256:8c766a",
            update_time=timestamp_pb2.Timestamp(seconds=1672578000),  # 2023-01-01 01:00 am
        ),
        artifactregistry.DockerImage(
            name="projects/moz-fx-data-experiments/locations/us/repositories/"
            + "gcr.io/dockerImages/unrelated@sha256:aaaaa",
            update_time=timestamp_pb2.Timestamp(seconds=1672578000),  # 2023-01-01 01:00 am
        ),
        artifactregistry.DockerImage(
            name="projects/moz-fx-data-experiments/locations/us/repositories/"
            + "gcr.io/dockerImages/jetstream@sha256:xxxxx",
            update_time=timestamp_pb2.Timestamp(seconds=1677675600),  # 2023-03-01 01:00 am
        ),
    ]


@pytest.fixture(scope="session", autouse=True)
def bq_client_mock():
    with mock.patch("jetstream.cli.BigQueryClient") as fixture:
        bigquery_mock_client = MagicMock()
        bigquery_mock_client.experiment_table_first_updated.return_value = dt.datetime(
            2023, 1, 1, tzinfo=UTC
        )
        fixture.return_value = bigquery_mock_client
        yield fixture
