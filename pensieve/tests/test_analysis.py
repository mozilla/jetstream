import datetime as dt
from datetime import timedelta
import json
import google
import mock
import os
import pytz
import pytest

from mozanalysis.metrics import Metric, DataSource, agg_sum

from pensieve.analysis import Analysis, BigQueryClient
from pensieve.experimenter import Experiment, Variant

test_project = "pensieve-integration-test"
test_dataset = "test"
static_dataset = (
    "test_data"  # contains the tables filled with test data required to run metrics analysis
)


@pytest.fixture
def experiments():
    return [
        Experiment(
            slug="test_slug",
            type="pref",
            start_date=dt.datetime(2019, 12, 1, tzinfo=pytz.utc),
            end_date=dt.datetime(2020, 3, 1, tzinfo=pytz.utc),
            proposed_enrollment=7,
            variants=[],
            normandy_slug="normandy-test-slug",
        ),
        Experiment(
            slug="test_slug",
            type="addon",
            start_date=dt.datetime(2019, 12, 1, tzinfo=pytz.utc),
            end_date=dt.datetime(2020, 3, 1, tzinfo=pytz.utc),
            proposed_enrollment=0,
            variants=[],
            normandy_slug=None,
        ),
    ]


def test_get_timelimits_if_ready(experiments):
    analysis = Analysis("test", "test", experiments[0])
    date = dt.datetime(2019, 12, 1, tzinfo=pytz.utc) + timedelta(days=13)
    assert analysis._get_timelimits_if_ready(date)

    date = dt.datetime(2019, 12, 1, tzinfo=pytz.utc) + timedelta(0)
    assert analysis._get_timelimits_if_ready(date) is None

    date = dt.datetime(2019, 12, 1, tzinfo=pytz.utc) + timedelta(2)
    assert analysis._get_timelimits_if_ready(date) is None


def test_regression_20200320():
    experiment_json = r"""
        {
          "experiment_url": "https://experimenter.services.mozilla.com/experiments/impact-of-level-2-etp-on-a-custom-distribution/",
          "type": "pref",
          "name": "Impact of Level 2 ETP on a Custom Distribution",
          "slug": "impact-of-level-2-etp-on-a-custom-distribution",
          "public_name": "Impact of Level 2 ETP",
          "status": "Live",
          "start_date": 1580169600000,
          "end_date": 1595721600000,
          "proposed_start_date": 1580169600000,
          "proposed_enrollment": null,
          "proposed_duration": 180,
          "normandy_slug": "pref-impact-of-level-2-etp-on-a-custom-distribution-release-72-80-bug-1607493",
          "normandy_id": 906,
          "other_normandy_ids": [],
          "variants": [
            {
              "description": "",
              "is_control": true,
              "name": "treatment",
              "ratio": 100,
              "slug": "treatment",
              "value": "true",
              "addon_release_url": null,
              "preferences": []
            }
          ]
        }
    """  # noqa
    experiment = Experiment.from_dict(json.loads(experiment_json))
    analysis = Analysis("test", "test", experiment)
    analysis.run(current_date=dt.datetime(2020, 3, 19), dry_run=True)


def test_regression_20200316():
    experiment_json = r"""
    {
      "experiment_url": "https://blah/experiments/search-tips-aka-nudges/",
      "type": "addon",
      "name": "Search Tips aka Nudges",
      "slug": "search-tips-aka-nudges",
      "public_name": "Search Tips",
      "public_description": "Search Tips are designed to increase engagement with the QuantumBar.",
      "status": "Live",
      "countries": [],
      "platform": "All Platforms",
      "start_date": 1578960000000,
      "end_date": 1584921600000,
      "population": "2% of Release Firefox 72.0 to 74.0",
      "population_percent": "2.0000",
      "firefox_channel": "Release",
      "firefox_min_version": "72.0",
      "firefox_max_version": "74.0",
      "addon_experiment_id": null,
      "addon_release_url": "https://bugzilla.mozilla.org/attachment.cgi?id=9120542",
      "pref_branch": null,
      "pref_name": null,
      "pref_type": null,
      "proposed_start_date": 1578960000000,
      "proposed_enrollment": 21,
      "proposed_duration": 69,
      "normandy_slug": "addon-search-tips-aka-nudges-release-72-74-bug-1603564",
      "normandy_id": 902,
      "other_normandy_ids": [],
      "variants": [
        {
          "description": "Standard address bar experience",
          "is_control": false,
          "name": "control",
          "ratio": 50,
          "slug": "control",
          "value": null,
          "addon_release_url": null,
          "preferences": []
        },
        {
          "description": "",
          "is_control": true,
          "name": "treatment",
          "ratio": 50,
          "slug": "treatment",
          "value": null,
          "addon_release_url": null,
          "preferences": []
        }
      ]
    }
    """
    experiment = Experiment.from_dict(json.loads(experiment_json))
    analysis = Analysis("test", "test", experiment)
    analysis.run(current_date=dt.datetime(2020, 3, 16), dry_run=True)


def bq_execute(cls, *args, **kwargs):
    # to use the test project and dataset, we need to change the SQL query generated by mozanalysis
    query = args[0].replace("moz-fx-data-shared-prod", test_project)
    query = query.replace("telemetry", "test_data")

    dataset = google.cloud.bigquery.dataset.DatasetReference.from_string(
        cls.dataset, default_project=cls.project,
    )

    destination_table = None

    if len(args) > 1:
        destination_table = args[1]

    kwargs = {}
    if destination_table:
        kwargs["destination"] = dataset.table(destination_table)
        kwargs["write_disposition"] = google.cloud.bigquery.job.WriteDisposition.WRITE_TRUNCATE
    config = google.cloud.bigquery.job.QueryJobConfig(default_dataset=dataset, **kwargs)
    job = cls.client.query(query, config)
    # block on result
    return job.result(max_results=1)

@pytest.mark.integration
def test_metrics():
    experiment = Experiment(
        slug="test-experiment",
        type="rollout",
        start_date=dt.datetime(2020, 3, 30, tzinfo=pytz.utc),
        end_date=dt.datetime(2020, 6, 1, tzinfo=pytz.utc),
        proposed_enrollment=7,
        variants=[
            Variant(is_control=False, slug="branch1", ratio=0.5),
            Variant(is_control=True, slug="branch2", ratio=0.5),
        ],
        normandy_slug="test-experiment",
    )

    # todo create class and move there
    project_id = os.getenv("GOOGLE_PROJECT_ID")

    if project_id is None:
        print("GOOGLE_PROJECT_ID is not set.")
        assert False

    with mock.patch.object(BigQueryClient, 'execute', new=bq_execute):
        analysis = Analysis(project_id, test_dataset, experiment)

        test_clients_daily = DataSource(
            name="clients_daily", from_expr=f"`{project_id}.test_data.clients_daily`",
        )

        test_active_hours = Metric(
            name="active_hours", data_source=test_clients_daily, select_expr=agg_sum("active_hours_sum")
        )

        analysis.STANDARD_METRICS = [test_active_hours]

        analysis.run(current_date=dt.datetime(2020, 4, 12), dry_run=False)

    client = google.cloud.bigquery.client.Client(project_id)
    query_job = client.query(f"""
        SELECT
          COUNT(*) as count
        FROM `{project_id}.{test_dataset}.test_experiment_week_1`
    """)

    result = query_job.result()

    for row in result:
        assert row["count"] == 2

    # todo: remove tables that are generated, but leave non-generated

