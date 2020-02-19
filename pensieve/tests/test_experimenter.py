import datetime as dt
import json
import pytz
from unittest.mock import Mock

import pytest

from pensieve.experimenter import ExperimentCollection, Experiment, Variant

EXPERIMENTER_FIXTURE = r"""
[
  {
    "experiment_url": "https://experimenter.services.mozilla.com/experiments/search-topsites/",
    "type": "addon",
    "name": "Activity Stream Search TopSites",
    "slug": "search-topsites",
    "public_name": "TopSites for Search",
    "public_description": "We believe we can deliver an enhanced product experience by exposing these Topsites in a new context, allowing users to navigate even more quickly and easily than they can today.",
    "status": "Complete",
    "client_matching": "Prefs: Exclude users with the following prefs:\r\n\r\nbrowser.newtabpage.activity-stream.feeds.topsites = false\r\nbrowser.privatebrowsing.autostart = true\r\n\r\nExperiments:\r\n\r\nAny additional filters:",
    "locales": [],
    "countries": [],
    "platform": "All Platforms",
    "start_date": 1568678400000,
    "end_date": 1574121600000,
    "population": "0.9% of Release Firefox 69.0",
    "population_percent": "0.9000",
    "firefox_channel": "Release",
    "firefox_min_version": "69.0",
    "firefox_max_version": null,
    "addon_experiment_id": "mythmon says this isn't necessary for new-style experiments like this one",
    "addon_release_url": "https://bugzilla.mozilla.org/attachment.cgi?id=9091835",
    "pref_branch": null,
    "pref_name": null,
    "pref_type": null,
    "proposed_start_date": 1568592000000,
    "proposed_enrollment": 14,
    "proposed_duration": 60,
    "variants": [
      {
        "description": "primary branch displaying Top Sites before the user starts typing",
        "is_control": false,
        "name": "treatment",
        "ratio": 50,
        "slug": "treatment",
        "value": "1",
        "addon_release_url": null,
        "preferences": []
      },
      {
        "description": "Standard address bar experience",
        "is_control": false,
        "name": "control",
        "ratio": 50,
        "slug": "control",
        "value": "0",
        "addon_release_url": null,
        "preferences": []
      }
    ],
    "changes": [
      {
        "changed_on": "2019-08-07T16:02:43.538514Z",
        "pretty_status": "Created Delivery",
        "new_status": "Draft",
        "old_status": null
      },
      {
        "changed_on": "2019-08-07T20:52:06.859236Z",
        "pretty_status": "Edited Delivery",
        "new_status": "Draft",
        "old_status": "Draft"
      }
    ]
  },
  {
    "experiment_url": "https://experimenter.services.mozilla.com/experiments/impact-of-level-2-etp-on-a-custom-distribution/",
    "type": "pref",
    "name": "Impact of Level 2 ETP on a Custom Distribution",
    "slug": "impact-of-level-2-etp-on-a-custom-distribution",
    "public_name": "Impact of Level 2 ETP",
    "public_description": "This study enables ETP for a known population to observe impacts on usage and revenue",
    "status": "Live",
    "client_matching": "Prefs: n/a\r\n\r\nExperiments: none (different G plugin means we'll ignore the main ETP Level 2 experiment)\r\n\r\nAny additional filters:\r\nnormandy.distribution must be one of the following two options:\r\n* isltd-g-aura-001\r\n* isltd-g-001\r\n    \r\nLess than 200k MAU should be targeted with this filtering.",
    "locales": [],
    "countries": [],
    "platform": "All Platforms",
    "start_date": null,
    "end_date": null,
    "population": "100% of Release Firefox 72.0 to 80.0",
    "population_percent": "100.0000",
    "firefox_channel": "Release",
    "firefox_min_version": "72.0",
    "firefox_max_version": "80.0",
    "addon_experiment_id": null,
    "addon_release_url": null,
    "pref_branch": "default",
    "pref_name": "privacy.annotate_channels.strict_list.enabled",
    "pref_type": "boolean",
    "proposed_start_date": 1580169600000,
    "proposed_enrollment": null,
    "proposed_duration": 180,
    "variants": [
      {
        "description": "this is actually the treatment branch (see background links or ask mconnor for clarity)",
        "is_control": true,
        "name": "treatment",
        "ratio": 100,
        "slug": "treatment",
        "value": "true",
        "addon_release_url": null,
        "preferences": []
      }
    ],
    "changes": [
      {
        "changed_on": "2020-01-07T15:35:19.880806Z",
        "pretty_status": "Created Delivery",
        "new_status": "Draft",
        "old_status": null
      },
      {
        "changed_on": "2020-01-07T15:38:15.351745Z",
        "pretty_status": "Edited Delivery",
        "new_status": "Draft",
        "old_status": "Draft"
      }
    ]
  }
]
"""  # noqa


@pytest.fixture
def mock_session():
    session = Mock()
    session.get.return_value.json.return_value = json.loads(EXPERIMENTER_FIXTURE)
    return session


@pytest.fixture
def experiment_collection(mock_session):
    return ExperimentCollection.from_experimenter(mock_session)


def test_from_experimenter(mock_session):
    collection = ExperimentCollection.from_experimenter(mock_session)
    mock_session.get.assert_called_once_with(ExperimentCollection.EXPERIMENTER_API_URL)
    assert len(collection.experiments) == 2
    assert isinstance(collection.experiments[0], Experiment)
    assert isinstance(collection.experiments[0].variants[0], Variant)
    assert len(collection.experiments[0].variants) == 2
    assert collection.experiments[0].start_date > dt.datetime(2019, 1, 1, tzinfo=pytz.utc)


def test_started_since(experiment_collection):
    recent = experiment_collection.started_since(dt.datetime(2019, 1, 1, tzinfo=pytz.utc))
    assert isinstance(recent, ExperimentCollection)
    assert len(recent.experiments) > 0
