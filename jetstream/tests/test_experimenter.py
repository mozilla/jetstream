import datetime as dt
import json
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import jsonschema
import pytest
import pytz
from metric_config_parser.experiment import Branch, Experiment

from jetstream.experimenter import (
    ExperimentCollection,
    ExperimentV1,
    ExperimentV6,
    Outcome,
    Variant,
)

EXPERIMENTER_FIXTURE_V1 = r"""
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
    "normandy_slug": "addon-activity-stream-search-topsites-release-69-1576277",
    "normandy_id": null,
    "other_normandy_ids": null,
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
        "is_control": true,
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
  },
  {
    "experiment_url":"https://experimenter.services.mozilla.com/experiments/doh-us-engagement-study-v2/",
    "type":"pref",
    "name":"DoH US Engagement Study V2",
    "slug":"doh-us-engagement-study-v2",
    "public_name":"DNS over HTTPS US Rollout",
    "public_description":"This Firefox experiment will measure the impact on user engagement and retention when DNS over HTTPS is rolled out in the United States. Users who are part of the study will receive a notification before DNS over HTTPS is enabled. Set network.trr.mode to ‘5’ in about:config to permanently disable DoH. This experiment does not collect personally-identifiable information, DNS queries, or answers.",
    "status":"Complete",
    "client_matching":"- 69.0.3 or higher (including 70.*)\r\n- Enrollment should be sticky over country\r\n- System addon doh-rollout@mozilla.org is installed\r\n\r\nThe staged rollout will want to avoid this experiment https://experimenter.services.mozilla.com/experiments/doh-us-staged-rollout-to-all-us-desktop-users/edit/",
    "locales":[],
    "platform":"All Windows",
    "start_date":1572393600000.0,
    "end_date":1576454400000.0,
    "population":"1% of Release Firefox 69.0 to 71.0",
    "population_percent":"1.0000",
    "firefox_channel":"Release",
    "firefox_min_version":"69.0",
    "firefox_max_version":"71.0",
    "addon_experiment_id":"None",
    "addon_release_url":"None",
    "normandy_slug": "pref-doh-us-engagement-study-v2-release-69-71-bug-1590831",
    "pref_branch":"default",
    "pref_name":"doh-rollout.enabled",
    "pref_type":"boolean",
    "proposed_start_date":1572307200000.0,
    "proposed_enrollment":7,
    "proposed_duration":69,
    "variants":[],
    "changes":[]
  }
]
"""  # noqa

EXPERIMENTER_FIXTURE_V6 = r"""
[
{
  "schemaVersion": "1",
  "application": "firefox-desktop",
  "id":"bug-1629000-rapid-testing-rapido-intake-1-release-79",
  "slug":"bug-1629098-rapid-please-reject-me-beta-86",
  "userFacingName":"",
  "userFacingDescription":" This is an empty CFR A/A experiment. The A/A experiment is being run to test the automation, effectiveness, and accuracy of the rapid experiments platform.\n    The experiment is an internal test, and Firefox users will not see any noticeable change and there will be no user impact.",
  "isEnrollmentPaused":false,
  "probeSets":[],
  "proposedEnrollment":7,
  "bucketConfig": {
    "randomizationUnit":"userId",
    "namespace":"bug-1629098-rapid-please-reject-me-beta-86",
    "start":0,
    "count":100,
    "total":10000 
  },
  "startDate":"2020-07-29",
  "endDate":null,
  "enrollmentEndDate":"2020-08-05",
  "branches":[{
      "slug":"treatment",
      "ratio":1,
      "feature": {"featureId": "foo", "enabled": false, "value": null}    
    },
    {
      "slug":"control",
      "ratio":1,
      "feature": {"featureId": "foo", "enabled": false, "value": null}    
    }
  ],
  "referenceBranch":"control",
  "filter_expression":"env.version|versionCompare('86.0') >= 0",
  "targeting":"[userId, \"bug-1629098-rapid-please-reject-me-beta-86\"]|bucketSample(0, 100, 10000) && localeLanguageCode == 'en' && region == 'US' && browserSettings.update.channel == 'beta'"
},
{
  "schemaVersion": "1",
  "application": "firefox-desktop",   
  "id":"bug-1629000-rapid-testing-rapido-intake-1-release-79",
    "slug":"bug-1629000-rapid-testing-rapido-intake-1-release-79",
    "userFacingName":"testing rapido intake 1",
    "userFacingDescription":" This is an empty CFR A/A experiment. The A/A experiment is being run to test the automation, effectiveness, and accuracy of the rapid experiments platform.\n    The experiment is an internal test, and Firefox users will not see any noticeable change and there will be no user impact.",
    "isEnrollmentPaused":false,
    "probeSets":[
      "fake_feature"
    ],
    "proposedEnrollment":14,
    "proposedDuration":30,
    "bucketConfig":{
      "randomizationUnit":"normandy_id",
      "namespace":"",
      "start":0,
      "count":0,
      "total":10000
    },
    "startDate":"2020-07-28",
    "endDate":null,
    "branches":[{
      "slug":"treatment",
      "ratio":1,
      "feature": {"featureId": "foo", "enabled": false, "value": null}     
      },
      {
        "slug":"control",
        "ratio":1,
        "feature": {"featureId": "foo", "enabled": false, "value": null}   
    }],
  "referenceBranch":"control",
  "filter_expression":"env.version|versionCompare('79.0') >= 0",
  "targeting":""
},
{   
  "id":null,
    "slug":null,
    "userFacingName":"some invalid experiment",
    "userFacingDescription":" This is an empty CFR A/A experiment. The A/A experiment is being run to test the automation, effectiveness, and accuracy of the rapid experiments platform.\n    The experiment is an internal test, and Firefox users will not see any noticeable change and there will be no user impact.",
    "isEnrollmentPaused":false,
    "proposedEnrollment":14,
    "bucketConfig":{
      "randomizationUnit":"normandy_id",
      "namespace":"",
      "start":0,
      "count":0,
      "total":10000
    },
    "startDate":null,
    "endDate":null,
    "branches":[],
  "referenceBranch":"control",
  "enabled":true,
  "targeting":null
}
]
"""  # noqa

FENIX_EXPERIMENT_FIXTURE = """
{
  "schemaVersion": "1.4.0",
  "slug": "fenix-bookmark-list-icon",
  "id": "fenix-bookmark-list-icon",
  "arguments": {},
  "application": "org.mozilla.fenix",
  "appName": "fenix",
  "appId": "org.mozilla.fenix",
  "channel": "nightly",
  "userFacingName": "Fenix Bookmark List Icon",
  "userFacingDescription": "If we make the save-bookmark and access-bookmarks icons more visually distinct,  users are more likely to know what icon to click to save their bookmarks. By changing the access-bookmarks icon, we believe that we will and can see an increase in engagement with the save to bookmarks icon.",
  "isEnrollmentPaused": true,
  "bucketConfig": {
    "randomizationUnit": "nimbus_id",
    "namespace": "fenix-bookmark-list-icon-1",
    "start": 0,
    "count": 10000,
    "total": 10000
  },
  "probeSets": [],
  "outcomes": [{
    "slug": "default-browser",
    "priority": "primary"
  }],
  "branches": [
    {
      "slug": "control",
      "ratio": 1
    },
    {
      "slug": "treatment",
      "ratio": 1
    }
  ],
  "targeting": "true",
  "startDate": "2021-02-09",
  "endDate": "2021-03-11",
  "proposedDuration": 28,
  "proposedEnrollment": 7,
  "referenceBranch": "control",
  "featureIds": []
}
"""  # noqa:E501

FIREFOX_IOS_EXPERIMENT_FIXTURE = """
{
   "schemaVersion":"1.4.0",
   "slug":"nimbus-aa-validation-for-ios",
   "id":"nimbus-aa-validation-for-ios",
   "application":"org.mozilla.ios.FirefoxBeta",
   "appName":"firefox_ios",
   "appId":"org.mozilla.ios.FirefoxBeta",
   "channel":"beta",
   "userFacingName":"Nimbus A/A Validation for iOS",
   "userFacingDescription":"Is Nimbus working? This experiment tries to find out.",
   "isEnrollmentPaused":true,
   "bucketConfig":{
      "randomizationUnit":"nimbus_id",
      "namespace":"nimbus-aa-validation-for-ios-1",
      "start":0,
      "count":8000,
      "total":10000
   },
   "probeSets":[],
   "outcomes":[],
   "branches":[
      {
         "slug":"a1",
         "ratio":60
      },
      {
         "slug":"a2",
         "ratio":40
      }
   ],
   "targeting":"true",
   "startDate":"2021-04-01",
   "endDate":null,
   "proposedDuration":28,
   "proposedEnrollment":7,
   "referenceBranch":"a1",
   "featureIds":[
      "nimbusValidation"
   ]
}
"""  # noqa:E501

KLAR_ANDROID_EXPERIMENT_FIXTURE = """
{
   "schemaVersion":"1.4.0",
   "slug":"klar-test",
   "id":"klar-test",
   "application":"org.mozilla.klar",
   "appName":"klar_android",
   "appId":"org.mozilla.klar",
   "channel":"beta",
   "userFacingName":"Klar test",
   "userFacingDescription":"Is Nimbus working? This experiment tries to find out.",
   "isEnrollmentPaused":true,
   "bucketConfig":{
      "randomizationUnit":"nimbus_id",
      "namespace":"klar-test-1",
      "start":0,
      "count":8000,
      "total":10000
   },
   "probeSets":[],
   "outcomes":[],
   "branches":[
      {
         "slug":"a1",
         "ratio":60
      },
      {
         "slug":"a2",
         "ratio":40
      }
   ],
   "targeting":"true",
   "startDate":"2021-04-01",
   "endDate":null,
   "proposedDuration":28,
   "proposedEnrollment":7,
   "referenceBranch":"a1",
   "featureIds":[
      "nimbusValidation"
   ]
}
"""  # noqa:E501

FOCUS_ANDROID_EXPERIMENT_FIXTURE = """
{
   "schemaVersion":"1.4.0",
   "slug":"focus-test",
   "id":"focus-test",
   "application":"org.mozilla.focus",
   "appName":"focus_android",
   "appId":"org.mozilla.focus",
   "channel":"beta",
   "userFacingName":"Focus test",
   "userFacingDescription":"Is Nimbus working? This experiment tries to find out.",
   "isEnrollmentPaused":true,
   "bucketConfig":{
      "randomizationUnit":"nimbus_id",
      "namespace":"focus-test-1",
      "start":0,
      "count":8000,
      "total":10000
   },
   "probeSets":[],
   "outcomes":[],
   "branches":[
      {
         "slug":"a1",
         "ratio":60
      },
      {
         "slug":"a2",
         "ratio":40
      }
   ],
   "targeting":"true",
   "startDate":"2021-04-01",
   "endDate":null,
   "proposedDuration":28,
   "proposedEnrollment":7,
   "referenceBranch":"a1",
   "featureIds":[
      "nimbusValidation"
   ]
}
"""  # noqa:E501


@pytest.fixture
def mock_session():
    def experimenter_fixtures(url):
        mocked_value = MagicMock()
        if url == ExperimentCollection.EXPERIMENTER_API_URL_V1:
            mocked_value.json.return_value = json.loads(EXPERIMENTER_FIXTURE_V1)
        elif url == ExperimentCollection.EXPERIMENTER_API_URL_V6:
            mocked_value.json.return_value = json.loads(EXPERIMENTER_FIXTURE_V6)
        else:
            raise Exception("Invalid Experimenter API call.")

        return mocked_value

    session = MagicMock()
    session.get = MagicMock(side_effect=experimenter_fixtures)
    return session


@pytest.fixture
def experiment_collection(mock_session):
    return ExperimentCollection.from_experimenter(mock_session)


def test_from_experimenter(mock_session):
    collection = ExperimentCollection.from_experimenter(mock_session)
    mock_session.get.assert_any_call(ExperimentCollection.EXPERIMENTER_API_URL_V1)
    mock_session.get.assert_any_call(ExperimentCollection.EXPERIMENTER_API_URL_V6)
    assert len(collection.experiments) == 6
    assert isinstance(collection.experiments[0], Experiment)
    assert isinstance(collection.experiments[0].branches[0], Branch)
    assert len(collection.experiments[0].branches) == 2
    assert collection.experiments[0].start_date > dt.datetime(2019, 1, 1, tzinfo=pytz.utc)
    assert len(collection.experiments[1].branches) == 2


def test_started_since(experiment_collection):
    recent = experiment_collection.started_since(dt.datetime(2019, 1, 1, tzinfo=pytz.utc))
    assert isinstance(recent, ExperimentCollection)
    assert len(recent.experiments) > 0


def test_normandy_experiment_slug(experiment_collection):
    normandy_slugs = list(map(lambda e: e.normandy_slug, experiment_collection.experiments))
    assert "addon-activity-stream-search-topsites-release-69-1576277" in normandy_slugs
    assert None in normandy_slugs
    assert "pref-doh-us-engagement-study-v2-release-69-71-bug-1590831" in normandy_slugs


def test_with_slug(experiment_collection):
    experiments = experiment_collection.with_slug("search-topsites")
    assert len(experiments.experiments) == 1
    assert experiments.experiments[0].experimenter_slug == "search-topsites"

    experiments = experiment_collection.with_slug(
        "addon-activity-stream-search-topsites-release-69-1576277"
    )
    assert len(experiments.experiments) == 1
    assert experiments.experiments[0].experimenter_slug == "search-topsites"
    assert (
        experiments.experiments[0].normandy_slug
        == "addon-activity-stream-search-topsites-release-69-1576277"
    )

    experiments = experiment_collection.with_slug("non-existing-slug")
    assert len(experiments.experiments) == 0


def test_convert_experiment_v1_to_experiment():
    experiment_v1 = ExperimentV1(
        slug="test-slug",
        normandy_slug="test_slug",
        status="Live",
        type="cfr",
        start_date=dt.datetime(2019, 1, 1),
        end_date=dt.datetime(2019, 1, 10),
        proposed_enrollment=14,
        variants=[
            Variant(is_control=True, slug="control", ratio=2),
            Variant(is_control=False, slug="treatment", ratio=1),
        ],
    )

    experiment = experiment_v1.to_experiment()

    assert experiment.experimenter_slug == "test-slug"
    assert experiment.normandy_slug == "test_slug"
    assert len(experiment.branches) == 2
    assert experiment.reference_branch == "control"
    assert experiment.is_high_population is False


def test_convert_experiment_v6_to_experiment():
    experiment_v6 = ExperimentV6(
        slug="test_slug",
        startDate=dt.datetime(2019, 1, 1),
        endDate=dt.datetime(2019, 1, 10),
        proposedEnrollment=14,
        branches=[Branch(slug="control", ratio=2), Branch(slug="treatment", ratio=1)],
        referenceBranch="control",
        isEnrollmentPaused=True,
    )

    experiment = experiment_v6.to_experiment()

    assert experiment.experimenter_slug is None
    assert experiment.normandy_slug == "test_slug"
    assert experiment.status == "Complete"
    assert experiment.type == "v6"
    assert len(experiment.branches) == 2
    assert experiment.reference_branch == "control"
    assert experiment.is_high_population is False
    assert experiment.outcomes == []
    assert experiment.is_enrollment_paused is True


def test_fixture_validates():
    schema = json.loads((Path(__file__).parent / "data/NimbusExperiment_v1.0.json").read_text())
    experiments = json.loads(EXPERIMENTER_FIXTURE_V6)
    [jsonschema.validate(e, schema) for e in experiments if e["slug"]]


def test_experiment_v6_status():
    experiment_live = ExperimentV6(
        slug="test_slug",
        startDate=dt.datetime(2019, 1, 1),
        endDate=dt.datetime.now() + timedelta(days=1),
        proposedEnrollment=14,
        branches=[Branch(slug="control", ratio=2), Branch(slug="treatment", ratio=1)],
        referenceBranch="control",
    )

    assert experiment_live.to_experiment().status == "Live"

    experiment_complete = ExperimentV6(
        slug="test_slug",
        startDate=dt.datetime(2019, 1, 1),
        endDate=dt.datetime.now() - timedelta(minutes=1),
        proposedEnrollment=14,
        branches=[Branch(slug="control", ratio=2), Branch(slug="treatment", ratio=1)],
        referenceBranch="control",
    )

    assert experiment_complete.to_experiment().status == "Complete"


def test_app_name():
    x = ExperimentV6.from_dict(json.loads(FENIX_EXPERIMENT_FIXTURE))
    assert x.appName == "fenix"
    assert x.appId == "org.mozilla.fenix"
    assert Outcome(slug="default-browser") in x.outcomes
    assert "default-browser" in x.to_experiment().outcomes


def test_ios_app_name():
    x = ExperimentV6.from_dict(json.loads(FIREFOX_IOS_EXPERIMENT_FIXTURE))
    assert x.appName == "firefox_ios"
    assert x.appId == "org.mozilla.ios.FirefoxBeta"
    assert x.outcomes == []
    assert x.to_experiment().outcomes == []


def test_klar_android_app_name():
    x = ExperimentV6.from_dict(json.loads(KLAR_ANDROID_EXPERIMENT_FIXTURE))
    assert x.appName == "klar_android"
    assert x.appId == "org.mozilla.klar"
    assert x.outcomes == []
    assert x.to_experiment().outcomes == []


def test_focus_android_app_name():
    x = ExperimentV6.from_dict(json.loads(FOCUS_ANDROID_EXPERIMENT_FIXTURE))
    assert x.appName == "focus_android"
    assert x.appId == "org.mozilla.focus"
    assert x.outcomes == []
    assert x.to_experiment().outcomes == []


def test_ended_after_or_live(experiment_collection):
    date = dt.datetime(2019, 1, 1, tzinfo=pytz.utc)
    recent = experiment_collection.ended_after_or_live(date)
    assert isinstance(recent, ExperimentCollection)
    assert len(recent.experiments) > 0
    assert all((e.status == "Live" or e.end_date >= date) for e in recent.experiments)
