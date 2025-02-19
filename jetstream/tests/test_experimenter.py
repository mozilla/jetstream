import datetime as dt
import json
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import jsonschema
import pytest
import pytz
from metric_config_parser.experiment import Branch, BucketConfig, Experiment

from jetstream.experimenter import (
    ExperimentCollection,
    NimbusExperiment,
    Outcome,
    Segment,
)

NIMBUS_EXPERIMENTER_FIXTURE = r"""
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
  "segments": [{
    "slug": "regular_users_v3"
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
   "segments":[],
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
"""

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
   "segments":[],
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
"""

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
   "segments":[],
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
"""


@pytest.fixture
def mock_session():
    def experimenter_fixtures(url):
        mocked_value = MagicMock()
        if url == ExperimentCollection.EXPERIMENTER_API_URL_V8:
            mocked_value.json.return_value = json.loads(NIMBUS_EXPERIMENTER_FIXTURE)
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
    mock_session.get.assert_any_call(ExperimentCollection.EXPERIMENTER_API_URL_V8)
    assert len(collection.experiments) == 3
    assert isinstance(collection.experiments[0], Experiment)
    assert isinstance(collection.experiments[0].branches[0], Branch)
    assert len(collection.experiments[0].branches) == 2
    assert collection.experiments[0].start_date > dt.datetime(2019, 1, 1, tzinfo=pytz.utc)
    assert len(collection.experiments[1].branches) == 2


def test_started_since(experiment_collection):
    recent = experiment_collection.started_since(dt.datetime(2019, 1, 1, tzinfo=pytz.utc))
    assert isinstance(recent, ExperimentCollection)
    assert len(recent.experiments) > 0


def test_with_slug(experiment_collection):
    experiments = experiment_collection.with_slug("bug-1629098-rapid-please-reject-me-beta-86")
    assert len(experiments.experiments) == 1
    assert experiments.experiments[0].experimenter_slug is None
    assert experiments.experiments[0].normandy_slug == "bug-1629098-rapid-please-reject-me-beta-86"

    experiments = experiment_collection.with_slug(
        "bug-1629000-rapid-testing-rapido-intake-1-release-79"
    )
    assert len(experiments.experiments) == 1
    assert experiments.experiments[0].experimenter_slug is None
    assert (
        experiments.experiments[0].normandy_slug
        == "bug-1629000-rapid-testing-rapido-intake-1-release-79"
    )

    experiments = experiment_collection.with_slug("non-existing-slug")
    assert len(experiments.experiments) == 0


def test_convert_nimbus_experiment_to_experiment():
    nimbus_experiment = NimbusExperiment(
        slug="test_slug",
        startDate=dt.datetime(2019, 1, 1),
        endDate=dt.datetime(2019, 1, 10),
        proposedEnrollment=14,
        branches=[Branch(slug="control", ratio=2), Branch(slug="treatment", ratio=1)],
        referenceBranch="control",
        isEnrollmentPaused=True,
        bucketConfig=BucketConfig(
            randomization_unit="test-randomization",
            namespace="test",
            start=1000,
            count=5000,
            total=10000,
        ),
    )

    experiment = nimbus_experiment.to_experiment()

    assert experiment.experimenter_slug is None
    assert experiment.normandy_slug == "test_slug"
    assert experiment.status == "Complete"
    assert experiment.type == "v6"
    assert len(experiment.branches) == 2
    assert experiment.reference_branch == "control"
    assert experiment.is_high_population is False
    assert experiment.outcomes == []
    assert experiment.segments == []
    assert experiment.is_enrollment_paused is True


def test_fixture_validates():
    schema = json.loads((Path(__file__).parent / "data/NimbusExperiment_v1.0.json").read_text())
    experiments = json.loads(NIMBUS_EXPERIMENTER_FIXTURE)
    [jsonschema.validate(e, schema) for e in experiments if e["slug"]]


def test_nimbus_experiment_status():
    experiment_live = NimbusExperiment(
        slug="test_slug",
        startDate=dt.datetime(2019, 1, 1),
        endDate=dt.datetime.now() + timedelta(days=1),
        proposedEnrollment=14,
        branches=[Branch(slug="control", ratio=2), Branch(slug="treatment", ratio=1)],
        referenceBranch="control",
        bucketConfig=BucketConfig(
            randomization_unit="test-randomization",
            namespace="test",
            start=1000,
            count=5000,
            total=10000,
        ),
    )

    assert experiment_live.to_experiment().status == "Live"

    experiment_complete = NimbusExperiment(
        slug="test_slug",
        startDate=dt.datetime(2019, 1, 1),
        endDate=dt.datetime.now() - timedelta(minutes=1),
        proposedEnrollment=14,
        branches=[Branch(slug="control", ratio=2), Branch(slug="treatment", ratio=1)],
        referenceBranch="control",
        bucketConfig=BucketConfig(
            randomization_unit="test-randomization",
            namespace="test",
            start=1000,
            count=5000,
            total=10000,
        ),
    )

    assert experiment_complete.to_experiment().status == "Complete"


def test_app_name():
    x = NimbusExperiment.from_dict(json.loads(FENIX_EXPERIMENT_FIXTURE))
    assert x.appName == "fenix"
    assert x.appId == "org.mozilla.fenix"
    assert Outcome(slug="default-browser") in x.outcomes
    assert "default-browser" in x.to_experiment().outcomes
    assert Segment(slug="regular_users_v3") in x.segments
    assert "regular_users_v3" in x.to_experiment().segments


def test_ios_app_name():
    x = NimbusExperiment.from_dict(json.loads(FIREFOX_IOS_EXPERIMENT_FIXTURE))
    assert x.appName == "firefox_ios"
    assert x.appId == "org.mozilla.ios.FirefoxBeta"
    assert x.outcomes == []
    assert x.to_experiment().outcomes == []
    assert x.segments == []
    assert x.to_experiment().segments == []


def test_klar_android_app_name():
    x = NimbusExperiment.from_dict(json.loads(KLAR_ANDROID_EXPERIMENT_FIXTURE))
    assert x.appName == "klar_android"
    assert x.appId == "org.mozilla.klar"
    assert x.outcomes == []
    assert x.to_experiment().outcomes == []
    assert x.segments == []
    assert x.to_experiment().segments == []


def test_focus_android_app_name():
    x = NimbusExperiment.from_dict(json.loads(FOCUS_ANDROID_EXPERIMENT_FIXTURE))
    assert x.appName == "focus_android"
    assert x.appId == "org.mozilla.focus"
    assert x.outcomes == []
    assert x.to_experiment().outcomes == []
    assert x.segments == []
    assert x.to_experiment().segments == []


def test_ended_after_or_live(experiment_collection):
    date = dt.datetime(2019, 1, 1, tzinfo=pytz.utc)
    recent = experiment_collection.ended_after_or_live(date)
    assert isinstance(recent, ExperimentCollection)
    assert len(recent.experiments) > 0
    assert all((e.status == "Live" or e.end_date >= date) for e in recent.experiments)


def test_of_type(experiment_collection):
    nimbus_experiments = experiment_collection.of_type("v6")
    for experiment in nimbus_experiments.experiments:
        assert experiment.type == "v6"

    legacy_experiments = experiment_collection.of_type("v1")
    for experiment in legacy_experiments.experiments:
        assert experiment.type == "v1"
