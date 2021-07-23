import datetime as dt
import json
import re
from datetime import timedelta
from textwrap import dedent
from unittest.mock import Mock

import mozanalysis.segments
import pytest
import pytz
import toml

import jetstream.analysis
from jetstream.analysis import Analysis, AnalysisPeriod
from jetstream.config import AnalysisSpec
from jetstream.errors import (
    ExplicitSkipException,
    HighPopulationException,
    NoEnrollmentPeriodException,
)
from jetstream.experimenter import ExperimentV1


def test_get_timelimits_if_ready(experiments):
    config = AnalysisSpec().resolve(experiments[0])
    config2 = AnalysisSpec().resolve(experiments[2])

    analysis = Analysis("test", "test", config)
    analysis2 = Analysis("test", "test", config2)

    date = dt.datetime(2019, 12, 1, tzinfo=pytz.utc) + timedelta(0)
    assert analysis._get_timelimits_if_ready(AnalysisPeriod.DAY, date) is None
    assert analysis._get_timelimits_if_ready(AnalysisPeriod.WEEK, date) is None

    date = dt.datetime(2019, 12, 1, tzinfo=pytz.utc) + timedelta(2)
    assert analysis._get_timelimits_if_ready(AnalysisPeriod.DAY, date) is None
    assert analysis._get_timelimits_if_ready(AnalysisPeriod.WEEK, date) is None

    date = dt.datetime(2019, 12, 1, tzinfo=pytz.utc) + timedelta(7)
    assert analysis._get_timelimits_if_ready(AnalysisPeriod.DAY, date)
    assert analysis._get_timelimits_if_ready(AnalysisPeriod.WEEK, date) is None

    date = dt.datetime(2019, 12, 1, tzinfo=pytz.utc) + timedelta(days=13)
    assert analysis._get_timelimits_if_ready(AnalysisPeriod.DAY, date)
    assert analysis._get_timelimits_if_ready(AnalysisPeriod.WEEK, date)

    date = dt.datetime(2020, 2, 29, tzinfo=pytz.utc)
    assert analysis._get_timelimits_if_ready(AnalysisPeriod.OVERALL, date) is None

    date = dt.datetime(2020, 3, 1, tzinfo=pytz.utc)
    assert analysis._get_timelimits_if_ready(AnalysisPeriod.OVERALL, date)
    assert analysis2._get_timelimits_if_ready(AnalysisPeriod.OVERALL, date) is None

    date = dt.datetime(2019, 12, 1, tzinfo=pytz.utc) + timedelta(days=34)
    assert analysis._get_timelimits_if_ready(AnalysisPeriod.DAYS_28, date)


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
    experiment = ExperimentV1.from_dict(json.loads(experiment_json)).to_experiment()
    config = AnalysisSpec().resolve(experiment)
    analysis = Analysis("test", "test", config)
    with pytest.raises(NoEnrollmentPeriodException):
        analysis.run(current_date=dt.datetime(2020, 3, 19, tzinfo=pytz.utc), dry_run=True)


def test_regression_20200316(monkeypatch):
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
    experiment = ExperimentV1.from_dict(json.loads(experiment_json)).to_experiment()
    config = AnalysisSpec().resolve(experiment)

    monkeypatch.setattr("jetstream.analysis.Analysis.ensure_enrollments", Mock())
    analysis = Analysis("test", "test", config)
    analysis.run(current_date=dt.datetime(2020, 3, 16, tzinfo=pytz.utc), dry_run=True)


def test_validate_doesnt_explode(experiments, monkeypatch):
    m = Mock()
    monkeypatch.setattr(jetstream.analysis, "dry_run_query", m)
    x = experiments[0]
    config = AnalysisSpec.default_for_experiment(x).resolve(x)
    Analysis("spam", "eggs", config).validate()
    assert m.call_count == 2


def test_analysis_doesnt_choke_on_segments(experiments, monkeypatch):
    conf = dedent(
        """
        [experiment]
        segments = ["regular_users_v3"]
        """
    )
    spec = AnalysisSpec.from_dict(toml.loads(conf))
    configured = spec.resolve(experiments[0])
    assert isinstance(configured.experiment.segments[0], mozanalysis.segments.Segment)
    monkeypatch.setattr("jetstream.analysis.Analysis.ensure_enrollments", Mock())
    Analysis("test", "test", configured).run(
        current_date=dt.datetime(2020, 1, 1, tzinfo=pytz.utc), dry_run=True
    )


def test_is_high_population_check(experiments):
    x = experiments[3]
    config = AnalysisSpec.default_for_experiment(x).resolve(x)

    with pytest.raises(HighPopulationException):
        Analysis("spam", "eggs", config).check_runnable()


def test_skip_works(experiments):
    conf = dedent(
        """
        [experiment]
        skip = true
        """
    )
    spec = AnalysisSpec.from_dict(toml.loads(conf))
    configured = spec.resolve(experiments[0])
    with pytest.raises(ExplicitSkipException):
        Analysis("test", "test", configured).run(
            current_date=dt.datetime(2020, 1, 1, tzinfo=pytz.utc), dry_run=True
        )


def test_fenix_experiments_use_right_datasets(fenix_experiments, monkeypatch):
    for experiment in fenix_experiments:
        called = 0

        def dry_run_query(query):
            nonlocal called
            called = called + 1
            dataset = re.sub(r"[^A-Za-z0-9_]", "_", experiment.app_id)
            assert dataset in query
            assert query.count(dataset) == query.count("org_mozilla")

        monkeypatch.setattr("jetstream.analysis.dry_run_query", dry_run_query)
        config = AnalysisSpec.default_for_experiment(experiment).resolve(experiment)
        Analysis("spam", "eggs", config).validate()
        assert called == 2


def test_firefox_ios_experiments_use_right_datasets(firefox_ios_experiments, monkeypatch):
    for experiment in firefox_ios_experiments:
        called = 0

        def dry_run_query(query):
            nonlocal called
            called = called + 1
            dataset = re.sub(r"[^A-Za-z0-9_]", "_", experiment.app_id).lower()
            assert dataset in query
            assert query.count(dataset) == query.count("org_mozilla_ios")

        monkeypatch.setattr("jetstream.analysis.dry_run_query", dry_run_query)
        config = AnalysisSpec.default_for_experiment(experiment).resolve(experiment)
        Analysis("spam", "eggs", config).validate()
        assert called == 2


def test_focus_android_experiments_use_right_datasets(focus_android_experiments, monkeypatch):
    for experiment in focus_android_experiments:
        called = 0

        def dry_run_query(query):
            nonlocal called
            called = called + 1
            dataset = re.sub(r"[^A-Za-z0-9_]", "_", experiment.app_id).lower()
            assert dataset in query
            assert query.count(dataset) == query.count("org_mozilla_focus")

        monkeypatch.setattr("jetstream.analysis.dry_run_query", dry_run_query)
        config = AnalysisSpec.default_for_experiment(experiment).resolve(experiment)
        Analysis("spam", "eggs", config).validate()
        assert called == 2


def test_klar_android_experiments_use_right_datasets(klar_android_experiments, monkeypatch):
    for experiment in klar_android_experiments:
        called = 0

        def dry_run_query(query):
            nonlocal called
            called = called + 1
            dataset = re.sub(r"[^A-Za-z0-9_]", "_", experiment.app_id).lower()
            assert dataset in query
            assert query.count(dataset) == query.count("org_mozilla_klar")

        monkeypatch.setattr("jetstream.analysis.dry_run_query", dry_run_query)
        config = AnalysisSpec.default_for_experiment(experiment).resolve(experiment)
        Analysis("spam", "eggs", config).validate()
        assert called == 2
