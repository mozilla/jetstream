import datetime as dt
import json
import re
from datetime import timedelta
from textwrap import dedent
from unittest.mock import Mock

import pandas as pd
import pytest
import pytz
import toml
from metric_config_parser import segment
from metric_config_parser.analysis import AnalysisSpec
from metric_config_parser.metric import AnalysisPeriod
from mozilla_nimbus_schemas.jetstream import AnalysisBasis

import jetstream.analysis
from jetstream.analysis import Analysis
from jetstream.config import ConfigLoader
from jetstream.errors import (
    EnrollmentNotCompleteException,
    ExplicitSkipException,
    HighPopulationException,
    NoEnrollmentPeriodException,
)
from jetstream.experimenter import ExperimentV1


def test_get_timelimits_if_ready(experiments):
    config = AnalysisSpec().resolve(experiments[0], ConfigLoader.configs)
    config2 = AnalysisSpec().resolve(experiments[2], ConfigLoader.configs)

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
    config = AnalysisSpec().resolve(experiment, ConfigLoader.configs)
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
    config = AnalysisSpec().resolve(experiment, ConfigLoader.configs)

    monkeypatch.setattr("jetstream.analysis.Analysis.ensure_enrollments", Mock())
    pre_start_time = dt.datetime.now(tz=pytz.utc)
    analysis = Analysis("test", "test", config)
    analysis.run(current_date=dt.datetime(2020, 3, 16, tzinfo=pytz.utc), dry_run=True)
    assert analysis.start_time is not None
    assert analysis.start_time >= pre_start_time


def test_validate_doesnt_explode(experiments, monkeypatch):
    m = Mock()
    monkeypatch.setattr(jetstream.analysis, "dry_run_query", m)
    x = experiments[0]
    config = AnalysisSpec.default_for_experiment(x, ConfigLoader.configs).resolve(
        x, ConfigLoader.configs
    )
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
    configured = spec.resolve(experiments[0], ConfigLoader.configs)
    assert isinstance(configured.experiment.segments[0], segment.Segment)
    monkeypatch.setattr("jetstream.analysis.Analysis.ensure_enrollments", Mock())
    Analysis("test", "test", configured).run(
        current_date=dt.datetime(2020, 1, 1, tzinfo=pytz.utc), dry_run=True
    )


def test_subset_to_segment(experiments):
    conf = dedent(
        """
        [experiment]
        segments = ["regular_users_v3"]

        [metrics.active_hours]
        analysis_bases = ["exposures", "enrollments"]
        """
    )

    spec = AnalysisSpec.from_dict(toml.loads(conf))
    configured = spec.resolve(experiments[0], ConfigLoader.configs)
    assert isinstance(configured.experiment.segments[0], segment.Segment)

    metrics_data = pd.DataFrame(
        [
            (None, None, "1"),
            (None, "1", None),
            (True, None, "1"),
            (True, "1", None),
            (False, None, "1"),
            (True, "1", None),
        ],
        columns=["regular_users_v3", "enrollment_date", "exposure_date"],
    )
    analysis = Analysis("test", "test", configured)
    all_enrollments = analysis.subset_to_segment("all", metrics_data, AnalysisBasis.ENROLLMENTS)
    all_enrollments = all_enrollments.compute()
    assert len(all_enrollments) == 3
    assert len(all_enrollments[all_enrollments["regular_users_v3"] == True]) == 2  # noqa: E712
    assert len(all_enrollments[all_enrollments["regular_users_v3"] == False]) == 0  # noqa: E712
    assert len(all_enrollments[all_enrollments["enrollment_date"] == "1"]) == 3

    all_exposures = analysis.subset_to_segment("all", metrics_data, AnalysisBasis.EXPOSURES)
    all_exposures = all_exposures.compute()
    assert len(all_exposures) == 3
    assert len(all_exposures[all_exposures["regular_users_v3"] == True]) == 1  # noqa: E712
    assert len(all_exposures[all_exposures["regular_users_v3"] == False]) == 1  # noqa: E712
    assert len(all_exposures[all_exposures["exposure_date"] == "1"]) == 3

    segment_enrollments = analysis.subset_to_segment(
        "regular_users_v3", metrics_data, AnalysisBasis.ENROLLMENTS
    )
    segment_enrollments = segment_enrollments.compute()
    assert len(segment_enrollments) == 2
    assert (
        len(segment_enrollments[segment_enrollments["regular_users_v3"] == True]) == 2  # noqa: E712
    )
    assert (
        len(segment_enrollments[segment_enrollments["regular_users_v3"] == False])  # noqa: E712
        == 0
    )
    assert len(segment_enrollments[segment_enrollments["enrollment_date"] == "1"]) == 2

    segment_exposures = analysis.subset_to_segment(
        "regular_users_v3", metrics_data, AnalysisBasis.EXPOSURES
    )
    segment_exposures = segment_exposures.compute()
    assert len(segment_exposures) == 1
    assert len(segment_exposures[segment_exposures["regular_users_v3"] == True]) == 1  # noqa: E712
    assert len(segment_exposures[segment_exposures["regular_users_v3"] == False]) == 0  # noqa: E712
    assert len(segment_exposures[segment_exposures["exposure_date"] == "1"]) == 1


def test_is_high_population_check(experiments):
    x = experiments[3]
    config = AnalysisSpec.default_for_experiment(x, ConfigLoader.configs).resolve(
        x, ConfigLoader.configs
    )

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
    configured = spec.resolve(experiments[0], ConfigLoader.configs)
    with pytest.raises(ExplicitSkipException):
        Analysis("test", "test", configured).run(
            current_date=dt.datetime(2020, 1, 1, tzinfo=pytz.utc), dry_run=True
        )


def test_skip_while_enrolling(experiments):
    config = AnalysisSpec().resolve(experiments[8], ConfigLoader.configs)
    with pytest.raises(EnrollmentNotCompleteException):
        Analysis("test", "test", config).run(
            current_date=dt.datetime(2020, 1, 1, tzinfo=pytz.utc), dry_run=True
        )


def test_custom_override_skips_enrollment_paused_check(experiments, monkeypatch):
    conf = dedent(
        """
        [experiment]
        enrollment_period = 7
        """
    )
    spec = AnalysisSpec.from_dict(toml.loads(conf))
    config = spec.resolve(experiments[8], ConfigLoader.configs)
    m = Mock()
    m.return_value = None
    monkeypatch.setattr("jetstream.analysis.Analysis._get_timelimits_if_ready", m)
    # no errors expected
    Analysis("test", "test", config).run(
        current_date=dt.datetime(2020, 1, 1, tzinfo=pytz.utc), dry_run=True
    )


def test_validation_working_while_enrolling(experiments):
    config = AnalysisSpec().resolve(experiments[8], ConfigLoader.configs)
    assert experiments[8].is_enrollment_paused is False
    try:
        Analysis("test", "test", config).validate()
    except Exception as e:
        assert False, f"Raised {e}"


def test_run_when_enrolling_complete(experiments, monkeypatch):
    config = AnalysisSpec().resolve(experiments[9], ConfigLoader.configs)
    m = Mock()
    m.return_value = None
    monkeypatch.setattr("jetstream.analysis.Analysis._get_timelimits_if_ready", m)
    # no errors expected
    Analysis("test", "test", config).run(
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
        config = AnalysisSpec.default_for_experiment(experiment, ConfigLoader.configs).resolve(
            experiment, ConfigLoader.configs
        )
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
        config = AnalysisSpec.default_for_experiment(experiment, ConfigLoader.configs).resolve(
            experiment, ConfigLoader.configs
        )
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
        config = AnalysisSpec.default_for_experiment(experiment, ConfigLoader.configs).resolve(
            experiment, ConfigLoader.configs
        )
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
        config = AnalysisSpec.default_for_experiment(experiment, ConfigLoader.configs).resolve(
            experiment, ConfigLoader.configs
        )
        Analysis("spam", "eggs", config).validate()
        assert called == 2
