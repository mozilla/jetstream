import datetime as dt
import json
import logging
import re
from datetime import timedelta
from textwrap import dedent
from unittest.mock import MagicMock, Mock

import pytest
import pytz
import toml
from metric_config_parser import segment
from metric_config_parser.analysis import AnalysisSpec
from metric_config_parser.data_source import DataSource
from metric_config_parser.experiment import Branch, BucketConfig, Experiment
from metric_config_parser.metric import AnalysisPeriod, Summary
from mozilla_nimbus_schemas.experiments import RandomizationUnit
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
from jetstream.experimenter import LegacyExperiment
from jetstream.metric import Metric

logger = logging.getLogger(__name__)


def _empty_analysis(experiments):
    x: Experiment = experiments[0]
    config = AnalysisSpec.default_for_experiment(x, ConfigLoader.configs).resolve(
        x, ConfigLoader.configs
    )
    return Analysis("spam", "eggs", config)


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
    assert analysis._get_timelimits_if_ready(AnalysisPeriod.PREENROLLMENT_WEEK, date) is None
    assert analysis._get_timelimits_if_ready(AnalysisPeriod.PREENROLLMENT_DAYS_28, date) is None

    date = dt.datetime(2019, 12, 1, tzinfo=pytz.utc) + timedelta(8)
    assert analysis._get_timelimits_if_ready(AnalysisPeriod.DAY, date)
    assert analysis._get_timelimits_if_ready(AnalysisPeriod.WEEK, date) is None
    assert analysis._get_timelimits_if_ready(AnalysisPeriod.PREENROLLMENT_WEEK, date)
    assert analysis._get_timelimits_if_ready(AnalysisPeriod.PREENROLLMENT_DAYS_28, date)

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
    experiment = LegacyExperiment.from_dict(json.loads(experiment_json)).to_experiment()
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
    experiment = LegacyExperiment.from_dict(json.loads(experiment_json)).to_experiment()
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
        pytest.fail(f"Raised {e}")


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

        def dry_run_query(query, exp=experiment):
            nonlocal called
            called = called + 1
            dataset = re.sub(r"[^A-Za-z0-9_]", "_", exp.app_id)
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

        def dry_run_query(query, exp=experiment):
            nonlocal called
            called = called + 1
            dataset = re.sub(r"[^A-Za-z0-9_]", "_", exp.app_id).lower()
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

        def dry_run_query(query, exp=experiment):
            nonlocal called
            called = called + 1
            dataset = re.sub(r"[^A-Za-z0-9_]", "_", exp.app_id).lower()
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

        def dry_run_query(query, exp=experiment):
            nonlocal called
            called = called + 1
            dataset = re.sub(r"[^A-Za-z0-9_]", "_", exp.app_id).lower()
            assert dataset in query
            assert query.count(dataset) == query.count("org_mozilla_klar")

        monkeypatch.setattr("jetstream.analysis.dry_run_query", dry_run_query)
        config = AnalysisSpec.default_for_experiment(experiment, ConfigLoader.configs).resolve(
            experiment, ConfigLoader.configs
        )
        Analysis("spam", "eggs", config).validate()
        assert called == 2


def test_create_subset_metric_table_query_univariate_basic(experiments):
    metric = Metric(
        name="metric_name",
        data_source=DataSource(name="test_data_source", from_expression="test.test"),
        select_expression="test",
        analysis_bases=[AnalysisBasis.ENROLLMENTS],
    )

    expected_query = dedent(
        """
    SELECT branch, metric_name
    FROM `test_experiment_enrollments_1`
    WHERE metric_name IS NOT NULL AND
    enrollment_date IS NOT NULL"""
    )

    actual_query = _empty_analysis(experiments)._create_subset_metric_table_query_univariate(
        "test_experiment_enrollments_1", "all", metric, AnalysisBasis.ENROLLMENTS
    )

    assert expected_query == actual_query


@pytest.mark.parametrize(("randomization_unit"), list(RandomizationUnit))
def test_create_subset_metric_table_query_covariate_basic(randomization_unit, monkeypatch):
    monkeypatch.setattr(
        "jetstream.analysis.Analysis._table_name", MagicMock(return_value="table_pre")
    )
    monkeypatch.setattr(
        "jetstream.bigquery_client.BigQueryClient.table_exists",
        MagicMock(return_value=True),
    )

    metric = Metric(
        name="metric_name",
        data_source=DataSource(name="test_data_source", from_expression="test.test"),
        select_expression="test",
        analysis_bases=[AnalysisBasis.ENROLLMENTS],
    )

    expected_query = dedent(
        """
        SELECT
            during.branch,
            during.metric_name,
            pre.metric_name AS metric_name_pre
        FROM (
            `test_experiment_enrollments_1` during
            LEFT JOIN `table_pre` pre
            USING (analysis_id, branch)
        )
        WHERE during.metric_name IS NOT NULL AND
        during.enrollment_date IS NOT NULL"""
    )

    exp = Experiment(
        experimenter_slug="test_slug",
        type="v6",
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
        enrollment_end_date=dt.datetime(2019, 12, 7, tzinfo=pytz.utc),
        bucket_config=BucketConfig(
            randomization_unit=randomization_unit,
            namespace="testing",
            start=0,
            count=10,
            total=100,
        ),
    )

    actual_query = _empty_analysis([exp])._create_subset_metric_table_query_covariate(
        "test_experiment_enrollments_1",
        "all",
        metric,
        AnalysisBasis.ENROLLMENTS,
        AnalysisPeriod.PREENROLLMENT_WEEK,
        "metric_name",
    )

    assert expected_query == actual_query


def test_create_subset_metric_table_query_covariate_missing_table_fallback(
    experiments, monkeypatch, caplog
):
    monkeypatch.setattr(
        "jetstream.analysis.Analysis._table_name", MagicMock(return_value="table_pre")
    )
    monkeypatch.setattr(
        "jetstream.bigquery_client.BigQueryClient.table_exists",
        MagicMock(return_value=False),
    )

    metric = Metric(
        name="metric_name",
        data_source=DataSource(name="test_data_source", from_expression="test.test"),
        select_expression="test",
        analysis_bases=[AnalysisBasis.ENROLLMENTS],
    )

    expected_query = dedent(
        """
    SELECT branch, metric_name
    FROM `test_experiment_enrollments_1`
    WHERE metric_name IS NOT NULL AND
    enrollment_date IS NOT NULL"""
    )

    actual_query = _empty_analysis(experiments)._create_subset_metric_table_query_covariate(
        "test_experiment_enrollments_1",
        "all",
        metric,
        AnalysisBasis.ENROLLMENTS,
        AnalysisPeriod.PREENROLLMENT_WEEK,
        "metric_name",
    )

    assert expected_query == actual_query

    # test that logging message was generated
    assert (
        "Covariate adjustment table table_pre does not exist, falling back to unadjusted inferences"
        in caplog.text
    )


def test_create_subset_metric_table_query_univariate_segment(experiments):
    metric = Metric(
        name="metric_name",
        data_source=DataSource(name="test_data_source", from_expression="test.test"),
        select_expression="test",
        analysis_bases=[AnalysisBasis.ENROLLMENTS],
    )

    expected_query = dedent(
        """
    SELECT branch, metric_name
    FROM `test_experiment_enrollments_1`
    WHERE metric_name IS NOT NULL AND
    enrollment_date IS NOT NULL
    AND mysegment = TRUE"""
    )

    actual_query = _empty_analysis(experiments)._create_subset_metric_table_query_univariate(
        "test_experiment_enrollments_1", "mysegment", metric, AnalysisBasis.ENROLLMENTS
    )

    assert expected_query == actual_query


@pytest.mark.parametrize(("randomization_unit"), list(RandomizationUnit))
def test_create_subset_metric_table_query_covariate_segment(randomization_unit, monkeypatch):
    monkeypatch.setattr(
        "jetstream.analysis.Analysis._table_name", MagicMock(return_value="table_pre")
    )
    monkeypatch.setattr(
        "jetstream.bigquery_client.BigQueryClient.table_exists",
        MagicMock(return_value=True),
    )

    metric = Metric(
        name="metric_name",
        data_source=DataSource(name="test_data_source", from_expression="test.test"),
        select_expression="test",
        analysis_bases=[AnalysisBasis.ENROLLMENTS],
    )

    expected_query = dedent(
        """
    SELECT
        during.branch,
        during.metric_name,
        pre.metric_name AS metric_name_pre
    FROM (
        `test_experiment_enrollments_1` during
        LEFT JOIN `table_pre` pre
        USING (analysis_id, branch)
    )
    WHERE during.metric_name IS NOT NULL AND
    during.enrollment_date IS NOT NULL
    AND during.mysegment = TRUE"""
    )

    exp = Experiment(
        experimenter_slug="test_slug",
        type="v6",
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
        enrollment_end_date=dt.datetime(2019, 12, 7, tzinfo=pytz.utc),
        bucket_config=BucketConfig(
            randomization_unit=randomization_unit,
            namespace="testing",
            start=0,
            count=10,
            total=100,
        ),
    )

    actual_query = _empty_analysis([exp])._create_subset_metric_table_query_covariate(
        "test_experiment_enrollments_1",
        "mysegment",
        metric,
        AnalysisBasis.ENROLLMENTS,
        AnalysisPeriod.PREENROLLMENT_WEEK,
        "metric_name",
    )

    assert expected_query == actual_query


def test_create_subset_metric_table_query_univariate_exposures(experiments):
    metric = Metric(
        name="metric_name",
        data_source=DataSource(name="test_data_source", from_expression="test.test"),
        select_expression="test",
        analysis_bases=[AnalysisBasis.EXPOSURES],
    )

    expected_query = dedent(
        """
    SELECT branch, metric_name
    FROM `test_experiment_exposures_1`
    WHERE metric_name IS NOT NULL AND
    enrollment_date IS NOT NULL AND exposure_date IS NOT NULL"""
    )

    actual_query = _empty_analysis(experiments)._create_subset_metric_table_query_univariate(
        "test_experiment_exposures_1", "all", metric, AnalysisBasis.EXPOSURES
    )

    assert expected_query == actual_query


@pytest.mark.parametrize(("randomization_unit"), list(RandomizationUnit))
def test_create_subset_metric_table_query_covariate_exposures(randomization_unit, monkeypatch):
    monkeypatch.setattr(
        "jetstream.analysis.Analysis._table_name", MagicMock(return_value="table_pre")
    )
    monkeypatch.setattr(
        "jetstream.bigquery_client.BigQueryClient.table_exists",
        MagicMock(return_value=True),
    )

    metric = Metric(
        name="metric_name",
        data_source=DataSource(name="test_data_source", from_expression="test.test"),
        select_expression="test",
        analysis_bases=[AnalysisBasis.ENROLLMENTS],
    )

    expected_query = dedent(
        """
    SELECT
        during.branch,
        during.metric_name,
        pre.metric_name AS metric_name_pre
    FROM (
        `test_experiment_enrollments_1` during
        LEFT JOIN `table_pre` pre
        USING (analysis_id, branch)
    )
    WHERE during.metric_name IS NOT NULL AND
    during.enrollment_date IS NOT NULL AND during.exposure_date IS NOT NULL"""
    )

    exp = Experiment(
        experimenter_slug="test_slug",
        type="v6",
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
        enrollment_end_date=dt.datetime(2019, 12, 7, tzinfo=pytz.utc),
        bucket_config=BucketConfig(
            randomization_unit=randomization_unit,
            namespace="testing",
            start=0,
            count=10,
            total=100,
        ),
    )

    actual_query = _empty_analysis([exp])._create_subset_metric_table_query_covariate(
        "test_experiment_enrollments_1",
        "all",
        metric,
        AnalysisBasis.EXPOSURES,
        AnalysisPeriod.PREENROLLMENT_WEEK,
        "metric_name",
    )

    assert expected_query == actual_query


def test_create_subset_metric_table_query_univariate_depends_on(experiments):
    upstream_1_metric = Metric(
        name="upstream_1",
        data_source=DataSource(name="test_data_source", from_expression="test.test"),
        select_expression="test",
        analysis_bases=[AnalysisBasis.ENROLLMENTS],
    )
    upstream_1 = Summary(upstream_1_metric, None, None)

    upstream_2_metric = Metric(
        name="upstream_2",
        data_source=DataSource(name="test_data_source", from_expression="test.test"),
        select_expression="test",
        analysis_bases=[AnalysisBasis.ENROLLMENTS],
    )

    upstream_2 = Summary(upstream_2_metric, None, None)

    metric = Metric(
        name="metric_name",
        data_source=DataSource(name="test_data_source", from_expression="test.test"),
        select_expression="test",
        analysis_bases=[AnalysisBasis.ENROLLMENTS],
        depends_on=[upstream_1, upstream_2],
    )

    expected_query = dedent(
        """
    SELECT branch, upstream_1, upstream_2, NULL AS metric_name
    FROM `test_experiment_enrollments_1`
    WHERE upstream_1 IS NOT NULL AND upstream_2 IS NOT NULL AND
    enrollment_date IS NOT NULL"""
    )

    actual_query = _empty_analysis(experiments)._create_subset_metric_table_query_univariate(
        "test_experiment_enrollments_1", "all", metric, AnalysisBasis.ENROLLMENTS
    )

    assert expected_query == actual_query


def test_create_subset_metric_table_query_covariate_depends_on(experiments, monkeypatch):
    monkeypatch.setattr(
        "jetstream.analysis.Analysis._table_name", MagicMock(return_value="table_pre")
    )
    upstream_1_metric = Metric(
        name="upstream_1",
        data_source=DataSource(name="test_data_source", from_expression="test.test"),
        select_expression="test",
        analysis_bases=[AnalysisBasis.ENROLLMENTS],
    )
    upstream_1 = Summary(upstream_1_metric, None, None)

    upstream_2_metric = Metric(
        name="upstream_2",
        data_source=DataSource(name="test_data_source", from_expression="test.test"),
        select_expression="test",
        analysis_bases=[AnalysisBasis.ENROLLMENTS],
    )

    upstream_2 = Summary(upstream_2_metric, None, None)

    metric = Metric(
        name="metric_name",
        data_source=DataSource(name="test_data_source", from_expression="test.test"),
        select_expression="test",
        analysis_bases=[AnalysisBasis.ENROLLMENTS],
        depends_on=[upstream_1, upstream_2],
    )

    with pytest.raises(
        ValueError,
        match=r"metrics with dependencies are not currently supported for covariate adjustment",
    ):
        _empty_analysis(experiments)._create_subset_metric_table_query_covariate(
            "test_experiment_enrollments_1",
            "all",
            metric,
            AnalysisBasis.ENROLLMENTS,
            AnalysisPeriod.PREENROLLMENT_WEEK,
            "metric_name",
        )


def test_create_subset_metric_table_query_univariate_unsupported_analysis_basis(
    experiments,
):
    metric = Metric(
        name="metric_name",
        data_source=DataSource(name="test_data_source", from_expression="test.test"),
        select_expression="test",
        analysis_bases=[AnalysisBasis.EXPOSURES],
    )
    analysis_basis = "non-basis"
    error_str = (
        f"AnalysisBasis {analysis_basis} not valid"
        + f"Allowed values are: {[AnalysisBasis.ENROLLMENTS, AnalysisBasis.EXPOSURES]}"
    )
    with pytest.raises(ValueError, match=re.escape(error_str)):
        _empty_analysis(experiments)._create_subset_metric_table_query_univariate(
            "test_experiment_exposures_1", "all", metric, analysis_basis
        )


def test_create_subset_metric_table_query_covariate_unsupported_analysis_basis(
    experiments, monkeypatch
):
    monkeypatch.setattr(
        "jetstream.bigquery_client.BigQueryClient.table_exists",
        MagicMock(return_value=True),
    )
    metric = Metric(
        name="metric_name",
        data_source=DataSource(name="test_data_source", from_expression="test.test"),
        select_expression="test",
        analysis_bases=[AnalysisBasis.EXPOSURES],
    )
    analysis_basis = "non-basis"
    error_str = (
        f"AnalysisBasis {analysis_basis} not valid"
        + f"Allowed values are: {[AnalysisBasis.ENROLLMENTS, AnalysisBasis.EXPOSURES]}"
    )
    with pytest.raises(ValueError, match=re.escape(error_str)):
        _empty_analysis(experiments)._create_subset_metric_table_query_covariate(
            "test_experiment_exposures_1",
            "all",
            metric,
            analysis_basis,
            AnalysisPeriod.PREENROLLMENT_WEEK,
            "metric_name",
        )


def test_create_subset_metric_table_query_use_covariate(experiments, monkeypatch):
    wrong_method = Mock(side_effect=Exception("the wrong query builder was called"))
    right_method = Mock()

    summary = MagicMock()
    summary.statistic.params = {
        "covariate_adjustment": {"metric": "my_metric", "period": "preenrollment_week"}
    }

    # all is correct and the function should call the covariate builder
    monkeypatch.setattr(
        "jetstream.analysis.Analysis._create_subset_metric_table_query_univariate",
        wrong_method,
    )
    monkeypatch.setattr(
        "jetstream.analysis.Analysis._create_subset_metric_table_query_covariate",
        right_method,
    )

    _empty_analysis(experiments)._create_subset_metric_table_query(
        "test_experiment_enrollments_1",
        "all",
        summary,
        AnalysisBasis.ENROLLMENTS,
        AnalysisPeriod.OVERALL,
    )


@pytest.mark.parametrize(("randomization_unit"), list(RandomizationUnit))
def test_create_subset_metric_table_query_use_covariate_explicit_metric(
    randomization_unit, monkeypatch
):
    monkeypatch.setattr(
        "jetstream.analysis.Analysis._table_name", MagicMock(return_value="table_pre")
    )
    monkeypatch.setattr(
        "jetstream.bigquery_client.BigQueryClient.table_exists",
        MagicMock(return_value=True),
    )

    summary = MagicMock()
    summary.statistic.params = {
        "covariate_adjustment": {"metric": "my_metric", "period": "preenrollment_week"}
    }

    metric = Metric(
        name="metric_name",
        data_source=DataSource(name="test_data_source", from_expression="test.test"),
        select_expression="test",
        analysis_bases=[AnalysisBasis.ENROLLMENTS],
    )
    summary.metric = metric

    expected_query = dedent(
        """
    SELECT
        during.branch,
        during.metric_name,
        pre.my_metric AS my_metric_pre
    FROM (
        `test_experiment_enrollments_1` during
        LEFT JOIN `table_pre` pre
        USING (analysis_id, branch)
    )
    WHERE during.metric_name IS NOT NULL AND
    during.enrollment_date IS NOT NULL"""
    )

    exp = Experiment(
        experimenter_slug="test_slug",
        type="v6",
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
        enrollment_end_date=dt.datetime(2019, 12, 7, tzinfo=pytz.utc),
        bucket_config=BucketConfig(
            randomization_unit=randomization_unit,
            namespace="testing",
            start=0,
            count=10,
            total=100,
        ),
    )

    actual_query = _empty_analysis([exp])._create_subset_metric_table_query(
        "test_experiment_enrollments_1",
        "all",
        summary,
        AnalysisBasis.ENROLLMENTS,
        AnalysisPeriod.OVERALL,
    )

    assert expected_query == actual_query


@pytest.mark.parametrize(("randomization_unit"), list(RandomizationUnit))
def test_create_subset_metric_table_query_use_covariate_implicit_metric(
    randomization_unit, monkeypatch
):
    monkeypatch.setattr(
        "jetstream.analysis.Analysis._table_name", MagicMock(return_value="table_pre")
    )
    monkeypatch.setattr(
        "jetstream.bigquery_client.BigQueryClient.table_exists",
        MagicMock(return_value=True),
    )

    summary = MagicMock()
    summary.statistic.params = {"covariate_adjustment": {"period": "preenrollment_week"}}

    metric = Metric(
        name="metric_name",
        data_source=DataSource(name="test_data_source", from_expression="test.test"),
        select_expression="test",
        analysis_bases=[AnalysisBasis.ENROLLMENTS],
    )
    summary.metric = metric

    expected_query = dedent(
        """
    SELECT
        during.branch,
        during.metric_name,
        pre.metric_name AS metric_name_pre
    FROM (
        `test_experiment_enrollments_1` during
        LEFT JOIN `table_pre` pre
        USING (analysis_id, branch)
    )
    WHERE during.metric_name IS NOT NULL AND
    during.enrollment_date IS NOT NULL"""
    )

    exp = Experiment(
        experimenter_slug="test_slug",
        type="v6",
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
        enrollment_end_date=dt.datetime(2019, 12, 7, tzinfo=pytz.utc),
        bucket_config=BucketConfig(
            randomization_unit=randomization_unit,
            namespace="testing",
            start=0,
            count=10,
            total=100,
        ),
    )

    actual_query = _empty_analysis([exp])._create_subset_metric_table_query(
        "test_experiment_enrollments_1",
        "all",
        summary,
        AnalysisBasis.ENROLLMENTS,
        AnalysisPeriod.OVERALL,
    )

    assert expected_query == actual_query


def test_create_subset_metric_table_query_use_univariate(experiments, monkeypatch):
    wrong_method = Mock(side_effect=Exception("the wrong query builder was called"))
    right_method = Mock()

    summary = MagicMock()
    summary.statistic.params = {}

    # no configured covariate_adjustment parameter, use univariate
    monkeypatch.setattr(
        "jetstream.analysis.Analysis._create_subset_metric_table_query_covariate",
        wrong_method,
    )
    monkeypatch.setattr(
        "jetstream.analysis.Analysis._create_subset_metric_table_query_univariate",
        right_method,
    )

    _empty_analysis(experiments)._create_subset_metric_table_query(
        "test_experiment_enrollments_1",
        "all",
        summary,
        AnalysisBasis.ENROLLMENTS,
        AnalysisPeriod.OVERALL,
    )


@pytest.mark.parametrize(("randomization_unit"), list(RandomizationUnit))
def test_create_subset_metric_table_query_complete_covariate(randomization_unit, monkeypatch):
    monkeypatch.setattr(
        "jetstream.analysis.Analysis._table_name", MagicMock(return_value="table_pre")
    )
    monkeypatch.setattr(
        "jetstream.bigquery_client.BigQueryClient.table_exists",
        MagicMock(return_value=True),
    )

    summary = MagicMock()
    summary.statistic.params = {
        "covariate_adjustment": {
            "metric": "my_metric",
            "period": "preenrollment_days28",
        }
    }

    metric = Metric(
        name="metric_name",
        data_source=DataSource(name="test_data_source", from_expression="test.test"),
        select_expression="test",
        analysis_bases=[AnalysisBasis.ENROLLMENTS],
    )
    summary.metric = metric

    expected_query = dedent(
        """
    SELECT
        during.branch,
        during.metric_name,
        pre.my_metric AS my_metric_pre
    FROM (
        `test_experiment_enrollments_1` during
        LEFT JOIN `table_pre` pre
        USING (analysis_id, branch)
    )
    WHERE during.metric_name IS NOT NULL AND
    during.enrollment_date IS NOT NULL"""
    )

    exp = Experiment(
        experimenter_slug="test_slug",
        type="v6",
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
        enrollment_end_date=dt.datetime(2019, 12, 7, tzinfo=pytz.utc),
        bucket_config=BucketConfig(
            randomization_unit=randomization_unit,
            namespace="testing",
            start=0,
            count=10,
            total=100,
        ),
    )

    actual_query = _empty_analysis([exp])._create_subset_metric_table_query(
        "test_experiment_enrollments_1",
        "all",
        summary,
        AnalysisBasis.ENROLLMENTS,
        AnalysisPeriod.PREENROLLMENT_WEEK,
    )

    assert expected_query == actual_query


def test_create_subset_metric_table_query_complete_univariate(experiments):
    summary = MagicMock()
    summary.statistic.params = {}

    metric = Metric(
        name="metric_name",
        data_source=DataSource(name="test_data_source", from_expression="test.test"),
        select_expression="test",
        analysis_bases=[AnalysisBasis.ENROLLMENTS],
    )
    summary.metric = metric

    expected_query = dedent(
        """
    SELECT branch, metric_name
    FROM `test_experiment_enrollments_1`
    WHERE metric_name IS NOT NULL AND
    enrollment_date IS NOT NULL"""
    )

    actual_query = _empty_analysis(experiments)._create_subset_metric_table_query(
        "test_experiment_enrollments_1",
        "all",
        summary,
        AnalysisBasis.ENROLLMENTS,
        AnalysisPeriod.PREENROLLMENT_WEEK,
    )

    assert expected_query == actual_query
