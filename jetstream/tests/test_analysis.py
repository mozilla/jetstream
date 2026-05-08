import datetime as dt
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
from mozilla_nimbus_schemas.experimenter_apis.experiments import RandomizationUnit
from mozilla_nimbus_schemas.jetstream import AnalysisBasis

import jetstream.analysis
from jetstream.analysis import Analysis
from jetstream.config import ConfigLoader
from jetstream.errors import (
    EnrollmentNotCompleteException,
    ExplicitSkipException,
    HighPopulationException,
    UnsupportedApplicationException,
)
from jetstream.metric import Metric

logger = logging.getLogger(__name__)


def _empty_analysis(experiments):
    exp: Experiment = experiments[0]
    config = AnalysisSpec.default_for_experiment(exp, ConfigLoader.configs).resolve(
        exp, ConfigLoader.configs
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


def test_validate_doesnt_explode(experiments, monkeypatch):
    m = Mock()
    m.return_value = -1
    monkeypatch.setattr(jetstream.analysis, "dry_run_query", m)
    exp = experiments[0]
    config = AnalysisSpec.default_for_experiment(exp, ConfigLoader.configs).resolve(
        exp, ConfigLoader.configs
    )
    Analysis("spam", "eggs", config).validate()
    assert m.call_count == 2


def test_validate_doesnt_explode_discrete_metric(experiments, monkeypatch):
    m = Mock()
    m.return_value = -1
    monkeypatch.setattr(jetstream.analysis, "dry_run_query", m)
    exp = experiments[0]
    config = AnalysisSpec.default_for_experiment(exp, ConfigLoader.configs).resolve(
        exp, ConfigLoader.configs
    )

    def bypass_mp_pool(_pool, func, args):
        class MockApplyResult:
            def __init__(self, func, args):
                self._func = func
                self._args = args

            def get(self, timeout=0):
                return self._func(*self._args)

        return MockApplyResult(func, args)

    monkeypatch.setattr("multiprocessing.pool.Pool.apply_async", bypass_mp_pool)

    Analysis("spam", "eggs", config).validate(metric_slugs=["active_hours", "retained"])

    # 1 for enrollments + 2 metrics
    assert m.call_count == 3


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
    exp = experiments[3]
    config = AnalysisSpec.default_for_experiment(exp, ConfigLoader.configs).resolve(
        exp, ConfigLoader.configs
    )

    with pytest.raises(HighPopulationException):
        Analysis("spam", "eggs", config).check_runnable()


def test_check_runnable_invalid_app(experiments):
    exp = Experiment(
        experimenter_slug="test_slug",
        type="v6",
        status="Live",
        start_date=dt.datetime(2019, 12, 1, tzinfo=pytz.utc),
        end_date=dt.datetime(2020, 3, 1, tzinfo=pytz.utc),
        proposed_enrollment=7,
        branches=[],
        normandy_slug="normandy-test-slug",
        reference_branch=None,
        is_high_population=False,
        app_name="invalid_app",
        app_id="invalid-app",
    )
    config = AnalysisSpec.default_for_experiment(exp, ConfigLoader.configs).resolve(
        exp, ConfigLoader.configs
    )

    with pytest.raises(UnsupportedApplicationException, match="normandy-test-slug -> invalid_app"):
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
        pytest.fail(f"Raised {e} (are you authenticated?)")


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
            return -1

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
            return -1

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
            return -1

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
            return -1

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
    FROM `test_experiment_enrollments_1` m

    WHERE metric_name IS NOT NULL AND
    enrollment_date IS NOT NULL"""
    )

    actual_query = _empty_analysis(experiments)._create_subset_metric_table_query_univariate(
        "test_experiment_enrollments_1",
        "all",
        metric,
        AnalysisBasis.ENROLLMENTS,
        AnalysisPeriod.WEEK,
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
        AnalysisPeriod.WEEK,
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
    FROM `test_experiment_enrollments_1` m

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
        AnalysisPeriod.WEEK,
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
    FROM `test_experiment_enrollments_1` m

    WHERE metric_name IS NOT NULL AND
    enrollment_date IS NOT NULL
    AND mysegment = TRUE"""
    )

    actual_query = _empty_analysis(experiments)._create_subset_metric_table_query_univariate(
        "test_experiment_enrollments_1",
        "mysegment",
        metric,
        AnalysisBasis.ENROLLMENTS,
        AnalysisPeriod.WEEK,
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
        AnalysisPeriod.WEEK,
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
    FROM `test_experiment_exposures_1` m

    WHERE metric_name IS NOT NULL AND
    enrollment_date IS NOT NULL AND m.exposure_date IS NOT NULL"""
    )

    actual_query = _empty_analysis(experiments)._create_subset_metric_table_query_univariate(
        "test_experiment_exposures_1",
        "all",
        metric,
        AnalysisBasis.EXPOSURES,
        AnalysisPeriod.WEEK,
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
        AnalysisPeriod.WEEK,
    )

    assert expected_query == actual_query


def test_create_subset_metric_table_query_univariate_depends_on(experiments, monkeypatch):
    # Return different dependency table names based on the data source argument (args[3])
    # so that both upstream metrics produce distinct LEFT JOINs.
    def table_name_side_effect(*args, **kwargs):
        data_source = args[3] if len(args) > 3 else kwargs.get("metric")
        if data_source == "data_source_a":
            return "dep_table_a_1"
        return "dep_table_b_1"

    monkeypatch.setattr(
        "jetstream.analysis.Analysis._table_name",
        MagicMock(side_effect=table_name_side_effect),
    )

    upstream_1_metric = Metric(
        name="upstream_1",
        data_source=DataSource(name="data_source_a", from_expression="test.test"),
        select_expression="test",
        analysis_bases=[AnalysisBasis.ENROLLMENTS],
    )
    upstream_1 = Summary(upstream_1_metric, None, None)

    upstream_2_metric = Metric(
        name="upstream_2",
        data_source=DataSource(name="data_source_b", from_expression="test.test"),
        select_expression="test",
        analysis_bases=[AnalysisBasis.ENROLLMENTS],
    )
    upstream_2 = Summary(upstream_2_metric, None, None)

    metric = Metric(
        name="ratio_metric",
        data_source=DataSource(name="test_data_source", from_expression="test.test"),
        select_expression=None,
        analysis_bases=[AnalysisBasis.ENROLLMENTS],
        depends_on=[upstream_1, upstream_2],
    )

    actual_query = _empty_analysis(experiments)._create_subset_metric_table_query_univariate(
        "dep_table_a_1",
        "all",
        metric,
        AnalysisBasis.ENROLLMENTS,
        AnalysisPeriod.DAY,
        discrete_metrics=True,
    )

    # can't assert whole query because order of metrics is not guaranteed
    assert "upstream_1" in actual_query
    assert "upstream_2" in actual_query
    assert "NULL AS ratio_metric" in actual_query
    assert "FROM `dep_table_a_1` m" in actual_query
    assert "LEFT JOIN `dep_table_b_1`" in actual_query
    assert "analysis_id" in actual_query
    assert "analysis_window_start" in actual_query
    assert "enrollment_date IS NOT NULL" in actual_query

    assert "LEFT JOIN `dep_table_a_1`" not in actual_query


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
            AnalysisPeriod.WEEK,
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
            "test_experiment_exposures_1",
            "all",
            metric,
            analysis_basis,
            AnalysisPeriod.WEEK,
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
            AnalysisPeriod.WEEK,
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
        AnalysisPeriod.WEEK,
    )

    assert expected_query == actual_query


@pytest.mark.parametrize(("randomization_unit"), list(RandomizationUnit))
def test_create_subset_metric_table_query_covariate_fallback(randomization_unit, monkeypatch):
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
    SELECT branch, metric_name
    FROM `test_experiment_enrollments_1` m

    WHERE metric_name IS NOT NULL AND
    enrollment_date IS NOT NULL"""
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

    # covariate statistic should fall back to univariate if current period is preenrollment
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
    FROM `test_experiment_enrollments_1` m

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


def test_run_continues_after_task_failure(experiments, monkeypatch, caplog):
    import threading

    exp = experiments[0]
    config = AnalysisSpec.default_for_experiment(exp, ConfigLoader.configs).resolve(
        exp, ConfigLoader.configs
    )
    analysis = Analysis("test", "test", config)

    monkeypatch.setattr("jetstream.analysis.Analysis.ensure_enrollments", Mock())
    monkeypatch.setattr("jetstream.analysis._dask_cluster", None)

    # Use threads (processes=False) so monkeypatches are visible inside dask workers.
    original_local_cluster = jetstream.analysis.LocalCluster
    monkeypatch.setattr(
        "jetstream.analysis.LocalCluster",
        lambda **kwargs: original_local_cluster(**{**kwargs, "processes": False, "n_workers": 1}),
    )

    mock_bq = MagicMock()
    monkeypatch.setattr("jetstream.analysis.BigQueryClient", Mock(return_value=mock_bq))

    # Raise on the first _table_name call that runs inside a dask worker thread.
    # _table_name is also called from the main thread during graph construction
    # (with the same arguments), so the thread check prevents a premature failure
    # before any task has been submitted.
    has_failed = threading.Event()
    main_thread = threading.main_thread()
    original_table_name = Analysis._table_name

    def patched_table_name(
        self, window_period, window_index, analysis_basis=None, metric=None, statistics=False
    ):
        if (
            metric is not None
            and not statistics
            and not has_failed.is_set()
            and threading.current_thread() is not main_thread
        ):
            has_failed.set()
            raise RuntimeError(f"simulated failure for data source {metric}")
        return original_table_name(
            self,
            window_period,
            window_index,
            analysis_basis=analysis_basis,
            metric=metric,
            statistics=statistics,
        )

    monkeypatch.setattr("jetstream.analysis.Analysis._table_name", patched_table_name)

    with caplog.at_level(logging.ERROR):
        analysis.run(
            current_date=dt.datetime(2020, 1, 10, tzinfo=pytz.utc),
            dry_run=True,
            discrete_metrics=True,
        )

    assert "simulated failure for data source" in caplog.text
    # Cascade failures (publish_view bound to the failing metric) share the same
    # exception and must not generate duplicate log entries.
    assert caplog.text.count("A task failed during analysis") == 1
    # publish_view ran for periods where all data sources succeeded
    assert mock_bq.execute.called


def test_metric_slugs_adds_depends_on_metrics(experiments, monkeypatch):
    config = AnalysisSpec.default_for_experiment(experiments[0], ConfigLoader.configs).resolve(
        experiments[0], ConfigLoader.configs
    )

    upstream_1_metric = Metric(
        name="upstream_1",
        data_source=DataSource(name="test_data_source", from_expression="test.test"),
        select_expression="test",
        analysis_bases=[AnalysisBasis.ENROLLMENTS],
    )
    upstream_2_metric = Metric(
        name="upstream_2",
        data_source=DataSource(name="test_data_source", from_expression="test.test"),
        select_expression="test",
        analysis_bases=[AnalysisBasis.ENROLLMENTS],
    )
    ratio_metric = Metric(
        name="ratio_metric",
        data_source=DataSource(name="test_data_source", from_expression="test.test"),
        select_expression=None,
        analysis_bases=[AnalysisBasis.ENROLLMENTS],
        depends_on=[
            Summary(upstream_1_metric, None, None),
            Summary(upstream_2_metric, None, None),
        ],
    )
    config.metrics[AnalysisPeriod.WEEK].append(Summary(ratio_metric, MagicMock(), []))

    monkeypatch.setattr("jetstream.analysis.Analysis.ensure_enrollments", Mock())
    monkeypatch.setattr(
        "jetstream.analysis.Analysis._get_timelimits_if_ready", MagicMock(return_value=MagicMock())
    )
    monkeypatch.setattr("jetstream.analysis.Analysis.calculate_metrics", MagicMock())
    monkeypatch.setattr("jetstream.analysis.Analysis.save_statistics", MagicMock())
    monkeypatch.setattr("jetstream.analysis.Analysis.publish_view", MagicMock())
    monkeypatch.setattr("jetstream.analysis.bind", lambda x, deps: x)
    monkeypatch.setattr("jetstream.analysis.LocalCluster", MagicMock())
    monkeypatch.setattr("jetstream.analysis.Client", MagicMock())

    metric_slugs = ["ratio_metric"]
    Analysis("test", "test", config).run(
        current_date=dt.datetime(2020, 1, 1, tzinfo=pytz.utc),
        dry_run=True,
        metric_slugs=metric_slugs,
    )

    assert "upstream_1" in metric_slugs
    assert "upstream_2" in metric_slugs
    assert "ratio_metric" in metric_slugs


def test_subset_metric_table_prerequisites_simple(experiments):
    """A plain metric with no covariate and no depends_on has no additional prerequisites."""
    summary = MagicMock()
    summary.statistic.params = {}
    summary.metric.depends_on = None

    prereqs = _empty_analysis(experiments)._subset_metric_table_prerequisites(
        summary,
        "normandy_test_slug_enrollments_week_1",
        AnalysisBasis.ENROLLMENTS,
        AnalysisPeriod.WEEK,
        False,
    )

    assert prereqs == set()


def test_subset_metric_table_prerequisites_covariate(experiments):
    """Covariate adjustment against preenrollment_week returns the covariate table name."""
    summary = MagicMock()
    summary.statistic.params = {
        "covariate_adjustment": {"metric": "my_metric", "period": "preenrollment_week"}
    }
    summary.metric.depends_on = None

    analysis = _empty_analysis(experiments)
    prereqs = analysis._subset_metric_table_prerequisites(
        summary,
        "normandy_test_slug_enrollments_week_1",
        AnalysisBasis.ENROLLMENTS,
        AnalysisPeriod.WEEK,
        False,
    )

    expected_covariate_table = analysis._table_name(
        AnalysisPeriod.PREENROLLMENT_WEEK.value, 1, AnalysisBasis.ENROLLMENTS
    )
    assert prereqs == {expected_covariate_table}


def test_subset_metric_table_prerequisites_covariate_skipped_for_preenrollment_period(experiments):
    """When the current period is a preenrollment period, covariate adjustment is not applied,
    so no extra prerequisite table should be returned."""
    summary = MagicMock()
    summary.statistic.params = {
        "covariate_adjustment": {"metric": "my_metric", "period": "preenrollment_week"}
    }
    summary.metric.depends_on = None

    prereqs = _empty_analysis(experiments)._subset_metric_table_prerequisites(
        summary,
        "normandy_test_slug_enrollments_preenrollment_week_1",
        AnalysisBasis.ENROLLMENTS,
        AnalysisPeriod.PREENROLLMENT_WEEK,
        False,
    )

    assert prereqs == set()


def test_subset_metric_table_prerequisites_discrete_depends_on(experiments):
    """For a discrete ratio metric whose dependencies span two data sources, the prerequisite
    list contains the cross-DS dependency table but not the primary table."""
    upstream_a = Metric(
        name="upstream_a",
        data_source=DataSource(name="data_source_a", from_expression="test.test"),
        select_expression="test",
        analysis_bases=[AnalysisBasis.ENROLLMENTS],
    )
    upstream_b = Metric(
        name="upstream_b",
        data_source=DataSource(name="data_source_b", from_expression="test.test"),
        select_expression="test",
        analysis_bases=[AnalysisBasis.ENROLLMENTS],
    )

    from jetstream.statistics import Summary as JetstreamSummary

    ratio_metric = Metric(
        name="ratio_metric",
        data_source=DataSource(name="data_source_a", from_expression="test.test"),
        select_expression=None,
        analysis_bases=[AnalysisBasis.ENROLLMENTS],
        depends_on=[
            JetstreamSummary(upstream_a, None, None),
            JetstreamSummary(upstream_b, None, None),
        ],
    )

    summary = MagicMock()
    summary.statistic.params = {}
    summary.metric = ratio_metric

    analysis = _empty_analysis(experiments)
    primary_table = analysis._table_name(
        AnalysisPeriod.WEEK.value, 1, AnalysisBasis.ENROLLMENTS, metric="data_source_a"
    )
    cross_ds_table = analysis._table_name(
        AnalysisPeriod.WEEK.value, 1, AnalysisBasis.ENROLLMENTS, metric="data_source_b"
    )

    prereqs = analysis._subset_metric_table_prerequisites(
        summary,
        primary_table,
        AnalysisBasis.ENROLLMENTS,
        AnalysisPeriod.WEEK,
        True,
    )

    assert cross_ds_table in prereqs
    assert primary_table not in prereqs


def test_run_covariate_bind_wires_cross_period_dep(experiments, monkeypatch):
    """When a metric uses covariate adjustment from preenrollment_week, run() must pass
    the preenrollment writer Delayed as a bind prerequisite for the weekly subset task."""
    config = AnalysisSpec.default_for_experiment(experiments[0], ConfigLoader.configs).resolve(
        experiments[0], ConfigLoader.configs
    )

    metric = Metric(
        name="active_hours",
        data_source=DataSource(name="clients_daily", from_expression="test.test"),
        select_expression="SUM(ah)",
        analysis_bases=[AnalysisBasis.ENROLLMENTS],
    )
    cov_statistic = MagicMock()
    cov_statistic.params = {
        "covariate_adjustment": {"metric": "active_hours", "period": "preenrollment_week"}
    }
    from jetstream.statistics import Summary as JetstreamSummary

    plain_statistic = MagicMock()
    plain_statistic.params = {}
    config.metrics = {
        AnalysisPeriod.WEEK: [JetstreamSummary(metric, cov_statistic, [])],
        AnalysisPeriod.PREENROLLMENT_WEEK: [JetstreamSummary(metric, plain_statistic, [])],
    }

    bind_calls: list[tuple] = []

    def capturing_bind(thing, deps):
        bind_calls.append((thing, list(deps)))
        return thing

    stats_mock = MagicMock()
    stats_mock.model_dump.return_value = []

    monkeypatch.setattr("jetstream.analysis.bind", capturing_bind)
    monkeypatch.setattr("jetstream.analysis.Analysis.ensure_enrollments", Mock())
    monkeypatch.setattr(
        "jetstream.analysis.Analysis._get_timelimits_if_ready", MagicMock(return_value=MagicMock())
    )
    monkeypatch.setattr("jetstream.analysis.Analysis.calculate_metrics", MagicMock())
    monkeypatch.setattr(
        "jetstream.analysis.Analysis.calculate_statistics", MagicMock(return_value=stats_mock)
    )
    monkeypatch.setattr("jetstream.analysis.Analysis.counts", MagicMock(return_value=stats_mock))
    monkeypatch.setattr("jetstream.analysis.Analysis.save_statistics", MagicMock())
    monkeypatch.setattr("jetstream.analysis.Analysis.publish_view", MagicMock())
    monkeypatch.setattr("jetstream.analysis.LocalCluster", MagicMock())
    monkeypatch.setattr("jetstream.analysis.Client", MagicMock())

    Analysis("test", "test", config).run(
        current_date=dt.datetime(2020, 1, 1, tzinfo=pytz.utc),
    )

    # At least one bind call for a subset_metric_table task should carry a non-empty deps list,
    # containing the preenrollment writer as the cross-period prerequisite.
    subset_binds_with_prereqs = [deps for _, deps in bind_calls if deps]
    assert subset_binds_with_prereqs, (
        "Expected at least one subset_metric_table bind call with prerequisites, "
        "but all bind calls had empty deps. The covariate cross-period edge is missing."
    )
