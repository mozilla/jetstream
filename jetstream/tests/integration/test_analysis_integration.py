import datetime
import datetime as dt
import json
from pathlib import Path

import dask
import jsonschema
import mozanalysis
import pytest
import pytz
from metric_config_parser.analysis import AnalysisSpec
from metric_config_parser.data_source import DataSource
from metric_config_parser.experiment import Branch, Experiment
from metric_config_parser.metric import AnalysisPeriod, Summary
from metric_config_parser.segment import Segment, SegmentDataSource
from metric_config_parser.statistic import Statistic
from mozanalysis.metrics import agg_sum
from mozilla_nimbus_schemas.jetstream import AnalysisBasis

from jetstream.analysis import Analysis
from jetstream.config import ConfigLoader
from jetstream.exposure_signal import ExposureSignal
from jetstream.logging import LogConfiguration
from jetstream.metric import Metric

TEST_DIR = Path(__file__).parent.parent


class TestAnalysisIntegration:
    def analysis_mock_run(
        self, monkeypatch, config, static_dataset, temporary_dataset, project_id, log_config=None
    ):
        orig_enrollments = mozanalysis.experiment.Experiment.build_enrollments_query
        orig_metrics = mozanalysis.experiment.Experiment.build_metrics_query

        def build_enrollments_query_test_project(instance, *args, **kwargs):
            # to use the test project and dataset, we need to change the SQL query
            # generated by mozanalysis
            query = orig_enrollments(instance, *args)
            query = query.replace("moz-fx-data-shared-prod", project_id)
            query = query.replace("telemetry", static_dataset)
            return query

        def build_metrics_query_test_project(instance, *args, **kwargs):
            # to use the test project and dataset, we need to change the SQL query
            # generated by mozanalysis
            query = orig_metrics(instance, *args)
            query = query.replace("moz-fx-data-shared-prod", project_id)
            query = query.replace("telemetry", static_dataset)
            return query

        orig_cluster = dask.distributed.LocalCluster.__init__

        def mock_local_cluster(
            instance, dashboard_address, processes, threads_per_worker, *args, **kwargs
        ):
            # if processes are used then `build_query_test_project` gets ignored
            return orig_cluster(
                instance,
                dashboard_address=dashboard_address,
                processes=False,
                threads_per_worker=threads_per_worker,
            )

        analysis = Analysis(project_id, temporary_dataset, config, log_config)

        monkeypatch.setattr(
            mozanalysis.experiment.Experiment,
            "build_enrollments_query",
            build_enrollments_query_test_project,
        )
        monkeypatch.setattr(
            mozanalysis.experiment.Experiment,
            "build_metrics_query",
            build_metrics_query_test_project,
        )
        monkeypatch.setattr(dask.distributed.LocalCluster, "__init__", mock_local_cluster)

        analysis.ensure_enrollments(dt.datetime(2020, 4, 12, tzinfo=pytz.utc))
        analysis.run(dt.datetime(2020, 4, 12, tzinfo=pytz.utc), dry_run=False)

    def test_metrics(self, monkeypatch, client, project_id, static_dataset, temporary_dataset):
        experiment = Experiment(
            experimenter_slug="test-experiment",
            type="rollout",
            status="Live",
            start_date=dt.datetime(2020, 3, 30, tzinfo=pytz.utc),
            end_date=dt.datetime(2020, 6, 1, tzinfo=pytz.utc),
            proposed_enrollment=7,
            is_enrollment_paused=True,
            branches=[Branch(slug="branch1", ratio=0.5), Branch(slug="branch2", ratio=0.5)],
            reference_branch="branch2",
            normandy_slug="test-experiment",
            is_high_population=False,
            app_name="firefox_desktop",
            app_id="firefox-desktop",
        )

        config = AnalysisSpec().resolve(experiment, ConfigLoader.configs)

        test_clients_daily = DataSource(
            name="clients_daily",
            from_expression=f"`{project_id}.test_data.clients_daily`",
        )

        test_active_hours = Metric(
            name="active_hours",
            data_source=test_clients_daily,
            select_expression=agg_sum("active_hours_sum"),
            analysis_bases=[AnalysisBasis.EXPOSURES, AnalysisBasis.ENROLLMENTS],
        )

        stat = Statistic(name="bootstrap_mean", params={})

        config.metrics = {AnalysisPeriod.WEEK: [Summary(test_active_hours, stat)]}

        self.analysis_mock_run(monkeypatch, config, static_dataset, temporary_dataset, project_id)

        query_job = client.client.query(
            f"""
            SELECT
              *
            FROM `{project_id}.{temporary_dataset}.test_experiment_exposures_week_1`
            ORDER BY enrollment_date DESC
        """
        )

        expected_metrics_results = [
            {
                "client_id": "bbbb",
                "branch": "branch2",
                "enrollment_date": datetime.date(2020, 4, 3),
                "num_enrollment_events": 1,
                "analysis_window_start": 0,
                "analysis_window_end": 6,
            },
            {
                "client_id": "aaaa",
                "branch": "branch1",
                "enrollment_date": datetime.date(2020, 4, 2),
                "num_enrollment_events": 1,
                "analysis_window_start": 0,
                "analysis_window_end": 6,
            },
        ]

        r = query_job.result()

        for i, row in enumerate(r):
            for k, v in expected_metrics_results[i].items():
                assert row[k] == v

        assert (
            client.client.get_table(
                f"{project_id}.{temporary_dataset}.test_experiment_exposures_weekly"
            )
            is not None
        )
        assert (
            client.client.get_table(
                f"{project_id}.{temporary_dataset}.test_experiment_enrollments_weekly"
            )
            is not None
        )
        assert (
            client.client.get_table(
                f"{project_id}.{temporary_dataset}.statistics_test_experiment_week_1"
            )
            is not None
        )

        stats = client.client.list_rows(
            f"{project_id}.{temporary_dataset}.statistics_test_experiment_week_1"
        ).to_dataframe()

        count_by_branch = stats.query("statistic == 'count'").set_index("branch")
        assert count_by_branch.loc["branch1", "point"][0] == 1.0
        assert count_by_branch.loc["branch2", "point"][0] == 1.0

        if count_by_branch.loc["branch2", "analysis_basis"][0] == "exposures":
            assert count_by_branch.loc["branch2", "analysis_basis"][1] == "enrollments"
        else:
            assert count_by_branch.loc["branch2", "analysis_basis"][0] == "enrollments"
            assert count_by_branch.loc["branch2", "analysis_basis"][1] == "exposures"

        assert (
            client.client.get_table(
                f"{project_id}.{temporary_dataset}.statistics_test_experiment_weekly"
            )
            is not None
        )

    def test_metrics_preenrollment(
        self, monkeypatch, client, project_id, static_dataset, temporary_dataset
    ):
        experiment = Experiment(
            experimenter_slug="test-experiment",
            type="rollout",
            status="Live",
            start_date=dt.datetime(2020, 3, 30, tzinfo=pytz.utc),
            end_date=dt.datetime(2020, 6, 1, tzinfo=pytz.utc),
            proposed_enrollment=12,
            is_enrollment_paused=True,
            branches=[Branch(slug="branch1", ratio=0.5), Branch(slug="branch2", ratio=0.5)],
            reference_branch="branch2",
            normandy_slug="test-experiment",
            is_high_population=False,
            app_name="firefox_desktop",
            app_id="firefox-desktop",
        )

        config = AnalysisSpec().resolve(experiment, ConfigLoader.configs)

        test_clients_daily = DataSource(
            name="clients_daily",
            from_expression=f"`{project_id}.test_data.clients_daily`",
        )

        test_active_hours = Metric(
            name="active_hours",
            data_source=test_clients_daily,
            select_expression=agg_sum("active_hours_sum"),
            analysis_bases=[AnalysisBasis.ENROLLMENTS],
        )

        stat = Statistic(name="bootstrap_mean", params={})

        config.metrics = {AnalysisPeriod.PREENROLLMENT_WEEK: [Summary(test_active_hours, stat)]}

        self.analysis_mock_run(monkeypatch, config, static_dataset, temporary_dataset, project_id)

        query_job = client.client.query(
            f"""
            SELECT
              *
            FROM `{project_id}.{temporary_dataset}.test_experiment_enrollments_week_preenrollment_1`
            ORDER BY enrollment_date DESC
        """
        )

        expected_metrics_results = [
            {
                "client_id": "bbbb",
                "branch": "branch2",
                "enrollment_date": datetime.date(2020, 4, 3),
                "num_enrollment_events": 1,
                "analysis_window_start": -7,
                "analysis_window_end": -1,
                "active_hours": 0.2,
            },
            {
                "client_id": "aaaa",
                "branch": "branch1",
                "enrollment_date": datetime.date(2020, 4, 2),
                "num_enrollment_events": 1,
                "analysis_window_start": -7,
                "analysis_window_end": -1,
                "active_hours": 2.8,
            },
        ]

        r = query_job.result()

        for i, row in enumerate(r):
            for k, v in expected_metrics_results[i].items():
                assert row[k] == v

    def test_metrics_with_exposure(
        self, monkeypatch, client, project_id, static_dataset, temporary_dataset
    ):
        experiment = Experiment(
            experimenter_slug="test-experiment",
            type="rollout",
            status="Live",
            start_date=dt.datetime(2020, 3, 30, tzinfo=pytz.utc),
            end_date=dt.datetime(2020, 6, 1, tzinfo=pytz.utc),
            proposed_enrollment=7,
            is_enrollment_paused=True,
            branches=[Branch(slug="branch1", ratio=0.5), Branch(slug="branch2", ratio=0.5)],
            reference_branch="branch2",
            normandy_slug="test-experiment",
            is_high_population=False,
            app_name="firefox_desktop",
            app_id="firefox-desktop",
        )

        config = AnalysisSpec().resolve(experiment, ConfigLoader.configs)

        test_clients_daily = DataSource(
            name="clients_daily",
            from_expression=f"`{project_id}.test_data.clients_daily`",
        )

        test_active_hours = Metric(
            name="active_hours",
            data_source=test_clients_daily,
            select_expression=agg_sum("active_hours_sum"),
            analysis_bases=[AnalysisBasis.EXPOSURES],
        )

        stat = Statistic(name="bootstrap_mean", params={})

        config.metrics = {AnalysisPeriod.WEEK: [Summary(test_active_hours, stat)]}
        config.experiment.exposure_signal = ExposureSignal(
            name="ad_exposure",
            data_source=test_clients_daily,
            select_expression="active_hours_sum > 0",
            friendly_name="Ad exposure",
            description="Clients have clicked on ad",
            window_start="enrollment_start",
            window_end="analysis_window_end",
        )

        self.analysis_mock_run(monkeypatch, config, static_dataset, temporary_dataset, project_id)

        query_job = client.client.query(
            f"""
            SELECT
              *
            FROM `{project_id}.{temporary_dataset}.test_experiment_exposures_week_1`
            ORDER BY enrollment_date DESC
        """
        )

        expected_metrics_results = [
            {
                "client_id": "bbbb",
                "branch": "branch2",
                "enrollment_date": datetime.date(2020, 4, 3),
                "num_enrollment_events": 1,
                "analysis_window_start": 0,
                "analysis_window_end": 6,
            },
            {
                "client_id": "aaaa",
                "branch": "branch1",
                "enrollment_date": datetime.date(2020, 4, 2),
                "num_enrollment_events": 1,
                "analysis_window_start": 0,
                "analysis_window_end": 6,
            },
        ]

        r = query_job.result()

        for i, row in enumerate(r):
            for k, v in expected_metrics_results[i].items():
                assert row[k] == v

        assert (
            client.client.get_table(
                f"{project_id}.{temporary_dataset}.test_experiment_exposures_weekly"
            )
            is not None
        )
        assert (
            client.client.get_table(
                f"{project_id}.{temporary_dataset}.statistics_test_experiment_week_1"
            )
            is not None
        )

        assert (
            client.client.get_table(
                f"{project_id}.{temporary_dataset}.statistics_test_experiment_weekly"
            )
            is not None
        )

    def test_metrics_with_depends_on(
        self, monkeypatch, client, project_id, static_dataset, temporary_dataset
    ):
        experiment = Experiment(
            experimenter_slug="test-experiment",
            type="rollout",
            status="Live",
            start_date=dt.datetime(2020, 3, 30, tzinfo=pytz.utc),
            end_date=dt.datetime(2020, 6, 1, tzinfo=pytz.utc),
            proposed_enrollment=7,
            is_enrollment_paused=True,
            branches=[Branch(slug="branch1", ratio=0.5), Branch(slug="branch2", ratio=0.5)],
            reference_branch="branch2",
            normandy_slug="test-experiment",
            is_high_population=False,
            app_name="firefox_desktop",
            app_id="firefox-desktop",
        )

        config = AnalysisSpec().resolve(experiment, ConfigLoader.configs)

        ratio_stat = Statistic(
            name="population_ratio",
            params={
                "numerator": "active_hours",
                "denominator": "active_hours_doubled",
                "num_samples": 10,
            },
        )
        bootstrap_stat = Statistic(name="bootstrap_mean", params={})

        test_clients_daily = DataSource(
            name="clients_daily",
            from_expression=f"`{project_id}.test_data.clients_daily`",
        )

        test_active_hours = Metric(
            name="active_hours",
            data_source=test_clients_daily,
            select_expression=agg_sum("active_hours_sum"),
            analysis_bases=[AnalysisBasis.EXPOSURES, AnalysisBasis.ENROLLMENTS],
        )

        test_active_hours_doubled = Metric(
            name="active_hours_doubled",
            data_source=test_clients_daily,
            select_expression=f'{agg_sum("active_hours_sum")} * 2',
            analysis_bases=[AnalysisBasis.EXPOSURES, AnalysisBasis.ENROLLMENTS],
        )

        test_active_hours_ratio = Metric(
            name="active_hours_ratio",
            select_expression=None,
            data_source=None,
            analysis_bases=[AnalysisBasis.EXPOSURES, AnalysisBasis.ENROLLMENTS],
            depends_on=[
                Summary(test_active_hours, bootstrap_stat),
                Summary(test_active_hours_doubled, bootstrap_stat),
            ],
        )

        config.metrics = {
            AnalysisPeriod.WEEK: [
                Summary(test_active_hours, bootstrap_stat),
                Summary(test_active_hours_doubled, bootstrap_stat),
                Summary(test_active_hours_ratio, ratio_stat),
            ]
        }

        self.analysis_mock_run(monkeypatch, config, static_dataset, temporary_dataset, project_id)

        query_job = client.client.query(
            f"""
            SELECT
              *
            FROM `{project_id}.{temporary_dataset}.test_experiment_enrollments_week_1`
            ORDER BY enrollment_date DESC
        """
        )

        expected_metrics_results = [
            {
                "client_id": "bbbb",
                "branch": "branch2",
                "enrollment_date": datetime.date(2020, 4, 3),
                "num_enrollment_events": 1,
                "analysis_window_start": 0,
                "analysis_window_end": 6,
                "active_hours_doubled": pytest.approx(0.6, rel=1e-5),
                "active_hours": pytest.approx(0.3, rel=1e-5),
            },
            {
                "client_id": "aaaa",
                "branch": "branch1",
                "enrollment_date": datetime.date(2020, 4, 2),
                "num_enrollment_events": 1,
                "analysis_window_start": 0,
                "active_hours_doubled": pytest.approx(1.6, rel=1e-5),
                "active_hours": pytest.approx(0.8, rel=1e-5),
            },
        ]

        for i, row in enumerate(query_job.result()):
            for k, v in expected_metrics_results[i].items():
                assert row[k] == v

        assert (
            client.client.get_table(
                f"{project_id}.{temporary_dataset}.test_experiment_enrollments_weekly"
            )
            is not None
        )
        assert (
            client.client.get_table(
                f"{project_id}.{temporary_dataset}.statistics_test_experiment_week_1"
            )
            is not None
        )

        stats = client.client.list_rows(
            f"{project_id}.{temporary_dataset}.statistics_test_experiment_week_1"
        ).to_dataframe()

        ratio_by_branch = stats.query(
            "metric == 'active_hours_ratio' and statistic == 'population_ratio' "
            + "and analysis_basis == 'enrollments' and comparison == 'relative_uplift'"
        ).set_index("branch")

        assert ratio_by_branch.loc["branch1", "point"] == 0.0

        ratio_by_branch = stats.query(
            "metric == 'active_hours_ratio' and statistic == 'population_ratio' "
            + "and analysis_basis == 'enrollments' and comparison == 'difference'"
        ).set_index("branch")

        assert ratio_by_branch.loc["branch1", "point"] == 0.0

        ratio_by_branch = stats.query(
            "metric == 'active_hours_ratio' and statistic == 'population_ratio' "
            + "and analysis_basis == 'enrollments' and comparison.isnull()"
        ).set_index("branch")

        assert (ratio_by_branch.loc["branch1", "point"] == 0.5).all()
        assert (ratio_by_branch.loc["branch2", "point"] == 0.5).all()

    def test_no_enrollments(
        self, monkeypatch, client, project_id, static_dataset, temporary_dataset
    ):
        experiment = Experiment(
            experimenter_slug="test-experiment-2",
            type="rollout",
            status="Live",
            start_date=dt.datetime(2020, 3, 30, tzinfo=pytz.utc),
            end_date=dt.datetime(2020, 6, 1, tzinfo=pytz.utc),
            proposed_enrollment=7,
            is_enrollment_paused=True,
            branches=[Branch(slug="a", ratio=0.5), Branch(slug="b", ratio=0.5)],
            reference_branch="a",
            normandy_slug="test-experiment-2",
            is_high_population=False,
            app_name="firefox_desktop",
            app_id="firefox-desktop",
        )

        config = AnalysisSpec().resolve(experiment, ConfigLoader.configs)

        test_clients_daily = DataSource(
            name="clients_daily",
            from_expression=f"`{project_id}.test_data.clients_daily`",
        )

        test_active_hours = Metric(
            name="active_hours",
            data_source=test_clients_daily,
            select_expression=agg_sum("active_hours_sum"),
        )

        stat = Statistic(name="bootstrap_mean", params={})

        config.metrics = {AnalysisPeriod.WEEK: [Summary(test_active_hours, stat)]}

        self.analysis_mock_run(monkeypatch, config, static_dataset, temporary_dataset, project_id)

        query_job = client.client.query(
            f"""
            SELECT
              *
            FROM `{project_id}.{temporary_dataset}.test_experiment_2_enrollments_week_1`
            ORDER BY enrollment_date DESC
        """
        )

        assert query_job.result().total_rows == 0

        stats = client.client.list_rows(
            f"{project_id}.{temporary_dataset}.statistics_test_experiment_2_week_1"
        ).to_dataframe()

        count_by_branch = stats.query("statistic == 'count'").set_index("branch")
        assert count_by_branch.loc["a", "point"][0] == 0.0
        assert count_by_branch.loc["a", "point"][1] == 0.0
        assert count_by_branch.loc["b", "point"][0] == 0.0
        assert count_by_branch.loc["b", "point"][1] == 0.0
        assert len(count_by_branch.loc["b", "analysis_basis"]) == 2

        assert (
            client.client.get_table(
                f"{project_id}.{temporary_dataset}.statistics_test_experiment_2_weekly"
            )
            is not None
        )

    def test_with_segments(
        self, monkeypatch, client, project_id, static_dataset, temporary_dataset
    ):
        experiment = Experiment(
            experimenter_slug="test-experiment",
            type="rollout",
            status="Live",
            start_date=dt.datetime(2020, 3, 30, tzinfo=pytz.utc),
            end_date=dt.datetime(2020, 6, 1, tzinfo=pytz.utc),
            proposed_enrollment=7,
            is_enrollment_paused=True,
            branches=[Branch(slug="branch1", ratio=0.5), Branch(slug="branch2", ratio=0.5)],
            reference_branch="branch2",
            normandy_slug="test-experiment",
            is_high_population=False,
            app_name="firefox_desktop",
            app_id="firefox-desktop",
        )

        config = AnalysisSpec().resolve(experiment, ConfigLoader.configs)

        test_clients_daily = DataSource(
            name="clients_daily",
            from_expression=f"`{project_id}.test_data.clients_daily`",
        )

        test_active_hours = Metric(
            name="active_hours",
            data_source=test_clients_daily,
            select_expression=agg_sum("active_hours_sum"),
        )

        test_clients_last_seen = SegmentDataSource(
            "clients_last_seen", f"`{project_id}.test_data.clients_last_seen`"
        )
        regular_user_v3 = Segment(
            "regular_user_v3",
            test_clients_last_seen,
            "COALESCE(LOGICAL_OR(is_regular_user_v3), FALSE)",
        )
        config.experiment.segments = [regular_user_v3]

        stat = Statistic(name="bootstrap_mean", params={})

        config.metrics = {AnalysisPeriod.WEEK: [Summary(test_active_hours, stat)]}

        self.analysis_mock_run(monkeypatch, config, static_dataset, temporary_dataset, project_id)

        query_job = client.client.query(
            f"""
            SELECT
              *
            FROM `{project_id}.{temporary_dataset}.test_experiment_enrollments_week_1`
            ORDER BY enrollment_date DESC
        """
        )

        expected_metrics_results = [
            {
                "client_id": "bbbb",
                "branch": "branch2",
                "enrollment_date": datetime.date(2020, 4, 3),
                "num_enrollment_events": 1,
                "analysis_window_start": 0,
                "analysis_window_end": 6,
                "regular_user_v3": True,
            },
            {
                "client_id": "aaaa",
                "branch": "branch1",
                "enrollment_date": datetime.date(2020, 4, 2),
                "num_enrollment_events": 1,
                "analysis_window_start": 0,
                "analysis_window_end": 6,
                "regular_user_v3": False,
            },
        ]

        for i, row in enumerate(query_job.result()):
            for k, v in expected_metrics_results[i].items():
                assert row[k] == v

        assert (
            client.client.get_table(
                f"{project_id}.{temporary_dataset}.test_experiment_enrollments_weekly"
            )
            is not None
        )
        assert (
            client.client.get_table(
                f"{project_id}.{temporary_dataset}.statistics_test_experiment_week_1"
            )
            is not None
        )

        stats = client.client.list_rows(
            f"{project_id}.{temporary_dataset}.statistics_test_experiment_week_1"
        ).to_dataframe()

        # Only one count per segment and branch, please
        assert (
            stats.query(
                "metric == 'identity' and statistic == 'count' and analysis_basis == 'enrollments'"
            )
            .groupby(["segment", "analysis_basis", "window_index", "branch"])
            .size()
            == 1
        ).all()

        count_by_branch = stats.query(
            "segment == 'all' and statistic == 'count' and analysis_basis == 'enrollments'"
        ).set_index("branch")
        assert count_by_branch.loc["branch1", "point"] == 1.0
        assert count_by_branch.loc["branch2", "point"] == 1.0
        assert count_by_branch.loc["branch2", "analysis_basis"] == "enrollments"

        assert (
            stats.query(
                "metric == 'identity' and statistic == 'count' and analysis_basis == 'exposures'"
            )
            .groupby(["segment", "analysis_basis", "window_index", "branch"])
            .size()
            == 1
        ).all()

        count_by_branch = stats.query(
            "segment == 'all' and statistic == 'count' and analysis_basis == 'exposures'"
        ).set_index("branch")

        assert count_by_branch.loc["branch1", "point"] == 1.0
        assert count_by_branch.loc["branch2", "point"] == 1.0
        assert count_by_branch.loc["branch2", "analysis_basis"] == "exposures"

        count_by_branch = stats.query(
            """
            segment == 'regular_user_v3' \
            and statistic == 'count' \
            and analysis_basis == 'enrollments'
            """
        ).set_index("branch")

        assert count_by_branch.loc["branch1", "point"] == 0.0
        assert count_by_branch.loc["branch2", "point"] == 1.0

        count_by_branch = stats.query(
            """
            segment == 'regular_user_v3' \
            and statistic == 'count' \
            and analysis_basis == 'exposures'
            """
        ).set_index("branch")

        assert count_by_branch.loc["branch1", "point"] == 0.0
        assert count_by_branch.loc["branch2", "point"] == 1.0

    def test_logging(self, monkeypatch, client, project_id, static_dataset, temporary_dataset):
        experiment = Experiment(
            experimenter_slug="test-experiment",
            type="rollout",
            status="Live",
            start_date=dt.datetime(2020, 3, 30, tzinfo=pytz.utc),
            end_date=dt.datetime(2020, 6, 1, tzinfo=pytz.utc),
            proposed_enrollment=7,
            is_enrollment_paused=True,
            branches=[Branch(slug="branch1", ratio=0.5), Branch(slug="branch2", ratio=0.5)],
            reference_branch="branch2",
            normandy_slug="test-experiment",
            is_high_population=False,
            app_name="firefox_desktop",
            app_id="firefox-desktop",
        )

        config = AnalysisSpec().resolve(experiment, ConfigLoader.configs)

        test_clients_daily = DataSource(
            name="clients_daily",
            from_expression=f"`{project_id}.test_data.clients_daily`",
        )

        test_active_hours = Metric(
            name="active_hours",
            data_source=test_clients_daily,
            select_expression=agg_sum("active_hours_sum"),
        )

        stat = Statistic(name="bootstrap_mean", params={"confidence_interval": 10})

        test_clients_last_seen = SegmentDataSource(
            "clients_last_seen", f"`{project_id}.test_data.clients_last_seen`"
        )
        regular_user_v3 = Segment(
            "regular_user_v3",
            test_clients_last_seen,
            "COALESCE(LOGICAL_OR(is_regular_user_v3), FALSE)",
        )
        config.experiment.segments = [regular_user_v3]
        config.metrics = {AnalysisPeriod.WEEK: [Summary(test_active_hours, stat)]}

        log_config = LogConfiguration(
            log_project_id=project_id,
            log_dataset_id=temporary_dataset,
            log_table_id="logs",
            log_to_bigquery=True,
            task_profiling_log_table_id="task_profiling_logs",
            task_monitoring_log_table_id="task_monitoring_logs",
            capacity=1,
        )
        self.analysis_mock_run(
            monkeypatch, config, static_dataset, temporary_dataset, project_id, log_config
        )

        assert client.client.get_table(f"{project_id}.{temporary_dataset}.logs") is not None

        logs = list(client.client.list_rows(f"{project_id}.{temporary_dataset}.logs"))

        assert len(logs) >= 1
        error_logs = [log for log in logs if log.get("log_level") == "ERROR"]
        error_logs.sort(key=lambda k: (k.get("segment"), k.get("analysis_basis")))

        assert len(error_logs) == 6
        assert (
            "Error while computing statistic bootstrap_mean for metric active_hours"
            in error_logs[0].get("message")
        )
        assert error_logs[0].get("log_level") == error_logs[1].get("log_level") == "ERROR"
        assert (
            error_logs[0].get("experiment") == error_logs[1].get("experiment") == "test-experiment"
        )
        assert error_logs[0].get("metric") == error_logs[1].get("metric") == "active_hours"
        assert error_logs[0].get("statistic") == error_logs[1].get("statistic") == "bootstrap_mean"
        assert (
            error_logs[0].get("analysis_basis")
            == error_logs[1].get("analysis_basis")
            == "enrollments"
        )
        assert error_logs[0].get("segment") == error_logs[1].get("segment") == "all"
        assert error_logs[0].get("source") == error_logs[1].get("source") == "jetstream"

        assert error_logs[2].get("log_level") == error_logs[3].get("log_level") == "ERROR"
        assert (
            error_logs[2].get("experiment") == error_logs[3].get("experiment") == "test-experiment"
        )
        assert error_logs[2].get("metric") == error_logs[3].get("metric") == "active_hours"
        assert error_logs[2].get("statistic") == error_logs[3].get("statistic") == "bootstrap_mean"
        assert (
            error_logs[2].get("analysis_basis")
            == error_logs[3].get("analysis_basis")
            == "exposures"
        )
        assert error_logs[2].get("segment") == error_logs[3].get("segment") == "all"

        assert error_logs[4].get("log_level") == "ERROR"
        assert error_logs[4].get("experiment") == "test-experiment"
        assert error_logs[4].get("metric") == "active_hours"
        assert error_logs[4].get("statistic") == "bootstrap_mean"
        assert error_logs[4].get("analysis_basis") == "enrollments"
        assert error_logs[4].get("segment") == "regular_user_v3"

        assert error_logs[5].get("log_level") == "ERROR"
        assert error_logs[5].get("experiment") == "test-experiment"
        assert error_logs[5].get("metric") == "active_hours"
        assert error_logs[5].get("statistic") == "bootstrap_mean"
        assert error_logs[5].get("analysis_basis") == "exposures"
        assert error_logs[5].get("segment") == "regular_user_v3"

    # wait for profiling results to land in BigQuery
    # todo: improve this test as it might lead to flakiness
    # sleep(10)

    # assert (
    #     client.client.get_table(f"{project_id}.{temporary_dataset}.task_profiling_logs")
    #     is not None
    # )

    # task_profiling_logs = list(
    #     client.client.list_rows(f"{project_id}.{temporary_dataset}.task_profiling_logs")
    # )
    # assert task_profiling_logs[0].get("max_cpu") >= 0

    # assert (
    #     client.client.get_table(f"{project_id}.{temporary_dataset}.task_monitoring_logs")
    #     is not None
    # )

    # task_monitoring_logs = list(
    #     client.client.list_rows(f"{project_id}.{temporary_dataset}.task_monitoring_logs")
    # )
    # assert len(task_monitoring_logs) > 0

    def test_statistics_export(
        self, monkeypatch, client, project_id, static_dataset, temporary_dataset
    ):
        experiment = Experiment(
            experimenter_slug="test-experiment",
            type="rollout",
            status="Live",
            start_date=dt.datetime(2020, 3, 30, tzinfo=pytz.utc),
            end_date=dt.datetime(2020, 6, 1, tzinfo=pytz.utc),
            proposed_enrollment=7,
            is_enrollment_paused=True,
            branches=[Branch(slug="branch1", ratio=0.5), Branch(slug="branch2", ratio=0.5)],
            reference_branch="branch2",
            normandy_slug="test-experiment",
            is_high_population=False,
            app_name="firefox_desktop",
            app_id="firefox-desktop",
        )

        config = AnalysisSpec().resolve(experiment, ConfigLoader.configs)

        test_clients_daily = DataSource(
            name="clients_daily",
            from_expression=f"`{project_id}.test_data.clients_daily`",
        )

        test_active_hours = Metric(
            name="active_hours",
            data_source=test_clients_daily,
            select_expression=agg_sum("active_hours_sum"),
        )

        test_clients_last_seen = SegmentDataSource(
            "clients_last_seen", f"`{project_id}.test_data.clients_last_seen`"
        )
        regular_user_v3 = Segment(
            "regular_user_v3",
            test_clients_last_seen,
            "COALESCE(LOGICAL_OR(is_regular_user_v3), FALSE)",
        )
        config.experiment.segments = [regular_user_v3]

        stat = Statistic(name="bootstrap_mean", params={})

        config.metrics = {AnalysisPeriod.WEEK: [Summary(test_active_hours, stat)]}

        self.analysis_mock_run(monkeypatch, config, static_dataset, temporary_dataset, project_id)

        query_job = client.client.query(
            f"""
            SELECT
              * EXCEPT(window_index),
              SAFE_CAST(window_index AS STRING) AS window_index
            FROM `{project_id}.{temporary_dataset}.statistics_test_experiment_weekly`
        """
        )

        statistics_export_data = query_job.to_dataframe().to_dict(orient="records")
        schema = json.loads(
            (Path(__file__).parent.parent / "data/Statistics_v1.0.json").read_text()
        )
        jsonschema.validate(statistics_export_data, schema)

    def test_subset_metric_table(
        self, monkeypatch, client, project_id, static_dataset, temporary_dataset
    ):
        experiment = Experiment(
            experimenter_slug="test-experiment",
            type="rollout",
            status="Live",
            start_date=dt.datetime(2020, 3, 30, tzinfo=pytz.utc),
            end_date=dt.datetime(2020, 6, 1, tzinfo=pytz.utc),
            proposed_enrollment=7,
            is_enrollment_paused=True,
            branches=[Branch(slug="branch1", ratio=0.5), Branch(slug="branch2", ratio=0.5)],
            reference_branch="branch2",
            normandy_slug="test-experiment",
            is_high_population=False,
            app_name="firefox_desktop",
            app_id="firefox-desktop",
        )

        config = AnalysisSpec().resolve(experiment, ConfigLoader.configs)

        test_clients_daily = DataSource(
            name="clients_daily",
            from_expression=f"`{project_id}.test_data.clients_daily`",
        )

        test_active_hours = Metric(
            name="active_hours",
            data_source=test_clients_daily,
            select_expression=agg_sum("active_hours_sum"),
            analysis_bases=[AnalysisBasis.EXPOSURES, AnalysisBasis.ENROLLMENTS],
        )

        stat = Statistic(name="bootstrap_mean", params={})

        config.metrics = {AnalysisPeriod.WEEK: [Summary(test_active_hours, stat)]}

        self.analysis_mock_run(monkeypatch, config, static_dataset, temporary_dataset, project_id)

        analysis = Analysis(project_id, temporary_dataset, config, None)

        exposures_results = (
            analysis.subset_metric_table(
                "test_experiment_exposures_week_1",
                "all",
                test_active_hours,
                AnalysisBasis.EXPOSURES,
            )
            .compute()
            .sort_values("branch")
            .reset_index(drop=True)
        )
        assert exposures_results.loc[0, "active_hours"] == pytest.approx(0.8, rel=1e-5)
        assert exposures_results.loc[1, "active_hours"] == pytest.approx(0.3, rel=1e-5)

        enrollments_results = (
            analysis.subset_metric_table(
                "test_experiment_enrollments_week_1",
                "all",
                test_active_hours,
                AnalysisBasis.ENROLLMENTS,
            )
            .compute()
            .sort_values("branch")
            .reset_index(drop=True)
        )
        assert enrollments_results.loc[0, "active_hours"] == pytest.approx(0.8, rel=1e-5)
        assert enrollments_results.loc[1, "active_hours"] == pytest.approx(0.3, rel=1e-5)
