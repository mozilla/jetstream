import datetime
import datetime as dt
import json
from pathlib import Path

import dask
import jsonschema
import mozanalysis
import pytz
from mozanalysis.metrics import AnalysisBasis, DataSource, agg_sum
from mozanalysis.segments import Segment, SegmentDataSource

from jetstream import AnalysisPeriod
from jetstream.analysis import Analysis
from jetstream.config import AnalysisSpec, Summary
from jetstream.experimenter import Branch, Experiment
from jetstream.exposure_signal import ExposureSignal
from jetstream.logging import LogConfiguration
from jetstream.metric import Metric
from jetstream.statistics import BootstrapMean

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
            branches=[Branch(slug="branch1", ratio=0.5), Branch(slug="branch2", ratio=0.5)],
            reference_branch="branch2",
            normandy_slug="test-experiment",
            is_high_population=False,
            app_name="firefox_desktop",
            app_id="firefox-desktop",
        )

        config = AnalysisSpec().resolve(experiment)

        test_clients_daily = DataSource(
            name="clients_daily",
            from_expr=f"`{project_id}.test_data.clients_daily`",
        )

        test_active_hours = Metric(
            name="active_hours",
            data_source=test_clients_daily,
            select_expression=agg_sum("active_hours_sum"),
            analysis_bases=[AnalysisBasis.EXPOSURES, AnalysisBasis.ENROLLMENTS],
        )

        config.metrics = {AnalysisPeriod.WEEK: [Summary(test_active_hours, BootstrapMean())]}

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
            branches=[Branch(slug="branch1", ratio=0.5), Branch(slug="branch2", ratio=0.5)],
            reference_branch="branch2",
            normandy_slug="test-experiment",
            is_high_population=False,
            app_name="firefox_desktop",
            app_id="firefox-desktop",
        )

        config = AnalysisSpec().resolve(experiment)

        test_clients_daily = DataSource(
            name="clients_daily",
            from_expr=f"`{project_id}.test_data.clients_daily`",
        )

        test_active_hours = Metric(
            name="active_hours",
            data_source=test_clients_daily,
            select_expression=agg_sum("active_hours_sum"),
            analysis_bases=[AnalysisBasis.EXPOSURES],
        )

        config.metrics = {AnalysisPeriod.WEEK: [Summary(test_active_hours, BootstrapMean())]}
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
            branches=[Branch(slug="a", ratio=0.5), Branch(slug="b", ratio=0.5)],
            reference_branch="a",
            normandy_slug="test-experiment-2",
            is_high_population=False,
            app_name="firefox_desktop",
            app_id="firefox-desktop",
        )

        config = AnalysisSpec().resolve(experiment)

        test_clients_daily = DataSource(
            name="clients_daily",
            from_expr=f"`{project_id}.test_data.clients_daily`",
        )

        test_active_hours = Metric(
            name="active_hours",
            data_source=test_clients_daily,
            select_expression=agg_sum("active_hours_sum"),
        )

        config.metrics = {AnalysisPeriod.WEEK: [Summary(test_active_hours, BootstrapMean())]}

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
        assert count_by_branch.loc["a", "point"] == 0.0
        assert count_by_branch.loc["b", "point"] == 0.0
        assert count_by_branch.loc["b", "analysis_basis"] == "enrollments"

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
            branches=[Branch(slug="branch1", ratio=0.5), Branch(slug="branch2", ratio=0.5)],
            reference_branch="branch2",
            normandy_slug="test-experiment",
            is_high_population=False,
            app_name="firefox_desktop",
            app_id="firefox-desktop",
        )

        config = AnalysisSpec().resolve(experiment)

        test_clients_daily = DataSource(
            name="clients_daily",
            from_expr=f"`{project_id}.test_data.clients_daily`",
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

        config.metrics = {AnalysisPeriod.WEEK: [Summary(test_active_hours, BootstrapMean())]}

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
            stats.query("metric == 'identity' and statistic == 'count'")
            .groupby(["segment", "analysis_basis", "window_index", "branch"])
            .size()
            == 1
        ).all()

        count_by_branch = stats.query("segment == 'all' and statistic == 'count'").set_index(
            "branch"
        )
        assert count_by_branch.loc["branch1", "point"] == 1.0
        assert count_by_branch.loc["branch2", "point"] == 1.0
        assert count_by_branch.loc["branch2", "analysis_basis"] == "enrollments"

    def test_logging(self, monkeypatch, client, project_id, static_dataset, temporary_dataset):
        experiment = Experiment(
            experimenter_slug="test-experiment",
            type="rollout",
            status="Live",
            start_date=dt.datetime(2020, 3, 30, tzinfo=pytz.utc),
            end_date=dt.datetime(2020, 6, 1, tzinfo=pytz.utc),
            proposed_enrollment=7,
            branches=[Branch(slug="branch1", ratio=0.5), Branch(slug="branch2", ratio=0.5)],
            reference_branch="branch2",
            normandy_slug="test-experiment",
            is_high_population=False,
            app_name="firefox_desktop",
            app_id="firefox-desktop",
        )

        config = AnalysisSpec().resolve(experiment)

        test_clients_daily = DataSource(
            name="clients_daily",
            from_expr=f"`{project_id}.test_data.clients_daily`",
        )

        test_active_hours = Metric(
            name="active_hours",
            data_source=test_clients_daily,
            select_expression=agg_sum("active_hours_sum"),
        )

        config.metrics = {
            AnalysisPeriod.WEEK: [Summary(test_active_hours, BootstrapMean(confidence_interval=10))]
        }

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
        assert (
            "Error while computing statistic bootstrap_mean for metric active_hours"
            in error_logs[0].get("message")
        )
        assert error_logs[0].get("log_level") == "ERROR"
        assert error_logs[0].get("experiment") == "test-experiment"
        assert error_logs[0].get("metric") == "active_hours"
        assert error_logs[0].get("statistic") == "bootstrap_mean"

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
            branches=[Branch(slug="branch1", ratio=0.5), Branch(slug="branch2", ratio=0.5)],
            reference_branch="branch2",
            normandy_slug="test-experiment",
            is_high_population=False,
            app_name="firefox_desktop",
            app_id="firefox-desktop",
        )

        config = AnalysisSpec().resolve(experiment)

        test_clients_daily = DataSource(
            name="clients_daily",
            from_expr=f"`{project_id}.test_data.clients_daily`",
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

        config.metrics = {AnalysisPeriod.WEEK: [Summary(test_active_hours, BootstrapMean())]}

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
