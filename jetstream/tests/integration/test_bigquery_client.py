import math
from datetime import datetime

import pytz
from google.cloud import bigquery
from mozilla_nimbus_schemas.jetstream import AnalysisBasis

from jetstream.statistics import StatisticResult, StatisticResultCollection


class TestBigQueryClient:
    def test_experiment_table_first_updated(self, client, temporary_dataset):
        earliest_timestamp = datetime(2022, 1, 1, 8, 0, 0, 0, tzinfo=pytz.utc)

        # table created after config loaded
        client.client.create_table(f"{temporary_dataset}.enrollments_test_experiment")
        client.add_labels_to_table(
            "enrollments_test_experiment",
            {"last_updated": str(int(earliest_timestamp.timestamp()))},
        )

        later_timestamp = datetime(2022, 1, 1, 9, 0, 0, 0, tzinfo=pytz.utc)
        client.client.create_table(f"{temporary_dataset}.statistics_test_experiment_day_1")
        client.add_labels_to_table(
            "statistics_test_experiment_day_1",
            {"last_updated": str(int(later_timestamp.timestamp()))},
        )

        assert client.experiment_table_first_updated("test-experiment") == earliest_timestamp

    def test_tables_matching_regex(self, client, temporary_dataset):
        client.client.create_table(f"{temporary_dataset}.enrollments_test_experiment")
        assert client.tables_matching_regex("^enrollments_.*$") == ["enrollments_test_experiment"]
        assert client.tables_matching_regex("nothing") == []

    def test_touch_tables(self, client, temporary_dataset):
        client.client.create_table(f"{temporary_dataset}.enrollments_test_experiment")
        client.client.create_table(f"{temporary_dataset}.statistics_test_experiment_week_0")
        client.client.create_table(f"{temporary_dataset}.statistics_test_experiment_day_12")
        client.client.create_table(f"{temporary_dataset}.test_foo_bar_day")

        client.touch_tables("test-experiment")

        enrollment_table = client.client.get_table(
            f"{temporary_dataset}.enrollments_test_experiment"
        )
        assert enrollment_table.labels
        assert enrollment_table.labels["last_updated"]

        stats_table = client.client.get_table(
            f"{temporary_dataset}.statistics_test_experiment_day_12"
        )
        assert stats_table.labels
        assert stats_table.labels["last_updated"]

        unrelated_table = client.client.get_table(f"{temporary_dataset}.test_foo_bar_day")
        assert unrelated_table.labels == {}

    def test_load_table_from_json(self, client, temporary_dataset):
        t0 = StatisticResult(
            metric="test_metric",
            statistic="test_statistic0",
            branch="control",
            parameter=0.12341234512323123,
            comparison=None,
            comparison_to_branch=None,
            ci_width=0.95,
            point=0.212412315123,
            lower=None,
            upper=None,
            segment="all",
            analysis_basis="exposures",
        )
        t1 = StatisticResult(
            metric="test_metric",
            statistic="test_statistic1",
            branch="control",
            parameter="0.0",
            comparison=None,
            comparison_to_branch=None,
            ci_width=0.95,
            point=0.212412315123,
            lower=None,
            upper=None,
            segment="all",
            analysis_basis="enrollments",
        )
        t2 = StatisticResult(
            metric="test_metric",
            statistic="test_statistic2",
            branch="control",
            parameter=None,
            comparison=None,
            comparison_to_branch=None,
            ci_width=0.95,
            point=math.nan,
            lower=math.nan,
            upper=math.nan,
            segment="all",
            analysis_basis=AnalysisBasis.EXPOSURES,
        )
        test_data = StatisticResultCollection.parse_obj([t0, t1, t2])

        job_config = bigquery.LoadJobConfig()
        job_config.schema = StatisticResult.bq_schema
        job_config.write_disposition = bigquery.job.WriteDisposition.WRITE_TRUNCATE

        table = "statistics_test_experiment_week_1"

        client.load_table_from_json(test_data.dict()["__root__"], table, job_config)

        table_ref = client.client.get_table(f"{temporary_dataset}.{table}")
        rows = client.client.list_rows(table_ref)
        results = list(rows)

        # ensure the results all got inserted and that values were properly parsed
        assert len(results) == 3
        found_stat0 = False
        found_stat2 = False
        for result in results:
            if result["statistic"] == "test_statistic0":
                found_stat0 = True
                assert float(result["parameter"]) == 0.123412
            if result["statistic"] == "test_statistic2":
                found_stat2 = True
                assert result["point"] is None
                assert result["point"] is not math.nan
                assert result["analysis_basis"] == "exposures"

        assert found_stat0
        assert found_stat2
