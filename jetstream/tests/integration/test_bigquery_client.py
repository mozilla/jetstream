from datetime import datetime

import pytz


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
