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
