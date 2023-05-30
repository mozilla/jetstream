class TestBigQueryClient:
    def test_experiment_table_first_updated(self, client, project_id, temporary_dataset):
        earliest_timestamp = client._current_timestamp_label()

        # table created after config loaded
        client.client.create_table(f"{temporary_dataset}.enrollments_test_experiment")
        client.add_labels_to_table(
            "enrollments_test_experiment",
            {"last_updated": earliest_timestamp},
        )
        client.client.create_table(f"{temporary_dataset}.statistics_test_experiment_day_1")
        client.add_labels_to_table(
            "statistics_test_experiment_day_1",
            {"last_updated": client._current_timestamp_label()},
        )

        assert client.experiment_table_first_updated("test-experiment") == earliest_timestamp
