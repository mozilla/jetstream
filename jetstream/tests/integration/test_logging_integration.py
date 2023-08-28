import logging

import pytest
from google.cloud import bigquery

from jetstream.logging import LogConfiguration


class TestLoggingIntegration:
    @pytest.fixture(autouse=True)
    def logging_table_setup(self, client, temporary_dataset, project_id):
        schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP"),
            bigquery.SchemaField("experiment", "STRING"),
            bigquery.SchemaField("metric", "STRING"),
            bigquery.SchemaField("statistic", "STRING"),
            bigquery.SchemaField("analysis_basis", "STRING"),
            bigquery.SchemaField("segment", "STRING"),
            bigquery.SchemaField("message", "STRING"),
            bigquery.SchemaField("log_level", "STRING"),
            bigquery.SchemaField("exception", "STRING"),
            bigquery.SchemaField("filename", "STRING"),
            bigquery.SchemaField("func_name", "STRING"),
            bigquery.SchemaField("exception_type", "STRING"),
            bigquery.SchemaField("source", "STRING"),
        ]

        table = bigquery.Table(f"{project_id}.{temporary_dataset}.logs", schema=schema)
        table = client.client.create_table(table)

        log_config = LogConfiguration(
            project_id,
            temporary_dataset,
            "logs",
            "task_profiling_logs",
            "task_monitoring_logs",
            log_to_bigquery=True,
            capacity=1,
        )
        log_config.setup_logger()

        yield
        client.client.delete_table(table, not_found_ok=True)

    def test_logging_to_bigquery(self, client, temporary_dataset, project_id):
        logger = logging.getLogger(__name__)
        logger.info("Do not write to BigQuery")
        logger.warning("Write warning to Bigquery")
        logger.error(
            "Write error to BigQuery",
            extra={
                "experiment": "test_experiment",
                "metric": "test_metric",
                "statistic": "test_statistic",
                "segment": "all",
                "analysis_basis": "enrollments",
            },
        )
        logger.exception(
            "Write exception to BigQuery",
            exc_info=Exception("Some exception"),
            extra={
                "experiment": "test_experiment",
                "metric": "test_metric",
                "statistic": "test_statistic",
                "segment": "test_segment",
                "analysis_basis": "exposures",
            },
        )

        result = list(
            client.client.query(f"SELECT * FROM {project_id}.{temporary_dataset}.logs").result()
        )
        assert any([r.message == "Write warning to Bigquery" for r in result])
        assert (
            any([r.message == "Do not write to BigQuery" and r.log_level == "WARN" for r in result])
            is False
        )
        assert any(
            [
                r.message == "Write error to BigQuery"
                and r.experiment == "test_experiment"
                and r.metric == "test_metric"
                and r.statistic == "test_statistic"
                and r.log_level == "ERROR"
                and r.segment == "all"
                and r.analysis_basis == "enrollments"
                and r.source == "jetstream"
                for r in result
            ]
        )
        assert any(
            [
                r.message == "Write exception to BigQuery"
                and r.experiment == "test_experiment"
                and r.metric == "test_metric"
                and r.statistic == "test_statistic"
                and r.log_level == "ERROR"
                and r.segment == "test_segment"
                and r.analysis_basis == "exposures"
                and "Exception('Some exception')" in r.exception
                for r in result
            ]
        )
