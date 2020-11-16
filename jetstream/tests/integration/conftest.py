import random
import string
from pathlib import Path

import pytest
from google.api_core.exceptions import NotFound
from google.cloud import bigquery

from jetstream.bigquery_client import BigQueryClient

TEST_DIR = Path(__file__).parent.parent


def pytest_runtest_setup(item):
    if "FLAKE8" in item.nodeid or "BLACK" in item.nodeid:
        return
    if not item.config.getoption("--integration", False):
        pytest.skip("Skipping integration test")


@pytest.fixture
def project_id():
    """Provide a BigQuery project ID."""
    return "jetstream-integration-test"


@pytest.fixture
def temporary_dataset(project_id):
    """Fixture for creating a random temporary BigQuery dataset."""
    # generate a random test dataset to avoid conflicts when running tests in parallel
    test_dataset = "test_" + "".join(random.choices(string.ascii_lowercase, k=12))

    client = bigquery.Client(project_id)
    client.create_dataset(test_dataset)

    yield test_dataset

    # cleanup and remove temporary dataset
    client.delete_dataset(test_dataset, delete_contents=True, not_found_ok=True)


@pytest.fixture
def client(project_id, temporary_dataset):
    """Provide a BigQuery client."""
    return BigQueryClient(project_id, temporary_dataset)


@pytest.fixture
def static_dataset(project_id):
    """Dataset with static test data."""
    bigquery_client = bigquery.Client(project_id)
    static_dataset = "test_data"

    for slug in ("clients_daily", "clients_last_seen", "events"):
        try:
            bigquery_client.get_table(f"{static_dataset}.{slug}")
        except NotFound:
            table_ref = bigquery_client.create_table(f"{static_dataset}.{slug}")
            job_config = bigquery.LoadJobConfig()
            job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
            job_config.autodetect = True

            table_source = TEST_DIR / "data" / f"test_{slug}.ndjson"

            with open(table_source, "rb") as source_file:
                job = bigquery_client.load_table_from_file(
                    source_file, table_ref, job_config=job_config
                )

            job.result()  # Waits for table load to complete.

    return static_dataset
