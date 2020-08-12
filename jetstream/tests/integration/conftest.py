from google.cloud import bigquery
from google.api_core.exceptions import NotFound
from pathlib import Path
import pytest
import random
import string

from jetstream.analysis import BigQueryClient

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
def temporary_dataset(request):
    """Fixture for creating a random temporary BigQuery dataset."""
    # generate a random test dataset to avoid conflicts when running tests in parallel
    test_dataset = "test_" + "".join(random.choice(string.ascii_lowercase) for i in range(12))

    project_id = request.getfixturevalue("project_id")
    client = bigquery.Client(project_id)
    client.create_dataset(test_dataset)

    yield test_dataset

    # cleanup and remove temporary dataset
    client.delete_dataset(test_dataset, delete_contents=True, not_found_ok=True)


@pytest.fixture
def client(request):
    """Provide a BigQuery client."""
    project_id = request.getfixturevalue("project_id")
    temporary_dataset = request.getfixturevalue("temporary_dataset")
    return BigQueryClient(project_id, temporary_dataset)


@pytest.fixture
def static_dataset(request):
    """Dataset with static test data."""
    clients_daily_source = TEST_DIR / "data" / "test_clients_daily.ndjson"
    events_source = TEST_DIR / "data" / "test_events.ndjson"

    project_id = request.getfixturevalue("project_id")
    bigquery_client = bigquery.Client(project_id)
    static_dataset = "test_data"

    try:
        bigquery_client.get_table(f"{static_dataset}.clients_daily")
    except NotFound:
        table_ref = bigquery_client.create_table(f"{static_dataset}.clients_daily")
        job_config = bigquery.LoadJobConfig()
        job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
        job_config.autodetect = True

        with open(clients_daily_source, "rb") as source_file:
            job = bigquery_client.load_table_from_file(
                source_file, table_ref, job_config=job_config
            )

        job.result()  # Waits for table load to complete.

    try:
        bigquery_client.get_table(f"{static_dataset}.events")
    except NotFound:
        table_ref = bigquery_client.create_table(f"{static_dataset}.events")
        job_config = bigquery.LoadJobConfig()
        job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
        job_config.autodetect = True

        with open(events_source, "rb") as source_file:
            job = bigquery_client.load_table_from_file(
                source_file, table_ref, job_config=job_config
            )

        job.result()  # Waits for table load to complete.

    return static_dataset
