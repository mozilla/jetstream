"""
Dry run generated queries.

Passes all queries defined under sql/ to a Cloud Function that will run the
queries with the dry_run option enabled.

We could provision BigQuery credentials to the CircleCI job to allow it to run
the queries directly, but there is no way to restrict permissions such that
only dry runs can be performed. In order to reduce risk of CI or local users
accidentally running queries during tests, leaking and overwriting production
data, we proxy the queries through the dry run service endpoint.
"""

import json
import logging
import os
import random
from typing import Any

import google.auth
import requests
import requests.exceptions
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.id_token import fetch_id_token

logger = logging.getLogger(__name__)

# https://github.com/mozilla-services/cloudops-infra/blob/master/projects/data-shared/tf/modules/cloudfunctions/src/bigquery_etl_dryrun/index.js
DRY_RUN_URL = "https://us-central1-moz-fx-data-shared-prod.cloudfunctions.net/dryrun"
BILLING_PROJECTS = [
    "moz-fx-data-backfill-10",
    "moz-fx-data-backfill-11",
    "moz-fx-data-backfill-12",
    "moz-fx-data-backfill-13",
    "moz-fx-data-backfill-14",
    "moz-fx-data-backfill-15",
    "moz-fx-data-backfill-16",
    "moz-fx-data-backfill-17",
    "moz-fx-data-backfill-18",
    "moz-fx-data-backfill-19",
    "moz-fx-data-backfill-20",
    "moz-fx-data-backfill-21",
    "moz-fx-data-backfill-22",
    "moz-fx-data-backfill-23",
    "moz-fx-data-backfill-24",
    "moz-fx-data-backfill-25",
    "moz-fx-data-backfill-26",
    "moz-fx-data-backfill-27",
    "moz-fx-data-backfill-28",
    "moz-fx-data-backfill-29",
    "moz-fx-data-backfill-31",
]


class DryRunFailedError(Exception):
    """Exception raised when dry run fails."""

    def __init__(self, error: Any, sql: str):
        self.sql = sql
        super().__init__(error)


def dry_run_query(sql: str, use_cloud_function: bool = True) -> int:
    """Dry run the provided SQL query.

    If ``use_cloud_function`` is True (default), the query is dry run via the
    dry run Cloud Function, which executes using its own service account.
    If False, the query is dry run directly using the current user's
    credentials.
    """
    if not use_cloud_function:
        return _dry_run_with_user_credentials(sql)

    try:
        # look for token created by the GitHub Actions workflow
        id_token = os.environ.get("GOOGLE_GHA_ID_TOKEN")
        if not id_token:
            auth_req = GoogleAuthRequest()
            creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            creds.refresh(auth_req)
            if hasattr(creds, "id_token"):
                # Get token from default credentials for the
                # current environment created via Cloud SDK run
                id_token = creds.id_token
            else:
                # If the environment variable GOOGLE_APPLICATION_CREDENTIALS
                # is set to service account JSON file,
                # then ID token is acquired using this service account credentials.
                id_token = fetch_id_token(auth_req, DRY_RUN_URL)

        billing_project = random.choice(BILLING_PROJECTS)
        r = requests.post(
            DRY_RUN_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {id_token}",
            },
            data=json.dumps(
                {
                    "dataset": "mozanalysis",
                    "project": "moz-fx-data-experiments",
                    "query": sql,
                    "billing_project": billing_project,
                }
            ).encode("utf8"),
        )
        response = r.json()
    except Exception:
        # This may be a JSONDecode exception or something else.
        # If we got a HTTP exception, that's probably the most interesting thing to raise.
        try:
            r.raise_for_status()
        except requests.exceptions.RequestException as request_exception:
            e = request_exception
        raise DryRunFailedError(e, sql) from e

    if response["valid"]:
        logger.info("Dry run OK")
        return int(response.get("bytesProcessed", -1))

    error = response["errors"][0] if "errors" in response and len(response["errors"]) == 1 else None

    if (
        error
        and error.get("code", None) in [400, 403]
        and "does not have bigquery.tables.create permission for dataset"
        in error.get("message", "")
    ):
        # We want the dryrun service to only have read permissions, so
        # we expect CREATE VIEW and CREATE TABLE to throw specific
        # exceptions.
        logger.info("Dry run OK")
        return int(response.get("bytesProcessed", -1))

    raise DryRunFailedError((error and error.get("message", None)) or response["errors"], sql=sql)


def _dry_run_with_user_credentials(sql: str) -> int:
    """Dry run a query directly using the current user's credentials.

    Unlike the Cloud Function, this executes the dry run as the authenticated
    user, so it can access tables the user has permission to read (e.g. personal
    tables in the ``analysis`` dataset) that the dry run service account cannot.
    """
    from google.cloud import bigquery

    try:
        client = bigquery.Client()
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        query_job = client.query(sql, job_config=job_config)
    except Exception as e:
        raise DryRunFailedError(e, sql) from e

    logger.info("Dry run OK")
    return int(query_job.total_bytes_processed or -1)
