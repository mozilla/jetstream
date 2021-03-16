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
from typing import Any

import requests
import requests.exceptions

logger = logging.getLogger(__name__)

# https://console.cloud.google.com/functions/details/us-central1/jetstream-dryrun?project=moz-fx-data-experiments
DRY_RUN_URL = "https://us-central1-moz-fx-data-experiments.cloudfunctions.net/jetstream-dryrun"


class DryRunFailedError(Exception):
    """Exception raised when dry run fails."""

    def __init__(self, error: Any, sql: str):
        self.sql = sql
        super().__init__(error)


def dry_run_query(sql: str) -> None:
    """Dry run the provided SQL query."""
    try:
        r = requests.post(
            DRY_RUN_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps({"dataset": "mozanalysis", "query": sql}).encode("utf8"),
        )
        response = r.json()
    except Exception as e:
        # This may be a JSONDecode exception or something else.
        # If we got a HTTP exception, that's probably the most interesting thing to raise.
        try:
            r.raise_for_status()
        except requests.exceptions.RequestException as request_exception:
            e = request_exception
        raise DryRunFailedError(e, sql)

    if response["valid"]:
        logger.info("Dry run OK")
        return

    if "errors" in response and len(response["errors"]) == 1:
        error = response["errors"][0]
    else:
        error = None

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
        return

    raise DryRunFailedError((error and error.get("message", None)) or response["errors"], sql=sql)
