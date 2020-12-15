from textwrap import dedent
from unittest import mock
from unittest.mock import patch, MagicMock

import json
import requests
import toml

from jetstream.config import AnalysisSpec
from jetstream.metadata import ExperimentMetadata, export_metadata
from jetstream.tests.test_probe_sets import EXPERIMENTER_FIXTURE_PROBESETS


@patch.object(requests.Session, "get")
def test_metadata_from_config(mock_get, experiments):
    config_str = dedent(
        """
        [metrics]
        weekly = ["view_about_logins", "my_cool_metric"]
        daily = ["my_cool_metric"]

        [metrics.my_cool_metric]
        data_source = "main"
        select_expression = "{{agg_histogram_mean('payload.content.my_cool_histogram')}}"
        friendly_name = "Cool metric"
        description = "Cool cool cool"
        bigger_is_better = true

        [metrics.my_cool_metric.statistics.bootstrap_mean]

        [metrics.view_about_logins.statistics.bootstrap_mean]
        """
    )

    mock_get.return_value.json.return_value = json.loads(EXPERIMENTER_FIXTURE_PROBESETS)

    spec = AnalysisSpec.from_dict(toml.loads(config_str))
    config = spec.resolve(experiments[4])
    metadata = ExperimentMetadata.from_config(config)

    assert "view_about_logins" in metadata.metrics
    assert metadata.metrics["view_about_logins"].bigger_is_better
    assert metadata.metrics["view_about_logins"].description != ""
    assert "my_cool_metric" in metadata.metrics
    assert metadata.metrics["my_cool_metric"].bigger_is_better
    assert metadata.metrics["my_cool_metric"].friendly_name == "Cool metric"
    assert metadata.metrics["my_cool_metric"].description == "Cool cool cool"

    assert "test-probe-set" in metadata.probesets
    assert "test-probe-1" in metadata.probesets["test-probe-set"]


@mock.patch("google.cloud.storage.Client")
def test_export_metadata(mock_storage_client, experiments):
    config_str = dedent(
        """
        [metrics]
        weekly = ["view_about_logins"]

        [metrics.view_about_logins.statistics.bootstrap_mean]
        """
    )

    spec = AnalysisSpec.from_dict(toml.loads(config_str))
    config = spec.resolve(experiments[0])

    mock_client = MagicMock()
    mock_storage_client.return_value = mock_client
    mock_bucket = MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_blob.upload_from_string.return_value = ""

    export_metadata(config, "test_bucket")

    mock_client.get_bucket.assert_called_once()
    mock_bucket.blob.assert_called_once()

    expected = json.loads(
        r"""
        {
            "metrics": {
                "view_about_logins": {
                    "friendly_name": "about:logins viewers",
                    "description": "Counts the number of clients that viewed about:logins.\n",
                    "bigger_is_better": true
                }
            },
            "probesets": {}
        }
    """
    )
    mock_blob.upload_from_string.assert_called_once_with(
        data=json.dumps(expected), content_type="application/json"
    )
