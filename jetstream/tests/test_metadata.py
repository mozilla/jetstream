import json
from textwrap import dedent
from unittest import mock
from unittest.mock import MagicMock, patch

import requests
import toml

from jetstream.config import AnalysisSpec
from jetstream.external_config import ExternalConfigCollection
from jetstream.metadata import ExperimentMetadata, export_metadata
from jetstream.statistics import StatisticResult


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
        bigger_is_better = false

        [metrics.my_cool_metric.statistics.bootstrap_mean]

        [metrics.view_about_logins.statistics.bootstrap_mean]
        """
    )

    spec = AnalysisSpec.from_dict(toml.loads(config_str))
    config = spec.resolve(experiments[4])
    metadata = ExperimentMetadata.from_config(config)

    assert StatisticResult.SCHEMA_VERSION == metadata.schema_version
    assert "view_about_logins" in metadata.metrics
    assert metadata.metrics["view_about_logins"].bigger_is_better
    assert metadata.metrics["view_about_logins"].description != ""
    assert "my_cool_metric" in metadata.metrics
    assert metadata.metrics["my_cool_metric"].bigger_is_better is False
    assert metadata.metrics["my_cool_metric"].friendly_name == "Cool metric"
    assert metadata.metrics["my_cool_metric"].description == "Cool cool cool"
    assert metadata.metrics["my_cool_metric"].analysis_bases == ["enrollments"]
    assert metadata.external_config is None


@patch.object(requests.Session, "get")
def test_metadata_reference_branch(mock_get, experiments):
    config_str = dedent(
        """
        [experiment]
        reference_branch = "a"

        [metrics]
        weekly = ["view_about_logins"]

        [metrics.view_about_logins.statistics.bootstrap_mean]
        """
    )

    spec = AnalysisSpec.from_dict(toml.loads(config_str))
    config = spec.resolve(experiments[4])
    metadata = ExperimentMetadata.from_config(config)

    assert metadata.external_config.reference_branch == "a"
    assert (
        metadata.external_config.url
        == ExternalConfigCollection.JETSTREAM_CONFIG_URL + "/blob/main/normandy-test-slug.toml"
    )

    config_str = dedent(
        """
        [metrics]
        weekly = ["view_about_logins"]

        [metrics.view_about_logins.statistics.bootstrap_mean]
        """
    )

    spec = AnalysisSpec.from_dict(toml.loads(config_str))
    config = spec.resolve(experiments[2])
    metadata = ExperimentMetadata.from_config(config)

    assert metadata.external_config is None


def test_metadata_with_outcomes(experiments, fake_outcome_resolver):
    config_str = dedent(
        """
        [metrics]
        weekly = ["view_about_logins"]

        [metrics.view_about_logins.statistics.bootstrap_mean]
        """
    )

    spec = AnalysisSpec.from_dict(toml.loads(config_str))
    config = spec.resolve(experiments[5])
    metadata = ExperimentMetadata.from_config(config)

    assert "view_about_logins" in metadata.metrics
    assert metadata.metrics["view_about_logins"].bigger_is_better
    assert metadata.metrics["view_about_logins"].description != ""

    assert "tastiness" in metadata.outcomes
    assert "performance" in metadata.outcomes
    assert "speed" in metadata.outcomes["performance"].default_metrics
    assert metadata.outcomes["tastiness"].friendly_name == "Tastiness outcomes"
    assert "meals_eaten" in metadata.outcomes["tastiness"].metrics
    assert metadata.outcomes["tastiness"].default_metrics == []


@patch.object(requests.Session, "get")
def test_metadata_from_config_missing_metadata(mock_get, experiments):
    config_str = dedent(
        """
        [metrics]
        weekly = ["view_about_logins", "my_cool_metric"]
        daily = ["my_cool_metric"]

        [metrics.my_cool_metric]
        data_source = "main"
        select_expression = "{{agg_histogram_mean('payload.content.my_cool_histogram')}}"
        analysis_bases = ["exposures"]

        [metrics.my_cool_metric.statistics.bootstrap_mean]

        [metrics.view_about_logins.statistics.bootstrap_mean]
        """
    )

    spec = AnalysisSpec.from_dict(toml.loads(config_str))
    config = spec.resolve(experiments[0])
    metadata = ExperimentMetadata.from_config(config)

    assert "my_cool_metric" in metadata.metrics
    assert metadata.metrics["my_cool_metric"].bigger_is_better
    assert metadata.metrics["my_cool_metric"].friendly_name == ""
    assert metadata.metrics["my_cool_metric"].description == ""
    assert metadata.metrics["my_cool_metric"].analysis_bases == ["exposures"]


@mock.patch("google.cloud.storage.Client")
def test_export_metadata(mock_storage_client, experiments):
    config_str = dedent(
        """
        [experiment]
        end_date = "2021-07-01"

        [metrics]
        weekly = ["view_about_logins", "my_cool_metric"]

        [metrics.my_cool_metric]
        data_source = "main"
        select_expression = "{{agg_histogram_mean('payload.content.my_cool_histogram')}}"

        [metrics.my_cool_metric.statistics.bootstrap_mean]

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

    export_metadata(config, "test_bucket", "project")

    mock_client.get_bucket.assert_called_once()
    mock_bucket.blob.assert_called_once()

    expected = json.loads(
        r"""
        {
            "metrics": {
                "view_about_logins": {
                    "friendly_name": "about:logins viewers",
                    "description": "Counts the number of clients that viewed about:logins.\n",
                    "bigger_is_better": true,
                    "analysis_bases": ["enrollments"]
                },
                "my_cool_metric": {
                    "friendly_name": "",
                    "description": "",
                    "bigger_is_better": true,
                    "analysis_bases": ["enrollments"]
                }
            },
            "outcomes": {},
            "external_config": {
                "end_date": "2021-07-01",
                "enrollment_period": null,
                "reference_branch": null,
                "skip": false,
                "start_date": null,
                "url": """
        + '"https://github.com/mozilla/jetstream-config/blob/main/normandy-test-slug.toml"'
        + r"""},
            "schema_version":"""
        + str(StatisticResult.SCHEMA_VERSION)
        + """
        }
    """
    )
    mock_blob.upload_from_string.assert_called_once_with(
        data=json.dumps(expected, sort_keys=True, indent=4), content_type="application/json"
    )
