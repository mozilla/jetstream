import datetime as dt
import json
from pathlib import Path
from textwrap import dedent
from typing import Callable
from unittest import mock
from unittest.mock import MagicMock, patch

import cattr
import jsonschema
import requests
import toml
from metric_config_parser.analysis import AnalysisSpec
from metric_config_parser.config import Outcome
from metric_config_parser.outcome import OutcomeSpec

from jetstream.config import METRIC_HUB_REPO, ConfigLoader, _ConfigLoader
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
    config = spec.resolve(experiments[4], ConfigLoader.configs)
    metadata = ExperimentMetadata.from_config(config)

    assert StatisticResult.SCHEMA_VERSION == metadata.schema_version
    assert "view_about_logins" in metadata.metrics
    assert metadata.metrics["view_about_logins"].bigger_is_better
    assert metadata.metrics["view_about_logins"].description != ""
    assert "my_cool_metric" in metadata.metrics
    assert metadata.metrics["my_cool_metric"].bigger_is_better is False
    assert metadata.metrics["my_cool_metric"].friendly_name == "Cool metric"
    assert metadata.metrics["my_cool_metric"].description == "Cool cool cool"
    assert metadata.metrics["my_cool_metric"].analysis_bases == ["enrollments", "exposures"]
    assert metadata.external_config is None
    assert metadata.analysis_start_time is None


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
    config = spec.resolve(experiments[4], ConfigLoader.configs)
    metadata = ExperimentMetadata.from_config(config)

    assert metadata.external_config.reference_branch == "a"
    assert (
        metadata.external_config.url
        == METRIC_HUB_REPO + "/blob/main/jetstream/normandy-test-slug.toml"
    )

    config_str = dedent(
        """
        [metrics]
        weekly = ["view_about_logins"]

        [metrics.view_about_logins.statistics.bootstrap_mean]
        """
    )

    spec = AnalysisSpec.from_dict(toml.loads(config_str))
    config = spec.resolve(experiments[2], ConfigLoader.configs)
    metadata = ExperimentMetadata.from_config(config)

    assert metadata.external_config is None


def test_metadata_with_outcomes(experiments):
    config_str = dedent(
        """
        [metrics]
        weekly = ["view_about_logins"]

        [metrics.view_about_logins.statistics.bootstrap_mean]
        """
    )

    performance_config = dedent(
        """
        friendly_name = "Performance outcomes"
        description = "Outcomes related to performance"
        default_metrics = ["speed"]

        [metrics.speed]
        data_source = "main"
        select_expression = "1"

        [metrics.speed.statistics.bootstrap_mean]
        """
    )

    tastiness_config = dedent(
        """
        friendly_name = "Tastiness outcomes"
        description = "Outcomes related to tastiness 😋"

        [metrics.meals_eaten]
        data_source = "meals"
        select_expression = "1"
        friendly_name = "Meals eaten"
        description = "Number of consumed meals"

        [metrics.meals_eaten.statistics.bootstrap_mean]
        num_samples = 10
        pre_treatments = ["remove_nulls"]

        [data_sources.meals]
        from_expression = "meals"
        client_id_column = "client_info.client_id"
        """
    )

    loader = _ConfigLoader()
    loader.configs.outcomes += [
        Outcome(
            slug="performance",
            spec=OutcomeSpec.from_dict(toml.loads(performance_config)),
            platform="firefox_desktop",
            commit_hash="000000",
        ),
        Outcome(
            slug="tastiness",
            spec=OutcomeSpec.from_dict(toml.loads(tastiness_config)),
            platform="firefox_desktop",
            commit_hash="000000",
        ),
    ]

    spec = AnalysisSpec.from_dict(toml.loads(config_str))
    config = spec.resolve(experiments[5], loader.configs)
    metadata = ExperimentMetadata.from_config(config, config_loader=loader)

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
    config = spec.resolve(experiments[0], ConfigLoader.configs)
    metadata = ExperimentMetadata.from_config(config)

    assert "my_cool_metric" in metadata.metrics
    assert metadata.metrics["my_cool_metric"].bigger_is_better
    assert metadata.metrics["my_cool_metric"].friendly_name == "My Cool Metric"
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
    config = spec.resolve(experiments[0], ConfigLoader.configs)

    mock_client = MagicMock()
    mock_storage_client.return_value = mock_client
    mock_bucket = MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_blob.upload_from_string.return_value = ""
    mock_analysis_start = dt.datetime.now()

    export_metadata(config, "test_bucket", "project", mock_analysis_start)

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
                    "analysis_bases": ["enrollments", "exposures"]
                },
                "my_cool_metric": {
                    "friendly_name": "My Cool Metric",
                    "description": "",
                    "bigger_is_better": true,
                    "analysis_bases": ["enrollments", "exposures"]
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
        + '"https://github.com/mozilla/metric-hub/blob/main/jetstream/normandy-test-slug.toml"'
        + r"""},
            "analysis_start_time": """
        + f'"{mock_analysis_start}"'
        + """,
            "schema_version":"""
        + str(StatisticResult.SCHEMA_VERSION)
        + """
        }
    """
    )
    mock_blob.upload_from_string.assert_called_once_with(
        data=json.dumps(expected, sort_keys=True, indent=4), content_type="application/json"
    )


@mock.patch("google.cloud.storage.Client")
def test_export_confidential_metadata(mock_storage_client, experiments):
    config_str = dedent(
        """
        [experiment]
        end_date = "2021-07-01"
        is_private = true
        dataset_id = "test"

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
    config = spec.resolve(experiments[0], ConfigLoader.configs)

    mock_client = MagicMock()
    mock_storage_client.return_value = mock_client
    mock_bucket = MagicMock()
    mock_client.get_bucket.return_value = mock_bucket
    mock_blob = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_blob.upload_from_string.return_value = ""
    mock_analysis_start = dt.datetime.now()

    export_metadata(config, "test_bucket", "project", mock_analysis_start)

    mock_client.get_bucket.assert_not_called()
    mock_bucket.blob.assert_not_called()


def test_metadata_schema(experiments):
    schema = json.loads((Path(__file__).parent / "data/Metadata_v1.0.json").read_text())
    converter = cattr.Converter()
    _date_to_json: Callable[[dt.date], str] = lambda d: d.strftime("%Y-%m-%d")
    converter.register_unstructure_hook(dt.date, _date_to_json)
    _datetime_to_json: Callable[[dt.datetime], str] = lambda dt: str(dt)
    converter.register_unstructure_hook(dt.datetime, _datetime_to_json)

    config_str = dedent(
        """
        [experiment]

        start_date = '2022-01-01'

        [metrics]
        weekly = ["view_about_logins"]

        [metrics.view_about_logins.statistics.bootstrap_mean]
        """
    )

    performance_config = dedent(
        """
        friendly_name = "Performance outcomes"
        description = "Outcomes related to performance"
        default_metrics = ["speed"]

        [metrics.speed]
        data_source = "main"
        select_expression = "1"

        [metrics.speed.statistics.bootstrap_mean]
        """
    )

    tastiness_config = dedent(
        """
        friendly_name = "Tastiness outcomes"
        description = "Outcomes related to tastiness 😋"

        [metrics.meals_eaten]
        data_source = "meals"
        select_expression = "1"
        friendly_name = "Meals eaten"
        description = "Number of consumed meals"

        [metrics.meals_eaten.statistics.bootstrap_mean]
        num_samples = 10
        pre_treatments = ["remove_nulls"]

        [data_sources.meals]
        from_expression = "meals"
        client_id_column = "client_info.client_id"
        """
    )

    loader = _ConfigLoader()
    loader.configs.outcomes += [
        Outcome(
            slug="performance",
            spec=OutcomeSpec.from_dict(toml.loads(performance_config)),
            platform="firefox_desktop",
            commit_hash="000000",
        ),
        Outcome(
            slug="tastiness",
            spec=OutcomeSpec.from_dict(toml.loads(tastiness_config)),
            platform="firefox_desktop",
            commit_hash="000000",
        ),
    ]

    spec = AnalysisSpec.from_dict(toml.loads(config_str))
    config = spec.resolve(experiments[5], loader.configs)
    metadata = ExperimentMetadata.from_config(config, dt.datetime.now(), config_loader=loader)

    jsonschema.validate(converter.unstructure(metadata), schema)
