from datetime import datetime, timezone
from unittest import mock
from unittest.mock import MagicMock

import pytest
import pytz

from jetstream.artifacts import ArtifactManager


class TestArtifactManager:
    def test_image_filter(self, docker_images):
        artifact_client = MagicMock()
        artifact_client.list_docker_images.return_value = docker_images

        artifact_manager = ArtifactManager(
            "moz-fx-data-experiments", "mozanalysis", "jetstream", artifact_client
        )
        assert len(artifact_manager.images) == 2
        assert "dockerImages/jetstream@sha256:8c766a" in artifact_manager.images[0].name

    def test_latest_image(self, docker_images):
        artifact_client = MagicMock()
        artifact_client.list_docker_images.return_value = docker_images

        artifact_manager = ArtifactManager(
            "moz-fx-data-experiments", "mozanalysis", "jetstream", artifact_client
        )
        latest_image = artifact_manager.latest_image()
        assert latest_image == "xxxxx"

    def test_invalid_latest_image(self, docker_images):
        artifact_client = MagicMock()
        artifact_client.list_docker_images.return_value = docker_images

        proj = "moz-fx-data-experiments"
        image = "not-existing"
        artifact_manager = ArtifactManager(proj, "mozanalysis", image, artifact_client)
        with pytest.raises(ValueError, match=f"No `{image}` docker image available in {proj}"):
            artifact_manager.latest_image()

    def test_image_for_date(self, docker_images):
        artifact_client = MagicMock()
        artifact_client.list_docker_images.return_value = docker_images

        artifact_manager = ArtifactManager(
            "moz-fx-data-experiments", "mozanalysis", "jetstream", artifact_client
        )
        image = artifact_manager._image_for_date(date=datetime.now(timezone.utc))
        assert image == "xxxxx"

        image = artifact_manager._image_for_date(date=pytz.UTC.localize(datetime(2023, 2, 1)))
        assert image == "8c766a"

        image = artifact_manager._image_for_date(date=pytz.UTC.localize(datetime(2019, 2, 1)))
        assert image == "8c766a"

    @pytest.mark.parametrize(
        ("tags", "expected_image"), [(["breaking"], "xxxxx"), (["non-breaking"], "8c766a")]
    )
    def test_image_for_slug(self, docker_images, tags, expected_image):
        artifact_client = MagicMock()
        # update tags on most recent image
        docker_images[2].tags = tags
        artifact_client.list_docker_images.return_value = docker_images

        with mock.patch("jetstream.artifacts.BigQueryClient") as mock_client:
            bigquery_mock_client = MagicMock()
            bigquery_mock_client.experiment_table_first_updated.return_value = datetime(
                2023, 1, 1, tzinfo=pytz.UTC
            )
            mock_client.return_value = bigquery_mock_client

            artifact_manager = ArtifactManager(
                "moz-fx-data-experiments", "mozanalysis", "jetstream", artifact_client
            )
            image = artifact_manager.image_for_slug("test-slug")
            assert image == expected_image
