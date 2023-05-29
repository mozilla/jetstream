from datetime import datetime
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

        artifact_manager = ArtifactManager(
            "moz-fx-data-experiments", "mozanalysis", "not-existing", artifact_client
        )
        with pytest.raises(ValueError):
            artifact_manager.latest_image()

    def test_image_for_date(self, docker_images):
        artifact_client = MagicMock()
        artifact_client.list_docker_images.return_value = docker_images

        artifact_manager = ArtifactManager(
            "moz-fx-data-experiments", "mozanalysis", "jetstream", artifact_client
        )
        image = artifact_manager._image_for_date(date=pytz.UTC.localize(datetime.utcnow()))
        assert image == "xxxxx"

        image = artifact_manager._image_for_date(date=pytz.UTC.localize(datetime(2023, 2, 1)))
        assert image == "8c766a"

        image = artifact_manager._image_for_date(date=pytz.UTC.localize(datetime(2019, 2, 1)))
        assert image == "8c766a"
