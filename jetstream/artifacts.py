from datetime import datetime, timezone

import attr
from google.cloud import artifactregistry

from jetstream.bigquery_client import BigQueryClient


@attr.s(auto_attribs=True, slots=True)
class ArtifactManager:
    """Access docker images in the artifact registry."""

    project: str
    dataset: str
    image: str
    _client: artifactregistry.ArtifactRegistryClient | None = None

    @property
    def client(self):
        self._client = self._client or artifactregistry.ArtifactRegistryClient()
        return self._client

    @property
    def images(self) -> list[artifactregistry.DockerImage]:
        """Images available in the artifact registry."""
        # list all docker images that are in artifact registry
        request = artifactregistry.ListDockerImagesRequest(
            parent=f"projects/{self.project}/locations/us/repositories/gcr.io",
        )
        result = self.client.list_docker_images(request=request)

        images = []

        for image_data in result:
            if f"/{self.image}@sha256" in image_data.name:
                images.append(image_data)

        return images

    def image_for_slug(self, slug: str) -> str:
        """
        Get the image that should be used to analyse the experiment with the provided slug.

        The image is determined based on the oldest last updated timestamp of the analysis results
        (in other words, the timestamp from the first time the experiment was analyzed).
        """
        client = BigQueryClient(self.project, self.dataset)
        last_updated = client.experiment_table_first_updated(slug=slug)

        if last_updated:
            return self._image_for_date(last_updated)
        else:
            return self.latest_image()

    def _image_for_date(self, date: datetime) -> str:
        """Return the hash for the image that was the latest on the provided date."""
        latest_updated = None
        earliest_uploaded = None

        # filter for the most recent jetstream image
        image: artifactregistry.DockerImage
        for image in self.images:
            # A note on the type ignore comments:
            # - mypy and the DockerImage docs both indicate that `upload_time`
            #   should be Timestamp type, but when we run the tests, they appear
            #   to be DatetimeWithNanoseconds instead. This code comparing
            #   `upload_time` with datetime objects has been working, so
            #   we ignore mypy here due to the conflicting errors.
            if not image:
                continue
            upload_time = image.upload_time

            # img: artifactregistry.DockerImage = image
            if (latest_updated is None and upload_time <= date) or (  # type: ignore
                latest_updated
                and latest_updated.upload_time < upload_time  # type: ignore
                and upload_time <= date  # type: ignore
            ):
                latest_updated = image

            # keep track of the earliest image available
            if (
                earliest_uploaded is None or upload_time <= earliest_uploaded.upload_time  # type: ignore
            ):
                earliest_uploaded = image

        if latest_updated:
            # return hash of image closest to the provided date
            return latest_updated.name.split("sha256:")[1]
        elif earliest_uploaded:
            # return hash of earliest image available if table got created before image got uploaded
            return earliest_uploaded.name.split("sha256:")[1]
        else:
            raise ValueError(f"No `{self.image}` docker image available in {self.project}")

    def latest_image(self) -> str:
        """Return the latest docker image hash."""
        return self._image_for_date(date=datetime.now(timezone.utc))
