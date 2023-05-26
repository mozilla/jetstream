from datetime import datetime
from typing import List, Optional

import attr
import pytz
from google.cloud import artifactregistry, bigquery
from pytz import UTC

from . import bq_normalize_name


@attr.s(auto_attribs=True, slots=True)
class ArtifactManager:
    """Access docker images in the artifact registry."""

    project: str
    _client: Optional[artifactregistry.ArtifactRegistryClient] = None

    @property
    def client(self):
        self._client = self._client or artifactregistry.ArtifactRegistryClient()
        return self._client

    @property
    def images(self) -> List[artifactregistry.DockerImage]:
        """Images available in the artifact registry."""
        # list all docker images that are in artifact registry
        request = artifactregistry.ListDockerImagesRequest(
            parent=f"projects/{self.project}/locations/us/repositories/gcr.io",
        )
        result = self.client.list_docker_images(request=request)

        images = []

        for image_data in result:
            if "/jetstream@sha256" in image_data.name:
                images.append(image_data)

        return images

    def image_for_slug(self, slug: str) -> str:
        """
        Get the image that should be used to analyse the experiment with the provided slug.

        The image is determined based on the oldes last updated timestamp of the analysis results.
        """
        last_updated = self._slug_last_updated(slug)

        if last_updated:
            return self._image_for_date(last_updated)
        else:
            return self._latest_image()

    def _slug_last_updated(self, slug: str) -> Optional[datetime]:
        """
        Determine the last updated timestamp of the slug based on the statistic table labels.

        Returns `None` if no table related to the experiment exists.
        """
        client = bigquery.Client(self.project)
        table_prefix = bq_normalize_name(slug)

        job = client.query(
            rf"""
            SELECT
                table_name,
                REGEXP_EXTRACT_ALL(
                    option_value,
                    '.*STRUCT\\(\"last_updated\", \"([^\"]+)\"\\).*'
                ) AS last_updated
            FROM
            {self.dataset}.INFORMATION_SCHEMA.TABLE_OPTIONS
            WHERE option_name = 'labels' AND table_name LIKE "statistics_{table_prefix}%"
            """
        )
        result = list(job.result())

        table_last_updated = None

        for row in result:
            if not len(row.last_updated):
                continue
            table_last_updated = UTC.localize(datetime.utcfromtimestamp(int(row.last_updated[0])))

        return table_last_updated

    def _image_for_date(self, date: datetime) -> str:
        """Return the hash for the image that was the latest on the provided date."""
        latest_updated = None
        earliest_uploaded = None

        # filter for the most recent jetstream image
        for image in self.images:
            updated_timestamp = image.update_time

            if (latest_updated is None and image.update_time <= date) or (
                latest_updated
                and latest_updated.update_time < updated_timestamp
                and image.update_time <= date
            ):
                latest_updated = image

            # keep track of the earliest image available
            if earliest_uploaded is None or image.update_time <= earliest_uploaded.update_time:
                earliest_uploaded = image

        if latest_updated:
            # return hash of image closest to the provided date
            return latest_updated.name.split("sha256:")[1]
        elif earliest_uploaded:
            # return hash of earliest image available if table got created before image got uploaded
            return earliest_uploaded.name.split("sha256:")[1]
        else:
            raise ValueError(f"No jetstream docker image available in {self.project}")

    def _latest_image(self) -> Optional[str]:
        """Return the latest docker image hash."""
        return self._image_for_date(date=pytz.UTC.localize(datetime.utcnow()))
