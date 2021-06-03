import attr
import logging

from typing import Optional

from .bigquery_log_handler import BigQueryLogHandler


@attr.s(auto_attribs=True)
class LogConfiguration:
    """Configuration for setting up logging."""

    log_project_id: Optional[str]
    log_dataset_id: Optional[str]
    log_table_id: Optional[str]
    log_to_bigquery: bool = False
    capacity: int = 50

    def setup_logger(self, client=None):
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s:%(asctime)s:%(name)s:%(message)s",
        )
        logger = logging.getLogger()

        if self.log_to_bigquery:
            bigquery_handler = BigQueryLogHandler(
                self.log_project_id, self.log_dataset_id, self.log_table_id, client, self.capacity
            )
            bigquery_handler.setLevel(logging.WARNING)
            logger.addHandler(bigquery_handler)
