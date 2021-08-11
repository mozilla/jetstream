import logging
from typing import Optional

import attr
import dask.distributed
from distributed.diagnostics.plugin import WorkerPlugin

from .bigquery_log_handler import BigQueryLogHandler


@attr.s(auto_attribs=True)
class LogConfiguration:
    """Configuration for setting up logging."""

    log_project_id: Optional[str]
    log_dataset_id: Optional[str]
    log_table_id: Optional[str]
    task_profiling_log_table_id: Optional[str]
    task_monitoring_log_table_id: Optional[str]
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


class LogPlugin(WorkerPlugin):
    """
    Dask worker plugin for initializing the logger.

    This ensures that the BigQuery logging handler gets initialized.
    """

    def __init__(self, log_config: LogConfiguration):
        self.log_config = log_config

    def setup(self, worker: dask.distributed.Worker):
        self.log_config.setup_logger()
