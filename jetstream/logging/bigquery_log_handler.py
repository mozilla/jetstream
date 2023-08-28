import datetime
from logging.handlers import BufferingHandler
from typing import Optional

from google.cloud import bigquery


class BigQueryLogHandler(BufferingHandler):
    """Custom logging handler for writing logs to BigQuery."""

    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        table_id: str,
        source: str,
        client: Optional[bigquery.Client] = None,
        capacity=50,
    ):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.client = client
        if client is None:
            self.client = bigquery.Client(project_id)
        self.source = source

        super().__init__(capacity)

    def _buffer_to_json(self, buffer):
        """Converts the records in the buffer to JSON."""
        return [
            {
                "timestamp": datetime.datetime.fromtimestamp(record.created).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "source": self.source if not hasattr(record, "source") else record.source,
                "experiment": None if not hasattr(record, "experiment") else record.experiment,
                "metric": None if not hasattr(record, "metric") else record.metric,
                "statistic": None if not hasattr(record, "statistic") else record.statistic,
                "analysis_basis": None
                if not hasattr(record, "analysis_basis")
                else record.analysis_basis,
                "segment": None if not hasattr(record, "segment") else record.segment,
                "message": record.getMessage(),
                "log_level": record.levelname,
                "exception": str(record.exc_info),
                "filename": record.filename,
                "func_name": record.funcName,
                "exception_type": None if not record.exc_info else record.exc_info[0].__name__,
            }
            for record in buffer
        ]

    def flush(self):
        """
        Override default flushing behaviour.

        Write the buffer to BigQuery.
        """
        self.acquire()
        try:
            if self.buffer:
                destination_table = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
                self.client.load_table_from_json(
                    self._buffer_to_json(self.buffer), destination_table
                ).result()
            self.buffer = []
        except Exception as e:
            print(f"Exception while flushing logs: {e}")
            pass
        finally:
            self.release()
