from google.cloud import bigquery
from logging.handlers import BufferingHandler
from typing import Optional


class BigQueryLogHandler(BufferingHandler):
    """Custom logging handler for writing logs to BigQuery."""

    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        table_id: str,
        client: Optional[bigquery.Client] = None,
        capacity=50,
    ):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.client = client
        if client is None:
            self.client = bigquery.Client(project_id)

        super().__init__(capacity)

    def _buffer_to_json(self, buffer):
        """Converts the records in the buffer to JSON."""
        return [
            {
                "submission_timestamp": record.created,
                "experiment": None if not hasattr(record, "experiment") else record.experiment,
                "message": record.msg,
                "log_level": record.levelname,
                "exception": str(record.exc_info),
                "filename": record.filename,
                "funcName": record.funcName,
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
