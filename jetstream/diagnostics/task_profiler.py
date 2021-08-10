from datetime import datetime
from typing import Optional

from dask.diagnostics import Profiler
from google.cloud import bigquery


class TaskProfiler(Profiler):
    """Extension of the dask `Profiler` to allows writing results to Bigquery."""

    def __init__(
        self,
        project_id: Optional[str],
        dataset_id: Optional[str],
        table_id: Optional[str],
        bigquery_client: Optional[bigquery.Client] = None,
    ):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.bigquery_client = bigquery_client

        if bigquery_client is None and project_id is not None:
            self.bigquery_client = bigquery.Client(project_id)

        super().__init__()

    def _results_to_json(self, experiment: str):
        """Convert the results to JSON."""
        print("Results: ")
        print(self.results)

        return [
            {
                "experiment": experiment,
                "key": r.key,
                "task": r.task,
                "start_time": datetime.fromtimestamp(r.start_time),
                "end_time": datetime.fromtimestamp(r.end_time),
                "worker_id": r.worker_id,
            }
            for r in self.results
        ]

    def write_to_bigquery(self, experiment: str):
        """Write results to BigQuery."""
        try:
            if self.bigquery_client:
                destination_table = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
                self.bigquery_client.load_table_from_json(
                    self._results_to_json(experiment=experiment), destination_table
                ).result()
            else:
                print("TaskProfiler not configured to write results to BigQuery.")
        except Exception as e:
            print(f"Exception while writing task profiling data to BigQuery: {e}")
