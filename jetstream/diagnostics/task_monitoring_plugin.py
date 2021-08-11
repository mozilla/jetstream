from datetime import datetime
from typing import Dict, List, Optional

from distributed.diagnostics.plugin import SchedulerPlugin
from distributed.scheduler import Scheduler
from google.cloud import bigquery


class TaskMonitoringPlugin(SchedulerPlugin):
    """Plugin to monitor Dask tasks."""

    def __init__(
        self,
        scheduler: Scheduler,
        project_id: Optional[str],
        dataset_id: Optional[str],
        table_id: Optional[str],
        experiment: Optional[str],
    ) -> None:
        SchedulerPlugin.__init__(self)
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.experiment = experiment
        self.scheduler = scheduler
        self.cache: List[Dict] = []

    def transition(self, key, start, finish, *args, **kwargs):
        """Called by the scheduler every time a task changes status."""
        tasks = [task for _, task in self.scheduler.tasks.items()]
        changed_tasks = [task for task in tasks if (task.key, task.state) not in self.cache]
        worker_address = kwargs["worker"]

        results = [
            {
                "experiment": self.experiment,
                "key": task.key,
                "timestamp": datetime.fromtimestamp(kwargs["startstops"][0]["stop"]).isoformat(),
                "worker_address": worker_address,
                "state": task.state,
            }
            for task in changed_tasks
        ]

        self.cache = [(task.key, task.state) for task in tasks]
        self._write_to_bigquery(results)

    def _write_to_bigquery(self, results: List[Dict]):
        """Write resource usage results to BigQuery."""
        try:
            if self.project_id and results != []:
                client = bigquery.Client(self.project_id)
                destination_table = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
                client.load_table_from_json(results, destination_table).result()
            elif self.project_id is None:
                print("ResourceProfilingPlugin not configured to write results to BigQuery.")
        except Exception as e:
            print(f"Exception while writing resource usage profiling data to BigQuery: {e}")
