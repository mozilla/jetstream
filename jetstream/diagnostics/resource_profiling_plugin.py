import os
import pickle
from collections import defaultdict
from datetime import datetime
from threading import Lock, Thread
from time import sleep
from typing import Any, Dict, List, Optional

import attr
from distributed.client import Client
from distributed.diagnostics.plugin import SchedulerPlugin
from distributed.scheduler import Scheduler
from google.cloud import bigquery
from psutil import Process


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ResourceUsage:
    """Recorded resource usages."""

    worker_address: str
    memory_usage: List[float]
    cpu_usage: List[float]


# See https://github.com/itamarst/dask-memusage/blob/22f1e44c6c3b32bd3125cd1f3fd1b512c568717f/
# dask_memusage.py#L32
def _process_memory():
    """Return process memory usage, in MB."""
    proc = Process(os.getpid())
    return sum([p.memory_info().rss / (1024 * 1024) for p in [proc]])


def _process_cpu():
    """Return process CPU usage."""
    proc = Process(os.getpid())
    return sum([p.cpu_percent(interval=0.01) for p in [proc]])


class WorkerResourceUsage(object):
    """Track memory usage for each worker."""

    def __init__(self, scheduler_address: str, update_freq: float):
        self._scheduler_address = scheduler_address
        self._lock = Lock()
        self._worker_memory: Dict[str, Any] = defaultdict(list)
        self._worker_cpu: Dict[str, Any] = defaultdict(list)
        self.update_freq = update_freq

    def start(self):
        """Start the thread."""
        t = Thread(target=self._fetch_resources, name="WorkerResources")
        t.setDaemon(True)
        t.start()

    def _add_memory(self, worker_address: str, mem: float):
        """Record memory time point for a worker."""
        self._worker_memory[worker_address].append(mem)

    def _add_cpu(self, worker_address: str, cpu: float):
        """Record CPU usage time point for a worker."""
        self._worker_cpu[worker_address].append(cpu)

    def _fetch_resources(self):
        """Retrieve worker resources."""
        client = Client(self._scheduler_address, timeout=30)
        while True:
            worker_memory = client.run(_process_memory)
            with self._lock:
                for worker, mem in worker_memory.items():
                    self._add_memory(worker, mem)

            worker_cpu = client.run(_process_cpu)
            with self._lock:
                for worker, cpu in worker_cpu.items():
                    self._add_cpu(worker, cpu)

            sleep(self.update_freq)

    def resources_for_task(self, worker_address: str):
        """The worker finished its previous task.
        Return its resource usage and then reset it.
        """
        with self._lock:
            mem_result = self._worker_memory[worker_address]
            if not mem_result:
                mem_result = [0]
            del self._worker_memory[worker_address]

            cpu_result = self._worker_cpu[worker_address]
            if not cpu_result:
                cpu_result = [0]
            del self._worker_cpu[worker_address]

            return ResourceUsage(
                worker_address=worker_address, memory_usage=mem_result, cpu_usage=cpu_result
            )


class ResourceProfilingPlugin(SchedulerPlugin):
    """Resource profiling plugin for Dask distributed."""

    def __init__(
        self,
        scheduler: Scheduler,
        project_id: Optional[str],
        dataset_id: Optional[str],
        table_id: Optional[str],
        experiment: Optional[str],
        update_freq: float = 10000.0,  # fetch resource usage every 10 seconds
    ):
        SchedulerPlugin.__init__(self)
        self.results: List[Dict] = []
        self.memory_usage = None
        self.scheduler = scheduler
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.experiment = experiment

        self._worker_resources = WorkerResourceUsage(scheduler.address, update_freq)
        self._worker_resources.start()

    def remove_worker(self, scheduler, worker, **kwargs):
        self._write_to_bigquery()
        return super().remove_worker(scheduler=scheduler, worker=worker, **kwargs)

    def transition(self, key, start, finish, *args, **kwargs):
        """Called by the scheduler every time a task changes status."""
        # if the task finished, record its memory usage:
        if start == "processing" and finish in ("memory", "erred"):
            worker_address = kwargs["worker"]
            resource_usage = self._worker_resources.resources_for_task(worker_address)
            self.results.append(
                {
                    "experiment": self.experiment,
                    "key": key,
                    "function": str(pickle.loads(self.scheduler.tasks[key].run_spec["function"])),
                    "args": str(pickle.loads(self.scheduler.tasks[key].run_spec["args"])),
                    "start": datetime.fromtimestamp(kwargs["startstops"][0]["start"]).isoformat(),
                    "end": datetime.fromtimestamp(kwargs["startstops"][0]["stop"]).isoformat(),
                    "worker_address": worker_address,
                    "max_memory": float(max(resource_usage.memory_usage)),
                    "min_memory": float(min(resource_usage.memory_usage)),
                    "max_cpu": float(max(resource_usage.cpu_usage)),
                    "min_cpu": float(min(resource_usage.cpu_usage)),
                }
            )

        # make sure that all tasks have finished and write all usage stats to BigQuery at once
        # this improves performances vs. writing results on every transition
        # https://distributed.dask.org/en/latest/scheduling-state.html#task-state
        finished_tasks = [
            task.state in ["released", "erred", "forgotten"]
            for _, task in self.scheduler.tasks.items()
        ]
        if all(finished_tasks):
            self._write_to_bigquery()

    def _write_to_bigquery(self):
        """Write resource usage results to BigQuery."""
        try:
            if self.project_id and self.results != []:
                client = bigquery.Client(self.project_id)
                destination_table = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
                client.load_table_from_json(self.results, destination_table).result()
                self.results = []
            elif self.project_id is None:
                print("ResourceProfilingPlugin not configured to write results to BigQuery.")
        except Exception as e:
            print(f"Exception while writing resource usage profiling data to BigQuery: {e}")
