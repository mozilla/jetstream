import os
from collections import defaultdict
from threading import Lock, Thread
from time import sleep
from typing import Dict, List, Optional

import attr
from distributed.client import Client
from distributed.diagnostics.plugin import SchedulerPlugin
from distributed.scheduler import Scheduler
from distributed.utils import Any
from google.cloud import bigquery
from psutil import Process
import pickle


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
    return sum(
        [p.memory_info().rss / (1024 * 1024) for p in [proc]]
    )


def _process_cpu():
    """Return process CPU usage."""
    proc = Process(os.getpid())
    print(proc.children(recursive=True))
    return sum([p.cpu_percent(interval=0.1) for p in [proc]])


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
        client: Optional[bigquery.Client] = None,
        update_freq: float = 10 #10000.0,  # fetch resource usage every 10 seconds
    ):
        SchedulerPlugin.__init__(self)
        self.memory_usage = None
        self.scheduler = scheduler
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.client = client

        if self.client is None:
            self.client = bigquery.Client(project_id)

        self._worker_resources = WorkerResourceUsage(scheduler.address, update_freq)
        self._worker_resources.start()

    def transition(self, key, start, finish, *args, **kwargs):
        """Called by the scheduler every time a task changes status."""
        # if the task finished, record its memory usage:
        if start == "processing" and finish in ("memory", "erred"):
            worker_address = kwargs["worker"]
            self.resource_usage = self._worker_resources.resources_for_task(worker_address)
            print(key)
            print([str(pickle.loads(t.run_spec['function'])) for _, t in self.scheduler.tasks.items()])
            print([pickle.loads(t.run_spec['args']) for _, t in self.scheduler.tasks.items()])
            # print([pickle.loads(t.run_spec['function']) for _, t in self.scheduler.tasks.items() for a in t])
            self._write_to_bigquery()
            # max_memory_usage = max(memory_usage)
            # min_memory_usage = min(memory_usage)

    def _write_to_bigquery(self):
        """Write resource usage results to BigQuery."""
        print(self.resource_usage)
