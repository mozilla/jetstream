from argo.workflows.client import ApiClient, Configuration, V1alpha1Api
import base64
from google.cloud.container_v1 import ClusterManagerClient
import google.auth
from pathlib import Path
from tempfile import NamedTemporaryFile
import time
from typing import Dict, Any
import yaml


def apply_parameters(manifest: Dict[Any, Any], parameters: Dict[str, Any]) -> Dict[Any, Any]:
    """Apply custom parameters to the workflow manifest."""
    workflow_parameters = manifest["spec"]["arguments"]["parameters"]
    for key, value in parameters.items():
        exists = False
        for workflow_param in workflow_parameters:
            # overwrite existing
            if workflow_param["name"] == key:
                workflow_param["value"] = value
                exists = True

        if not exists:
            # append non-existing parameters
            workflow_parameters.append({"name": key, "value": value})

    return manifest


def submit_workflow(
    project_id: str,
    zone: str,
    cluster_id: str,
    workflow_file: Path,
    parameters: Dict[str, Any],
    monitor_status: bool = False,
):
    """Submit a workflow to Argo."""

    config = get_config(project_id, zone, cluster_id)
    api_client = ApiClient(configuration=config)
    api = V1alpha1Api(api_client=api_client)
    manifest = yaml.safe_load(workflow_file.read_text())
    manifest = apply_parameters(manifest, parameters)

    workflow = api.create_namespaced_workflow("argo", manifest)

    if monitor_status:
        finished = False

        print("Worflow running")
        print(workflow.status)

        while not finished:
            workflow = api.get_namespaced_workflow(
                workflow.metadata.namespace, workflow.metadata.name
            )

            if workflow.status.finished_at is not None:
                finished = True
                if workflow.status.phase == "Failed":
                    raise Exception(f"Workflow execution failed: {workflow.status}")

            time.sleep(1)


def get_config(project_id: str, zone: str, cluster_id: str):
    """Get the kubernetes config."""
    cluster_manager_client = ClusterManagerClient()
    cluster = cluster_manager_client.get_cluster(
        name=f"projects/{project_id}/locations/{zone}/clusters/{cluster_id}"
    )

    credentials, project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds, projects = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)

    configuration = Configuration()
    configuration.host = f"https://{cluster.endpoint}"
    with NamedTemporaryFile(delete=False) as ca_cert:
        ca_cert.write(base64.b64decode(cluster.master_auth.cluster_ca_certificate))
    configuration.ssl_ca_cert = ca_cert.name
    configuration.api_key_prefix["authorization"] = "Bearer"
    configuration.api_key["authorization"] = creds.token

    return configuration
