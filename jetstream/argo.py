import base64
import logging
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Optional

import google.auth
import yaml
from argo.workflows.client import ApiClient, Configuration, V1alpha1Api
from google.cloud.container_v1 import ClusterManagerClient

logger = logging.getLogger(__name__)


def apply_parameters(manifest: Dict[Any, Any], parameters: Dict[str, Any]) -> Dict[Any, Any]:
    """Apply custom parameters to the workflow manifest."""
    # Currently, there is no option for providing custom parameters for workflows.
    # apply_parameters works around this limitation by modifying the parsed manifest
    # and injecting custom parameters.
    workflow_parameters = manifest["spec"]["arguments"]["parameters"]
    for key, value in parameters.items():
        exists = False
        for workflow_param in workflow_parameters:
            # overwrite existing
            if workflow_param["name"] == key:
                # the array needs to be encoded as string
                # Argo doesn't support ' in it's configuration so replace with "
                workflow_param["value"] = str(value).replace("'", '"')
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
    cluster_ip: Optional[str] = None,
    cluster_cert: Optional[str] = None,
) -> bool:
    """Submit a workflow to Argo and return success."""
    api = get_api(project_id, zone, cluster_id, cluster_ip, cluster_cert)
    manifest = yaml.safe_load(workflow_file.read_text())
    manifest = apply_parameters(manifest, parameters)

    workflow = api.create_namespaced_workflow("argo", manifest)

    failed = False
    if monitor_status:
        finished = False

        logger.info("Argo workflow is running")
        # link to logs
        logger.info(
            "To connect to Argo dashboard forward port by running: "
            + f"gcloud container clusters get-credentials jetstream --zone {zone} "
            + f"--project {project_id} && "
            + "kubectl port-forward --namespace argo $(kubectl get pod --namespace argo "
            + "--selector='app=argo-server' --output jsonpath='{.items[0].metadata.name}') "
            + "8080:2746"
        )
        logger.info("The dashboard can be accessed via 127.0.0.1:8080")

        while not finished:
            try:
                workflow = api.get_namespaced_workflow(
                    workflow.metadata.namespace, workflow.metadata.name
                )
            except Exception:
                # Unauthorized status gets returned when the access token expires (after ca. 1h)
                # Refresh the access token and try again
                api = get_api(project_id, zone, cluster_id, cluster_ip, cluster_cert)
                workflow = api.get_namespaced_workflow(
                    workflow.metadata.namespace, workflow.metadata.name
                )

            if workflow.status and workflow.status.finished_at is not None:
                finished = True
                if workflow.status.phase == "Failed":
                    raise Exception(f"Workflow execution failed: {workflow.status}")
                    failed = True

            time.sleep(1)

    # check status of pods
    all_pods_succeeded = True

    if workflow.status and workflow.status.nodes:
        all_pods_succeeded = all(
            [
                node.phase == "Succeeded"
                for _, node in workflow.status.nodes.items()
                if node.type == "Pod"
            ]
        )

    return not failed and all_pods_succeeded


def get_api(project_id, zone, cluster_id, cluster_ip, cluster_cert):
    """Get Argo API handle."""
    config = get_config(project_id, zone, cluster_id, cluster_ip, cluster_cert)
    api_client = ApiClient(configuration=config)
    return V1alpha1Api(api_client=api_client)


def get_config(
    project_id: str,
    zone: str,
    cluster_id: str,
    cluster_ip: Optional[str] = None,
    cluster_cert: Optional[str] = None,
):
    """Get the kubernetes cluster config."""

    if not cluster_ip and not cluster_cert:
        cluster_manager_client = ClusterManagerClient()
        cluster = cluster_manager_client.get_cluster(
            name=f"projects/{project_id}/locations/{zone}/clusters/{cluster_id}"
        )
        cluster_ip = cluster.endpoint
        cluster_cert = str(cluster.master_auth.cluster_ca_certificate)
    elif not (cluster_ip and cluster_cert):
        raise Exception(
            "Cluster IP and cluster certificate required when cluster configuration "
            "is provided explicitly."
        )

    creds, projects = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])

    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)  # refresh token

    configuration = Configuration()
    configuration.host = f"https://{cluster_ip}"
    with NamedTemporaryFile(delete=False) as ca_cert:
        ca_cert.write(base64.b64decode(cluster_cert))
    configuration.ssl_ca_cert = ca_cert.name
    configuration.api_key_prefix["authorization"] = "Bearer"
    configuration.api_key["authorization"] = creds.token  # valid for one hour

    return configuration
