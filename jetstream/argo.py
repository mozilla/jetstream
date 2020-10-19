from argo.workflows.client import ApiClient, Configuration, V1alpha1Api
import base64
from google.cloud.container_v1 import ClusterManagerClient
import google.auth
from pathlib import Path
from tempfile import NamedTemporaryFile
import yaml


def submit_workflow(
    project_id: str, zone: str, cluster_id: str, workflow_file: Path, monitor_status: bool = False
):
    """Submit a workflow to Argo."""

    config = get_config(project_id, zone, cluster_id)
    api_client = ApiClient(configuration=config)
    api = V1alpha1Api(api_client=api_client)
    manifest = yaml.safe_load(workflow_file.read_text())

    api.create_namespaced_workflow("argo", manifest)

    if monitor_status:
        pass
        # todo


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
