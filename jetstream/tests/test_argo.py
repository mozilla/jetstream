from textwrap import dedent
from unittest import mock

import yaml

from jetstream import argo
from jetstream.cli import ArgoExecutorStrategy

MINIMAL_WORKFLOW = dedent(
    """
    apiVersion: argoproj.io/v1alpha1
    kind: Workflow
    metadata:
        generateName: jetstream-
    spec:
        entrypoint: jetstream
        arguments:
            parameters:
            - name: date
    """
)


class TestArgo:
    def test_apply_parameters(self):
        manifest = yaml.safe_load(MINIMAL_WORKFLOW)
        updated_manifest = argo.apply_parameters(manifest, {"date": "2020-01-01"})
        assert updated_manifest["spec"]["arguments"]["parameters"][0]["name"] == "date"
        assert updated_manifest["spec"]["arguments"]["parameters"][0]["value"] == "2020-01-01"

    def test_apply_parameters_overwrite(self):
        manifest = yaml.safe_load(MINIMAL_WORKFLOW)
        manifest["spec"]["arguments"]["parameters"][0]["value"] = "2020-12-12"

        updated_manifest = argo.apply_parameters(manifest, {"date": "2020-01-01"})
        assert updated_manifest["spec"]["arguments"]["parameters"][0]["name"] == "date"
        assert updated_manifest["spec"]["arguments"]["parameters"][0]["value"] == "2020-01-01"

    def test_apply_parameters_add(self):
        manifest = yaml.safe_load(MINIMAL_WORKFLOW)
        manifest["spec"]["arguments"]["parameters"][0]["value"] = "2020-12-12"

        updated_manifest = argo.apply_parameters(manifest, {"date": "2020-01-01", "slug": "test"})
        assert updated_manifest["spec"]["arguments"]["parameters"][0]["name"] == "date"
        assert updated_manifest["spec"]["arguments"]["parameters"][0]["value"] == "2020-01-01"
        assert updated_manifest["spec"]["arguments"]["parameters"][1]["name"] == "slug"
        assert updated_manifest["spec"]["arguments"]["parameters"][1]["value"] == "test"

    def test_experiment_injection(self):
        with open(ArgoExecutorStrategy.RUN_WORKFLOW) as workflow_file:
            manifest = yaml.safe_load(workflow_file)
            updated_manifest = argo.apply_parameters(
                manifest,
                {
                    "experiments": [
                        {"date": "2020-01-01", "slug": "a", "image_hash": "abc"},
                        {"date": "2020-01-02", "slug": "b", "image_hash": "abc"},
                    ]
                },
            )

            assert updated_manifest["spec"]["arguments"]["parameters"][0]["name"] == "experiments"
            assert (
                updated_manifest["spec"]["arguments"]["parameters"][0]["value"]
                == '[{"date": "2020-01-01", "slug": "a", "image_hash": "abc"}, '
                + '{"date": "2020-01-02", "slug": "b", "image_hash": "abc"}]'
            )

    def test_submit_workflow_status_without_nodes(self, tmp_path):
        workflow_file = tmp_path / "workflow.yaml"
        workflow_file.write_text(MINIMAL_WORKFLOW)

        created_workflow = {
            "metadata": {"namespace": "argo", "name": "jetstream-abc"},
            "status": {"phase": "Succeeded"},
        }
        with mock.patch.object(argo.ArgoApi, "create_workflow", return_value=created_workflow):
            assert (
                argo.submit_workflow(
                    "project", "zone", "cluster", workflow_file, {"date": "2020-01-01"}
                )
                is True
            )

    def test_submit_workflow_nodes_failure_detected(self, tmp_path):
        workflow_file = tmp_path / "workflow.yaml"
        workflow_file.write_text(MINIMAL_WORKFLOW)

        created_workflow = {
            "metadata": {"namespace": "argo", "name": "jetstream-abc"},
            "status": {
                "nodes": {
                    "node1": {"name": "step-one(0)", "type": "Pod", "phase": "Failed"},
                }
            },
        }
        with mock.patch.object(argo.ArgoApi, "create_workflow", return_value=created_workflow):
            assert (
                argo.submit_workflow(
                    "project", "zone", "cluster", workflow_file, {"date": "2020-01-01"}
                )
                is False
            )
