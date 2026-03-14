"""Tests for Docker and Kubernetes deployment artifacts."""

import re
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def k8s_dir(project_root):
    """Return the k8s directory."""
    return project_root / "k8s"


def test_dockerfile_exists(project_root):
    """Verify Dockerfile exists."""
    dockerfile = project_root / "Dockerfile"
    assert dockerfile.exists(), "Dockerfile not found"


def test_dockerfile_has_multistage(project_root):
    """Verify Dockerfile uses multi-stage build."""
    dockerfile = project_root / "Dockerfile"
    content = dockerfile.read_text()
    from_count = len(re.findall(r"^FROM\s+", content, re.MULTILINE))
    assert from_count >= 2, f"Expected multi-stage build (>=2 FROM), found {from_count}"


def test_dockerfile_exposes_port_8000(project_root):
    """Verify Dockerfile exposes port 8000."""
    dockerfile = project_root / "Dockerfile"
    content = dockerfile.read_text()
    assert re.search(r"^EXPOSE\s+8000", content, re.MULTILINE), "EXPOSE 8000 not found"


def test_dockerfile_has_healthcheck(project_root):
    """Verify Dockerfile includes HEALTHCHECK directive."""
    dockerfile = project_root / "Dockerfile"
    content = dockerfile.read_text()
    assert "HEALTHCHECK" in content, "HEALTHCHECK directive not found"


def test_dockerignore_exists(project_root):
    """Verify .dockerignore exists."""
    dockerignore = project_root / ".dockerignore"
    assert dockerignore.exists(), ".dockerignore not found"


def test_dockerignore_excludes_common_dirs(project_root):
    """Verify .dockerignore excludes tests, docs, figures."""
    dockerignore = project_root / ".dockerignore"
    content = dockerignore.read_text()
    assert "tests/" in content, "tests/ not excluded"
    assert "docs/" in content, "docs/ not excluded"
    assert "figures/" in content, "figures/ not excluded"


def test_k8s_manifests_exist(k8s_dir):
    """Verify all required K8s manifests exist."""
    required = ["namespace.yaml", "configmap.yaml", "deployment.yaml", "service.yaml", "pvc.yaml"]
    for manifest in required:
        assert (k8s_dir / manifest).exists(), f"Missing k8s/{manifest}"


def test_k8s_deployment_has_probes(k8s_dir):
    """Verify deployment has liveness and readiness probes."""
    deployment = k8s_dir / "deployment.yaml"
    content = yaml.safe_load(deployment.read_text())

    containers = content["spec"]["template"]["spec"]["containers"]
    assert len(containers) > 0, "No containers in deployment"

    container = containers[0]
    assert "livenessProbe" in container, "No livenessProbe in deployment"
    assert "readinessProbe" in container, "No readinessProbe in deployment"

    # Verify probes use /health endpoint
    assert container["livenessProbe"]["httpGet"]["path"] == "/health"
    assert container["readinessProbe"]["httpGet"]["path"] == "/health"


def test_k8s_deployment_has_resource_limits(k8s_dir):
    """Verify deployment has resource requests and limits."""
    deployment = k8s_dir / "deployment.yaml"
    content = yaml.safe_load(deployment.read_text())

    containers = content["spec"]["template"]["spec"]["containers"]
    container = containers[0]

    assert "resources" in container, "No resources section in deployment"
    assert "requests" in container["resources"], "No resource requests"
    assert "limits" in container["resources"], "No resource limits"

    # Verify specific resources are defined
    requests = container["resources"]["requests"]
    limits = container["resources"]["limits"]
    assert "cpu" in requests and "memory" in requests
    assert "cpu" in limits and "memory" in limits


def test_k8s_configmap_has_model_type(k8s_dir):
    """Verify ConfigMap has PCSAFT_MODEL_TYPE set to rf."""
    configmap = k8s_dir / "configmap.yaml"
    content = yaml.safe_load(configmap.read_text())

    assert "data" in content, "No data section in ConfigMap"
    assert "PCSAFT_MODEL_TYPE" in content["data"], "PCSAFT_MODEL_TYPE not in ConfigMap"
    assert content["data"]["PCSAFT_MODEL_TYPE"] == "rf", "MODEL_TYPE should be 'rf'"


def test_k8s_deployment_has_volume_mount(k8s_dir):
    """Verify deployment mounts PVC for submissions."""
    deployment = k8s_dir / "deployment.yaml"
    content = yaml.safe_load(deployment.read_text())

    containers = content["spec"]["template"]["spec"]["containers"]
    container = containers[0]

    assert "volumeMounts" in container, "No volumeMounts in deployment"
    volume_mounts = container["volumeMounts"]
    assert any(vm["name"] == "submissions" for vm in volume_mounts), "No submissions mount"

    # Verify volumes reference PVC
    volumes = content["spec"]["template"]["spec"]["volumes"]
    assert any(v["name"] == "submissions" for v in volumes), "No submissions volume"


def test_k8s_service_type_clusterip(k8s_dir):
    """Verify service is ClusterIP type."""
    service = k8s_dir / "service.yaml"
    content = yaml.safe_load(service.read_text())

    assert content["spec"]["type"] == "ClusterIP", "Service should be ClusterIP"
    assert content["spec"]["ports"][0]["port"] == 80, "Service port should be 80"
    assert content["spec"]["ports"][0]["targetPort"] == 8000, "Target port should be 8000"


def test_k8s_namespace_is_pcsaft(k8s_dir):
    """Verify namespace is 'pcsaft'."""
    namespace = k8s_dir / "namespace.yaml"
    content = yaml.safe_load(namespace.read_text())

    assert content["metadata"]["name"] == "pcsaft", "Namespace should be 'pcsaft'"


def test_k8s_pvc_requests_storage(k8s_dir):
    """Verify PVC requests 1Gi storage."""
    pvc = k8s_dir / "pvc.yaml"
    content = yaml.safe_load(pvc.read_text())

    assert "spec" in content, "No spec in PVC"
    assert "resources" in content["spec"], "No resources in PVC spec"
    assert content["spec"]["resources"]["requests"]["storage"] == "1Gi"
