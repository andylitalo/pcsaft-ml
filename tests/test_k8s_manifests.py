"""Tests for Step 35 GKE migration k8s manifests."""

from pathlib import Path

import yaml

K8S_DIR = Path(__file__).parent.parent / "k8s"


def test_portal_deployment_yaml():
    """Verify portal deployment manifest is valid YAML with required fields."""
    path = K8S_DIR / "portal-deployment.yaml"
    assert path.exists(), "k8s/portal-deployment.yaml not found"
    content = yaml.safe_load(path.read_text())

    assert content["kind"] == "Deployment"
    assert content["metadata"]["name"] == "pcsaft-portal"
    assert content["metadata"]["namespace"] == "pcsaft"

    containers = content["spec"]["template"]["spec"]["containers"]
    assert len(containers) > 0
    container = containers[0]
    assert container["ports"][0]["containerPort"] == 8501
    assert "livenessProbe" in container
    assert "readinessProbe" in container
    assert "resources" in container


def test_portal_service_yaml():
    """Verify portal service targets correct port."""
    path = K8S_DIR / "portal-service.yaml"
    assert path.exists(), "k8s/portal-service.yaml not found"
    content = yaml.safe_load(path.read_text())

    assert content["kind"] == "Service"
    assert content["metadata"]["name"] == "pcsaft-portal"
    assert content["spec"]["ports"][0]["port"] == 80
    assert content["spec"]["ports"][0]["targetPort"] == 8501
    assert content["spec"]["type"] == "ClusterIP"


def test_ingress_yaml():
    """Verify ingress exposes portal at /."""
    path = K8S_DIR / "ingress.yaml"
    assert path.exists(), "k8s/ingress.yaml not found"
    content = yaml.safe_load(path.read_text())

    assert content["kind"] == "Ingress"
    rules = content["spec"]["rules"]
    assert len(rules) > 0
    paths = rules[0]["http"]["paths"]
    assert any(p["path"] == "/" for p in paths)
    portal_path = next(p for p in paths if p["path"] == "/")
    assert portal_path["backend"]["service"]["name"] == "pcsaft-portal"


def test_configmap_model_type():
    """Verify configmap sets PCSAFT_MODEL_TYPE."""
    path = K8S_DIR / "configmap.yaml"
    content = yaml.safe_load(path.read_text())

    assert "PCSAFT_MODEL_TYPE" in content["data"]
    assert content["data"]["PCSAFT_MODEL_TYPE"] == "gnn"


def test_hpa_yaml():
    """Verify HPA targets API deployment."""
    path = K8S_DIR / "api-hpa.yaml"
    assert path.exists(), "k8s/api-hpa.yaml not found"
    content = yaml.safe_load(path.read_text())

    assert content["kind"] == "HorizontalPodAutoscaler"
    ref = content["spec"]["scaleTargetRef"]
    assert ref["kind"] == "Deployment"
    assert ref["name"] == "pcsaft-api"
    assert content["spec"]["minReplicas"] == 1
    assert content["spec"]["maxReplicas"] == 5


def test_portal_env_uses_internal_api():
    """Verify PCSAFT_API_URL points at http://pcsaft-api:80."""
    path = K8S_DIR / "portal-deployment.yaml"
    content = yaml.safe_load(path.read_text())

    containers = content["spec"]["template"]["spec"]["containers"]
    container = containers[0]
    env_vars = {e["name"]: e["value"] for e in container["env"]}
    assert "PCSAFT_API_URL" in env_vars
    assert env_vars["PCSAFT_API_URL"] == "http://pcsaft-api:80"
