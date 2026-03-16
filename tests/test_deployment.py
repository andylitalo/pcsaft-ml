"""Tests for deployment artifacts (Dockerfiles, docker-compose)."""

import pathlib

import yaml

PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def test_portal_dockerfile_exists():
    """Verify Dockerfile.portal exists in project root."""
    dockerfile = PROJECT_ROOT / "Dockerfile.portal"
    assert dockerfile.exists(), "Dockerfile.portal not found in project root"


def test_api_dockerfile_exists():
    """Verify Dockerfile exists in project root."""
    dockerfile = PROJECT_ROOT / "Dockerfile"
    assert dockerfile.exists(), "Dockerfile not found in project root"


def test_docker_compose_exists():
    """Verify docker-compose.yaml exists and is valid YAML."""
    compose_file = PROJECT_ROOT / "docker-compose.yaml"
    assert compose_file.exists(), "docker-compose.yaml not found in project root"

    # Verify it's valid YAML
    with open(compose_file) as f:
        config = yaml.safe_load(f)

    assert "services" in config, "docker-compose.yaml missing 'services' key"
    assert "api" in config["services"], "docker-compose.yaml missing 'api' service"
    assert "portal" in config["services"], "docker-compose.yaml missing 'portal' service"


def test_portal_dockerfile_exposes_8501():
    """Verify Dockerfile.portal exposes port 8501."""
    dockerfile = PROJECT_ROOT / "Dockerfile.portal"
    content = dockerfile.read_text()

    assert "EXPOSE 8501" in content, "Dockerfile.portal does not EXPOSE 8501"


def test_api_dockerfile_exposes_8000():
    """Verify Dockerfile exposes port 8000."""
    dockerfile = PROJECT_ROOT / "Dockerfile"
    content = dockerfile.read_text()

    assert "EXPOSE 8000" in content, "Dockerfile does not EXPOSE 8000"


def test_portal_dockerfile_healthcheck():
    """Verify Dockerfile.portal includes a healthcheck."""
    dockerfile = PROJECT_ROOT / "Dockerfile.portal"
    content = dockerfile.read_text()

    assert "HEALTHCHECK" in content, "Dockerfile.portal missing HEALTHCHECK"
    assert (
        "_stcore/health" in content
    ), "Dockerfile.portal healthcheck doesn't use Streamlit health endpoint"


def test_api_dockerfile_healthcheck():
    """Verify Dockerfile includes a healthcheck."""
    dockerfile = PROJECT_ROOT / "Dockerfile"
    content = dockerfile.read_text()

    assert "HEALTHCHECK" in content, "Dockerfile missing HEALTHCHECK"


def test_docker_compose_api_health_depends():
    """Verify portal depends_on api with health condition."""
    compose_file = PROJECT_ROOT / "docker-compose.yaml"

    with open(compose_file) as f:
        config = yaml.safe_load(f)

    portal = config["services"]["portal"]
    assert "depends_on" in portal, "portal service missing depends_on"

    # Can be dict (with condition) or list (without condition)
    depends_on = portal["depends_on"]
    if isinstance(depends_on, dict):
        assert "api" in depends_on, "portal doesn't depend on api"
        assert depends_on["api"]["condition"] == "service_healthy", \
            "portal doesn't wait for api to be healthy"
    else:
        # If it's just a list, that's also valid but less optimal
        assert "api" in depends_on, "portal doesn't depend on api"


def test_docker_compose_environment_vars():
    """Verify expected environment variables are set in docker-compose."""
    compose_file = PROJECT_ROOT / "docker-compose.yaml"

    with open(compose_file) as f:
        config = yaml.safe_load(f)

    # API should have PCSAFT_MODEL_TYPE
    api_env = config["services"]["api"].get("environment", {})
    if isinstance(api_env, dict):
        assert "PCSAFT_MODEL_TYPE" in api_env, "api service missing PCSAFT_MODEL_TYPE"
    elif isinstance(api_env, list):
        # Environment can be list of strings like "KEY=value"
        env_keys = [e.split("=")[0] for e in api_env]
        assert "PCSAFT_MODEL_TYPE" in env_keys, "api service missing PCSAFT_MODEL_TYPE"

    # Portal should have PCSAFT_API_URL
    portal_env = config["services"]["portal"].get("environment", {})
    if isinstance(portal_env, dict):
        assert "PCSAFT_API_URL" in portal_env, "portal service missing PCSAFT_API_URL"
    elif isinstance(portal_env, list):
        env_keys = [e.split("=")[0] for e in portal_env]
        assert "PCSAFT_API_URL" in env_keys, "portal service missing PCSAFT_API_URL"
