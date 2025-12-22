"""Functional tests using WebTest.

See: http://webtest.readthedocs.org/
"""
from app.registry.model_config import ModelConfig


def test_model_config_accepts_kimi_backend():
    """Test that ModelConfig accepts 'kimi' as a valid backend."""
    config = ModelConfig(
        name="kimi-test",
        backend="kimi",
        api_model="Kimi-K2-Thinking",
        base_url="https://example.openai.azure.com/openai/v1",
    )
    assert config.backend == "kimi"
    assert config.api_model == "Kimi-K2-Thinking"


class TestModels:
    """Models."""

    def test_models_endpoint_returns_400(self, testapp):
        """Ensure /models endpoint returns HTTP 400 wihtout auth."""
        testapp.get("/models", status=400)

    def test_models_endpoint_returns_200(self, testapp):
        """Ensure /models endpoint returns HTTP 400 with auth."""
        response = testapp.get(
            "/models",
            status=200,
            headers={"Authorization": "Bearer test-service-api-key"},
        )
        content = response.body.decode("utf-8")
        assert '"gpt-high"' in content
        assert '"gpt-medium"' in content
        assert '"gpt-low"' in content
        assert '"gpt-minimal"' in content

    def test_models_endpoint_returns_configured_models(self, testapp):
        """Test /models endpoint returns models from registry."""
        response = testapp.get(
            "/models",
            status=200,
            headers={"Authorization": "Bearer test-service-api-key"},
        )
        data = response.json
        assert data["object"] == "list"

        model_ids = [m["id"] for m in data["data"]]
        assert "gpt-high" in model_ids
        assert "gpt-medium" in model_ids
        assert "claude-sonnet-4-5" in model_ids

    def test_health_endpoint_returns_200(self, testapp):
        """Ensure /health endpoint returns HTTP 200 without auth."""
        testapp.get("/health", status=200)
