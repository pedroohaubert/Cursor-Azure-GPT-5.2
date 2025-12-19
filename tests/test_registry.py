"""Tests for model registry."""
import pytest
from app.registry.registry import ModelRegistry
from app.exceptions import ModelNotFoundError


def test_registry_loads_valid_config(tmp_path):
    """Test that registry loads a valid YAML configuration."""
    config_file = tmp_path / "models.yaml"
    config_file.write_text("""
models:
  test-model:
    backend: azure
    api_model: gpt-5
""")

    registry = ModelRegistry(str(config_file))
    assert registry.get_model_config("test-model") is not None


def test_registry_rejects_unknown_model(tmp_path):
    """Test that registry raises error for unlisted model."""
    config_file = tmp_path / "models.yaml"
    config_file.write_text("""
models:
  test-model:
    backend: azure
    api_model: gpt-5
""")

    registry = ModelRegistry(str(config_file))
    with pytest.raises(ModelNotFoundError):
        registry.get_model_config("unknown-model")


def test_registry_returns_backend_type(tmp_path):
    """Test that registry returns correct backend type."""
    config_file = tmp_path / "models.yaml"
    config_file.write_text("""
models:
  claude-test:
    backend: anthropic
    api_model: claude-sonnet-4.5-20250514
""")

    registry = ModelRegistry(str(config_file))
    config = registry.get_model_config("claude-test")
    assert config.backend == "anthropic"
    assert config.api_model == "claude-sonnet-4.5-20250514"
