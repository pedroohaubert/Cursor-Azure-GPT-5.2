"""Tests for adapter factory."""
import pytest
from app.adapters.factory import AdapterFactory
from app.adapters.base import BaseAdapter
from app.registry.model_config import ModelConfig
from app.exceptions import ServiceConfigurationError


def test_factory_creates_azure_adapter():
    """Test factory creates AzureAdapter for azure backend."""
    config = ModelConfig(
        name="test-azure",
        backend="azure",
        api_model="gpt-5",
        reasoning_effort="high"
    )

    adapter = AdapterFactory.create_adapter(config)
    assert adapter is not None
    assert isinstance(adapter, BaseAdapter)


def test_factory_rejects_unknown_backend():
    """Test ModelConfig validation rejects unknown backend."""
    # ModelConfig should reject unknown backends in __post_init__
    with pytest.raises(ValueError, match="Unsupported backend"):
        config = ModelConfig(
            name="test-unknown",
            backend="unknown",
            api_model="some-model"
        )
