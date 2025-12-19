"""Factory for creating backend-specific adapters."""
from typing import TYPE_CHECKING

from .base import BaseAdapter
from ..registry.model_config import ModelConfig
from ..exceptions import ServiceConfigurationError

if TYPE_CHECKING:
    from ..azure.adapter import AzureAdapter


class AdapterFactory:
    """Factory for instantiating the correct adapter based on backend type."""

    @staticmethod
    def create_adapter(model_config: ModelConfig) -> BaseAdapter:
        """Create appropriate adapter instance for the model configuration.

        Args:
            model_config: Configuration specifying backend and model details

        Returns:
            Instance of BaseAdapter subclass (AzureAdapter, AnthropicAdapter, etc.)

        Raises:
            ServiceConfigurationError: If backend is not supported
        """
        backend = model_config.backend

        if backend == "azure":
            # Import here to avoid circular dependency
            from ..azure.adapter import AzureAdapter
            return AzureAdapter(model_config)
        elif backend == "anthropic":
            # Import here to avoid circular dependency
            from ..anthropic.adapter import AnthropicAdapter
            return AnthropicAdapter(model_config)
        else:
            raise ServiceConfigurationError(
                f"Unsupported backend: {backend}. "
                f"Supported backends: azure, anthropic"
            )
