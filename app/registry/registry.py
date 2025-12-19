"""Model registry for loading and validating model configurations."""
from pathlib import Path
from typing import Dict
import yaml

from .model_config import ModelConfig
from ..exceptions import ModelNotFoundError, ServiceConfigurationError


class ModelRegistry:
    """Registry for managing model configurations from YAML file."""

    def __init__(self, config_path: str):
        """Initialize registry by loading YAML configuration.

        Args:
            config_path: Path to models.yaml configuration file

        Raises:
            ServiceConfigurationError: If config file is invalid
        """
        self._models: Dict[str, ModelConfig] = {}
        self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        """Load and parse YAML configuration file."""
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            raise ServiceConfigurationError(
                f"Model configuration file not found: {config_path}"
            )
        except yaml.YAMLError as e:
            raise ServiceConfigurationError(
                f"Invalid YAML in model configuration: {e}"
            )

        if not isinstance(data, dict) or "models" not in data:
            raise ServiceConfigurationError(
                'Configuration must have top-level "models" key'
            )

        models_dict = data["models"]
        if not isinstance(models_dict, dict):
            raise ServiceConfigurationError(
                '"models" must be a dictionary'
            )

        for model_name, model_data in models_dict.items():
            try:
                config = ModelConfig(
                    name=model_name,
                    backend=model_data["backend"],
                    api_model=model_data["api_model"],
                    reasoning_effort=model_data.get("reasoning_effort"),
                    deployment_name=model_data.get("deployment_name"),
                    summary_level=model_data.get("summary_level"),
                    verbosity_level=model_data.get("verbosity_level"),
                    truncation_strategy=model_data.get("truncation_strategy"),
                    max_tokens=model_data.get("max_tokens"),
                    base_url=model_data.get("base_url"),
                    extra=model_data.get("extra"),
                )
                self._models[model_name] = config
            except (KeyError, ValueError) as e:
                raise ServiceConfigurationError(
                    f"Invalid configuration for model '{model_name}': {e}"
                )

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a specific model.

        Args:
            model_name: Name of the model (e.g., "claude-sonnet-4-5")

        Returns:
            ModelConfig for the requested model

        Raises:
            ModelNotFoundError: If model is not in whitelist
        """
        if model_name not in self._models:
            available = ", ".join(self._models.keys())
            raise ModelNotFoundError(
                f"Model '{model_name}' is not configured. "
                f"Available models: {available}"
            )
        return self._models[model_name]

    def list_models(self) -> list[str]:
        """Return list of all configured model names."""
        return list(self._models.keys())
