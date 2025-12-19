# Multi-Backend Model Router Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform the proxy from a single Azure backend into a multi-backend router supporting Azure and Anthropic APIs with model-based routing.

**Architecture:** Registry-based routing with YAML configuration defining model-to-backend mappings. Factory pattern creates backend-specific adapters implementing a common interface. Only whitelisted models can be called.

**Tech Stack:** Flask, requests, PyYAML (new), anthropic SDK (new), existing SSE/logging infrastructure

---

## Task 1: Create Model Registry Configuration Schema

**Files:**
- Create: `app/models.yaml`
- Create: `app/registry/__init__.py`
- Create: `app/registry/model_config.py`
- Create: `app/registry/registry.py`

**Step 1: Write the failing test for model registry**

Create: `tests/test_registry.py`

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_registry.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'app.registry'"

**Step 3: Install PyYAML dependency**

Run: `pip install pyyaml && echo "pyyaml>=6.0.1" >> requirements/prod.txt`

**Step 4: Create ModelConfig dataclass**

Create: `app/registry/model_config.py`

```python
"""Model configuration dataclass."""
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    name: str
    backend: str  # "azure" or "anthropic"
    api_model: str

    # Azure-specific
    reasoning_effort: Optional[str] = None
    deployment_name: Optional[str] = None
    summary_level: Optional[str] = None
    verbosity_level: Optional[str] = None
    truncation_strategy: Optional[str] = None

    # Anthropic-specific
    max_tokens: Optional[int] = None

    # Common
    extra: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.backend not in {"azure", "anthropic"}:
            raise ValueError(f"Unsupported backend: {self.backend}")
```

**Step 5: Create ModelRegistry class**

Create: `app/registry/registry.py`

```python
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
```

Create: `app/registry/__init__.py`

```python
"""Model registry package."""
from .registry import ModelRegistry
from .model_config import ModelConfig

__all__ = ["ModelRegistry", "ModelConfig"]
```

**Step 6: Add ModelNotFoundError exception**

Modify: `app/exceptions.py` (add after ClientClosedConnection)

```python
class ModelNotFoundError(ValueError):
    """Exception raised when requested model is not in configuration."""
    pass
```

**Step 7: Create initial models.yaml**

Create: `app/models.yaml`

```yaml
# Model Configuration
# Only models listed here can be called through the proxy

models:
  # Azure GPT-5 models with reasoning efforts
  gpt-high:
    backend: azure
    api_model: gpt-5
    reasoning_effort: high
    summary_level: detailed
    verbosity_level: medium

  gpt-medium:
    backend: azure
    api_model: gpt-5
    reasoning_effort: medium
    summary_level: detailed
    verbosity_level: medium

  gpt-low:
    backend: azure
    api_model: gpt-5
    reasoning_effort: low
    summary_level: detailed
    verbosity_level: medium

  gpt-minimal:
    backend: azure
    api_model: gpt-5
    reasoning_effort: minimal
    summary_level: detailed
    verbosity_level: medium

  # Anthropic Claude models (to be implemented)
  claude-sonnet-4-5:
    backend: anthropic
    api_model: claude-sonnet-4.5-20250514
    max_tokens: 8192

  claude-opus-4-5:
    backend: anthropic
    api_model: claude-opus-4.5-20251101
    max_tokens: 16384
```

**Step 8: Run tests to verify they pass**

Run: `pytest tests/test_registry.py -v`

Expected: PASS (3 tests)

**Step 9: Commit**

```bash
git add app/registry/ app/models.yaml app/exceptions.py tests/test_registry.py requirements/prod.txt
git commit -m "feat: add model registry with YAML configuration

- Add ModelRegistry class to load and validate model configs
- Add ModelConfig dataclass for type-safe config access
- Add models.yaml with initial Azure and Anthropic models
- Add ModelNotFoundError for whitelist enforcement
- Add PyYAML dependency"
```

---

## Task 2: Create Base Adapter Interface

**Files:**
- Create: `app/adapters/__init__.py`
- Create: `app/adapters/base.py`
- Create: `app/adapters/factory.py`

**Step 1: Write failing test for adapter interface**

Create: `tests/test_adapter_factory.py`

```python
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
    """Test factory raises error for unknown backend."""
    config = ModelConfig(
        name="test-unknown",
        backend="unknown",
        api_model="some-model"
    )

    with pytest.raises(ServiceConfigurationError):
        AdapterFactory.create_adapter(config)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_adapter_factory.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'app.adapters'"

**Step 3: Create BaseAdapter abstract class**

Create: `app/adapters/base.py`

```python
"""Base adapter interface for all backend adapters."""
from abc import ABC, abstractmethod
from typing import Any, Dict
from flask import Request, Response

from ..registry.model_config import ModelConfig


class BaseAdapter(ABC):
    """Abstract base class for backend adapters.

    All backend adapters (Azure, Anthropic, etc.) must implement this interface.
    """

    def __init__(self, model_config: ModelConfig):
        """Initialize adapter with model configuration.

        Args:
            model_config: Configuration for the model this adapter handles
        """
        self.model_config = model_config
        self.inbound_model: str = model_config.name

    @abstractmethod
    def forward(self, req: Request) -> Response:
        """Forward the Flask request to backend and return adapted response.

        This is the main entry point. Implementations should:
        1. Adapt the OpenAI request format to backend format
        2. Call the backend API
        3. Adapt the backend response to OpenAI format
        4. Return Flask Response with SSE stream

        Args:
            req: Flask request object with OpenAI Chat Completions format

        Returns:
            Flask Response with OpenAI-compatible SSE stream
        """
        pass

    @abstractmethod
    def adapt_request(self, req: Request) -> Dict[str, Any]:
        """Adapt OpenAI request to backend-specific format.

        Args:
            req: Flask request with OpenAI format

        Returns:
            Dict suitable for requests.request(**kwargs)
        """
        pass

    @abstractmethod
    def adapt_response(self, backend_response: Any) -> Response:
        """Adapt backend response to OpenAI format.

        Args:
            backend_response: Response from backend API (requests.Response)

        Returns:
            Flask Response with OpenAI Chat Completions chunks
        """
        pass
```

**Step 4: Create AdapterFactory**

Create: `app/adapters/factory.py`

```python
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
```

Create: `app/adapters/__init__.py`

```python
"""Backend adapters package."""
from .base import BaseAdapter
from .factory import AdapterFactory

__all__ = ["BaseAdapter", "AdapterFactory"]
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_adapter_factory.py::test_factory_rejects_unknown_backend -v`

Expected: PASS (1 test - the azure test will fail until we refactor AzureAdapter)

**Step 6: Commit**

```bash
git add app/adapters/ tests/test_adapter_factory.py
git commit -m "feat: add base adapter interface and factory

- Add BaseAdapter abstract class defining adapter interface
- Add AdapterFactory for creating backend-specific adapters
- Use strategy pattern for backend selection"
```

---

## Task 3: Refactor AzureAdapter to Implement BaseAdapter

**Files:**
- Modify: `app/azure/adapter.py`
- Modify: `app/azure/request_adapter.py`
- Modify: `app/azure/response_adapter.py`

**Step 1: Update AzureAdapter to inherit from BaseAdapter**

Modify: `app/azure/adapter.py`

OLD (lines 1-36):
```python
"""Azure adapter orchestrating request/response transformations."""

from __future__ import annotations

import json
import re
from typing import Optional

import requests
from flask import Request, Response

from ..common.logging import console
from ..common.recording import record_payload

# Local adapters
from .request_adapter import RequestAdapter
from .response_adapter import ResponseAdapter


class AzureAdapter:
    """Orchestrate forwarding of a Flask Request to Azure's Responses API.

    Provides a Completions-compatible interface to the caller by composing a
    RequestAdapter (pre-request transformations) and a ResponseAdapter
    (post-request transformations). The adapters receive a reference to this
    instance for shared per-request state (models).
    """

    # Per-request state (streaming completions only)
    inbound_model: Optional[str] = None

    def __init__(self) -> None:
        """Initialize child adapters and shared state references."""
        # Composition: child adapters get a reference to this orchestrator
        self.request_adapter = RequestAdapter(self)
        self.response_adapter = ResponseAdapter(self)
```

NEW:
```python
"""Azure adapter orchestrating request/response transformations."""

from __future__ import annotations

import json
import re
from typing import Optional

import requests
from flask import Request, Response

from ..adapters.base import BaseAdapter
from ..common.logging import console
from ..common.recording import record_payload
from ..registry.model_config import ModelConfig

# Local adapters
from .request_adapter import RequestAdapter
from .response_adapter import ResponseAdapter


class AzureAdapter(BaseAdapter):
    """Orchestrate forwarding of a Flask Request to Azure's Responses API.

    Provides a Completions-compatible interface to the caller by composing a
    RequestAdapter (pre-request transformations) and a ResponseAdapter
    (post-request transformations). The adapters receive a reference to this
    instance for shared per-request state (models).
    """

    def __init__(self, model_config: ModelConfig) -> None:
        """Initialize child adapters with model configuration.

        Args:
            model_config: Configuration for this Azure model
        """
        super().__init__(model_config)
        # Composition: child adapters get a reference to this orchestrator
        self.request_adapter = RequestAdapter(self)
        self.response_adapter = ResponseAdapter(self)
```

**Step 2: Update AzureAdapter.forward to use adapt_request/adapt_response**

Modify: `app/azure/adapter.py`

OLD (lines 39-57):
```python
    def forward(self, req: Request) -> Response:
        """Forward the Flask request upstream and adapt the response back.

        High-level flow:
        1) RequestAdapter builds the upstream request kwargs and stores state
           on this adapter (models).
        2) Perform the upstream HTTP call using a short-lived requests call.
        3) ResponseAdapter converts the upstream response into a Flask Response.
        """
        request_kwargs = self.request_adapter.adapt(req)

        record_payload(request_kwargs.get("json", {}), "upstream_request")

        # Perform upstream request with kwargs directly (no long-lived session)
        resp = requests.request(**request_kwargs)
        if resp.status_code != 200:
            return self._handle_azure_error(resp, request_kwargs)

        return self.response_adapter.adapt(resp)
```

NEW:
```python
    def forward(self, req: Request) -> Response:
        """Forward the Flask request upstream and adapt the response back.

        High-level flow:
        1) RequestAdapter builds the upstream request kwargs and stores state
           on this adapter (models).
        2) Perform the upstream HTTP call using a short-lived requests call.
        3) ResponseAdapter converts the upstream response into a Flask Response.
        """
        request_kwargs = self.adapt_request(req)

        record_payload(request_kwargs.get("json", {}), "upstream_request")

        # Perform upstream request with kwargs directly (no long-lived session)
        resp = requests.request(**request_kwargs)
        if resp.status_code != 200:
            return self._handle_azure_error(resp, request_kwargs)

        return self.adapt_response(resp)

    def adapt_request(self, req: Request) -> dict:
        """Adapt OpenAI request to Azure Responses API format.

        Args:
            req: Flask request with OpenAI Chat Completions format

        Returns:
            Dict suitable for requests.request(**kwargs)
        """
        return self.request_adapter.adapt(req)

    def adapt_response(self, backend_response) -> Response:
        """Adapt Azure streaming response to OpenAI format.

        Args:
            backend_response: requests.Response from Azure Responses API

        Returns:
            Flask Response with OpenAI Chat Completions chunks
        """
        return self.response_adapter.adapt(backend_response)
```

**Step 3: Update RequestAdapter to use model_config**

Modify: `app/azure/request_adapter.py`

OLD (lines 185-234):
```python
    def adapt(self, req: Request) -> Dict[str, Any]:
        """Build requests.request kwargs for the Azure Responses API call.

        Maps inputs to the Responses schema and returns a dict suitable for
        requests.request(**kwargs).
        """
        # Reset per-request state
        self.adapter.inbound_model = None

        # Parse request body
        payload = req.get_json(silent=True, force=False)

        # Determine target model: prefer env AZURE_MODEL/AZURE_DEPLOYMENT
        inbound_model = payload.get("model") if isinstance(payload, dict) else None
        self.adapter.inbound_model = inbound_model

        settings = current_app.config

        upstream_headers = self._copy_request_headers_for_azure(
            req, api_key=settings["AZURE_API_KEY"]
        )

        # Map Chat/Completions to Responses (always streaming)
        messages = payload.get("messages") or []

        responses_body = (
            self._messages_to_responses_input_and_instructions(messages)
            if isinstance(messages, list)
            else {"input": None, "instructions": None}
        )

        responses_body["model"] = settings["AZURE_DEPLOYMENT"]

        # Transform tools and tool choice
        responses_body["tools"] = self._transform_tools_for_responses(
            payload.get("tools", [])
        )
        responses_body["tool_choice"] = payload.get("tool_choice")

        responses_body["prompt_cache_key"] = payload.get("user")

        # Always streaming
        responses_body["stream"] = True

        reasoning_effort = inbound_model.replace("gpt-", "").lower()
        if reasoning_effort not in {"high", "medium", "low", "minimal"}:
            raise CursorConfigurationError(
                "Model name must be either gpt-high, gpt-medium, gpt-low, or gpt-minimal."
                f"\n\nGot: {inbound_model}"
            )
```

NEW:
```python
    def adapt(self, req: Request) -> Dict[str, Any]:
        """Build requests.request kwargs for the Azure Responses API call.

        Maps inputs to the Responses schema and returns a dict suitable for
        requests.request(**kwargs).
        """
        # Parse request body
        payload = req.get_json(silent=True, force=False)

        settings = current_app.config

        upstream_headers = self._copy_request_headers_for_azure(
            req, api_key=settings["AZURE_API_KEY"]
        )

        # Map Chat/Completions to Responses (always streaming)
        messages = payload.get("messages") or []

        responses_body = (
            self._messages_to_responses_input_and_instructions(messages)
            if isinstance(messages, list)
            else {"input": None, "instructions": None}
        )

        # Use deployment name from config or fall back to env var
        responses_body["model"] = (
            self.adapter.model_config.deployment_name or settings["AZURE_DEPLOYMENT"]
        )

        # Transform tools and tool choice
        responses_body["tools"] = self._transform_tools_for_responses(
            payload.get("tools", [])
        )
        responses_body["tool_choice"] = payload.get("tool_choice")

        responses_body["prompt_cache_key"] = payload.get("user")

        # Always streaming
        responses_body["stream"] = True

        # Get reasoning effort from model config
        reasoning_effort = self.adapter.model_config.reasoning_effort
        if not reasoning_effort:
            raise CursorConfigurationError(
                f"Model '{self.adapter.model_config.name}' is missing reasoning_effort configuration"
            )
```

OLD (lines 236-253):
```python
        responses_body["reasoning"] = {
            "effort": reasoning_effort,
        }

        # Concise is not supported by GPT-5,
        # but allowing it for now to be able to test it on other models
        if settings["AZURE_SUMMARY_LEVEL"] in {"auto", "detailed", "concise"}:
            responses_body["reasoning"]["summary"] = settings["AZURE_SUMMARY_LEVEL"]
        else:
            raise ServiceConfigurationError(
                "AZURE_SUMMARY_LEVEL must be either auto, detailed, or concise."
                f"\n\nGot: {settings['AZURE_SUMMARY_LEVEL']}"
            )

        # No need to pass verbosity if it's set to medium, as it's the model's default
        if settings["AZURE_VERBOSITY_LEVEL"] in {"low", "high"}:
            responses_body["text"] = {"verbosity": settings["AZURE_VERBOSITY_LEVEL"]}

        responses_body["store"] = False
```

NEW:
```python
        responses_body["reasoning"] = {
            "effort": reasoning_effort,
        }

        # Use summary level from model config or fall back to env var
        summary_level = (
            self.adapter.model_config.summary_level or settings["AZURE_SUMMARY_LEVEL"]
        )
        if summary_level in {"auto", "detailed", "concise"}:
            responses_body["reasoning"]["summary"] = summary_level
        else:
            raise ServiceConfigurationError(
                "summary_level must be either auto, detailed, or concise."
                f"\n\nGot: {summary_level}"
            )

        # Use verbosity level from model config or fall back to env var
        verbosity_level = (
            self.adapter.model_config.verbosity_level or settings["AZURE_VERBOSITY_LEVEL"]
        )
        if verbosity_level in {"low", "high"}:
            responses_body["text"] = {"verbosity": verbosity_level}

        responses_body["store"] = False
```

OLD (lines 255-258):
```python
        responses_body["stream_options"] = {"include_obfuscation": False}

        if settings["AZURE_TRUNCATION"] == "auto":
            responses_body["truncation"] = settings["AZURE_TRUNCATION"]
```

NEW:
```python
        responses_body["stream_options"] = {"include_obfuscation": False}

        # Use truncation strategy from model config or fall back to env var
        truncation = (
            self.adapter.model_config.truncation_strategy or settings["AZURE_TRUNCATION"]
        )
        if truncation == "auto":
            responses_body["truncation"] = truncation
```

**Step 4: Run existing Azure tests to verify they still pass**

Run: `pytest tests/test_replays.py tests/test_azure_errors.py -v`

Expected: Tests may fail due to missing model_config in test fixtures

**Step 5: Update test fixtures to include model_config**

Modify: `tests/conftest.py` (add helper function)

```python
from app.registry.model_config import ModelConfig


def create_test_azure_config(reasoning_effort="high"):
    """Create test Azure model configuration."""
    return ModelConfig(
        name=f"test-gpt-{reasoning_effort}",
        backend="azure",
        api_model="gpt-5",
        reasoning_effort=reasoning_effort,
        summary_level="detailed",
        verbosity_level="medium",
    )
```

**Step 6: Run tests again**

Run: `pytest tests/test_replays.py tests/test_azure_errors.py -v`

Expected: PASS

**Step 7: Run adapter factory test**

Run: `pytest tests/test_adapter_factory.py::test_factory_creates_azure_adapter -v`

Expected: PASS

**Step 8: Commit**

```bash
git add app/azure/ tests/conftest.py
git commit -m "refactor: make AzureAdapter implement BaseAdapter interface

- AzureAdapter now inherits from BaseAdapter
- Use model_config for reasoning_effort, summary_level, etc.
- Remove hardcoded model name parsing (gpt-high/medium/low)
- Update tests to use ModelConfig fixtures"
```

---

## Task 4: Update Blueprint to Use Registry and Factory

**Files:**
- Modify: `app/blueprint.py`
- Modify: `app/settings.py`

**Step 1: Write failing test for blueprint routing**

Modify: `tests/test_models.py`

Add test:
```python
def test_models_endpoint_returns_configured_models(client, tmp_path):
    """Test /models endpoint returns models from registry."""
    # This test will verify that /models returns the models from models.yaml
    response = client.get("/models")
    assert response.status_code == 200

    data = response.get_json()
    assert data["object"] == "list"

    model_ids = [m["id"] for m in data["data"]]
    assert "gpt-high" in model_ids
    assert "gpt-medium" in model_ids
    assert "claude-sonnet-4-5" in model_ids
```

**Step 2: Run test to verify current behavior**

Run: `pytest tests/test_models.py -v`

Expected: FAIL (hardcoded models don't include claude-sonnet-4-5)

**Step 3: Add MODEL_CONFIG_PATH to settings**

Modify: `app/settings.py`

Add after line 12:
```python
import os

# Model registry configuration
MODEL_CONFIG_PATH = env.str(
    "MODEL_CONFIG_PATH",
    default=os.path.join(os.path.dirname(__file__), "models.yaml")
)
```

**Step 4: Update blueprint to use registry and factory**

Modify: `app/blueprint.py`

OLD (lines 1-17):
```python
"""Flask blueprint and request routing for the proxy service.

This module defines the application blueprint, configures logging, and
forwards incoming HTTP requests to the configured backend implementation.
"""

from flask import Blueprint, current_app, jsonify, request

from .auth import require_auth
from .azure.adapter import AzureAdapter
from .common.logging import log_request
from .common.recording import (
    increment_last_recording,
    init_last_recording,
    record_payload,
)
from .exceptions import ConfigurationError
```

NEW:
```python
"""Flask blueprint and request routing for the proxy service.

This module defines the application blueprint, configures logging, and
forwards incoming HTTP requests to the configured backend implementation.
"""

from flask import Blueprint, current_app, jsonify, request

from .adapters.factory import AdapterFactory
from .auth import require_auth
from .common.logging import log_request
from .common.recording import (
    increment_last_recording,
    init_last_recording,
    record_payload,
)
from .exceptions import ConfigurationError, ModelNotFoundError
from .registry.registry import ModelRegistry

# Global registry (initialized in app factory)
_registry: ModelRegistry = None


def init_registry(config_path: str) -> None:
    """Initialize the global model registry.

    Args:
        config_path: Path to models.yaml configuration file
    """
    global _registry
    _registry = ModelRegistry(config_path)


def get_registry() -> ModelRegistry:
    """Get the initialized model registry.

    Returns:
        ModelRegistry instance

    Raises:
        RuntimeError: If registry not initialized
    """
    if _registry is None:
        raise RuntimeError("Model registry not initialized. Call init_registry() first.")
    return _registry
```

OLD (lines 42-56):
```python
@blueprint.route("/", defaults={"path": ""}, methods=ALL_METHODS)
@blueprint.route("/<path:path>", methods=ALL_METHODS)
@require_auth
def catch_all(path: str):
    """Forward any request path to the Azure backend.

    Logs the incoming request and forwards it to the selected backend
    implementation, returning the backend's response. If forwarding fails,
    returns a 502 JSON error payload.
    """
    if current_app.config.get("LOG_CONTEXT"):
        log_request(request)
    init_last_recording()
    increment_last_recording()
    record_payload(request.json, "downstream_request")
    adapter = AzureAdapter()
    return adapter.forward(request)
```

NEW:
```python
@blueprint.route("/", defaults={"path": ""}, methods=ALL_METHODS)
@blueprint.route("/<path:path>", methods=ALL_METHODS)
@require_auth
def catch_all(path: str):
    """Forward any request path to the appropriate backend.

    Logs the incoming request, determines the model from the request,
    looks up the configuration, creates the appropriate adapter,
    and forwards to the backend. Returns a 400 error if model is not configured.
    """
    if current_app.config.get("LOG_CONTEXT"):
        log_request(request)
    init_last_recording()
    increment_last_recording()

    # Extract model name from request
    payload = request.get_json(silent=True, force=False)
    model_name = payload.get("model") if isinstance(payload, dict) else None

    if not model_name:
        return jsonify({
            "error": {
                "message": "Missing 'model' field in request",
                "type": "invalid_request_error",
                "param": "model",
                "code": None
            }
        }), 400

    # Get model configuration
    try:
        registry = get_registry()
        model_config = registry.get_model_config(model_name)
    except ModelNotFoundError as e:
        return jsonify({
            "error": {
                "message": str(e),
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found"
            }
        }), 400

    record_payload(request.json, "downstream_request")

    # Create appropriate adapter and forward
    adapter = AdapterFactory.create_adapter(model_config)
    return adapter.forward(request)
```

OLD (lines 58-82):
```python
@blueprint.route("/models", methods=["GET"])
@blueprint.route("/v1/models", methods=["GET"])
@require_auth
def models():
    """Return a list of available models."""
    models = [
        "gpt-high",
        "gpt-medium",
        "gpt-low",
        "gpt-minimal",
    ]
    return jsonify(
        {
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": 1686935002,
                    "owned_by": "openai",
                }
                for model in models
            ],
        }
    )
```

NEW:
```python
@blueprint.route("/models", methods=["GET"])
@blueprint.route("/v1/models", methods=["GET"])
@require_auth
def models():
    """Return a list of available models from registry."""
    registry = get_registry()
    model_list = registry.list_models()

    return jsonify(
        {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": 1686935002,
                    "owned_by": "system",
                }
                for model_name in model_list
            ],
        }
    )
```

**Step 5: Update app factory to initialize registry**

Modify: `app/app.py`

OLD (lines 1-8):
```python
"""The app module, containing the app factory function."""

from flask import Flask
from rich.traceback import install as install_rich_traceback

from . import commands
from .blueprint import blueprint
```

NEW:
```python
"""The app module, containing the app factory function."""

from flask import Flask
from rich.traceback import install as install_rich_traceback

from . import commands
from .blueprint import blueprint, init_registry
```

OLD (lines 10-20):
```python
def create_app(config_object="app.settings"):
    """Create application factory, as explained here: http://flask.pocoo.org/docs/patterns/appfactories/.

    :param config_object: The configuration object to use.
    """
    app = Flask(__name__.split(".")[0])
    app.config.from_object(config_object)
    configure_logging(app)
    register_commands(app)
    register_blueprints(app)
    return app
```

NEW:
```python
def create_app(config_object="app.settings"):
    """Create application factory, as explained here: http://flask.pocoo.org/docs/patterns/appfactories/.

    :param config_object: The configuration object to use.
    """
    app = Flask(__name__.split(".")[0])
    app.config.from_object(config_object)
    configure_logging(app)
    configure_registry(app)
    register_commands(app)
    register_blueprints(app)
    return app


def configure_registry(app):
    """Initialize the model registry."""
    init_registry(app.config["MODEL_CONFIG_PATH"])
```

**Step 6: Run tests to verify routing works**

Run: `pytest tests/test_models.py -v`

Expected: PASS

**Step 7: Run all tests**

Run: `pytest tests/ -v --tb=short`

Expected: Most tests pass (Anthropic tests will fail - not implemented yet)

**Step 8: Commit**

```bash
git add app/blueprint.py app/app.py app/settings.py tests/test_models.py
git commit -m "feat: integrate registry and factory into request routing

- Blueprint now uses ModelRegistry to validate model names
- AdapterFactory creates appropriate backend adapter
- /models endpoint returns configured models from registry
- Add MODEL_CONFIG_PATH setting for config file location
- Return 400 error for unconfigured models"
```

---

## Task 5: Implement Anthropic Adapter

**Files:**
- Create: `app/anthropic/__init__.py`
- Create: `app/anthropic/adapter.py`
- Create: `app/anthropic/request_adapter.py`
- Create: `app/anthropic/response_adapter.py`

**Step 1: Install anthropic SDK**

Run: `pip install anthropic && echo "anthropic>=0.42.0" >> requirements/prod.txt`

**Step 2: Write failing test for Anthropic adapter**

Create: `tests/test_anthropic_adapter.py`

```python
"""Tests for Anthropic adapter."""
import pytest
from unittest.mock import Mock, patch
from app.anthropic.adapter import AnthropicAdapter
from app.registry.model_config import ModelConfig


def create_test_anthropic_config():
    """Create test Anthropic model configuration."""
    return ModelConfig(
        name="test-claude",
        backend="anthropic",
        api_model="claude-sonnet-4.5-20250514",
        max_tokens=8192,
    )


def test_anthropic_adapter_initialization():
    """Test AnthropicAdapter can be initialized."""
    config = create_test_anthropic_config()
    adapter = AnthropicAdapter(config)

    assert adapter.model_config == config
    assert adapter.inbound_model == "test-claude"


def test_anthropic_request_conversion(app):
    """Test conversion of OpenAI format to Anthropic Messages API."""
    config = create_test_anthropic_config()
    adapter = AnthropicAdapter(config)

    with app.test_request_context(
        json={
            "model": "test-claude",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ],
            "temperature": 0.7,
            "max_tokens": 1024
        }
    ):
        from flask import request
        request_data = adapter.adapt_request(request)

        # Verify Anthropic Messages API format
        assert "model" in request_data["json"]
        assert request_data["json"]["model"] == "claude-sonnet-4.5-20250514"
        assert "messages" in request_data["json"]
        assert request_data["json"]["stream"] is True

        # System message should be in separate 'system' field
        assert "system" in request_data["json"]
        assert request_data["json"]["system"] == "You are a helpful assistant."

        # Only user message should be in messages array
        messages = request_data["json"]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello!"
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/test_anthropic_adapter.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'app.anthropic'"

**Step 4: Create Anthropic RequestAdapter**

Create: `app/anthropic/request_adapter.py`

```python
"""Request adaptation for Anthropic Messages API."""
from typing import Any, Dict, List
from flask import Request, current_app


class AnthropicRequestAdapter:
    """Convert OpenAI Chat Completions requests to Anthropic Messages API format."""

    def __init__(self, adapter: Any):
        """Initialize with reference to parent AnthropicAdapter."""
        self.adapter = adapter

    def _extract_system_messages(self, messages: List[Dict]) -> str:
        """Extract and concatenate all system messages.

        Anthropic API requires system messages in a separate 'system' parameter,
        not in the messages array.
        """
        system_parts = []
        for msg in messages:
            if msg.get("role") in {"system", "developer"}:
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_parts.append(content)
                elif isinstance(content, list):
                    # Extract text from content array
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            system_parts.append(item.get("text", ""))

        return "\n\n".join(system_parts) if system_parts else ""

    def _convert_messages(self, messages: List[Dict]) -> List[Dict]:
        """Convert OpenAI messages to Anthropic format.

        Filters out system messages (handled separately) and converts
        the rest to Anthropic's message format.
        """
        anthropic_messages = []

        for msg in messages:
            role = msg.get("role")

            # Skip system messages (handled in _extract_system_messages)
            if role in {"system", "developer"}:
                continue

            content = msg.get("content", "")

            # Convert role (OpenAI 'assistant' -> Anthropic 'assistant')
            # OpenAI 'user' -> Anthropic 'user'
            if role in {"user", "assistant"}:
                anthropic_messages.append({
                    "role": role,
                    "content": content
                })
            elif role == "tool":
                # Tool responses in Anthropic format
                anthropic_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id", ""),
                            "content": content
                        }
                    ]
                })

            # Handle tool calls from assistant
            if tool_calls := msg.get("tool_calls"):
                tool_use_content = []
                for tool_call in tool_calls:
                    function = tool_call.get("function", {})
                    tool_use_content.append({
                        "type": "tool_use",
                        "id": tool_call.get("id", ""),
                        "name": function.get("name", ""),
                        "input": function.get("arguments", "{}")
                    })

                if tool_use_content:
                    # Append tool_use to last assistant message or create new one
                    if anthropic_messages and anthropic_messages[-1]["role"] == "assistant":
                        if isinstance(anthropic_messages[-1]["content"], str):
                            anthropic_messages[-1]["content"] = [
                                {"type": "text", "text": anthropic_messages[-1]["content"]}
                            ]
                        anthropic_messages[-1]["content"].extend(tool_use_content)
                    else:
                        anthropic_messages.append({
                            "role": "assistant",
                            "content": tool_use_content
                        })

        return anthropic_messages

    def _convert_tools(self, tools: List[Dict]) -> List[Dict]:
        """Convert OpenAI tool definitions to Anthropic format."""
        if not tools:
            return []

        anthropic_tools = []
        for tool in tools:
            if tool.get("type") != "function":
                continue

            function = tool.get("function", {})
            anthropic_tools.append({
                "name": function.get("name", ""),
                "description": function.get("description", ""),
                "input_schema": function.get("parameters", {})
            })

        return anthropic_tools

    def adapt(self, req: Request) -> Dict[str, Any]:
        """Convert OpenAI request to Anthropic Messages API request kwargs.

        Args:
            req: Flask request with OpenAI Chat Completions format

        Returns:
            Dict suitable for requests.request(**kwargs)
        """
        payload = req.get_json(silent=True, force=False)
        messages = payload.get("messages", [])

        # Build Anthropic request body
        anthropic_body = {
            "model": self.adapter.model_config.api_model,
            "messages": self._convert_messages(messages),
            "max_tokens": (
                payload.get("max_tokens") or
                self.adapter.model_config.max_tokens or
                8192
            ),
            "stream": True,
        }

        # Add system message if present
        system = self._extract_system_messages(messages)
        if system:
            anthropic_body["system"] = system

        # Add optional parameters
        if "temperature" in payload:
            anthropic_body["temperature"] = payload["temperature"]

        if "top_p" in payload:
            anthropic_body["top_p"] = payload["top_p"]

        # Convert tools
        if tools := payload.get("tools"):
            anthropic_body["tools"] = self._convert_tools(tools)

        # Get Anthropic API key from environment
        settings = current_app.config
        api_key = settings.get("ANTHROPIC_API_KEY")
        if not api_key:
            from ..exceptions import ServiceConfigurationError
            raise ServiceConfigurationError(
                "ANTHROPIC_API_KEY not set in environment"
            )

        # Build request kwargs
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        request_kwargs = {
            "method": "POST",
            "url": "https://api.anthropic.com/v1/messages",
            "headers": headers,
            "json": anthropic_body,
            "stream": True,
            "timeout": (60, None),
        }

        return request_kwargs
```

**Step 5: Create Anthropic ResponseAdapter**

Create: `app/anthropic/response_adapter.py`

```python
"""Response adaptation for Anthropic Messages API streaming."""
import json
import random
import time
from string import ascii_letters, digits
from typing import Any, Dict, Iterable, Optional

from flask import Response, current_app, stream_with_context
from rich.live import Live

from ..common.logging import console, create_message_panel
from ..exceptions import ClientClosedConnection


class AnthropicResponseAdapter:
    """Convert Anthropic Messages API streaming responses to OpenAI format."""

    def __init__(self, adapter: Any):
        """Initialize with reference to parent AnthropicAdapter."""
        self.adapter = adapter
        self._chat_completion_id: Optional[str] = None
        self._tool_calls_count: int = 0

    @staticmethod
    def _create_chat_completion_id() -> str:
        """Generate OpenAI-compatible chat completion ID."""
        alphabet = ascii_letters + digits
        return "chatcmpl-" + "".join(random.choices(alphabet, k=24))

    def _build_completion_chunk(
        self,
        *,
        delta: Optional[Dict[str, Any]] = None,
        finish_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build OpenAI Chat Completions chunk."""
        return {
            "id": self._chat_completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.adapter.inbound_model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta or {},
                    "finish_reason": finish_reason,
                }
            ],
        }

    def _parse_sse_line(self, line: bytes) -> Optional[Dict[str, Any]]:
        """Parse a single SSE line into event data.

        Anthropic SSE format:
        event: message_start
        data: {"type": "message_start", ...}
        """
        line_str = line.decode("utf-8").strip()

        if line_str.startswith("data: "):
            data_str = line_str[6:]  # Remove "data: " prefix
            try:
                return json.loads(data_str)
            except json.JSONDecodeError:
                return None

        return None

    def _handle_anthropic_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert Anthropic streaming event to OpenAI chunk.

        Anthropic events:
        - message_start: Initial message metadata
        - content_block_start: Start of content block (text or tool_use)
        - content_block_delta: Incremental content
        - content_block_stop: End of content block
        - message_delta: Message-level updates
        - message_stop: End of stream
        """
        event_type = event.get("type")

        if event_type == "message_start":
            # First chunk with role
            return self._build_completion_chunk(
                delta={"role": "assistant", "content": ""}
            )

        elif event_type == "content_block_start":
            # Start of text or tool_use block
            content_block = event.get("content_block", {})
            block_type = content_block.get("type")

            if block_type == "tool_use":
                self._tool_calls_count += 1
                return self._build_completion_chunk(
                    delta={
                        "tool_calls": [
                            {
                                "index": self._tool_calls_count - 1,
                                "id": content_block.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": content_block.get("name", ""),
                                    "arguments": ""
                                }
                            }
                        ]
                    }
                )

            return None

        elif event_type == "content_block_delta":
            delta = event.get("delta", {})
            delta_type = delta.get("type")

            if delta_type == "text_delta":
                # Text content
                return self._build_completion_chunk(
                    delta={"content": delta.get("text", "")}
                )

            elif delta_type == "input_json_delta":
                # Tool call arguments
                return self._build_completion_chunk(
                    delta={
                        "tool_calls": [
                            {
                                "index": self._tool_calls_count - 1,
                                "function": {
                                    "arguments": delta.get("partial_json", "")
                                }
                            }
                        ]
                    }
                )

        elif event_type == "message_delta":
            # Check for stop reason
            delta = event.get("delta", {})
            stop_reason = delta.get("stop_reason")

            if stop_reason == "end_turn":
                return self._build_completion_chunk(finish_reason="stop")
            elif stop_reason == "tool_use":
                return self._build_completion_chunk(finish_reason="tool_calls")

        return None

    def adapt(self, upstream_resp: Any) -> Response:
        """Convert Anthropic streaming response to OpenAI SSE format.

        Args:
            upstream_resp: requests.Response from Anthropic API

        Returns:
            Flask Response with OpenAI-compatible SSE stream
        """

        @stream_with_context
        def generate() -> Iterable[bytes]:
            self._chat_completion_id = self._create_chat_completion_id()
            self._tool_calls_count = 0

            completion_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": "",
                "tool_calls": [],
            }

            events = 0
            buffer = b""

            with Live(None, console=console, refresh_per_second=2) as live:
                try:
                    for chunk in upstream_resp.iter_content(chunk_size=8192):
                        buffer += chunk

                        # Process complete lines
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)

                            if not line.strip():
                                continue

                            event_data = self._parse_sse_line(line)
                            if not event_data:
                                continue

                            if current_app.config.get("LOG_COMPLETION"):
                                if events > 1:
                                    live.update(create_message_panel(completion_msg, 1, 1))
                                events += 1

                            chunk_dict = self._handle_anthropic_event(event_data)
                            if chunk_dict:
                                # Yield as SSE
                                yield f"data: {json.dumps(chunk_dict)}\n\n".encode("utf-8")

                                # Update completion message for logging
                                if current_app.config.get("LOG_COMPLETION"):
                                    delta = chunk_dict.get("choices", [{}])[0].get("delta", {})
                                    if "content" in delta:
                                        completion_msg["content"] += delta["content"]

                    # Send final chunk if no finish_reason was sent
                    if events > 0:
                        final_chunk = self._build_completion_chunk(
                            finish_reason="stop" if self._tool_calls_count == 0 else "tool_calls"
                        )
                        yield f"data: {json.dumps(final_chunk)}\n\n".encode("utf-8")

                    yield b"data: [DONE]\n\n"

                    if current_app.config.get("LOG_COMPLETION"):
                        live.update(create_message_panel(completion_msg, 1, 1))

                except GeneratorExit:
                    raise ClientClosedConnection(
                        "Client closed connection during streaming response"
                    ) from None
                finally:
                    upstream_resp.close()

        headers = {
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }

        return Response(
            generate(),
            status=getattr(upstream_resp, "status_code", 200),
            headers=headers,
        )
```

**Step 6: Create Anthropic main adapter**

Create: `app/anthropic/adapter.py`

```python
"""Anthropic adapter for Messages API."""
import requests
from flask import Request, Response

from ..adapters.base import BaseAdapter
from ..common.logging import console
from ..common.recording import record_payload
from ..registry.model_config import ModelConfig

from .request_adapter import AnthropicRequestAdapter
from .response_adapter import AnthropicResponseAdapter


class AnthropicAdapter(BaseAdapter):
    """Adapter for Anthropic Messages API.

    Converts OpenAI Chat Completions format to Anthropic Messages API
    and streams responses back in OpenAI format.
    """

    def __init__(self, model_config: ModelConfig):
        """Initialize Anthropic adapter with model configuration."""
        super().__init__(model_config)
        self.request_adapter = AnthropicRequestAdapter(self)
        self.response_adapter = AnthropicResponseAdapter(self)

    def forward(self, req: Request) -> Response:
        """Forward request to Anthropic API and return adapted response."""
        request_kwargs = self.adapt_request(req)

        record_payload(request_kwargs.get("json", {}), "upstream_request")

        # Call Anthropic API
        resp = requests.request(**request_kwargs)

        if resp.status_code != 200:
            return self._handle_anthropic_error(resp, request_kwargs)

        return self.adapt_response(resp)

    def adapt_request(self, req: Request) -> dict:
        """Adapt OpenAI request to Anthropic Messages API format."""
        return self.request_adapter.adapt(req)

    def adapt_response(self, backend_response) -> Response:
        """Adapt Anthropic streaming response to OpenAI format."""
        return self.response_adapter.adapt(backend_response)

    def _handle_anthropic_error(self, resp, request_kwargs) -> Response:
        """Handle Anthropic API errors."""
        try:
            resp_content = resp.json()
        except ValueError:
            resp_content = resp.text

        console.rule(f"[red]Anthropic API request failed with status code {resp.status_code}[/red]")
        console.print(f"Response: {resp_content}")

        error_message = (
            f"Anthropic API error (status {resp.status_code}): {resp_content}\n"
            "Check your ANTHROPIC_API_KEY and model configuration."
        )

        return Response(
            error_message,
            status=resp.status_code if resp.status_code != 401 else 400,
        )
```

Create: `app/anthropic/__init__.py`

```python
"""Anthropic adapter package."""
from .adapter import AnthropicAdapter

__all__ = ["AnthropicAdapter"]
```

**Step 7: Add ANTHROPIC_API_KEY to settings**

Modify: `app/settings.py`

Add after AZURE settings:
```python
# Anthropic configuration
ANTHROPIC_API_KEY = env.str("ANTHROPIC_API_KEY", "")
```

**Step 8: Run Anthropic adapter tests**

Run: `pytest tests/test_anthropic_adapter.py -v`

Expected: PASS

**Step 9: Run all tests**

Run: `pytest tests/ -v`

Expected: PASS (all tests should pass now)

**Step 10: Commit**

```bash
git add app/anthropic/ app/settings.py tests/test_anthropic_adapter.py requirements/prod.txt
git commit -m "feat: implement Anthropic Messages API adapter

- Add AnthropicAdapter implementing BaseAdapter interface
- Convert OpenAI Chat Completions to Anthropic Messages API
- Stream Anthropic responses as OpenAI-compatible SSE chunks
- Handle system messages, tools, and streaming correctly
- Add ANTHROPIC_API_KEY configuration
- Add anthropic SDK dependency"
```

---

## Task 6: Documentation and Testing

**Files:**
- Create: `docs/MULTI_BACKEND.md`
- Modify: `README.md`
- Modify: `.env.example`

**Step 1: Create multi-backend documentation**

Create: `docs/MULTI_BACKEND.md`

```markdown
# Multi-Backend Router

This proxy now supports multiple AI backends through a registry-based routing system.

## How It Works

1. **Model Registry**: `app/models.yaml` defines all available models
2. **Backend Routing**: Client requests specify `model` name, proxy routes to correct backend
3. **Format Conversion**: Each backend adapter converts OpenAI ↔ backend format
4. **Unified Interface**: Client always uses OpenAI Chat Completions format

## Supported Backends

### Azure OpenAI Responses API
- Models: `gpt-high`, `gpt-medium`, `gpt-low`, `gpt-minimal`
- Reasoning models with configurable effort levels
- Requires: `AZURE_API_KEY`, `AZURE_BASE_URL`, `AZURE_DEPLOYMENT`

### Anthropic Messages API
- Models: `claude-sonnet-4-5`, `claude-opus-4-5`
- High-quality reasoning and coding models
- Requires: `ANTHROPIC_API_KEY`

## Configuration

### Adding a New Model

Edit `app/models.yaml`:

```yaml
models:
  my-new-model:
    backend: anthropic  # or azure
    api_model: claude-sonnet-4.5-20250514
    max_tokens: 8192
```

### Model Configuration Options

**Azure models:**
- `backend`: "azure"
- `api_model`: Azure model name (e.g., "gpt-5")
- `reasoning_effort`: "minimal" | "low" | "medium" | "high"
- `deployment_name`: (optional) Azure deployment name
- `summary_level`: (optional) "auto" | "detailed" | "concise"
- `verbosity_level`: (optional) "low" | "medium" | "high"
- `truncation_strategy`: (optional) "auto" | "disabled"

**Anthropic models:**
- `backend`: "anthropic"
- `api_model`: Anthropic model ID (e.g., "claude-sonnet-4.5-20250514")
- `max_tokens`: (optional) Default max tokens (default: 8192)

## Environment Variables

```bash
# Azure Backend
AZURE_API_KEY=your-azure-key
AZURE_BASE_URL=https://your-resource.openai.azure.com
AZURE_DEPLOYMENT=gpt-5
AZURE_API_VERSION=2025-04-01-preview

# Anthropic Backend
ANTHROPIC_API_KEY=your-anthropic-key

# Model Registry
MODEL_CONFIG_PATH=app/models.yaml  # optional, defaults to app/models.yaml
```

## Usage Examples

### Using Azure Model

```bash
curl http://localhost:5000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $SERVICE_API_KEY" \
  -d '{
    "model": "gpt-high",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Using Anthropic Model

```bash
curl http://localhost:5000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $SERVICE_API_KEY" \
  -d '{
    "model": "claude-sonnet-4-5",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Listing Available Models

```bash
curl http://localhost:5000/models \
  -H "Authorization: Bearer $SERVICE_API_KEY"
```

Returns:
```json
{
  "object": "list",
  "data": [
    {"id": "gpt-high", "object": "model", ...},
    {"id": "claude-sonnet-4-5", "object": "model", ...}
  ]
}
```

## Error Handling

**Model not configured:**
```json
{
  "error": {
    "message": "Model 'unknown-model' is not configured. Available models: gpt-high, claude-sonnet-4-5, ...",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

**Missing API key:**
```json
{
  "error": {
    "message": "ANTHROPIC_API_KEY not set in environment",
    "type": "configuration_error"
  }
}
```

## Architecture

```
Client Request (OpenAI format)
    ↓
ModelRegistry.get_model_config(model_name)
    ↓
AdapterFactory.create_adapter(config)
    ↓
├─ AzureAdapter → Azure Responses API
└─ AnthropicAdapter → Anthropic Messages API
    ↓
Response (OpenAI format SSE stream)
```

## Adding a New Backend

To add a new backend (e.g., Google Gemini):

1. Create `app/gemini/adapter.py` implementing `BaseAdapter`
2. Implement `adapt_request()` and `adapt_response()` methods
3. Add backend to `AdapterFactory.create_adapter()`:
   ```python
   elif backend == "gemini":
       from ..gemini.adapter import GeminiAdapter
       return GeminiAdapter(model_config)
   ```
4. Add models to `app/models.yaml`:
   ```yaml
   gemini-pro:
     backend: gemini
     api_model: gemini-pro
   ```
5. Add required env vars to `app/settings.py`

## Testing

Run tests for all backends:
```bash
pytest tests/ -v
```

Test specific backend:
```bash
pytest tests/test_anthropic_adapter.py -v
pytest tests/test_azure_errors.py -v
```
```

**Step 2: Update README.md**

Modify: `README.md`

Add section after "Features":

```markdown
## Multi-Backend Support

This proxy now supports multiple AI backends with a unified OpenAI-compatible interface:

- **Azure OpenAI Responses API**: GPT-5 reasoning models with configurable effort levels
- **Anthropic Messages API**: Claude Sonnet and Opus models

See [docs/MULTI_BACKEND.md](docs/MULTI_BACKEND.md) for detailed configuration and usage.

### Quick Start

1. Configure your models in `app/models.yaml`
2. Set API keys in `.env`:
   ```bash
   ANTHROPIC_API_KEY=your-key
   AZURE_API_KEY=your-key
   ```
3. Use any configured model by name:
   ```bash
   curl -X POST http://localhost:5000/chat/completions \
     -d '{"model": "claude-sonnet-4-5", "messages": [...]}'
   ```
```

**Step 3: Update .env.example**

Modify: `.env.example`

Add Anthropic configuration:

```bash
# Anthropic Configuration (optional)
ANTHROPIC_API_KEY=sk-ant-...

# Model Registry
MODEL_CONFIG_PATH=app/models.yaml
```

**Step 4: Commit documentation**

```bash
git add docs/MULTI_BACKEND.md README.md .env.example
git commit -m "docs: add multi-backend documentation

- Add comprehensive multi-backend guide
- Document model configuration format
- Add usage examples for all backends
- Update README with quick start
- Update .env.example with new variables"
```

---

## Task 7: Integration Testing and Verification

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

Create: `tests/test_integration.py`

```python
"""Integration tests for multi-backend routing."""
import pytest
from unittest.mock import patch, Mock
import json


def test_azure_model_routing(client):
    """Test that gpt-high routes to Azure backend."""
    with patch("app.azure.adapter.requests.request") as mock_request:
        # Mock Azure streaming response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content = Mock(return_value=[
            b'event: response.output_text.delta\n',
            b'data: {"delta": "Hello"}\n\n'
        ])
        mock_request.return_value = mock_response

        response = client.post(
            "/chat/completions",
            json={
                "model": "gpt-high",
                "messages": [{"role": "user", "content": "Hi"}]
            }
        )

        assert response.status_code == 200
        # Verify Azure API was called
        assert mock_request.called
        call_kwargs = mock_request.call_args[1]
        assert "responses" in call_kwargs["url"]


def test_anthropic_model_routing(client):
    """Test that claude-sonnet-4-5 routes to Anthropic backend."""
    with patch("app.anthropic.adapter.requests.request") as mock_request:
        # Mock Anthropic streaming response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content = Mock(return_value=[
            b'data: {"type": "message_start"}\n\n',
            b'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}\n\n'
        ])
        mock_request.return_value = mock_response

        response = client.post(
            "/chat/completions",
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hi"}]
            }
        )

        assert response.status_code == 200
        # Verify Anthropic API was called
        assert mock_request.called
        call_kwargs = mock_request.call_args[1]
        assert "anthropic.com" in call_kwargs["url"]


def test_unknown_model_returns_error(client):
    """Test that requesting unknown model returns 400."""
    response = client.post(
        "/chat/completions",
        json={
            "model": "unknown-model-xyz",
            "messages": [{"role": "user", "content": "Hi"}]
        }
    )

    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "not configured" in data["error"]["message"]
    assert "model_not_found" in data["error"]["code"]


def test_missing_model_field_returns_error(client):
    """Test that missing model field returns 400."""
    response = client.post(
        "/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hi"}]
        }
    )

    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "Missing 'model' field" in data["error"]["message"]
```

**Step 2: Run integration tests**

Run: `pytest tests/test_integration.py -v`

Expected: PASS

**Step 3: Run full test suite**

Run: `pytest tests/ -v --cov=app --cov-report=term-missing`

Expected: PASS with >80% coverage

**Step 4: Manual testing checklist**

Test manually:

```bash
# 1. Start server
flask run

# 2. Test Azure model
curl -X POST http://localhost:5000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $SERVICE_API_KEY" \
  -d '{"model": "gpt-high", "messages": [{"role": "user", "content": "Say hello"}]}'

# 3. Test Anthropic model (if ANTHROPIC_API_KEY is set)
curl -X POST http://localhost:5000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $SERVICE_API_KEY" \
  -d '{"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "Say hello"}]}'

# 4. Test /models endpoint
curl http://localhost:5000/models \
  -H "Authorization: Bearer $SERVICE_API_KEY"

# 5. Test error handling
curl -X POST http://localhost:5000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $SERVICE_API_KEY" \
  -d '{"model": "nonexistent", "messages": [{"role": "user", "content": "Hi"}]}'
```

**Step 5: Commit integration tests**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for multi-backend routing

- Test Azure model routing
- Test Anthropic model routing
- Test unknown model error handling
- Test missing model field validation"
```

---

## Execution Complete

Plan saved to `docs/plans/2025-12-19-multi-backend-router.md`.

### Summary

This plan transforms the proxy into a multi-backend router:

1. **Model Registry** - YAML-based whitelist of allowed models
2. **Adapter Pattern** - Unified interface for all backends
3. **Azure Refactoring** - Use model config instead of hardcoded names
4. **Anthropic Support** - Full OpenAI ↔ Anthropic conversion
5. **Request Routing** - Model name determines backend
6. **Documentation** - Comprehensive usage guide

### Key Benefits

- ✅ Whitelist enforcement (only configured models work)
- ✅ Easy to add new models (edit YAML, no code changes)
- ✅ Clean architecture (factory pattern, adapter interface)
- ✅ Backward compatible (existing Azure models still work)
- ✅ Extensible (easy to add more backends)

### Testing Strategy

- Unit tests for each component (registry, factory, adapters)
- Integration tests for routing
- Replay-based tests for Azure (existing)
- Mock-based tests for Anthropic
- Manual testing checklist

### Estimated Complexity

- **Registry & Factory**: Simple (2-3 hours)
- **Azure Refactoring**: Medium (3-4 hours) - careful not to break existing behavior
- **Anthropic Adapter**: Medium (4-5 hours) - format conversion complexity
- **Integration & Testing**: Medium (2-3 hours)
- **Total**: 11-15 hours for experienced developer with zero context

### Next Steps After Implementation

1. Add more Anthropic models (Haiku, etc.)
2. Add other backends (Google Gemini, Mistral, etc.)
3. Add model aliasing (e.g., "best-coder" → "claude-sonnet-4-5")
4. Add usage tracking per model
5. Add rate limiting per backend
