# Kimi-K2-Thinking Integration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add support for Kimi-K2-Thinking model via OpenAI Completions API to the multi-backend proxy.

**Architecture:** Create a new "kimi" backend adapter that handles OpenAI Chat Completions API natively. Since Kimi-K2-Thinking already speaks OpenAI format, the adapter will be simpler than Azure (no Responses API conversion) or Anthropic (no Messages API conversion). The adapter only needs to route requests to the correct endpoint with proper authentication.

**Tech Stack:** Flask, requests, YAML configuration, pytest

---

## Task 1: Update ModelConfig for Kimi Backend

**Files:**
- Modify: `app/registry/model_config.py:30-33`
- Test: `tests/test_models.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_models.py
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py::test_model_config_accepts_kimi_backend -v`
Expected: FAIL with "Unsupported backend: kimi"

**Step 3: Update ModelConfig validation**

In `app/registry/model_config.py`, update the `__post_init__` method:

```python
def __post_init__(self):
    """Validate configuration after initialization."""
    if self.backend not in {"azure", "anthropic", "kimi"}:
        raise ValueError(f"Unsupported backend: {self.backend}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py::test_model_config_accepts_kimi_backend -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/registry/model_config.py tests/test_models.py
git commit -m "feat: add kimi backend support to ModelConfig"
```

---

## Task 2: Create Kimi Request Adapter

**Files:**
- Create: `app/kimi/__init__.py`
- Create: `app/kimi/request_adapter.py`
- Test: `tests/test_kimi_adapter.py`

**Step 1: Write the failing test**

```python
# Create tests/test_kimi_adapter.py
import pytest
from flask import Flask
from app.kimi.request_adapter import KimiRequestAdapter
from app.registry.model_config import ModelConfig


@pytest.fixture
def mock_adapter():
    """Create a mock adapter with Kimi model config."""
    class MockKimiAdapter:
        def __init__(self):
            self.model_config = ModelConfig(
                name="kimi-k2-thinking",
                backend="kimi",
                api_model="Kimi-K2-Thinking",
                base_url="https://test.openai.azure.com/openai/v1",
                max_tokens=4096
            )
    return MockKimiAdapter()


def test_request_adapter_builds_correct_url(mock_adapter, app):
    """Test that request adapter builds correct API URL."""
    from flask import request as flask_request

    with app.test_request_context(
        json={
            "model": "kimi-k2-thinking",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7
        }
    ):
        adapter = KimiRequestAdapter(mock_adapter)
        request_kwargs = adapter.adapt(flask_request)

        assert request_kwargs["method"] == "POST"
        assert request_kwargs["url"] == "https://test.openai.azure.com/openai/v1/chat/completions"
        assert "Authorization" in request_kwargs["headers"]
        assert request_kwargs["json"]["model"] == "Kimi-K2-Thinking"


def test_request_adapter_preserves_messages(mock_adapter, app):
    """Test that request adapter preserves OpenAI message format."""
    from flask import request as flask_request

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]

    with app.test_request_context(
        json={"model": "kimi-k2-thinking", "messages": messages}
    ):
        adapter = KimiRequestAdapter(mock_adapter)
        request_kwargs = adapter.adapt(flask_request)

        assert request_kwargs["json"]["messages"] == messages
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_kimi_adapter.py::test_request_adapter_builds_correct_url -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.kimi'"

**Step 3: Create Kimi request adapter**

Create `app/kimi/__init__.py`:

```python
"""Kimi adapter for OpenAI Chat Completions API."""
```

Create `app/kimi/request_adapter.py`:

```python
"""Request adapter for Kimi models via OpenAI Chat Completions API."""
from typing import Any, Dict
from flask import Request, current_app


class KimiRequestAdapter:
    """Convert OpenAI request for Kimi backend.

    Since Kimi uses OpenAI Chat Completions API format natively,
    this adapter primarily handles endpoint routing and authentication.
    """

    def __init__(self, adapter: Any):
        """Initialize with reference to parent KimiAdapter."""
        self.adapter = adapter

    def adapt(self, req: Request) -> Dict[str, Any]:
        """Adapt OpenAI request for Kimi API.

        Args:
            req: Flask request with OpenAI Chat Completions format

        Returns:
            Dict suitable for requests.request(**kwargs)
        """
        payload = req.get_json(silent=True, force=False)

        # Get API key from environment
        settings = current_app.config
        api_key = settings.get("KIMI_API_KEY")
        if not api_key:
            from ..exceptions import ServiceConfigurationError
            raise ServiceConfigurationError(
                "KIMI_API_KEY not set in environment"
            )

        # Build request body - mostly pass-through with model name mapping
        kimi_body = {
            "model": self.adapter.model_config.api_model,
            "messages": payload.get("messages", []),
        }

        # Add optional parameters if present
        if "temperature" in payload:
            kimi_body["temperature"] = payload["temperature"]

        if "top_p" in payload:
            kimi_body["top_p"] = payload["top_p"]

        if "max_tokens" in payload:
            kimi_body["max_tokens"] = payload["max_tokens"]
        elif self.adapter.model_config.max_tokens:
            kimi_body["max_tokens"] = self.adapter.model_config.max_tokens

        if "stream" in payload:
            kimi_body["stream"] = payload["stream"]
        else:
            kimi_body["stream"] = True  # Default to streaming

        # Add tools if present (Kimi supports OpenAI tools format)
        if "tools" in payload:
            kimi_body["tools"] = payload["tools"]

        if "tool_choice" in payload:
            kimi_body["tool_choice"] = payload["tool_choice"]

        # Build request kwargs
        base_url = self.adapter.model_config.base_url
        if not base_url:
            raise ServiceConfigurationError(
                "base_url must be set for Kimi models "
                "(e.g., https://xxx.openai.azure.com/openai/v1)"
            )

        # Endpoint is /chat/completions
        url = f"{base_url.rstrip('/')}/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        request_kwargs = {
            "method": "POST",
            "url": url,
            "headers": headers,
            "json": kimi_body,
            "stream": True,
            "timeout": (60, None),
        }

        current_app.logger.info(
            f"[Kimi] Request to {url} with model {kimi_body['model']}"
        )

        return request_kwargs
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_kimi_adapter.py -v`
Expected: PASS (both tests)

**Step 5: Commit**

```bash
git add app/kimi/ tests/test_kimi_adapter.py
git commit -m "feat: add Kimi request adapter with OpenAI format"
```

---

## Task 3: Create Kimi Response Adapter

**Files:**
- Create: `app/kimi/response_adapter.py`
- Test: `tests/test_kimi_adapter.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_kimi_adapter.py
def test_response_adapter_passes_through_streaming(mock_adapter):
    """Test that response adapter passes through OpenAI SSE stream."""
    from app.kimi.response_adapter import KimiResponseAdapter
    from unittest.mock import Mock

    # Mock streaming response from Kimi API
    mock_response = Mock()
    mock_response.iter_lines.return_value = [
        b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"Kimi-K2-Thinking","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}',
        b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"Kimi-K2-Thinking","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}',
        b'data: [DONE]'
    ]
    mock_response.headers = {"content-type": "text/event-stream"}

    adapter = KimiResponseAdapter(mock_adapter)
    response = adapter.adapt(mock_response)

    assert response.status_code == 200
    assert response.mimetype == "text/event-stream"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_kimi_adapter.py::test_response_adapter_passes_through_streaming -v`
Expected: FAIL with "cannot import name 'KimiResponseAdapter'"

**Step 3: Create Kimi response adapter**

Create `app/kimi/response_adapter.py`:

```python
"""Response adapter for Kimi models."""
from flask import Response, stream_with_context, current_app
from ..common.recording import record_payload


class KimiResponseAdapter:
    """Adapt Kimi streaming response to OpenAI format.

    Since Kimi already returns OpenAI Chat Completions format,
    this adapter is a simple pass-through with logging.
    """

    def __init__(self, adapter):
        """Initialize with reference to parent KimiAdapter."""
        self.adapter = adapter

    def adapt(self, backend_response) -> Response:
        """Pass through Kimi streaming response.

        Args:
            backend_response: requests.Response from Kimi API

        Returns:
            Flask Response with OpenAI Chat Completions SSE stream
        """
        def generate():
            """Stream SSE chunks from Kimi API."""
            for line in backend_response.iter_lines():
                if not line:
                    continue

                # Log chunks if enabled
                if current_app.config.get("LOG_COMPLETION"):
                    current_app.logger.debug(f"[Kimi] {line.decode('utf-8')}")

                # Pass through the SSE line
                yield line + b'\n'

        return Response(
            stream_with_context(generate()),
            status=200,
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_kimi_adapter.py::test_response_adapter_passes_through_streaming -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/kimi/response_adapter.py tests/test_kimi_adapter.py
git commit -m "feat: add Kimi response adapter with pass-through streaming"
```

---

## Task 4: Create Kimi Main Adapter

**Files:**
- Create: `app/kimi/adapter.py`
- Test: `tests/test_kimi_adapter.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_kimi_adapter.py
def test_kimi_adapter_forwards_request(mock_adapter, app):
    """Test that KimiAdapter forwards request correctly."""
    from app.kimi.adapter import KimiAdapter
    from unittest.mock import patch, Mock
    from flask import request as flask_request

    config = ModelConfig(
        name="kimi-k2-thinking",
        backend="kimi",
        api_model="Kimi-K2-Thinking",
        base_url="https://test.openai.azure.com/openai/v1"
    )

    with app.test_request_context(
        json={
            "model": "kimi-k2-thinking",
            "messages": [{"role": "user", "content": "Hello"}]
        }
    ):
        with patch('app.kimi.adapter.requests.request') as mock_request:
            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.iter_lines.return_value = [
                b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"Kimi-K2-Thinking","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}',
                b'data: [DONE]'
            ]
            mock_request.return_value = mock_response

            adapter = KimiAdapter(config)
            response = adapter.forward(flask_request)

            assert response.status_code == 200
            assert mock_request.called
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_kimi_adapter.py::test_kimi_adapter_forwards_request -v`
Expected: FAIL with "cannot import name 'KimiAdapter'"

**Step 3: Create Kimi main adapter**

Create `app/kimi/adapter.py`:

```python
"""Kimi adapter orchestrating request/response transformations."""
import requests
from flask import Request, Response

from ..adapters.base import BaseAdapter
from ..common.logging import console
from ..common.recording import record_payload
from ..registry.model_config import ModelConfig

from .request_adapter import KimiRequestAdapter
from .response_adapter import KimiResponseAdapter


class KimiAdapter(BaseAdapter):
    """Adapter for Kimi models via OpenAI Chat Completions API.

    Kimi-K2-Thinking natively supports OpenAI Chat Completions format,
    so this adapter is simpler than Azure or Anthropic adapters.
    """

    def __init__(self, model_config: ModelConfig):
        """Initialize Kimi adapter with model configuration."""
        super().__init__(model_config)
        self.request_adapter = KimiRequestAdapter(self)
        self.response_adapter = KimiResponseAdapter(self)

    def forward(self, req: Request) -> Response:
        """Forward request to Kimi API and return adapted response."""
        request_kwargs = self.adapt_request(req)

        record_payload(request_kwargs.get("json", {}), "upstream_request")

        # Call Kimi API
        resp = requests.request(**request_kwargs)

        if resp.status_code != 200:
            return self._handle_kimi_error(resp, request_kwargs)

        return self.adapt_response(resp)

    def adapt_request(self, req: Request) -> dict:
        """Adapt OpenAI request for Kimi API."""
        return self.request_adapter.adapt(req)

    def adapt_response(self, backend_response) -> Response:
        """Adapt Kimi streaming response to OpenAI format."""
        return self.response_adapter.adapt(backend_response)

    def _handle_kimi_error(self, resp, request_kwargs) -> Response:
        """Handle Kimi API errors."""
        try:
            resp_content = resp.json()
        except ValueError:
            resp_content = resp.text

        console.rule(f"[red]Kimi API request failed with status code {resp.status_code}[/red]")
        console.print(f"Response: {resp_content}")

        error_message = (
            f"Kimi API error (status {resp.status_code}): {resp_content}\n"
            "Check your KIMI_API_KEY and model configuration."
        )

        return Response(
            error_message,
            status=resp.status_code if resp.status_code != 401 else 400,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_kimi_adapter.py::test_kimi_adapter_forwards_request -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/kimi/adapter.py tests/test_kimi_adapter.py
git commit -m "feat: add Kimi main adapter orchestrating request/response flow"
```

---

## Task 5: Update Adapter Factory

**Files:**
- Modify: `app/adapters/factory.py:38-42`
- Test: `tests/test_adapter_factory.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_adapter_factory.py
def test_factory_creates_kimi_adapter():
    """Test factory creates KimiAdapter for kimi backend."""
    from app.registry.model_config import ModelConfig

    config = ModelConfig(
        name="test-kimi",
        backend="kimi",
        api_model="Kimi-K2-Thinking",
        base_url="https://test.openai.azure.com/openai/v1"
    )

    adapter = AdapterFactory.create_adapter(config)
    assert adapter is not None
    assert isinstance(adapter, BaseAdapter)
    # Verify it's specifically a KimiAdapter
    from app.kimi.adapter import KimiAdapter
    assert isinstance(adapter, KimiAdapter)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_adapter_factory.py::test_factory_creates_kimi_adapter -v`
Expected: FAIL with "Unsupported backend: kimi"

**Step 3: Update adapter factory**

In `app/adapters/factory.py`, add kimi backend support:

```python
@staticmethod
def create_adapter(model_config: ModelConfig) -> BaseAdapter:
    """Create appropriate adapter instance for the model configuration.

    Args:
        model_config: Configuration specifying backend and model details

    Returns:
        Instance of BaseAdapter subclass (AzureAdapter, AnthropicAdapter, KimiAdapter, etc.)

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
    elif backend == "kimi":
        # Import here to avoid circular dependency
        from ..kimi.adapter import KimiAdapter
        return KimiAdapter(model_config)
    else:
        raise ServiceConfigurationError(
            f"Unsupported backend: {backend}. "
            f"Supported backends: azure, anthropic, kimi"
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_adapter_factory.py::test_factory_creates_kimi_adapter -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/adapters/factory.py tests/test_adapter_factory.py
git commit -m "feat: add Kimi backend to adapter factory"
```

---

## Task 6: Add Kimi Configuration to Settings

**Files:**
- Modify: `app/settings.py:44-50`

**Step 1: Add KIMI_API_KEY configuration**

In `app/settings.py`, add Kimi configuration after Anthropic section:

```python
# Anthropic configuration
ANTHROPIC_API_KEY = env.str("ANTHROPIC_API_KEY", "")

# Azure AD Service Principal (for Responses API with Claude in production)
# Only needed if using api_format: responses in models.yaml
AZURE_CLIENT_ID = env.str("AZURE_CLIENT_ID", "")
AZURE_CLIENT_SECRET = env.str("AZURE_CLIENT_SECRET", "")
AZURE_TENANT_ID = env.str("AZURE_TENANT_ID", "")

# Kimi configuration
KIMI_API_KEY = env.str("KIMI_API_KEY", "")
```

**Step 2: Commit**

```bash
git add app/settings.py
git commit -m "feat: add KIMI_API_KEY to settings"
```

---

## Task 7: Add Kimi Model to Configuration

**Files:**
- Modify: `app/models.yaml:69-76`

**Step 1: Add Kimi-K2-Thinking model configuration**

In `app/models.yaml`, add Kimi model configuration after Claude models:

```yaml
  claude-opus-4-5:
    backend: anthropic
    api_model: claude-opus-4-5
    api_format: messages  # Use OpenAI Responses API (better thinking + tools support!)
    base_url: https://cyrela-ia-foundry.openai.azure.com/anthropic
    max_tokens: 64000

  # Kimi models via Azure AI Foundry
  # Uses OpenAI Chat Completions API natively
  # Set KIMI_API_KEY to your Azure Foundry API key
  kimi-k2-thinking:
    backend: kimi
    api_model: Kimi-K2-Thinking
    base_url: https://cyrela-ia-foundry.openai.azure.com/openai/v1
    max_tokens: 4096
```

**Step 2: Commit**

```bash
git add app/models.yaml
git commit -m "feat: add kimi-k2-thinking model to configuration"
```

---

## Task 8: Update Documentation

**Files:**
- Modify: `README.md:31-46`

**Step 1: Update Multi-Backend Support section**

In `README.md`, update the Multi-Backend Support section to mention Kimi:

```markdown
## Multi-Backend Support

This proxy now supports multiple AI backends with a unified OpenAI-compatible interface:

- **Azure OpenAI Responses API**: GPT-5 reasoning models with configurable effort levels
- **Anthropic Messages API**: Claude Sonnet and Opus models
- **Kimi Chat Completions API**: Kimi-K2-Thinking reasoning model

See [docs/MULTI_BACKEND.md](docs/MULTI_BACKEND.md) for detailed configuration and usage.

### Quick Start

1. Configure your models in `app/models.yaml`
2. Set API keys in `.env`:
   ```bash
   ANTHROPIC_API_KEY=your-key
   AZURE_API_KEY=your-key
   KIMI_API_KEY=your-key
   ```
3. Use any configured model by name:
   ```bash
   curl -X POST http://localhost:5000/chat/completions \
     -d '{"model": "kimi-k2-thinking", "messages": [...]}'
   ```
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add Kimi backend to README"
```

---

## Task 9: Add Integration Test

**Files:**
- Test: `tests/test_integration.py`

**Step 1: Write integration test for Kimi model**

```python
# Add to tests/test_integration.py
def test_kimi_model_endpoint_integration(app, client):
    """Test that Kimi model can be called through the proxy."""
    from unittest.mock import patch, Mock

    with patch('app.kimi.adapter.requests.request') as mock_request:
        # Mock successful streaming response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"Kimi-K2-Thinking","choices":[{"index":0,"delta":{"role":"assistant","content":"The"},"finish_reason":null}]}',
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"Kimi-K2-Thinking","choices":[{"index":0,"delta":{"content":" capital"},"finish_reason":null}]}',
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"Kimi-K2-Thinking","choices":[{"index":0,"delta":{"content":" of"},"finish_reason":null}]}',
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"Kimi-K2-Thinking","choices":[{"index":0,"delta":{"content":" France"},"finish_reason":null}]}',
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"Kimi-K2-Thinking","choices":[{"index":0,"delta":{"content":" is"},"finish_reason":null}]}',
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"Kimi-K2-Thinking","choices":[{"index":0,"delta":{"content":" Paris"},"finish_reason":"stop"}]}',
            b'data: [DONE]'
        ]
        mock_request.return_value = mock_response

        response = client.post(
            '/chat/completions',
            json={
                "model": "kimi-k2-thinking",
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"}
                ]
            },
            headers={"Authorization": "Bearer test-key"}
        )

        assert response.status_code == 200
        assert response.mimetype == "text/event-stream"

        # Verify the request was made with correct parameters
        assert mock_request.called
        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["json"]["model"] == "Kimi-K2-Thinking"
        assert "Authorization" in call_kwargs["headers"]
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/test_integration.py::test_kimi_model_endpoint_integration -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration test for Kimi model endpoint"
```

---

## Task 10: Run Full Test Suite

**Step 1: Run all tests**

Run: `pytest -v`
Expected: All tests PASS

**Step 2: Run linter**

Run: `flask lint --check`
Expected: No linting errors

**Step 3: Final commit if any lint fixes needed**

```bash
git add .
git commit -m "style: apply linting fixes"
```

---

## Task 11: Update .env.example

**Files:**
- Modify: `.env.example`

**Step 1: Add KIMI_API_KEY to environment example**

Add Kimi configuration section to `.env.example`:

```bash
# Kimi Configuration
# API key from Azure AI Foundry for Kimi models
KIMI_API_KEY=your-kimi-api-key-here
```

**Step 2: Commit**

```bash
git add .env.example
git commit -m "docs: add KIMI_API_KEY to .env.example"
```

---

## Task 12: Manual Testing

**Step 1: Set up local environment**

1. Copy `.env.example` to `.env` if not already done
2. Add your actual Kimi API key to `.env`:
   ```
   KIMI_API_KEY=your-actual-key
   ```

**Step 2: Start the service**

Run: `flask run -p 8080`

**Step 3: Test with curl**

```bash
curl -X POST http://localhost:8080/chat/completions \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k2-thinking",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7
  }'
```

Expected: Streaming response with OpenAI format chunks containing the answer "Paris"

**Step 4: Verify model listing**

```bash
curl -X GET http://localhost:8080/v1/models \
  -H "Authorization: Bearer change-me"
```

Expected: JSON list including "kimi-k2-thinking" model

**Step 5: Document results**

Create `docs/plans/2025-12-22-kimi-testing-results.md` with:
- Test results
- Any issues encountered
- Performance observations

---

## Completion Checklist

- [ ] ModelConfig accepts "kimi" backend
- [ ] Kimi request adapter created
- [ ] Kimi response adapter created
- [ ] Kimi main adapter created
- [ ] Adapter factory supports Kimi
- [ ] Settings include KIMI_API_KEY
- [ ] Model configuration includes kimi-k2-thinking
- [ ] README updated with Kimi support
- [ ] Integration tests pass
- [ ] Full test suite passes
- [ ] Linting passes
- [ ] .env.example updated
- [ ] Manual testing successful

---

## Notes

**YAGNI Principles Applied:**
- No complex message transformations (Kimi uses OpenAI format natively)
- No custom error handling beyond standard error responses
- No additional configuration options beyond what's necessary

**Testing Strategy:**
- Unit tests for each adapter component
- Integration test for end-to-end flow
- Manual testing with real API

**Commit Strategy:**
- Frequent small commits after each passing test
- Descriptive commit messages following conventional commits format
- Each task produces at least one commit
