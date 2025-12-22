"""Tests for Kimi adapter."""
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


def test_response_adapter_passes_through_streaming(mock_adapter, app):
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

    with app.test_request_context():
        adapter = KimiResponseAdapter(mock_adapter)
        response = adapter.adapt(mock_response)

        assert response.status_code == 200
        assert response.mimetype == "text/event-stream"
