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
