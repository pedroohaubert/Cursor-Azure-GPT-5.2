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
            },
            headers={"Authorization": "Bearer test-service-api-key"}
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
            },
            headers={"Authorization": "Bearer test-service-api-key"}
        )

        assert response.status_code == 200
        # Verify Anthropic API was called (through Foundry or direct)
        assert mock_request.called
        call_kwargs = mock_request.call_args[1]
        # Check if URL contains either Foundry endpoint or direct Anthropic API
        assert "anthropic" in call_kwargs["url"]  # Works for both endpoints


def test_unknown_model_returns_error(client):
    """Test that requesting unknown model returns 400."""
    response = client.post(
        "/chat/completions",
        json={
            "model": "unknown-model-xyz",
            "messages": [{"role": "user", "content": "Hi"}]
        },
        headers={"Authorization": "Bearer test-service-api-key"}
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
        },
        headers={"Authorization": "Bearer test-service-api-key"}
    )

    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "Missing 'model' field" in data["error"]["message"]
