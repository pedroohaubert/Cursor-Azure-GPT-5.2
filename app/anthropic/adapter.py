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

    Supports both:
    - Direct Anthropic API (api.anthropic.com)
    - Azure AI Foundry (custom base_url in model config)
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
