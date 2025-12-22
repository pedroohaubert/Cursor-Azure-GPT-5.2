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
            "Check your AZURE_API_KEY and model configuration."
        )

        return Response(
            error_message,
            status=resp.status_code if resp.status_code != 401 else 400,
        )
