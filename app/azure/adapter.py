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

    # Public API
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

    def _handle_azure_error(self, resp: Response, request_kwargs) -> Response:

        try:
            resp_content = resp.json()
        except ValueError:
            resp_content = resp.text

        body = request_kwargs.get("json", {})
        body["instructions"] = body.get("instructions", "no instructions")[:16] + "..."
        body["tools"] = f"...redacted {len(body.get('tools', 'no tools'))} tools..."
        body["input"] = (
            f"...redacted {len(body.get('input', 'no input'))} input items..."
        )
        body["prompt_cache_key"] = re.sub(
            r"(...)(.*)(...)",
            "\\1***\\3",
            body.get("prompt_cache_key", "no prompt_cache_key"),
        )
        report = {
            "endpoint": re.sub(
                r"(//.)(.*?)(.\.)", "\\1***\\3", request_kwargs.get("url")
            ),
            "azure_status_code": resp.status_code,
            "azure_response": resp_content,
            "request_body": body,
        }
        # Precompute pretty JSON to avoid backslashes inside f-string expressions
        report_pretty = json.dumps(report, indent=4).replace("\n", "\n\t")
        error_message = (
            '\nCheck "azure_response" for the error details:\n'
            f"\t{report_pretty}\n"
            "If the issue persists, report it to:\n"
            "\thttps://github.com/gabrii/Cursor-Azure-GPT-5/issues\n"
            "Including all the details above"
        )
        console.rule(f"[red]Request failed with status code {resp.status_code}[/red]")
        console.print(error_message)
        return Response(
            error_message,
            status=resp.status_code if resp.status_code != 401 else 400,
        )
