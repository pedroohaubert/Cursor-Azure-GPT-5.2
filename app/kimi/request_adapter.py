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

        # Get API key from environment (uses same Azure Foundry key)
        settings = current_app.config
        api_key = settings.get("AZURE_API_KEY")
        if not api_key:
            from ..exceptions import ServiceConfigurationError
            raise ServiceConfigurationError(
                "AZURE_API_KEY not set in environment"
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
            from ..exceptions import ServiceConfigurationError
            raise ServiceConfigurationError(
                "base_url must be set for Kimi models "
                "(e.g., https://xxx.cognitiveservices.azure.com/openai/deployments/Kimi-K2-Thinking)"
            )

        # Endpoint is /chat/completions with api-version
        url = f"{base_url.rstrip('/')}/chat/completions?api-version=2024-05-01-preview"

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
        current_app.logger.debug(
            f"[Kimi] Request body: {kimi_body}"
        )

        return request_kwargs
