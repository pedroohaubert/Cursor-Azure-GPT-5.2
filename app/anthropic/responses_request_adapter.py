"""Request adaptation for Claude models via OpenAI Responses API."""
from typing import Any, Dict, List
from flask import Request, current_app

try:
    from azure.identity import DefaultAzureCredential
    from azure.core.credentials import AccessToken
    AZURE_IDENTITY_AVAILABLE = True
except ImportError:
    AZURE_IDENTITY_AVAILABLE = False
    DefaultAzureCredential = None
    AccessToken = None


class AnthropicResponsesRequestAdapter:
    """Convert OpenAI Chat Completions requests to OpenAI Responses API format for Claude models.

    OpenAI Responses API endpoint for Claude in Azure Foundry:
    https://<project>.services.ai.azure.com/api/projects/<project-name>/openai

    This uses the OpenAI SDK format, not the Anthropic Messages API format.
    """

    def __init__(self, adapter: Any):
        """Initialize with reference to parent AnthropicAdapter."""
        self.adapter = adapter
        self._credential = None
        self._token_cache = None

    def _get_bearer_token(self) -> str:
        """Get Azure AD bearer token for Responses API authentication.

        Supports multiple authentication methods:
        1. Service Principal (for production Docker):
           - Set AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID
        2. DefaultAzureCredential (for local dev):
           - Uses az login, managed identity, etc.

        Requires scope: https://ai.azure.com/.default
        """
        if not AZURE_IDENTITY_AVAILABLE:
            from ..exceptions import ServiceConfigurationError
            raise ServiceConfigurationError(
                "azure-identity package is required for Responses API. "
                "Install with: pip install azure-identity"
            )

        # Initialize credential if needed
        if self._credential is None:
            # Check if Service Principal credentials are available
            settings = current_app.config
            client_id = settings.get("AZURE_CLIENT_ID")
            client_secret = settings.get("AZURE_CLIENT_SECRET")
            tenant_id = settings.get("AZURE_TENANT_ID")

            if client_id and client_secret and tenant_id:
                # Use Service Principal (for production)
                from azure.identity import ClientSecretCredential
                current_app.logger.info(
                    "[Claude Responses API] Using Service Principal authentication"
                )
                self._credential = ClientSecretCredential(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    client_secret=client_secret
                )
            else:
                # Use DefaultAzureCredential (for local dev with az login)
                current_app.logger.info(
                    "[Claude Responses API] Using DefaultAzureCredential (requires az login)"
                )
                self._credential = DefaultAzureCredential()

        # Get token with AI Foundry scope
        scope = "https://ai.azure.com/.default"

        try:
            token = self._credential.get_token(scope)
            current_app.logger.debug(
                f"[Claude Responses API] Obtained Azure AD token (expires: {token.expires_on})"
            )
            return token.token
        except Exception as e:
            from ..exceptions import ServiceConfigurationError
            raise ServiceConfigurationError(
                f"Failed to obtain Azure AD token. "
                f"For production Docker, set: AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID. "
                f"For local dev, run 'az login'. "
                f"Error: {e}"
            )

    def _convert_messages_to_input(self, messages: List[Dict]) -> str:
        """Convert OpenAI messages to simple text input for Responses API.

        For now, we'll use a simple concatenation approach. The Responses API
        accepts a string input, which is simpler than Messages API.

        In the future, we could support the full input format with images, etc.
        """
        text_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Extract text from content (can be string or array)
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                # Extract text from content array
                text = ""
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text += item.get("text", "")
            else:
                text = str(content)

            # Format as conversation
            if role == "system":
                text_parts.append(f"System: {text}")
            elif role == "user":
                text_parts.append(f"User: {text}")
            elif role == "assistant":
                text_parts.append(f"Assistant: {text}")
            elif role == "tool":
                # Tool results
                tool_name = msg.get("name", "tool")
                text_parts.append(f"Tool Result ({tool_name}): {text}")

        return "\n\n".join(text_parts)

    def adapt(self, req: Request) -> Dict[str, Any]:
        """Convert OpenAI request to Claude Responses API request kwargs.

        Args:
            req: Flask request with OpenAI Chat Completions format

        Returns:
            Dict suitable for requests.request(**kwargs)
        """
        payload = req.get_json(silent=True, force=False)
        messages = payload.get("messages", [])

        # Get API key from environment
        settings = current_app.config
        api_key = settings.get("ANTHROPIC_API_KEY")
        if not api_key:
            from ..exceptions import ServiceConfigurationError
            raise ServiceConfigurationError(
                "ANTHROPIC_API_KEY not set in environment"
            )

        # Build OpenAI Responses API request body
        responses_body = {
            "model": self.adapter.model_config.api_model,
            "input": self._convert_messages_to_input(messages),
        }

        # Add max_output_tokens if configured
        max_tokens = payload.get("max_tokens") or self.adapter.model_config.max_tokens or 64000
        responses_body["max_output_tokens"] = max_tokens

        # Add temperature if present
        if "temperature" in payload:
            responses_body["temperature"] = payload["temperature"]

        # Add top_p if present
        if "top_p" in payload:
            responses_body["top_p"] = payload["top_p"]

        # Add tools if present
        tools = payload.get("tools")
        if tools:
            # Convert OpenAI tools format to Responses API format
            converted_tools = []
            for tool in tools:
                if tool.get("type") == "function":
                    function = tool.get("function", {})
                    converted_tools.append({
                        "type": "function",
                        "name": function.get("name", ""),
                        "description": function.get("description", ""),
                        "parameters": function.get("parameters", {})
                    })
            if converted_tools:
                responses_body["tools"] = converted_tools

        # Build request kwargs
        # Use base_url from config if available, otherwise construct from project endpoint
        base_url = self.adapter.model_config.base_url
        if not base_url:
            raise ServiceConfigurationError(
                "base_url must be set for Responses API (e.g., "
                "https://xxx.services.ai.azure.com/api/projects/xxx/openai)"
            )

        # For Responses API, the endpoint is /responses
        # Base URL should be: https://xxx.services.ai.azure.com/api/projects/xxx/openai
        url = f"{base_url.rstrip('/')}/responses"

        # Get Azure AD bearer token for authentication
        bearer_token = self._get_bearer_token()

        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json",
        }

        request_kwargs = {
            "method": "POST",
            "url": url,
            "headers": headers,
            "json": responses_body,
            "stream": True,  # Always stream for compatibility
            "timeout": (60, None),
        }

        current_app.logger.info(
            f"[Claude Responses API] Request to {url} with model {responses_body['model']}"
        )
        current_app.logger.info(
            f"[Claude Responses API] max_output_tokens: {responses_body.get('max_output_tokens')}"
        )

        return request_kwargs
