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

    def _convert_content(self, content: Any) -> Any:
        """Convert OpenAI content format to Anthropic format.

        Handles text, images, and mixed content arrays.
        OpenAI image format: {"type": "image_url", "image_url": {"url": "data:..."}}
        Anthropic image format: {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
        """
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            anthropic_content = []
            for item in content:
                if not isinstance(item, dict):
                    continue

                item_type = item.get("type")

                if item_type == "text":
                    anthropic_content.append(item)

                elif item_type == "image_url":
                    # Convert OpenAI image format to Anthropic
                    image_url = item.get("image_url", {})
                    url = image_url.get("url", "")

                    if url.startswith("data:"):
                        # Parse data URL: data:image/jpeg;base64,/9j/4AAQ...
                        try:
                            header, data = url.split(",", 1)
                            media_type = header.split(":")[1].split(";")[0]

                            anthropic_content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": data
                                }
                            })
                        except (ValueError, IndexError):
                            # Invalid data URL, skip
                            pass

                else:
                    # Pass through other types (tool_result, etc.)
                    anthropic_content.append(item)

            return anthropic_content if anthropic_content else content

        return content

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
                    "content": self._convert_content(content)
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

        # Determine max_tokens
        # max_tokens is the TOTAL limit (thinking + response combined)
        # thinking_budget is part of max_tokens, not additive
        max_tokens = (
            payload.get("max_tokens") or
            self.adapter.model_config.max_tokens or
            64000  # High default for long responses
        )

        # Build Anthropic request body
        anthropic_body = {
            "model": self.adapter.model_config.api_model,
            "messages": self._convert_messages(messages),
            "max_tokens": max_tokens,
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

        # Add extended thinking if configured
        if thinking_budget := self.adapter.model_config.thinking_budget:
            anthropic_body["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget
            }

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

        # Use custom base_url if provided (for Azure Foundry)
        # Otherwise use default Anthropic API endpoint
        base_url = self.adapter.model_config.base_url
        if base_url:
            # Azure Foundry format: https://xxx.openai.azure.com/anthropic
            url = f"{base_url.rstrip('/')}/v1/messages"
        else:
            # Default Anthropic API
            url = "https://api.anthropic.com/v1/messages"

        request_kwargs = {
            "method": "POST",
            "url": url,
            "headers": headers,
            "json": anthropic_body,
            "stream": True,
            "timeout": (60, None),
        }

        return request_kwargs
