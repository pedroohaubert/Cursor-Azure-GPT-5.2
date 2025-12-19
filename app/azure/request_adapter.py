"""Request adaptation helpers for Azure Responses API.

This module defines RequestAdapter, which transforms incoming OpenAI-style
requests into Azure Responses API request parameters.
"""

from __future__ import annotations

from typing import Any, Dict, List

from flask import Request, current_app

from ..exceptions import CursorConfigurationError, ServiceConfigurationError


class RequestAdapter:
    """Handle pre-request adaptation for the Azure Responses API.

    Transforms OpenAI Completions/Chat-style inputs into Azure Responses API
    request parameters suitable for streaming completions in this codebase.
    Returns request_kwargs for requests.request(**kwargs). Also sets
    per-request state on the adapter (model).
    """

    def __init__(self, adapter: Any) -> None:
        """Initialize the adapter with a reference to the AzureAdapter."""
        self.adapter = adapter  # AzureAdapter instance for shared config/env

    # ---- Helpers (kept local to minimize cross-module coupling) ----
    def _extract_text_from_content(self, content: Any) -> str:
        """Extract text from content, handling both string and array formats.

        OpenAI Chat API can send content as:
        - A string: "Hello"
        - An array: [{"type": "text", "text": "Hello"}, {"type": "image_url", ...}]

        Returns the concatenated text from all text-type content items.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "text":
                        text_parts.append(item.get("text", ""))
                    elif "text" in item:
                        text_parts.append(str(item.get("text", "")))
            return "".join(text_parts)
        return str(content) if content is not None else ""

    def _convert_content_to_responses_format(
        self, content: Any, role: str
    ) -> List[Dict[str, Any]]:
        """Convert OpenAI Chat API content format to Azure Responses API format.

        OpenAI Chat API can send content as:
        - A string: "Hello"
        - An array: [{"type": "text", "text": "Hello"}, {"type": "image_url", "image_url": {"url": "..."}}]

        Azure Responses API expects:
        - [{"type": "input_text", "text": "Hello"}, {"type": "input_image", "image_url": "..."}]

        Returns a list of content items in Azure Responses API format.
        """
        if isinstance(content, str):
            content_type = "input_text" if role == "user" else "output_text"
            return [{"type": content_type, "text": content}]

        if isinstance(content, list):
            responses_content: List[Dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    continue

                item_type = item.get("type")
                if item_type == "text":
                    content_type = "input_text" if role == "user" else "output_text"
                    text = item.get("text", "")
                    if text:
                        responses_content.append({"type": content_type, "text": text})
                elif item_type == "image_url":
                    image_url_obj = item.get("image_url", {})
                    image_url = (
                        image_url_obj.get("url")
                        if isinstance(image_url_obj, dict)
                        else image_url_obj
                    )
                    if image_url:
                        responses_content.append(
                            {"type": "input_image", "image_url": image_url}
                        )

            return responses_content if responses_content else []

        return []

    def _copy_request_headers_for_azure(
        self, src: Request, *, api_key: str
    ) -> Dict[str, str]:
        headers: Dict[str, str] = {k: v for k, v in src.headers.items()}
        headers.pop("Host", None)
        # Azure prefers api-key header
        headers.pop("Authorization", None)
        headers["api-key"] = api_key
        return headers

    def _messages_to_responses_input_and_instructions(
        self, messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        instructions_parts: List[str] = []
        input_items: List[Dict[str, Any]] = []

        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if role in {"system", "developer"}:
                text_content = self._extract_text_from_content(content)
                instructions_parts.append(text_content)
                continue
            # For user/assistant/tools as inputs
            if role == "tool":
                call_id = m.get("tool_call_id")
                text_content = self._extract_text_from_content(content)

                item = {
                    "type": "function_call_output",
                    "output": text_content,
                    "status": "completed",
                    "call_id": call_id,
                }
                input_items.append(item)
            else:
                responses_content = self._convert_content_to_responses_format(
                    content, role or "user"
                )
                if responses_content:
                    item = {
                        "role": role or "user",
                        "content": responses_content,
                    }
                    input_items.append(item)

                if tool_calls := m.get("tool_calls"):
                    for tool_call in tool_calls:
                        function = tool_call.get("function", {})
                        call_id = tool_call.get("id")
                        item = {
                            "type": "function_call",
                            "name": function.get("name"),
                            "arguments": function.get("arguments"),
                            "call_id": call_id,
                        }
                        input_items.append(item)

        instructions = "\n\n".join(instructions_parts) if instructions_parts else None
        return {
            "instructions": instructions,
            "input": input_items if input_items else None,
        }

    def _transform_tools_for_responses(self, tools: Any) -> Any:
        out: List[Dict[str, Any]] = []
        if not isinstance(tools, list):
            current_app.logger.debug(
                "Skipping tool transformation because tools payload is not a list: %r",
                tools,
            )
            return out

        for tool in tools:
            function = tool.get("function")
            transformed: Dict[str, Any] = {
                "type": "function",
                "name": function.get("name"),
                "description": function.get("description"),
                "parameters": function.get("parameters"),
                "strict": False,
            }
            out.append(transformed)
        return out

    # ---- Main adaptation (always streaming completions-like) ----
    def adapt(self, req: Request) -> Dict[str, Any]:
        """Build requests.request kwargs for the Azure Responses API call.

        Maps inputs to the Responses schema and returns a dict suitable for
        requests.request(**kwargs).
        """
        # Parse request body
        payload = req.get_json(silent=True, force=False)

        settings = current_app.config

        upstream_headers = self._copy_request_headers_for_azure(
            req, api_key=settings["AZURE_API_KEY"]
        )

        # Map Chat/Completions to Responses (always streaming)
        messages = payload.get("messages") or []

        responses_body = (
            self._messages_to_responses_input_and_instructions(messages)
            if isinstance(messages, list)
            else {"input": None, "instructions": None}
        )

        # Use deployment name from config or fall back to env var
        responses_body["model"] = (
            self.adapter.model_config.deployment_name or settings["AZURE_DEPLOYMENT"]
        )

        # Transform tools and tool choice
        responses_body["tools"] = self._transform_tools_for_responses(
            payload.get("tools", [])
        )
        responses_body["tool_choice"] = payload.get("tool_choice")

        responses_body["prompt_cache_key"] = payload.get("user")

        # Always streaming
        responses_body["stream"] = True

        # Get reasoning effort from model config
        reasoning_effort = self.adapter.model_config.reasoning_effort
        if not reasoning_effort:
            raise CursorConfigurationError(
                f"Model '{self.adapter.model_config.name}' is missing reasoning_effort configuration"
            )

        responses_body["reasoning"] = {
            "effort": reasoning_effort,
        }

        # Use summary level from model config or fall back to env var
        summary_level = (
            self.adapter.model_config.summary_level or settings["AZURE_SUMMARY_LEVEL"]
        )
        if summary_level in {"auto", "detailed", "concise"}:
            responses_body["reasoning"]["summary"] = summary_level
        else:
            raise ServiceConfigurationError(
                "summary_level must be either auto, detailed, or concise."
                f"\n\nGot: {summary_level}"
            )

        # Use verbosity level from model config or fall back to env var
        verbosity_level = (
            self.adapter.model_config.verbosity_level or settings["AZURE_VERBOSITY_LEVEL"]
        )
        if verbosity_level in {"low", "high"}:
            responses_body["text"] = {"verbosity": verbosity_level}

        responses_body["store"] = False
        responses_body["stream_options"] = {"include_obfuscation": False}

        # Use truncation strategy from model config or fall back to env var
        truncation = (
            self.adapter.model_config.truncation_strategy or settings["AZURE_TRUNCATION"]
        )
        if truncation == "auto":
            responses_body["truncation"] = truncation

        request_kwargs: Dict[str, Any] = {
            "method": "POST",
            "url": settings["AZURE_RESPONSES_API_URL"],
            "headers": upstream_headers,
            "json": responses_body,
            "data": None,
            "stream": True,
            "timeout": (60, None),
        }
        return request_kwargs
