"""Response adaptation for Claude models via OpenAI Responses API streaming."""
import json
import random
import time
from string import ascii_letters, digits
from typing import Any, Dict, Iterable, Optional

from flask import Response, current_app, stream_with_context
from rich.live import Live

from ..common.logging import console, create_message_panel
from ..exceptions import ClientClosedConnection


class AnthropicResponsesResponseAdapter:
    """Convert OpenAI Responses API streaming responses to OpenAI Chat Completions format.

    The Responses API returns events in a different format than Messages API,
    but we still need to convert them to OpenAI Chat Completions SSE format
    for compatibility with clients.
    """

    def __init__(self, adapter: Any):
        """Initialize with reference to parent AnthropicAdapter."""
        self.adapter = adapter
        self._chat_completion_id: Optional[str] = None

    @staticmethod
    def _create_chat_completion_id() -> str:
        """Generate OpenAI-compatible chat completion ID."""
        alphabet = ascii_letters + digits
        return "chatcmpl-" + "".join(random.choices(alphabet, k=24))

    def _build_completion_chunk(
        self,
        *,
        delta: Optional[Dict[str, Any]] = None,
        finish_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build OpenAI Chat Completions chunk."""
        return {
            "id": self._chat_completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.adapter.inbound_model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta or {},
                    "finish_reason": finish_reason,
                }
            ],
        }

    def _parse_sse_line(self, line: bytes) -> Optional[Dict[str, Any]]:
        """Parse a single SSE line into event data.

        OpenAI Responses API SSE format:
        event: response.content.delta
        data: {"delta": {"text": "Hello"}}
        """
        line_str = line.decode("utf-8").strip()

        if line_str.startswith("data: "):
            data_str = line_str[6:]  # Remove "data: " prefix

            # Skip [DONE] marker
            if data_str == "[DONE]":
                return None

            try:
                return json.loads(data_str)
            except json.JSONDecodeError:
                return None

        return None

    def _handle_responses_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert OpenAI Responses API event to OpenAI Chat Completions chunk.

        Responses API events (simplified - will need to handle full format):
        - response.created: Initial response metadata
        - response.content.delta: Incremental content
        - response.done: End of stream
        """
        event_type = event.get("type")

        if event_type == "response.created":
            # First chunk with role
            return self._build_completion_chunk(
                delta={"role": "assistant", "content": ""}
            )

        elif event_type == "response.content.delta":
            # Text content delta
            delta = event.get("delta", {})
            text = delta.get("text", "")

            if text:
                return self._build_completion_chunk(
                    delta={"content": text}
                )

        elif event_type == "response.done":
            # End of stream
            return self._build_completion_chunk(finish_reason="stop")

        # Try to handle generic delta format (simplified approach for initial version)
        if "delta" in event:
            delta = event["delta"]

            # Check for text content
            if "text" in delta:
                return self._build_completion_chunk(
                    delta={"content": delta["text"]}
                )

        return None

    def adapt(self, upstream_resp: Any) -> Response:
        """Convert Responses API streaming response to OpenAI SSE format.

        Args:
            upstream_resp: requests.Response from Responses API

        Returns:
            Flask Response with OpenAI-compatible SSE stream
        """

        @stream_with_context
        def generate() -> Iterable[bytes]:
            self._chat_completion_id = self._create_chat_completion_id()

            completion_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": "",
            }

            events = 0
            buffer = b""

            with Live(None, console=console, refresh_per_second=2) as live:
                try:
                    for chunk in upstream_resp.iter_content(chunk_size=8192):
                        buffer += chunk

                        # Process complete lines
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)

                            if not line.strip():
                                continue

                            event_data = self._parse_sse_line(line)
                            if not event_data:
                                continue

                            if current_app.config.get("LOG_COMPLETION"):
                                if events > 1:
                                    live.update(create_message_panel(completion_msg, 1, 1))
                                events += 1

                            chunk_dict = self._handle_responses_event(event_data)
                            if chunk_dict:
                                # Yield as SSE
                                yield f"data: {json.dumps(chunk_dict)}\n\n".encode("utf-8")

                                # Update completion message for logging
                                if current_app.config.get("LOG_COMPLETION"):
                                    delta = chunk_dict.get("choices", [{}])[0].get("delta", {})
                                    if "content" in delta:
                                        completion_msg["content"] += delta["content"]

                    # Send final chunk if no finish_reason was sent
                    if events > 0:
                        final_chunk = self._build_completion_chunk(finish_reason="stop")
                        yield f"data: {json.dumps(final_chunk)}\n\n".encode("utf-8")

                    yield b"data: [DONE]\n\n"

                    if current_app.config.get("LOG_COMPLETION"):
                        live.update(create_message_panel(completion_msg, 1, 1))

                except GeneratorExit:
                    raise ClientClosedConnection(
                        "Client closed connection during streaming response"
                    ) from None
                finally:
                    upstream_resp.close()

        headers = {
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }

        return Response(
            generate(),
            status=getattr(upstream_resp, "status_code", 200),
            headers=headers,
        )
