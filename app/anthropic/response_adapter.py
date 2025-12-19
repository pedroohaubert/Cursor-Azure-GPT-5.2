"""Response adaptation for Anthropic Messages API streaming."""
import json
import random
import time
from string import ascii_letters, digits
from typing import Any, Dict, Iterable, Optional

from flask import Response, current_app, stream_with_context
from rich.live import Live

from ..common.logging import console, create_message_panel
from ..exceptions import ClientClosedConnection


class AnthropicResponseAdapter:
    """Convert Anthropic Messages API streaming responses to OpenAI format."""

    def __init__(self, adapter: Any):
        """Initialize with reference to parent AnthropicAdapter."""
        self.adapter = adapter
        self._chat_completion_id: Optional[str] = None
        self._tool_calls_count: int = 0

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

        Anthropic SSE format:
        event: message_start
        data: {"type": "message_start", ...}
        """
        line_str = line.decode("utf-8").strip()

        if line_str.startswith("data: "):
            data_str = line_str[6:]  # Remove "data: " prefix
            try:
                return json.loads(data_str)
            except json.JSONDecodeError:
                return None

        return None

    def _handle_anthropic_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert Anthropic streaming event to OpenAI chunk.

        Anthropic events:
        - message_start: Initial message metadata
        - content_block_start: Start of content block (text, tool_use, or thinking)
        - content_block_delta: Incremental content
        - content_block_stop: End of content block
        - message_delta: Message-level updates
        - message_stop: End of stream
        """
        event_type = event.get("type")

        if event_type == "message_start":
            # First chunk with role
            return self._build_completion_chunk(
                delta={"role": "assistant", "content": ""}
            )

        elif event_type == "content_block_start":
            # Start of text, tool_use, or thinking block
            content_block = event.get("content_block", {})
            block_type = content_block.get("type")

            if block_type == "tool_use":
                self._tool_calls_count += 1
                return self._build_completion_chunk(
                    delta={
                        "tool_calls": [
                            {
                                "index": self._tool_calls_count - 1,
                                "id": content_block.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": content_block.get("name", ""),
                                    "arguments": ""
                                }
                            }
                        ]
                    }
                )

            elif block_type == "thinking":
                # Start of thinking block - send marker
                return self._build_completion_chunk(
                    delta={"thinking": ""}
                )

            return None

        elif event_type == "content_block_delta":
            delta = event.get("delta", {})
            delta_type = delta.get("type")

            if delta_type == "text_delta":
                # Text content
                return self._build_completion_chunk(
                    delta={"content": delta.get("text", "")}
                )

            elif delta_type == "thinking_delta":
                # Thinking content - expose as separate field
                return self._build_completion_chunk(
                    delta={"thinking": delta.get("thinking", "")}
                )

            elif delta_type == "input_json_delta":
                # Tool call arguments
                return self._build_completion_chunk(
                    delta={
                        "tool_calls": [
                            {
                                "index": self._tool_calls_count - 1,
                                "function": {
                                    "arguments": delta.get("partial_json", "")
                                }
                            }
                        ]
                    }
                )

        elif event_type == "message_delta":
            # Check for stop reason
            delta = event.get("delta", {})
            stop_reason = delta.get("stop_reason")

            if stop_reason == "end_turn":
                return self._build_completion_chunk(finish_reason="stop")
            elif stop_reason == "tool_use":
                return self._build_completion_chunk(finish_reason="tool_calls")

        return None

    def adapt(self, upstream_resp: Any) -> Response:
        """Convert Anthropic streaming response to OpenAI SSE format.

        Args:
            upstream_resp: requests.Response from Anthropic API

        Returns:
            Flask Response with OpenAI-compatible SSE stream
        """

        @stream_with_context
        def generate() -> Iterable[bytes]:
            self._chat_completion_id = self._create_chat_completion_id()
            self._tool_calls_count = 0

            completion_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": "",
                "tool_calls": [],
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

                            chunk_dict = self._handle_anthropic_event(event_data)
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
                        final_chunk = self._build_completion_chunk(
                            finish_reason="stop" if self._tool_calls_count == 0 else "tool_calls"
                        )
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
