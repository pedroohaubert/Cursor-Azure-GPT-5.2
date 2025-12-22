"""Response adapter for Kimi models."""
from flask import Response, stream_with_context, current_app
from ..common.recording import record_payload


class KimiResponseAdapter:
    """Adapt Kimi streaming response to OpenAI format.

    Since Kimi already returns OpenAI Chat Completions format,
    this adapter is a simple pass-through with logging.
    """

    def __init__(self, adapter):
        """Initialize with reference to parent KimiAdapter."""
        self.adapter = adapter

    def adapt(self, backend_response) -> Response:
        """Pass through Kimi streaming response.

        Args:
            backend_response: requests.Response from Kimi API

        Returns:
            Flask Response with OpenAI Chat Completions SSE stream
        """
        def generate():
            """Stream SSE chunks from Kimi API."""
            for line in backend_response.iter_lines():
                if not line:
                    continue

                # Log chunks if enabled
                if current_app.config.get("LOG_COMPLETION"):
                    current_app.logger.debug(f"[Kimi] {line.decode('utf-8')}")

                # Pass through the SSE line
                yield line + b'\n'

        return Response(
            stream_with_context(generate()),
            status=200,
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
        )
