"""Base adapter interface for all backend adapters."""
from abc import ABC, abstractmethod
from typing import Any, Dict
from flask import Request, Response

from ..registry.model_config import ModelConfig


class BaseAdapter(ABC):
    """Abstract base class for backend adapters.

    All backend adapters (Azure, Anthropic, etc.) must implement this interface.
    """

    def __init__(self, model_config: ModelConfig):
        """Initialize adapter with model configuration.

        Args:
            model_config: Configuration for the model this adapter handles
        """
        self.model_config = model_config
        self.inbound_model: str = model_config.name

    @abstractmethod
    def forward(self, req: Request) -> Response:
        """Forward the Flask request to backend and return adapted response.

        This is the main entry point. Implementations should:
        1. Adapt the OpenAI request format to backend format
        2. Call the backend API
        3. Adapt the backend response to OpenAI format
        4. Return Flask Response with SSE stream

        Args:
            req: Flask request object with OpenAI Chat Completions format

        Returns:
            Flask Response with OpenAI-compatible SSE stream
        """
        pass

    @abstractmethod
    def adapt_request(self, req: Request) -> Dict[str, Any]:
        """Adapt OpenAI request to backend-specific format.

        Args:
            req: Flask request with OpenAI format

        Returns:
            Dict suitable for requests.request(**kwargs)
        """
        pass

    @abstractmethod
    def adapt_response(self, backend_response: Any) -> Response:
        """Adapt backend response to OpenAI format.

        Args:
            backend_response: Response from backend API (requests.Response)

        Returns:
            Flask Response with OpenAI Chat Completions chunks
        """
        pass
