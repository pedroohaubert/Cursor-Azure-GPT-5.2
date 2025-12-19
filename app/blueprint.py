"""Flask blueprint and request routing for the proxy service.

This module defines the application blueprint, configures logging, and
forwards incoming HTTP requests to the configured backend implementation.
"""

from flask import Blueprint, current_app, jsonify, request

from .adapters.factory import AdapterFactory
from .auth import require_auth
from .common.logging import log_request
from .common.recording import (
    increment_last_recording,
    init_last_recording,
    record_payload,
)
from .exceptions import ConfigurationError, ModelNotFoundError
from .registry.registry import ModelRegistry

# Global registry (initialized in app factory)
_registry: ModelRegistry = None


def init_registry(config_path: str) -> None:
    """Initialize the global model registry.

    Args:
        config_path: Path to models.yaml configuration file
    """
    global _registry
    _registry = ModelRegistry(config_path)


def get_registry() -> ModelRegistry:
    """Get the initialized model registry.

    Returns:
        ModelRegistry instance

    Raises:
        RuntimeError: If registry not initialized
    """
    if _registry is None:
        raise RuntimeError("Model registry not initialized. Call init_registry() first.")
    return _registry


blueprint = Blueprint("blueprint", __name__)

ALL_METHODS = [
    "GET",
    "POST",
    "PUT",
    "PATCH",
    "DELETE",
    "OPTIONS",
    "HEAD",
    "TRACE",
]


@blueprint.route("/health", methods=["GET"])
def health():
    """Return a simple health check payload."""
    return jsonify({"status": "ok"})


@blueprint.route("/", defaults={"path": ""}, methods=ALL_METHODS)
@blueprint.route("/<path:path>", methods=ALL_METHODS)
@require_auth
def catch_all(path: str):
    """Forward any request path to the appropriate backend.

    Logs the incoming request, determines the model from the request,
    looks up the configuration, creates the appropriate adapter,
    and forwards to the backend. Returns a 400 error if model is not configured.
    """
    if current_app.config.get("LOG_CONTEXT"):
        log_request(request)
    init_last_recording()
    increment_last_recording()

    # Extract model name from request
    payload = request.get_json(silent=True, force=False)
    model_name = payload.get("model") if isinstance(payload, dict) else None

    if not model_name:
        return jsonify({
            "error": {
                "message": "Missing 'model' field in request",
                "type": "invalid_request_error",
                "param": "model",
                "code": None
            }
        }), 400

    # Get model configuration
    try:
        registry = get_registry()
        model_config = registry.get_model_config(model_name)
    except ModelNotFoundError as e:
        return jsonify({
            "error": {
                "message": str(e),
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found"
            }
        }), 400

    record_payload(request.json, "downstream_request")

    # Create appropriate adapter and forward
    adapter = AdapterFactory.create_adapter(model_config)
    return adapter.forward(request)


@blueprint.route("/models", methods=["GET"])
@blueprint.route("/v1/models", methods=["GET"])
@require_auth
def models():
    """Return a list of available models from registry."""
    registry = get_registry()
    model_list = registry.list_models()

    return jsonify(
        {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": 1686935002,
                    "owned_by": "system",
                }
                for model_name in model_list
            ],
        }
    )


@blueprint.errorhandler(ConfigurationError)
def configuration_error(e: ConfigurationError):
    """Return a 400 JSON error payload for ValueError."""
    return e.get_response_content(), 400
