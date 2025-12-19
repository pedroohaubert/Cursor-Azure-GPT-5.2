"""Model configuration dataclass."""
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    name: str
    backend: str  # "azure" or "anthropic"
    api_model: str

    # Azure-specific
    reasoning_effort: Optional[str] = None
    deployment_name: Optional[str] = None
    summary_level: Optional[str] = None
    verbosity_level: Optional[str] = None
    truncation_strategy: Optional[str] = None

    # Anthropic-specific
    max_tokens: Optional[int] = None
    base_url: Optional[str] = None  # For Azure Foundry: https://xxx.openai.azure.com/anthropic
    thinking_budget: Optional[int] = None  # Extended thinking budget in tokens
    api_format: Optional[str] = None  # "messages" (default) or "responses" for OpenAI Responses API

    # Common
    extra: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.backend not in {"azure", "anthropic"}:
            raise ValueError(f"Unsupported backend: {self.backend}")
