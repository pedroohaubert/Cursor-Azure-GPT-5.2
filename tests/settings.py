"""Settings module for test app."""
import os

ENV = "development"
TESTING = True

SERVICE_API_KEY = "test-service-api-key"

# Model registry configuration (use the app's models.yaml for tests)
MODEL_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "models.yaml")

AZURE_API_VERSION = "2025-04-01-preview"
AZURE_BASE_URL = "https://test-resource.openai.azure.com"
AZURE_API_KEY = "test-api-key"
AZURE_DEPLOYMENT = "gpt-5"
AZURE_SUMMARY_LEVEL = "detailed"
AZURE_VERBOSITY_LEVEL = "medium"
AZURE_TRUNCATION = "disabled"

RECORD_TRAFFIC = False
LOG_CONTEXT = True
LOG_COMPLETION = True


AZURE_RESPONSES_API_URL = (
    f"{AZURE_BASE_URL}/openai/responses?api-version={AZURE_API_VERSION}"
)

# Anthropic configuration
ANTHROPIC_API_KEY = "test-anthropic-key"
