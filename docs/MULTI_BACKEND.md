# Multi-Backend Router

This proxy now supports multiple AI backends through a registry-based routing system.

## How It Works

1. **Model Registry**: `app/models.yaml` defines all available models
2. **Backend Routing**: Client requests specify `model` name, proxy routes to correct backend
3. **Format Conversion**: Each backend adapter converts OpenAI ↔ backend format
4. **Unified Interface**: Client always uses OpenAI Chat Completions format

## Supported Backends

### Azure OpenAI Responses API
- Models: `gpt-high`, `gpt-medium`, `gpt-low`, `gpt-minimal`
- Reasoning models with configurable effort levels
- Requires: `AZURE_API_KEY`, `AZURE_BASE_URL`, `AZURE_DEPLOYMENT`

### Anthropic Messages API
- Models: `claude-sonnet-4-5`, `claude-opus-4-5`, `claude-haiku-4-5`
- High-quality reasoning and coding models
- Requires: `ANTHROPIC_API_KEY`

## Configuration

### Adding a New Model

Edit `app/models.yaml`:

```yaml
models:
  my-new-model:
    backend: anthropic  # or azure
    api_model: claude-sonnet-4.5-20250514
    max_tokens: 8192
```

### Model Configuration Options

**Azure models:**
- `backend`: "azure"
- `api_model`: Azure model name (e.g., "gpt-5")
- `reasoning_effort`: "minimal" | "low" | "medium" | "high"
- `deployment_name`: (optional) Azure deployment name
- `summary_level`: (optional) "auto" | "detailed" | "concise"
- `verbosity_level`: (optional) "low" | "medium" | "high"
- `truncation_strategy`: (optional) "auto" | "disabled"

**Anthropic models:**
- `backend`: "anthropic"
- `api_model`: Anthropic model ID (e.g., "claude-sonnet-4.5-20250514")
- `max_tokens`: (optional) Default max tokens (default: 8192)

## Environment Variables

```bash
# Azure Backend
AZURE_API_KEY=your-azure-key
AZURE_BASE_URL=https://your-resource.openai.azure.com
AZURE_DEPLOYMENT=gpt-5
AZURE_API_VERSION=2025-04-01-preview

# Anthropic Backend
ANTHROPIC_API_KEY=your-anthropic-key

# Model Registry
MODEL_CONFIG_PATH=app/models.yaml  # optional, defaults to app/models.yaml
```

## Usage Examples

### Using Azure Model

```bash
curl http://localhost:5000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $SERVICE_API_KEY" \
  -d '{
    "model": "gpt-high",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Using Anthropic Model

```bash
curl http://localhost:5000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $SERVICE_API_KEY" \
  -d '{
    "model": "claude-sonnet-4-5",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Listing Available Models

```bash
curl http://localhost:5000/models \
  -H "Authorization: Bearer $SERVICE_API_KEY"
```

Returns:
```json
{
  "object": "list",
  "data": [
    {"id": "gpt-high", "object": "model", ...},
    {"id": "claude-sonnet-4-5", "object": "model", ...}
  ]
}
```

## Error Handling

**Model not configured:**
```json
{
  "error": {
    "message": "Model 'unknown-model' is not configured. Available models: gpt-high, claude-sonnet-4-5, ...",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

**Missing API key:**
```json
{
  "error": {
    "message": "ANTHROPIC_API_KEY not set in environment",
    "type": "configuration_error"
  }
}
```

## Architecture

```
Client Request (OpenAI format)
    ↓
ModelRegistry.get_model_config(model_name)
    ↓
AdapterFactory.create_adapter(config)
    ↓
├─ AzureAdapter → Azure Responses API
└─ AnthropicAdapter → Anthropic Messages API
    ↓
Response (OpenAI format SSE stream)
```

## Adding a New Backend

To add a new backend (e.g., Google Gemini):

1. Create `app/gemini/adapter.py` implementing `BaseAdapter`
2. Implement `adapt_request()` and `adapt_response()` methods
3. Add backend to `AdapterFactory.create_adapter()`:
   ```python
   elif backend == "gemini":
       from ..gemini.adapter import GeminiAdapter
       return GeminiAdapter(model_config)
   ```
4. Add models to `app/models.yaml`:
   ```yaml
   gemini-pro:
     backend: gemini
     api_model: gemini-pro
   ```
5. Add required env vars to `app/settings.py`

## Testing

Run tests for all backends:
```bash
pytest tests/ -v
```

Test specific backend:
```bash
pytest tests/test_anthropic_adapter.py -v
pytest tests/test_azure_errors.py -v
```
