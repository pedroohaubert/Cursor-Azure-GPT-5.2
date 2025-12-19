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
- **Supports Azure AI Foundry**: Use custom `base_url` to route through Azure Foundry

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
- `api_model`: Anthropic model ID (e.g., "claude-sonnet-4.5-20250514") or deployment name (for Foundry)
- `max_tokens`: (optional) Default max tokens (default: 64000)
- `thinking_budget`: (optional) Extended thinking budget in tokens (default: 32000)
- `base_url`: (optional) Custom endpoint URL (for Azure AI Foundry: `https://xxx.openai.azure.com/anthropic`)

**Example for Azure AI Foundry:**
```yaml
claude-haiku-4-5:
  backend: anthropic
  api_model: claude-haiku-4-5  # Foundry deployment name
  base_url: https://cyrela-ia-foundry.openai.azure.com/anthropic
  max_tokens: 8192
```

## Environment Variables

```bash
# Azure Backend
AZURE_API_KEY=your-azure-key
AZURE_BASE_URL=https://your-resource.openai.azure.com
AZURE_DEPLOYMENT=gpt-5
AZURE_API_VERSION=2025-04-01-preview

# Anthropic Backend (or Azure AI Foundry key)
# For Azure AI Foundry: Use your Foundry API key (configured in models.yaml base_url)
# For direct Anthropic: Use sk-ant-... key
ANTHROPIC_API_KEY=your-anthropic-or-foundry-key

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

## Extended Thinking with Anthropic Models

Claude models support **extended thinking** - enhanced reasoning capabilities where Claude shows its step-by-step thought process before delivering the final answer.

### Features Enabled

**Interleaved Thinking** (Beta: `interleaved-thinking-2025-05-14`)
- Claude can think **between tool calls**
- Reasons about tool results before deciding what to do next
- Chains multiple tool calls with reasoning steps in between
- Only supported for Claude 4 models (Opus 4.5, Opus 4.1, Opus 4, Sonnet 4)

**Current Configuration:**
- `max_tokens`: 64,000 (maximum output limit for Claude models)
- `thinking_budget`: 32,000 (32k for thinking, leaves 32k for final response)
- Interleaved thinking: **Enabled by default**

### How Thinking Appears in Responses

Claude's responses include two types of content:

1. **`delta.thinking`**: Claude's internal reasoning process (streaming)
2. **`delta.content`**: The final answer/response (streaming)

Example streaming output:
```json
// First: Thinking stream
{"delta": {"thinking": "I need to calculate 27 * 453..."}}
{"delta": {"thinking": "Let me break this down step by step..."}}
{"delta": {"thinking": "27 * 400 = 10800, 27 * 50 = 1350..."}}

// Then: Final response stream
{"delta": {"content": "27 × 453 = **12,231**"}}
```

### Thinking with Tool Use

When using tools (function calling), Claude will:
1. Think about which tool to call
2. Call the tool
3. **Think about the tool result** (interleaved thinking)
4. Decide next action or provide final answer

This creates more sophisticated multi-step reasoning workflows.

### Token Limits

- `max_tokens`: Total limit for thinking + response combined
- `thinking_budget`: Maximum tokens Claude can use for thinking
- With interleaved thinking, `thinking_budget` represents the total across all thinking blocks in one turn
- Actual thinking usage may be less than the budget

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
