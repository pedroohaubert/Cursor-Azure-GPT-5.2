# Setting up Responses API with Claude Models (Production)

## The Problem

The OpenAI Responses API with Claude models requires **Azure AD authentication**, not simple API keys. This means you need to set up a Service Principal for production Docker deployments.

## Two Options

### Option 1: Use Messages API (Simple - Works Now)

In `app/models.yaml`, set:
```yaml
claude-sonnet-4-5:
  backend: anthropic
  api_format: messages  # Use Messages API instead of Responses API
  base_url: https://cyrela-ia-foundry.openai.azure.com/anthropic
  max_tokens: 64000
  thinking_budget: 32000
```

**Pros:**
- ✅ Works with your existing ANTHROPIC_API_KEY
- ✅ No Azure AD setup needed
- ✅ Simple configuration

**Cons:**
- ❌ Thinking disabled in multi-turn conversations with tools
- ❌ Some limitations with extended thinking

### Option 2: Use Responses API (Better - Requires Setup)

Requires creating an Azure Service Principal and configuring environment variables.

## Creating a Service Principal

### Step 1: Create Service Principal in Azure Portal

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to **Azure Active Directory** → **App registrations** → **New registration**
3. Name it (e.g., "Claude Proxy Service Principal")
4. Click **Register**

### Step 2: Create Client Secret

1. In your app registration, go to **Certificates & secrets**
2. Click **New client secret**
3. Add description (e.g., "Production Docker")
4. Set expiration (recommended: 24 months)
5. Click **Add**
6. **COPY THE SECRET VALUE NOW** (you can't see it again!)

### Step 3: Get Required IDs

From your app registration overview page, copy:
- **Application (client) ID**
- **Directory (tenant) ID**

### Step 4: Grant Permissions to Foundry Project

1. Go to your **AI Foundry project** in Azure Portal
2. Navigate to **Access control (IAM)**
3. Click **Add** → **Add role assignment**
4. Select role: **Cognitive Services OpenAI User** or **Cognitive Services User**
5. In the **Members** tab, search for your Service Principal name
6. Select it and click **Review + assign**

### Step 5: Configure Environment Variables

Add these to your `.env` file or EasyPanel environment variables:

```bash
# Azure Service Principal for Responses API
AZURE_CLIENT_ID=<your-application-client-id>
AZURE_CLIENT_SECRET=<your-client-secret-value>
AZURE_TENANT_ID=<your-directory-tenant-id>
```

### Step 6: Update models.yaml

```yaml
claude-sonnet-4-5:
  backend: anthropic
  api_format: responses  # Use Responses API
  base_url: https://cyrela-ia-foundry.services.ai.azure.com/api/projects/cyrela-hub-chat/openai
  max_tokens: 64000
```

### Step 7: Rebuild and Restart

```bash
docker-compose down
docker-compose build
docker-compose up -d
```

## Verification

Check logs to confirm authentication:
```bash
docker-compose logs -f web
```

You should see:
```
[Claude Responses API] Using Service Principal authentication
[Claude Responses API] Obtained Azure AD token (expires: ...)
```

## Common Errors

### 401 Unauthorized
- **Cause**: Service Principal doesn't have permissions
- **Solution**: Grant "Cognitive Services OpenAI User" role to the SP on your Foundry project

### "audience is incorrect"
- **Cause**: Token has wrong scope
- **Solution**: Make sure the code requests scope `https://ai.azure.com/.default` (already configured)

### "AADSTS7000215: Invalid client secret"
- **Cause**: Wrong client secret or expired
- **Solution**: Create new client secret and update AZURE_CLIENT_SECRET

### "Failed to obtain Azure AD token"
- **Cause**: Missing environment variables
- **Solution**: Verify AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID are all set

## Recommendation

**For Production (Docker/EasyPanel):**
- Use **Responses API** with Service Principal (better thinking + tools support)
- Follow the setup above

**For Quick Testing:**
- Use **Messages API** with API key (simpler, works now)
- Switch to Responses API later if needed

## Security Notes

- 🔒 **Never commit** AZURE_CLIENT_SECRET to git
- 🔒 Use EasyPanel's secret management for production
- 🔒 Rotate client secrets regularly (every 6-12 months)
- 🔒 Use minimum required permissions (Cognitive Services User is enough)
