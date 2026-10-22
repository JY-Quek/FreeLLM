# FreeLLM

Imagine you and a couple of friends all have Ollama or Lemonade installed onto your PC, and you want to share the inference workload easily. FreeLLM is a lightweight self-hosted gateway that coordinates multiple inference clients for Ollama or Lemonade. It exposes a single OpenAI-style API endpoint, balances requests across available clients, and gives you a simple dashboard for managing API keys.

## Quick Start

### 1. Start the server

```bash
cd server
pip install -r requirements.txt
python server.py --secret <shared-secret>
```

If you do not pass `--admin-key`, the server generates one for the current run and prints it to the console. Open `<server_url>`, sign in with that admin key, and create an API key from the dashboard.

### 2. Start one or more clients

```bash
cd client
pip install -r requirements.txt
python client.py --server <server_url> --secret <shared-secret>
```

Use `--engine lemonade` to run Lemonade instead of Ollama. Run more than one client if you want the server to spread inference load across them.

### 3. Call the API

Requests are routed to available clients for inference.

Basic request:

```bash
curl -X POST <server_url>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <api_key>" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

Target a specific model:

```bash
curl -X POST <server_url>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <api_key>" \
  -d '{"model":"Qwen/Qwen3.5-0.8B","messages":[{"role":"user","content":"Hello!"}]}'
```

List available models:

```bash
curl <server_url>/v1/models -H "Authorization: Bearer <api_key>"
```

## Updating Model Mapping

If you want to change how public model names map to runtime model names, set `FREELLM_MODEL_MAPS_JSON` before starting the server.

Example:

```bash
export FREELLM_MODEL_MAPS_JSON='{"ollama":{"MyModel":"my-model:latest"},"lemonade":{"MyModel":"MyModel-GGUF"}}'
```

If you prefer, you can also edit the default mapping in [server/config.py](server/config.py).
