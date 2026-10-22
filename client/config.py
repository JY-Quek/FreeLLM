"""Client configuration."""

import os

# Client release version, reported to the server and shown in the dashboard.
CLIENT_VERSION = "1.0.0"

# Default engine used by the client installer / launcher.
DEFAULT_ENGINE = os.getenv("FREELLM_ENGINE", "ollama").strip().lower()

# Default FreeLLM server the client registers with. Override with --server or
# the FREELLM_SERVER_URL environment variable.
SERVER_URL = os.getenv("FREELLM_SERVER_URL", "http://127.0.0.1:8000")
SERVER_PORT = int(os.getenv("FREELLM_SERVER_PORT", "8000"))

OLLAMA_HOST = os.getenv("FREELLM_OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = int(os.getenv("FREELLM_OLLAMA_PORT", "11434"))

LEMONADE_HOST = os.getenv("FREELLM_LEMONADE_HOST", "127.0.0.1")
LEMONADE_PORT = int(os.getenv("FREELLM_LEMONADE_PORT", "13305"))

HEARTBEAT_INTERVAL = int(os.getenv("FREELLM_HEARTBEAT_INTERVAL", "10"))

INFERENCE_PORT = int(os.getenv("FREELLM_CLIENT_PORT", "8080"))
SECRET = os.getenv("FREELLM_SECRET", "").strip()

# Default reasoning behaviour for models that support it (Qwen3.x and friends).
# Callers override per-request with `think`, or bypass this entirely by sending
# `reasoning_effort` / `reasoning` / `chat_template_kwargs` themselves.
ENABLE_THINKING = os.getenv("FREELLM_ENABLE_THINKING", "0") != "0"

# Hard ceiling on generated tokens when the caller does not ask for one.
# Reasoning models otherwise keep emitting <think> until the context window is
# exhausted, which looks like a hang. 0 disables the cap.
MAX_OUTPUT_TOKENS = int(os.getenv("FREELLM_MAX_OUTPUT_TOKENS", "2048"))

# How long to wait for the local runtime to answer a chat request.
RUNTIME_TIMEOUT = int(os.getenv("FREELLM_RUNTIME_TIMEOUT", "120"))

# Also prepend Qwen3's "/no_think" control token when reasoning is disabled on
# Lemonade. Needed for chat templates that predate the `enable_thinking` kwarg.
# Set to "0" if your model's template handles the kwarg and dislikes the token.
LEMONADE_NO_THINK_PREFIX = os.getenv("FREELLM_LEMONADE_NO_THINK", "1") != "0"

SUPPORTED_ENGINES = ("ollama", "lemonade")

# Populated by the FreeLLM server during registration / heartbeat. The client
# treats the server as the source of truth for public-name -> runtime-id maps.
ENGINE_MODEL_MAPS: dict[str, dict[str, str]] = {
    "ollama": {},
    "lemonade": {},
}


def set_engine_model_maps(model_maps: dict[str, dict[str, str]] | None) -> None:
    updated = {engine: {} for engine in SUPPORTED_ENGINES}
    if isinstance(model_maps, dict):
        for engine, engine_map in model_maps.items():
            chosen = str(engine).strip().lower()
            if chosen not in updated or not isinstance(engine_map, dict):
                continue
            updated[chosen] = {
                str(public_name): str(runtime_name)
                for public_name, runtime_name in engine_map.items()
            }

    ENGINE_MODEL_MAPS.clear()
    ENGINE_MODEL_MAPS.update(updated)


def normalize_engine(engine: str | None = None) -> str:
    chosen = (engine or DEFAULT_ENGINE).strip().lower()
    return chosen if chosen in SUPPORTED_ENGINES else "ollama"


def engine_host(engine: str | None = None) -> str:
    chosen = normalize_engine(engine)
    return OLLAMA_HOST if chosen == "ollama" else LEMONADE_HOST


def engine_port(engine: str | None = None) -> int:
    chosen = normalize_engine(engine)
    return OLLAMA_PORT if chosen == "ollama" else LEMONADE_PORT


def engine_model_map(engine: str | None = None) -> dict[str, str]:
    return ENGINE_MODEL_MAPS[normalize_engine(engine)]
