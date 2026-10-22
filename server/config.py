"""Server configuration."""

import json
import os
import secrets
from pathlib import Path

HOST = os.getenv("FREELLM_SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("FREELLM_SERVER_PORT", "8000"))
CLIENT_TIMEOUT = int(os.getenv("FREELLM_CLIENT_TIMEOUT", "60"))
DEFAULT_MODEL = os.getenv("FREELLM_DEFAULT_MODEL", "Qwen/Qwen3.5-0.8B")
CLUSTER_NAME = os.getenv("FREELLM_CLUSTER_NAME", "Demo")

# Sub-path the app is served under when sitting behind a reverse proxy
# (e.g. "/freellm" for https://example.com/freellm). Empty = served at root.
ROOT_PATH = os.getenv("FREELLM_ROOT_PATH", "").rstrip("/")

SECRET = os.getenv("FREELLM_SECRET", "").strip()


def _load_model_maps() -> dict[str, dict[str, str]]:
    raw = os.getenv("FREELLM_MODEL_MAPS_JSON", "").strip()
    if raw:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid FREELLM_MODEL_MAPS_JSON: {exc}") from exc
        if isinstance(data, dict):
            return {
                str(engine).strip().lower(): {
                    str(public_name): str(runtime_name)
                    for public_name, runtime_name in engine_map.items()
                }
                for engine, engine_map in data.items()
                if isinstance(engine_map, dict)
            }
        raise ValueError("FREELLM_MODEL_MAPS_JSON must decode to an object")

    return {
        "ollama": {
            "Qwen/Qwen3.5-0.8B": "qwen3.5:0.8b",
            "google/gemma-4-E2B-it": "gemma4:e2b",
        },
        "lemonade": {
            "Qwen/Qwen3.5-0.8B": "Qwen3.5-0.8B-GGUF",
            "google/gemma-4-E2B-it": "Gemma-4-E2B-it-GGUF",
        },
    }


MODEL_MAPS = _load_model_maps()


def normalize_engine(engine: str | None) -> str:
    chosen = (engine or "").strip().lower()
    return chosen if chosen in MODEL_MAPS else ""


def public_models(engine: str | None = None) -> list[str]:
    chosen = normalize_engine(engine)
    if chosen:
        return sorted(MODEL_MAPS.get(chosen, {}))

    names = {
        public_name
        for engine_map in MODEL_MAPS.values()
        for public_name in engine_map
    }
    return sorted(names)

# Admin key — used to sign in to the dashboard and manage API keys.
# Falls back to the legacy FREELLM_API_KEY for backward compatibility.
ADMIN_KEY = os.getenv("FREELLM_ADMIN_KEY", "").strip()
# File where API keys generated from the dashboard are persisted.
API_KEYS_FILE = os.getenv("FREELLM_API_KEYS_FILE", str(Path(__file__).parent / "api_keys.json"))
# File where per-client assigned model overrides are persisted.
ASSIGNMENTS_FILE = os.getenv("FREELLM_ASSIGNMENTS_FILE", str(Path(__file__).parent / "assignments.json"))
