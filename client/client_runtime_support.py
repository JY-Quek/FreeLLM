"""Client bootstrap helpers for FreeLLM."""

import json
import logging
import os
import subprocess
import sys
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path

import config


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _resource_path(name: str) -> Path:
    if _is_frozen():
        base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    else:
        base = Path(__file__).resolve().parent.parent
    return base / name


def _windows_subprocess_kwargs() -> dict:
    if sys.platform != "win32":
        return {}

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0
    return {
        "creationflags": subprocess.CREATE_NO_WINDOW,
        "startupinfo": startupinfo,
    }


def _ensure_std_streams() -> None:
    devnull = open(os.devnull, "w", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = devnull
    if sys.stderr is None:
        sys.stderr = devnull


_ensure_std_streams()


def _get_log_dir() -> Path:
    if _is_frozen():
        base = os.environ.get("APPDATA") or os.path.expanduser("~/.freellm")
        directory = Path(base) / "FreeLLM"
    else:
        directory = Path(__file__).resolve().parent
    directory.mkdir(parents=True, exist_ok=True)
    return directory


LOGO_FILE = _resource_path("FreeLLM-logo-dark-transparent.png")
LOG_DIR = _get_log_dir()
LOG_FILE = LOG_DIR / "client.log"
SETTINGS_FILE = LOG_DIR / "settings.json"
RUNTIME_STATE_FILE = LOG_DIR / "runtime.json"


def _setup_logging() -> None:
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)

    handlers: list[logging.Handler] = [file_handler]
    if not _is_frozen() and sys.stdout is not None:
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        handlers.append(stream)

    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)


_setup_logging()
log = logging.getLogger("freellm.client")
log.info("Logging to %s", LOG_FILE)


def load_settings() -> dict:
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        log.warning("Ignoring unreadable settings file %s: %s", SETTINGS_FILE, exc)
        return {}


def save_setting(key: str, value) -> None:
    settings = load_settings()
    settings[key] = value
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as fh:
            json.dump(settings, fh, indent=2)
    except OSError as exc:
        log.error("Could not save settings to %s: %s", SETTINGS_FILE, exc)


def load_or_create_client_id() -> str:
    settings = load_settings()
    saved_client_id = settings.get("client_id")
    if isinstance(saved_client_id, str) and saved_client_id.strip():
        return saved_client_id.strip()

    client_id = str(uuid.uuid4())[:8]
    save_setting("client_id", client_id)
    return client_id


def _save_runtime_state(engine: str, pid: int) -> None:
    try:
        with open(RUNTIME_STATE_FILE, "w", encoding="utf-8") as fh:
            json.dump({"engine": config.normalize_engine(engine), "pid": pid}, fh)
    except OSError as exc:
        log.warning("Could not save runtime state to %s: %s", RUNTIME_STATE_FILE, exc)


def _load_runtime_state() -> dict:
    try:
        with open(RUNTIME_STATE_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        log.warning("Ignoring unreadable runtime state file %s: %s", RUNTIME_STATE_FILE, exc)
        return {}


def _clear_runtime_state() -> None:
    try:
        RUNTIME_STATE_FILE.unlink(missing_ok=True)
    except OSError as exc:
        log.warning("Could not clear runtime state file %s: %s", RUNTIME_STATE_FILE, exc)


def _owned_runtime_pid(engine: str | None = None) -> int | None:
    data = _load_runtime_state()
    saved_engine = data.get("engine")
    saved_pid = data.get("pid")
    if engine is not None and config.normalize_engine(saved_engine) != config.normalize_engine(engine):
        return None
    if isinstance(saved_pid, int) and saved_pid > 0:
        return saved_pid
    return None