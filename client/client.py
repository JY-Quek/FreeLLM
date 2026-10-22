"""FreeLLM Client — manages a local inference runtime and registers with the central server."""

import argparse
import asyncio
import json
import os
import platform
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote, urlparse

import httpx
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

import config
from client_runtime_support import (
    LOG_DIR,
    LOG_FILE,
    LOGO_FILE,
    RUNTIME_STATE_FILE,
    SETTINGS_FILE,
    _clear_runtime_state,
    _get_log_dir,
    _owned_runtime_pid,
    _resource_path,
    _save_runtime_state,
    _setup_logging,
    _windows_subprocess_kwargs,
    load_or_create_client_id,
    load_settings,
    log,
    save_setting,
)
from request_adapter import build_request, to_public_model, to_runtime_model
from tray import TrayController


class LoadModelRequest(BaseModel):
    model_name: str


async def verify_secret(authorization: str = Header(...)):
    if authorization != f"Bearer {config.SECRET}":
        raise HTTPException(403, "Invalid secret")


CLIENT_ID = load_or_create_client_id()
HOSTNAME = socket.gethostname()
busy = False
requests_handled = 0


def _total_ram_gb() -> float:
    """Best-effort total physical RAM in GB using only the standard library."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return round(stat.ullTotalPhys / (1024 ** 3), 1)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3), 1)
    except Exception:
        return 0.0


def _collect_gpu_specs() -> dict:
    """Collect the primary GPU name and memory when available."""
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM | ConvertTo-Json -Compress",
                ],
                capture_output=True,
                text=True,
                timeout=4,
                check=False,
                **_windows_subprocess_kwargs(),
            )
            output = (result.stdout or "").strip()
            if output:
                data = json.loads(output)
                adapters = data if isinstance(data, list) else [data]
                adapters = [adapter for adapter in adapters if isinstance(adapter, dict)]
                adapters.sort(
                    key=lambda adapter: 0 if str(adapter.get("Name", "")).lower().startswith("microsoft basic") else 1,
                )
                adapter = adapters[-1] if adapters else None
                if adapter:
                    gpu_name = str(adapter.get("Name") or "").strip()
                    gpu_memory_bytes = float(adapter.get("AdapterRAM") or 0)
                    payload = {}
                    if gpu_name:
                        payload["gpu_name"] = gpu_name
                    if gpu_memory_bytes > 0:
                        gpu_memory_mb = gpu_memory_bytes / (1024 ** 2)
                        payload["gpu_memory_mb"] = round(gpu_memory_mb)
                        payload["gpu_vram_gb"] = round(gpu_memory_mb / 1024, 1)
                    if payload:
                        return payload
        except Exception:
            pass

    return {}


def _collect_specs() -> dict:
    """Collect static machine specifications once at startup."""
    try:
        specs = {
            "os": f"{platform.system()} {platform.release()}".strip(),
            "os_version": platform.version(),
            "arch": platform.machine(),
            "processor": platform.processor() or platform.machine(),
            "cpu_cores": os.cpu_count() or 0,
            "ram_gb": _total_ram_gb(),
            "client_version": config.CLIENT_VERSION,
        }
        specs.update(_collect_gpu_specs())
        return specs
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("Could not collect specs: %s", exc)
        return {}


def _registration_payload(advertise_url: str) -> dict:
    specs = dict(SPECS)
    specs["runtime_engine"] = runtime_engine
    return {
        "id": CLIENT_ID,
        "url": advertise_url,
        "model": current_model,
        "engine": runtime_engine,
        "busy": busy,
        "hostname": HOSTNAME,
        "specs": specs,
        "requests": requests_handled,
    }


SPECS = _collect_specs()
current_model: str = ""
runtime_engine: str = config.normalize_engine()
runtime_process: subprocess.Popen | None = None
runtime_base_url: str = ""
_state_lock = asyncio.Lock()
# Serializes runtime restarts so concurrent requests do not spawn duplicates.
_runtime_lock = asyncio.Lock()
# Event loop running the service, so the tray thread can schedule work onto it.
_main_loop: asyncio.AbstractEventLoop | None = None


def _engine_label(engine: str | None = None) -> str:
    chosen = config.normalize_engine(engine or runtime_engine)
    return "Ollama" if chosen == "ollama" else chosen.capitalize()


def _engine_model_map() -> dict[str, str]:
    return config.engine_model_map(runtime_engine)


def _update_model_maps(model_maps: dict | None) -> None:
    config.set_engine_model_maps(model_maps)


def _find_engine_executable(engine: str) -> str:
    chosen = config.normalize_engine(engine)
    if chosen == "ollama":
        path = shutil.which("ollama")
        if path:
            return path
        for d in (os.environ.get("LOCALAPPDATA", ""), os.environ.get("PROGRAMFILES", "")):
            p = os.path.join(d, "Programs", "Ollama", "ollama.exe") if d else ""
            if p and os.path.isfile(p):
                return p
        raise FileNotFoundError("Ollama executable not found. Please install Ollama first.")

    # Only lemond (headless server) and LemonadeServer (Windows tray app with an
    # embedded server) actually host the HTTP API. `lemonade` is a thin CLI
    # client that talks to an already-running server, so it must never be used
    # to start one.
    for name in ("lemond", "LemonadeServer"):
        path = shutil.which(name)
        if path:
            return path

    # MSI install roots: per-user is the default, all-users requires elevation.
    roots = [
        os.path.join(d, sub)
        for d, sub in (
            (os.environ.get("LOCALAPPDATA", ""), "lemonade_server"),
            (os.environ.get("PROGRAMFILES", ""), "Lemonade Server"),
        )
        if d
    ]
    for base in roots:
        for name in ("lemond.exe", "LemonadeServer.exe"):
            candidate = os.path.join(base, "bin", name)
            if os.path.isfile(candidate):
                return candidate
    raise FileNotFoundError(
        "Lemonade server executable (lemond/LemonadeServer) not found. "
        "Please install Lemonade first."
    )


def _is_engine_installed(engine: str) -> bool:
    try:
        _find_engine_executable(engine)
    except FileNotFoundError:
        return False
    return True


def _installed_engines() -> list[str]:
    return [
        engine for engine in sorted(config.SUPPORTED_ENGINES) if _is_engine_installed(engine)
    ]


def _find_runtime_executable() -> str:
    return _find_engine_executable(runtime_engine)


def _runtime_endpoints() -> list[tuple[str, int]]:
    endpoints: list[tuple[str, int]] = []
    if runtime_base_url:
        parsed = urlparse(runtime_base_url)
        if parsed.hostname and parsed.port:
            endpoints.append((parsed.hostname, parsed.port))

    default_endpoint = (config.engine_host(runtime_engine), config.engine_port(runtime_engine))
    if default_endpoint not in endpoints:
        endpoints.append(default_endpoint)
    return endpoints


def start_runtime(host: str, port: int) -> subprocess.Popen:
    runtime_exe = _find_runtime_executable()
    log.info("Found %s at %s", _engine_label(), runtime_exe)

    env = os.environ.copy()
    if runtime_engine == "ollama":
        # Ollama reads its bind address from the environment.
        env["OLLAMA_HOST"] = f"{host}:{port}"
        cmd = [runtime_exe, "serve"]
    else:
        # lemond/LemonadeServer have no "serve" subcommand; a positional
        # argument would be read as their cache directory. Bind via flags.
        cmd = [runtime_exe, "--port", str(port), "--host", host]

    # Redirect runtime output to a log file. Using subprocess.PIPE without
    # continuously draining it deadlocks once the OS pipe buffer fills up.
    runtime_log_path = LOG_DIR / f"{runtime_engine}.log"
    runtime_log = open(runtime_log_path, "a", encoding="utf-8", buffering=1)

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=runtime_log,
        stderr=subprocess.STDOUT,
        **_windows_subprocess_kwargs(),
    )
    log.info(
        "%s server started (PID %d) on %s:%d — logging to %s",
        _engine_label(), proc.pid, host, port, runtime_log_path,
    )
    _save_runtime_state(runtime_engine, proc.pid)
    return proc


def _stop_runtime() -> None:
    """Stop the runtime, but only if this client is the one that started it."""
    global runtime_process
    proc = runtime_process
    runtime_process = None
    pid = proc.pid if proc is not None else _owned_runtime_pid(runtime_engine)
    if pid is None:
        return
    _clear_runtime_state()
    log.info("Stopping %s server (PID %d)", _engine_label(), pid)
    if sys.platform == "win32":
        # Ollama can leave child processes behind on Windows when only the
        # parent process receives terminate(). Kill the full tree instead.
        result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            **_windows_subprocess_kwargs(),
        )
        if result.returncode not in (0, 128):
            log.warning(
                "taskkill failed for %s server (PID %d): %s",
                _engine_label(),
                pid,
                (result.stderr or result.stdout).strip() or f"exit {result.returncode}",
            )
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        return

    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


async def wait_for_runtime(base_url: str, timeout: float = 30.0):
    async with httpx.AsyncClient() as http:
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            try:
                resp = await http.get(f"{base_url}/api/tags", timeout=3)
                if resp.status_code == 200:
                    log.info("%s API is ready", _engine_label())
                    return
            except httpx.RequestError:
                pass
            await asyncio.sleep(1)
    raise TimeoutError(f"{_engine_label()} did not become ready in time")


def _is_runtime_running(host: str, port: int) -> bool:
    """Check whether the selected runtime is already responding on host:port."""
    try:
        resp = httpx.get(f"http://{host}:{port}/api/tags", timeout=2)
        return resp.status_code == 200
    except httpx.RequestError:
        return False


async def _bring_up_runtime(engine: str) -> None:
    """Reuse or launch `engine`, then block until its API answers.

    Sets the module-level runtime globals on success.
    """
    global runtime_process, runtime_base_url

    host = config.engine_host(engine)
    port = config.engine_port(engine)
    if await asyncio.to_thread(_is_runtime_running, host, port):
        log.info("Using existing %s at http://%s:%d", _engine_label(engine), host, port)
    else:
        port = await asyncio.to_thread(find_free_port, port, host)
        runtime_process = start_runtime(host, port)

    runtime_base_url = f"http://{host}:{port}"
    await wait_for_runtime(
        runtime_base_url, timeout=60.0 if engine == "lemonade" else 30.0
    )


async def _ensure_runtime_available() -> bool:
    """Restart the selected runtime if it has been killed or stopped responding."""
    global runtime_process, runtime_base_url

    async def _alive() -> tuple[str, int] | None:
        for host, port in _runtime_endpoints():
            if await asyncio.to_thread(_is_runtime_running, host, port):
                return host, port
        return None

    live = await _alive()
    if live is not None:
        runtime_base_url = f"http://{live[0]}:{live[1]}"
        if runtime_process is not None and runtime_process.poll() is not None:
            runtime_process = None
        return False

    async with _runtime_lock:
        live = await _alive()
        if live is not None:
            runtime_base_url = f"http://{live[0]}:{live[1]}"
            if runtime_process is not None and runtime_process.poll() is not None:
                runtime_process = None
            return False

        if runtime_process is not None and runtime_process.poll() is not None:
            runtime_process = None

        log.warning("%s is not responding; restarting it", _engine_label())
        await _bring_up_runtime(runtime_engine)
        return True


async def switch_engine(engine: str) -> None:
    """Swap the inference engine at runtime and remember the choice."""
    global busy, runtime_engine, current_model

    engine = config.normalize_engine(engine)
    if engine == runtime_engine:
        return

    async with _state_lock:
        if busy:
            log.warning("Cannot switch to %s while the client is busy", _engine_label(engine))
            return
        busy = True

    previous_engine = runtime_engine
    if tray is not None:
        tray.set_status("switching")
    try:
        log.info("Switching engine: %s -> %s", _engine_label(previous_engine), _engine_label(engine))
        _stop_runtime()
        runtime_engine = engine
        await _bring_up_runtime(engine)

        # current_model holds the public name, so it carries across engines.
        if current_model:
            await pull_model(runtime_base_url, current_model)

        save_setting("engine", engine)
        log.info("Now serving with %s at %s", _engine_label(), runtime_base_url)
        if tray is not None:
            tray.set_status("running")
    except Exception as exc:
        log.error("Could not switch to %s: %s", _engine_label(engine), exc)
        # The previous runtime was already stopped, so fall back to it rather
        # than leaving the client with no engine at all.
        _stop_runtime()
        runtime_engine = previous_engine
        try:
            await _bring_up_runtime(previous_engine)
            log.info("Kept %s at %s", _engine_label(), runtime_base_url)
            if tray is not None:
                tray.set_status("error", f"{_engine_label(engine)} unavailable")
        except Exception as restore_exc:
            log.error("Could not restore %s: %s", _engine_label(), restore_exc)
            if tray is not None:
                tray.set_status("error", "no inference engine available")
    finally:
        async with _state_lock:
            busy = False


def request_engine_change(engine: str) -> None:
    """Entry point for the tray thread: schedule a switch on the service loop."""
    loop = _main_loop
    if loop is None:
        log.warning("Ignoring engine change: the service is not running yet")
        return
    asyncio.run_coroutine_threadsafe(switch_engine(engine), loop)


async def ensure_model_loaded(base_url: str, model: str):
    """Ask the local runtime to materialize a pulled model into memory."""
    model = to_runtime_model(model, _engine_model_map())
    log.info("Loading model '%s' into %s memory", model, _engine_label())
    async with httpx.AsyncClient(timeout=config.RUNTIME_TIMEOUT) as http:
        resp = await http.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": "", "stream": False},
        )
        resp.raise_for_status()
    log.info("Model '%s' loaded into %s memory", model, _engine_label())


async def pull_model(base_url: str, model: str):
    model = to_runtime_model(model, _engine_model_map())
    model_available = False
    async with httpx.AsyncClient(timeout=600) as http:
        # Check if model already exists
        resp = await http.get(f"{base_url}/api/tags")
        if resp.status_code == 200:
            tags = resp.json()
            existing = [m["name"] for m in tags.get("models", [])]
            # Both runtimes expose model tags; match with or without :latest.
            model_variants = {model, model + ":latest"}
            if model_variants & set(existing):
                log.info("Model '%s' already available", model)
                model_available = True
        if not model_available:
            log.info("Pulling model '%s' (this may take a while)...", model)
            resp = await http.post(
                f"{base_url}/api/pull",
                json={"name": model, "stream": False},
                timeout=1800,
            )
            resp.raise_for_status()
            log.info("Model '%s' pulled successfully", model)
    await ensure_model_loaded(base_url, model)


async def fetch_model_from_server(server_url: str, advertise_url: str) -> str:
    async with httpx.AsyncClient(verify=False) as http:
        resp = await http.post(
            f"{server_url}/register",
            json=_registration_payload(advertise_url),
            headers={"Authorization": f"Bearer {config.SECRET}"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        _update_model_maps(data.get("model_maps"))
        return data.get("model", "")


# Singleton tray controller (None when no tray is in use).
tray: TrayController | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runtime_process, runtime_base_url, current_model, _main_loop

    _main_loop = asyncio.get_running_loop()
    runtime_host = app.state.runtime_host
    runtime_port = app.state.runtime_port
    runtime_base_url = f"http://{runtime_host}:{runtime_port}"

    try:
        if app.state.runtime_external:
            log.info("Using existing %s at %s", _engine_label(), runtime_base_url)
        else:
            runtime_process = start_runtime(runtime_host, runtime_port)

        try:
            await wait_for_runtime(
                runtime_base_url,
                timeout=60.0 if runtime_engine == "lemonade" else 30.0,
            )
        except TimeoutError:
            log.error("%s failed to become ready", _engine_label())
            _stop_runtime()
            raise

        # Ask the server which model to use
        assigned_model = ""
        try:
            assigned_model = await fetch_model_from_server(
                app.state.server_url, app.state.advertise_url
            )
            log.info("Server assigned model: '%s'", assigned_model)
        except Exception as exc:
            log.warning("Could not fetch model from server: %s. Waiting for server assignment.", exc)

        model_to_use = assigned_model or app.state.initial_model
        if model_to_use:
            await pull_model(runtime_base_url, model_to_use)
            current_model = to_public_model(model_to_use, _engine_model_map())
        else:
            log.info("No model assigned yet. Client will wait for server to assign a model.")
    except Exception as exc:
        if tray is not None:
            tray.set_status("error", str(exc) or exc.__class__.__name__)
        raise

    # Start heartbeat
    task = asyncio.create_task(heartbeat_loop(app.state.server_url, app.state.advertise_url))
    ws_task = asyncio.create_task(ws_loop(app.state.server_url))
    if tray is not None:
        tray.set_status("running")
    yield
    ws_task.cancel()
    task.cancel()

    # Shutdown runtime if we started it ourselves.
    _stop_runtime()
    _main_loop = None

app = FastAPI(lifespan=lifespan)

_CHAT_PATH = "/v1/chat/completions"


@app.get("/health")
def health():
    return {"id": CLIENT_ID, "busy": busy, "model": current_model, "status": "ok"}


@app.post("/load_model", dependencies=[Depends(verify_secret)])
async def load_model(req: LoadModelRequest):
    global busy, current_model
    async with _state_lock:
        if busy:
            raise HTTPException(503, "Client is busy, cannot switch model")
        busy = True
    try:
        await _ensure_runtime_available()
        await pull_model(runtime_base_url, req.model_name)
        async with _state_lock:
            current_model = to_public_model(req.model_name, _engine_model_map())
        return {"status": "ok", "model": current_model}
    finally:
        async with _state_lock:
            busy = False
async def run_chat(req: dict) -> dict:
    """Run one chat request against the local runtime and return its OpenAI response."""
    payload = build_request(
        req,
        current_model=current_model,
        runtime_engine=runtime_engine,
        model_map=_engine_model_map(),
        enable_thinking_default=config.ENABLE_THINKING,
        max_output_tokens=config.MAX_OUTPUT_TOKENS,
        lemonade_no_think_prefix=config.LEMONADE_NO_THINK_PREFIX,
    )
    restarted = await _ensure_runtime_available()
    if restarted and current_model:
        await pull_model(runtime_base_url, current_model)
    log.info(
        "Sending to %s: %s%s (model=%s, reasoning_effort=%s)",
        _engine_label(), runtime_base_url, _CHAT_PATH, current_model,
        payload.get("reasoning_effort", "default"),
    )
    async with httpx.AsyncClient(timeout=config.RUNTIME_TIMEOUT) as http:
        resp = await http.post(f"{runtime_base_url}{_CHAT_PATH}", json=payload)
        resp.raise_for_status()
        data = resp.json()

    data["model"] = current_model  # report the public name, not the local model id
    return data


@app.post("/v1/chat/completions", dependencies=[Depends(verify_secret)])
async def chat_completions(req: dict):
    global busy, requests_handled
    async with _state_lock:
        if not current_model:
            raise HTTPException(503, "No model loaded yet — waiting for server assignment")
        if busy:
            raise HTTPException(503, "Client is busy")
        busy = True
    try:
        data = await run_chat(req)
        requests_handled += 1
        return data
    finally:
        async with _state_lock:
            busy = False


async def heartbeat_loop(server_url: str, advertise_url: str):
    global current_model, busy
    headers = {"Authorization": f"Bearer {config.SECRET}"}
    async with httpx.AsyncClient(verify=False) as http:
        while True:
            try:
                resp = await http.post(
                    f"{server_url}/register",
                    json=_registration_payload(advertise_url),
                    headers=headers,
                    timeout=5,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    _update_model_maps(data.get("model_maps"))
                    server_model = to_public_model(data.get("model", ""), _engine_model_map())
                    if server_model and server_model != current_model and not busy:
                        log.info("Server requests model change: '%s' -> '%s'", current_model, server_model)
                        asyncio.create_task(_switch_model(server_model))
            except Exception as exc:
                log.warning("Heartbeat failed: %s", exc)
            await asyncio.sleep(config.HEARTBEAT_INTERVAL)


async def _switch_model(model: str):
    global busy, current_model
    async with _state_lock:
        if busy:
            return
        busy = True
    try:
        await _ensure_runtime_available()
        await pull_model(runtime_base_url, model)
        async with _state_lock:
            current_model = to_public_model(model, _engine_model_map())
        log.info("Switched to model '%s'", current_model)
    except Exception as exc:
        log.error("Failed to switch model: %s", exc)
    finally:
        async with _state_lock:
            busy = False


async def ws_loop(server_url: str):
    global busy, current_model
    ws_url = server_url.replace("https://", "wss://").replace("http://", "ws://")
    # Pass the secret both as a header and as a query-string token. Reverse
    # proxies (e.g. IIS ARR) often strip the Authorization header on the
    # WebSocket upgrade, so the query parameter is the reliable channel.
    ws_url = f"{ws_url}/ws/{CLIENT_ID}?token={quote(config.SECRET, safe='')}"

    # Disable TLS verification for wss connections (self-signed / untrusted CA).
    ssl_ctx = None
    if ws_url.startswith("wss://"):
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

    while True:
        try:
            async with ws_connect(
                ws_url,
                additional_headers={"Authorization": f"Bearer {config.SECRET}"},
                ssl=ssl_ctx,
                compression=None,  # permessage-deflate is dropped by some reverse proxies (IIS/ARR) -> 1006 resets
            ) as ws:
                log.info("WebSocket connected to server at %s", ws_url)
                async for raw in ws:
                    msg = json.loads(raw)
                    request_id = msg.get("request_id")
                    msg_type = msg.get("type")
                    payload = msg.get("payload", {})

                    if msg_type == "chat":
                        asyncio.create_task(_handle_ws_chat(ws, request_id, payload))
                    elif msg_type == "load_model":
                        asyncio.create_task(_handle_ws_load_model(ws, request_id, payload))
        except ConnectionClosed as exc:
            code = getattr(exc, "code", None)
            reason = getattr(exc, "reason", "") or ""
            log.warning(
                "WebSocket closed (code=%s reason=%r). Reconnecting in 5s...",
                code, reason,
            )
        except Exception as exc:
            log.warning("WebSocket error: %s. Reconnecting in 5s...", exc)
        await asyncio.sleep(5)


async def _handle_ws_chat(ws, request_id: str, payload: dict):
    global busy, requests_handled
    log.info("WS chat received (request_id=%s, model=%s)", request_id, current_model)
    async with _state_lock:
        if not current_model:
            reject = {"type": "error", "request_id": request_id, "payload": {"detail": "No model loaded"}}
        elif busy:
            reject = {"type": "error", "request_id": request_id, "payload": {"detail": "Client is busy"}}
        else:
            reject = None
            busy = True
    if reject:
        log.warning("WS chat rejected: %s", reject["payload"]["detail"])
        await ws.send(json.dumps(reject))
        return
    try:
        data = await run_chat(payload)
        usage = data.get("usage") or {}
        tokens_used = usage.get("total_tokens") or (
            usage.get("completion_tokens", 0) + usage.get("prompt_tokens", 0)
        )
        log.info("%s responded (%d tokens)", _engine_label(), tokens_used)
        requests_handled += 1
        await ws.send(json.dumps({
            "type": "chat_response",
            "request_id": request_id,
            "payload": data,
        }))
    except Exception as exc:
        log.error("WS chat handler failed: %r", exc)
        await ws.send(json.dumps(
            {"type": "error", "request_id": request_id, "payload": {"detail": str(exc) or repr(exc)}}
        ))
    finally:
        async with _state_lock:
            busy = False


async def _handle_ws_load_model(ws, request_id: str, payload: dict):
    global busy, current_model
    async with _state_lock:
        if busy:
            reject = {"type": "error", "request_id": request_id, "payload": {"detail": "Client is busy"}}
        else:
            reject = None
            busy = True
    if reject:
        await ws.send(json.dumps(reject))
        return
    try:
        model_name = payload.get("model_name", "")
        await _ensure_runtime_available()
        await pull_model(runtime_base_url, model_name)
        async with _state_lock:
            current_model = to_public_model(model_name, _engine_model_map())
        await ws.send(json.dumps({
            "type": "load_model_response",
            "request_id": request_id,
            "payload": {"status": "ok", "model": current_model},
        }))
    except Exception as exc:
        await ws.send(json.dumps(
            {"type": "error", "request_id": request_id, "payload": {"detail": str(exc)}}
        ))
    finally:
        async with _state_lock:
            busy = False


def get_local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"

def find_free_port(start: int, host: str = "0.0.0.0") -> int:
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}-{start + 99}")


def main():
    global tray

    parser = argparse.ArgumentParser(description="FreeLLM Client")
    parser.add_argument(
        "--engine",
        choices=sorted(config.SUPPORTED_ENGINES),
        default=None,
        help="Inference engine to use (overrides the tray Settings choice)",
    )
    parser.add_argument("--port", type=int, default=config.INFERENCE_PORT)
    parser.add_argument("--server", default=config.SERVER_URL, help="FreeLLM server URL")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--secret", default=config.SECRET, help="Shared secret for server auth")
    parser.add_argument("--model", default="", help="Model to pull on startup (overridden by server assignment)")
    parser.add_argument("--runtime-host", "--ollama-host", dest="runtime_host", default="", help="Inference engine bind host")
    parser.add_argument("--runtime-port", "--ollama-port", dest="runtime_port", type=int, default=0, help="Inference engine port")
    parser.add_argument("--no-tray", action="store_true", help="Disable system-tray icon")
    args = parser.parse_args()

    global runtime_engine
    # Precedence: explicit CLI flag > choice saved from the tray > environment.
    runtime_engine = config.normalize_engine(
        args.engine or load_settings().get("engine") or config.DEFAULT_ENGINE
    )

    config.SECRET = args.secret

    if not config.SECRET:
        log.warning("FREELLM_SECRET not set — pass --secret or set it in the environment.")

    port = find_free_port(args.port, args.host)
    if port != args.port:
        log.info("Port %d busy, using %d", args.port, port)

    advertise_ip = get_local_ip()
    app.state.server_url = args.server
    app.state.advertise_url = f"http://{advertise_ip}:{port}"
    app.state.initial_model = args.model
    app.state.runtime_engine = runtime_engine
    app.state.runtime_host = args.runtime_host or config.engine_host(runtime_engine)

    runtime_port = args.runtime_port or config.engine_port(runtime_engine)
    expected_runtime_port = runtime_port

    # Reuse an existing runtime on the requested port if one is already running.
    # Lemonade auto-starts on Windows, so this is its common path.
    if _is_runtime_running(app.state.runtime_host, runtime_port):
        log.info("Existing %s detected on %s:%d — will reuse it", _engine_label(), app.state.runtime_host, runtime_port)
        app.state.runtime_external = True
    else:
        runtime_port = find_free_port(runtime_port, app.state.runtime_host)
        if runtime_port != expected_runtime_port:
            log.info("%s port %d busy, using %d", _engine_label(), expected_runtime_port, runtime_port)
        app.state.runtime_external = False
    app.state.runtime_port = runtime_port

    log.info("Client %s starting on %s:%d", CLIENT_ID, args.host, port)
    log.info("Will register with server at %s", args.server)
    log.info("%s will run on %s:%d (model assigned by server)", _engine_label(), app.state.runtime_host, runtime_port)

    uv_config = uvicorn.Config(app, host=args.host, port=port, log_level="warning")
    server = uvicorn.Server(uv_config)

    # Enable tray by default on Windows when frozen; opt-in elsewhere.
    use_tray = not args.no_tray and sys.platform == "win32"

    if not use_tray:
        server.run()
        return

    tray = TrayController(
        logo_file=LOGO_FILE,
        log_file=LOG_FILE,
        logger=log,
        installed_engines=_installed_engines,
        engine_label=_engine_label,
    )

    def _serve():
        try:
            server.run()
        except Exception as exc:
            log.exception("Server crashed: %s", exc)
            if tray is not None:
                tray.set_status("error", str(exc) or exc.__class__.__name__)

    server_thread = threading.Thread(target=_serve, name="uvicorn", daemon=True)
    server_thread.start()

    def _shutdown():
        log.info("Shutting down server…")
        server.should_exit = True
        # Force-exit if it does not stop gracefully in time
        server_thread.join(timeout=15)
        if server_thread.is_alive():
            log.warning("Server did not stop gracefully; forcing exit")
            _stop_runtime()
            os._exit(0)

    try:
        tray.run(
            _shutdown,
            on_engine_change=request_engine_change,
            engine_getter=lambda: runtime_engine,
        )
    except KeyboardInterrupt:
        _shutdown()
    finally:
        # If tray exited but server is still alive (e.g. pystray not available), wait on server.
        if server_thread.is_alive():
            server_thread.join()


if __name__ == "__main__":
    main()
