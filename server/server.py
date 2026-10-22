"""FreeLLM Server — routes inference requests to available clients."""

import argparse
import asyncio
import json
import logging
import os
import secrets
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("freellm.server")


class RegisterRequest(BaseModel):
    id: str
    url: str
    model: str = ""
    engine: str = ""
    busy: bool = False
    hostname: str = ""
    specs: dict = {}
    requests: int = 0

class AssignModelRequest(BaseModel):
    client_id: str
    model_name: str


def normalize_model(name: str) -> str:
    """Lower-case a model name and drop an implicit ':latest' tag."""
    name = (name or "").strip().lower()
    if name.endswith(":latest"):
        name = name[: -len(":latest")]
    return name


def model_matches(requested: str, served: str) -> bool:
    """True when a client running `served` can answer a `requested` model. An
    empty request matches any client, keeping the round-robin behaviour for
    callers that don't specify a model."""
    if not requested:
        return True
    return normalize_model(requested) == normalize_model(served)


class ClientPool:

    def __init__(self, assignments_path: Path | None = None):
        self._clients: dict[str, dict] = {}  # id -> {url, last_seen, model, busy}
        self._ws: dict[str, WebSocket] = {}  # id -> WebSocket
        self._assignments_path = assignments_path
        self._assigned: dict[str, str] = self._load_assignments()  # id -> assigned model override
        self._lock = asyncio.Lock()
        self._idx = 0

    def _load_assignments(self) -> dict[str, str]:
        if not self._assignments_path or not self._assignments_path.exists():
            return {}
        try:
            data = json.loads(self._assignments_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Failed to load client assignments from %s: %s", self._assignments_path, exc)
            return {}
        if not isinstance(data, dict):
            log.warning("Ignoring invalid client assignments file %s", self._assignments_path)
            return {}
        return {
            str(client_id): str(model_name)
            for client_id, model_name in data.items()
            if str(client_id).strip() and str(model_name).strip()
        }

    def _persist_assignments(self) -> None:
        if not self._assignments_path:
            return
        try:
            self._assignments_path.write_text(
                json.dumps(self._assigned, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("Failed to persist client assignments to %s: %s", self._assignments_path, exc)

    async def register(self, client_id: str, url: str, model: str = "", engine: str = "", busy: bool = False, hostname: str = "", specs: dict | None = None, requests: int = 0):
        async with self._lock:
            existing = self._clients.get(client_id, {})
            first_seen = existing.get("first_seen", time.time())
            self._clients[client_id] = {
                "url": url,
                "last_seen": time.time(),
                "first_seen": first_seen,
                "model": model,
                "engine": config.normalize_engine(engine),
                "busy": busy,
                "hostname": hostname,
                "specs": specs or {},
                "requests": requests,
            }
        log.info(
            "Registered client %s (%s) at %s (engine=%s model=%s busy=%s)",
            client_id,
            hostname or "unknown",
            url,
            config.normalize_engine(engine) or "unknown",
            model,
            busy,
        )

    async def set_ws(self, client_id: str, ws: WebSocket):
        async with self._lock:
            self._ws[client_id] = ws

    async def remove_ws(self, client_id: str):
        async with self._lock:
            self._ws.pop(client_id, None)

    async def remove(self, client_id: str):
        async with self._lock:
            existed = self._clients.pop(client_id, None) is not None
            self._ws.pop(client_id, None)
        if existed:
            log.info("Removed client %s", client_id)

    async def get_ws(self, client_id: str) -> WebSocket | None:
        async with self._lock:
            return self._ws.get(client_id)

    async def assign_model(self, client_id: str, model: str):
        async with self._lock:
            self._assigned[client_id] = model
            self._persist_assignments()

    async def get_assigned_model(self, client_id: str) -> str:
        async with self._lock:
            return self._assigned.get(client_id, "")

    async def _remove_stale_locked(self):
        cutoff = time.time() - config.CLIENT_TIMEOUT
        stale = [k for k, v in self._clients.items() if v["last_seen"] < cutoff]
        for k in stale:
            del self._clients[k]
            self._ws.pop(k, None)
            log.info("Removed stale client %s", k)

    async def remove_stale(self):
        async with self._lock:
            await self._remove_stale_locked()

    async def pick(self, model: str = "") -> tuple[str, str] | None:
        async with self._lock:
            await self._remove_stale_locked()
            available = sorted(
                k for k, v in self._clients.items()
                if not v.get("busy", False) and model_matches(model, v.get("model", ""))
            )
            if not available:
                return None
            self._idx = self._idx % len(available)
            chosen = available[self._idx]
            self._idx += 1
            return chosen, self._clients[chosen]["url"]

    async def models(self) -> list[str]:
        async with self._lock:
            await self._remove_stale_locked()
            return sorted({v["model"] for v in self._clients.values() if v.get("model")})

    async def all(self) -> list[dict]:
        async with self._lock:
            await self._remove_stale_locked()
            return [
                {
                    "id": k,
                    "url": v["url"],
                    "last_seen": v["last_seen"],
                    "first_seen": v.get("first_seen", v["last_seen"]),
                    "model": v.get("model", ""),
                    "assigned_model": self._assigned.get(k, ""),
                    "engine": v.get("engine", ""),
                    "busy": v.get("busy", False),
                    "hostname": v.get("hostname", ""),
                    "specs": v.get("specs", {}),
                    "requests": v.get("requests", 0),
                }
                for k, v in self._clients.items()
            ]

    async def get_client(self, client_id: str) -> dict | None:
        async with self._lock:
            client = self._clients.get(client_id)
            return dict(client) if client else None


class ApiKeyStore:
    """Persistent store of API keys generated from the dashboard."""

    def __init__(self, path: Path):
        self._path = path
        self._keys: dict[str, dict] = {}  # key -> {key, name, created}
        self._lock = asyncio.Lock()
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                self._keys = {entry["key"]: entry for entry in data}
                log.info("Loaded %d API key(s) from %s", len(self._keys), self._path)
            except Exception as exc:
                log.warning("Failed to load API keys from %s: %s", self._path, exc)

    def _persist(self):
        try:
            self._path.write_text(json.dumps(list(self._keys.values()), indent=2))
        except Exception as exc:
            log.warning("Failed to persist API keys to %s: %s", self._path, exc)

    def valid(self, key: str) -> bool:
        return key in self._keys

    async def create(self, name: str = "") -> dict:
        async with self._lock:
            key = "fllm-" + secrets.token_urlsafe(32)
            entry = {"key": key, "name": name.strip(), "created": time.time()}
            self._keys[key] = entry
            self._persist()
        log.info("Generated API key '%s' (name=%s)", key[:12] + "…", entry["name"] or "unnamed")
        return entry

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._keys:
                del self._keys[key]
                self._persist()
                log.info("Deleted API key %s…", key[:12])
                return True
            return False

    async def all(self) -> list[dict]:
        async with self._lock:
            return sorted(self._keys.values(), key=lambda e: e.get("created", 0), reverse=True)


pool = ClientPool(Path(config.ASSIGNMENTS_FILE))
api_keys = ApiKeyStore(Path(config.API_KEYS_FILE))
http_client: httpx.AsyncClient | None = None
pending_requests: dict[str, asyncio.Future] = {}
request_client: dict[str, str] = {}  # request_id -> client_id


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(
        timeout=120,
        headers={"Authorization": f"Bearer {config.SECRET}"},
    )
    yield
    await http_client.aclose()

app = FastAPI(title="FreeLLM Server", lifespan=lifespan, root_path=config.ROOT_PATH)


async def dispatch_chat(req: dict) -> dict:
    """Route one chat request to an available client and return its raw result."""
    requested_model = (req.get("model") or "").strip()
    tried: set[str] = set()
    while True:
        target = await pool.pick(requested_model)
        if target is None or target[0] in tried:
            if requested_model:
                raise HTTPException(503, f"No available client serving model '{requested_model}'")
            raise HTTPException(503, "No available clients")

        client_id, client_url = target
        tried.add(client_id)

        # Try WebSocket first
        ws = await pool.get_ws(client_id)
        if ws:
            try:
                result = await ws_request(client_id, ws, "chat", req)
                if result.get("type") == "error":
                    log.warning("WebSocket %s returned error: %s", client_id, result.get("payload"))
                else:
                    return result.get("payload", {})
            except (asyncio.TimeoutError, WebSocketDisconnect, Exception) as exc:
                log.warning("WebSocket request to %s failed: %r", client_id, exc)

        # Fall back to HTTP (also used when WS missing or WS failed)
        try:
            resp = await http_client.post(f"{client_url}/v1/chat/completions", json=req)
            if resp.status_code == 503:
                continue  # client busy, try next
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError:
            continue
        except httpx.RequestError as exc:
            log.warning("Client %s unreachable: %s", client_id, exc)
            continue


async def verify_api_key(authorization: str = Header(...)):
    token = authorization[7:].strip() if authorization.startswith("Bearer ") else ""
    if not api_keys.valid(token):
        raise HTTPException(401, "Invalid API key")


async def verify_admin(authorization: str = Header(...)):
    if authorization != f"Bearer {config.ADMIN_KEY}":
        raise HTTPException(401, "Invalid admin key")


async def verify_client_secret(authorization: str = Header(...)):
    if authorization != f"Bearer {config.SECRET}":
        raise HTTPException(403, "Invalid client secret")


class SetDefaultModelRequest(BaseModel):
    model_name: str


class CreateApiKeyRequest(BaseModel):
    name: str = ""


@app.post("/register", dependencies=[Depends(verify_client_secret)])
async def register(req: RegisterRequest):
    await pool.register(req.id, req.url, req.model, req.engine, req.busy, req.hostname, req.specs, req.requests)
    assigned = await pool.get_assigned_model(req.id)
    return {
        "status": "ok",
        "model": assigned or config.DEFAULT_MODEL,
        "model_maps": config.MODEL_MAPS,
    }


@app.get("/default_model", dependencies=[Depends(verify_admin)])
async def get_default_model():
    return {"model": config.DEFAULT_MODEL}


@app.post("/default_model", dependencies=[Depends(verify_admin)])
async def set_default_model(req: SetDefaultModelRequest):
    allowed_models = config.public_models()
    if req.model_name not in allowed_models:
        raise HTTPException(400, f"Model '{req.model_name}' is not configured in MODEL_MAPS")
    config.DEFAULT_MODEL = req.model_name
    log.info("Default model set to '%s'", req.model_name)
    return {"status": "ok", "model": config.DEFAULT_MODEL}


@app.get("/clients", dependencies=[Depends(verify_admin)])
async def list_clients():
    return await pool.all()


@app.get("/api_keys", dependencies=[Depends(verify_admin)])
async def list_api_keys():
    return await api_keys.all()


@app.post("/api_keys", dependencies=[Depends(verify_admin)])
async def create_api_key(req: CreateApiKeyRequest):
    return await api_keys.create(req.name)


@app.delete("/api_keys/{key}", dependencies=[Depends(verify_admin)])
async def delete_api_key(key: str):
    if not await api_keys.delete(key):
        raise HTTPException(404, "API key not found")
    return {"status": "ok"}


async def ws_request(client_id: str, ws: WebSocket, msg_type: str, payload: dict, timeout: float = 120) -> dict:
    request_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    pending_requests[request_id] = fut
    request_client[request_id] = client_id
    try:
        await ws.send_json({"type": msg_type, "request_id": request_id, "payload": payload})
        return await asyncio.wait_for(fut, timeout=timeout)
    finally:
        pending_requests.pop(request_id, None)
        request_client.pop(request_id, None)


@app.websocket("/ws/{client_id}")
async def ws_endpoint(ws: WebSocket, client_id: str):
    await ws.accept()
    # Authenticate via the Authorization header, falling back to a `token`
    # query parameter. Reverse proxies (e.g. IIS ARR) often strip custom
    # headers on the WebSocket upgrade, but the query string is always
    # forwarded as part of the URL.
    auth = ws.headers.get("authorization", "")
    token = ws.query_params.get("token", "")
    if auth != f"Bearer {config.SECRET}" and token != config.SECRET:
        await ws.close(code=1008)
        return
    await pool.set_ws(client_id, ws)
    log.info("WebSocket connected for client %s", client_id)
    try:
        while True:
            data = await ws.receive_json()
            request_id = data.get("request_id")
            if request_id and request_id in pending_requests:
                fut = pending_requests[request_id]
                if not fut.done():
                    fut.set_result(data)
    except WebSocketDisconnect:
        log.info("WebSocket disconnected for client %s", client_id)
    except Exception as exc:
        log.warning("WebSocket error for client %s: %s", client_id, exc)
    finally:
        await pool.remove_ws(client_id)
        orphaned = [rid for rid, cid in request_client.items() if cid == client_id]
        for rid in orphaned:
            fut = pending_requests.pop(rid, None)
            if fut and not fut.done():
                fut.cancel()
            request_client.pop(rid, None)


@app.post("/assign_model", dependencies=[Depends(verify_admin)])
async def assign_model(req: AssignModelRequest):
    client = await pool.get_client(req.client_id)
    if not client:
        raise HTTPException(404, "Client not found")
    url = client["url"]
    engine = client.get("engine", "")
    allowed_models = config.public_models(engine)
    if not allowed_models:
        raise HTTPException(409, "Client has no known engine or no models are configured for it")
    if req.model_name not in allowed_models:
        raise HTTPException(
            400,
            f"Model '{req.model_name}' is not allowed for engine '{engine or 'unknown'}'",
        )
    await pool.assign_model(req.client_id, req.model_name)

    # Try WebSocket first
    ws = await pool.get_ws(req.client_id)
    if ws:
        try:
            result = await ws_request(req.client_id, ws, "load_model", {"model_name": req.model_name}, timeout=300)
            if result.get("type") == "error":
                raise HTTPException(502, result.get("payload", {}).get("detail", "Client error"))
            return result.get("payload", {})
        except asyncio.TimeoutError:
            raise HTTPException(504, "Client timed out loading model")
        except WebSocketDisconnect:
            log.warning("WebSocket lost during assign_model for %s, trying HTTP", req.client_id)

    # Fall back to HTTP
    try:
        resp = await http_client.post(f"{url}/load_model", json={"model_name": req.model_name}, timeout=300)
        resp.raise_for_status()
        return resp.json()
    except httpx.RequestError as exc:
        raise HTTPException(502, f"Failed to reach client: {exc}")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(exc.response.status_code, exc.response.text)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html = (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")
    # Inject the sub-path the dashboard is served under so all asset/API
    # URLs resolve correctly when behind a reverse proxy (e.g. /freellm).
    return html.replace("__BASE__", config.ROOT_PATH).replace(
        '"__MODEL_MAPS__"',
        json.dumps(config.MODEL_MAPS),
    ).replace(
        '"__CLUSTER_NAME__"',
        json.dumps(config.CLUSTER_NAME),
    )


@app.get("/logo.png", include_in_schema=False)
async def logo():
    return FileResponse(Path(__file__).parent.parent / "FreeLLM-logo-dark-transparent.png")


@app.get("/favicon.png", include_in_schema=False)
async def favicon():
    return FileResponse(Path(__file__).parent.parent / "favicon.png")


@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    names = await pool.models()
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "created": 0, "owned_by": "freellm"}
            for name in names
        ],
    }


@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(req: dict):
    if req.get("stream"):
        raise HTTPException(400, "Streaming is not supported")
    return await dispatch_chat(req)


def main():
    parser = argparse.ArgumentParser(description="FreeLLM Server")
    parser.add_argument("--host", default=config.HOST)
    parser.add_argument("--port", type=int, default=config.PORT)
    parser.add_argument("--secret", default=config.SECRET, help="Shared secret for client auth")
    parser.add_argument("--admin-key", default=config.ADMIN_KEY, help="Admin key for the dashboard / API key management")
    parser.add_argument("--default-model", default=config.DEFAULT_MODEL, help="Default Ollama model for clients")
    parser.add_argument("--root-path", default=config.ROOT_PATH, help="Sub-path when behind a reverse proxy, e.g. /freellm")
    args = parser.parse_args()

    config.SECRET = args.secret
    config.ADMIN_KEY = args.admin_key
    config.DEFAULT_MODEL = args.default_model
    config.ROOT_PATH = (args.root_path or "").rstrip("/")
    app.root_path = config.ROOT_PATH

    if not config.SECRET:
        log.warning("FREELLM_SECRET not set — pass --secret or set it in the environment.")
    admin_from_env = os.getenv("FREELLM_ADMIN_KEY") or os.getenv("FREELLM_API_KEY")
    if not config.ADMIN_KEY:
        config.ADMIN_KEY = "fllm-" + secrets.token_urlsafe(24)
        if not admin_from_env and "--admin-key" not in sys.argv:
            log.warning("Admin key not set — generated one for this run: %s", config.ADMIN_KEY)

    log.info("Server starting on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, root_path=config.ROOT_PATH)


if __name__ == "__main__":
    main()
