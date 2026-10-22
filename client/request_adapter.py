"""Request shaping helpers for the FreeLLM client."""

_CALLER_THINKING_FIELDS = ("reasoning_effort", "reasoning", "chat_template_kwargs")
_MAX_TOKEN_FIELDS = ("max_tokens", "max_completion_tokens", "max_output_tokens")
_NO_THINK_PREFIX = "/no_think"


def _copy_messages(req: dict) -> list:
    return [dict(m) if isinstance(m, dict) else m for m in (req.get("messages") or [])]


def _prepend_no_think(messages: list) -> None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and _NO_THINK_PREFIX not in content:
            messages[index] = {**message, "content": f"{_NO_THINK_PREFIX}\n{content}"}
        return


def _explicit_enable_thinking(payload: dict):
    top_level = payload.get("enable_thinking")
    if isinstance(top_level, bool):
        return top_level
    kwargs = payload.get("chat_template_kwargs")
    if isinstance(kwargs, dict) and isinstance(kwargs.get("enable_thinking"), bool):
        return kwargs["enable_thinking"]
    return None


def _resolve_thinking(payload: dict, default_enabled: bool):
    think = payload.pop("think", None)
    explicit_enable = _explicit_enable_thinking(payload)
    if think is None:
        think = explicit_enable

    if isinstance(think, str):
        payload.setdefault("reasoning_effort", think)
        return None

    explicit_reasoning = any(
        field in payload for field in _CALLER_THINKING_FIELDS if field != "chat_template_kwargs"
    )
    if explicit_reasoning or isinstance(payload.get("reasoning_effort"), str):
        return None
    if isinstance(think, bool):
        return think
    if "chat_template_kwargs" in payload or isinstance(payload.get("enable_thinking"), bool):
        return None
    return default_enabled


def _apply_thinking(payload: dict, think, runtime_engine: str, lemonade_no_think_prefix: bool) -> None:
    if think is None:
        return
    if not think:
        payload.setdefault("reasoning_effort", "none")
    if runtime_engine != "lemonade":
        return

    payload["enable_thinking"] = think
    kwargs = dict(payload.get("chat_template_kwargs") or {})
    kwargs.setdefault("enable_thinking", think)
    payload["chat_template_kwargs"] = kwargs
    if not think and lemonade_no_think_prefix:
        _prepend_no_think(payload["messages"])


def to_runtime_model(name: str, model_map: dict[str, str]) -> str:
    wanted = (name or "").strip().lower()
    for public, runtime_name in model_map.items():
        if public.lower() == wanted:
            return runtime_name
    return name


def to_public_model(name: str, model_map: dict[str, str]) -> str:
    wanted = (name or "").strip().lower()
    for public, runtime_name in model_map.items():
        if runtime_name.lower() == wanted or public.lower() == wanted:
            return public
    return (name or "").strip()


def build_request(
    req: dict,
    *,
    current_model: str,
    runtime_engine: str,
    model_map: dict[str, str],
    enable_thinking_default: bool,
    max_output_tokens: int,
    lemonade_no_think_prefix: bool,
) -> dict:
    payload = dict(req)
    payload["messages"] = _copy_messages(req)
    _apply_thinking(
        payload,
        _resolve_thinking(payload, enable_thinking_default),
        runtime_engine,
        lemonade_no_think_prefix,
    )
    payload["model"] = to_runtime_model((req.get("model") or current_model).strip(), model_map)
    payload["stream"] = False
    if max_output_tokens > 0 and not any(field in payload for field in _MAX_TOKEN_FIELDS):
        payload["max_tokens"] = max_output_tokens
    return payload