from __future__ import annotations

import datetime as dt
import json
import os
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .paths import config_dir


_ROUTE_LOG_DEFAULT = "/tmp/freeclaw-route.log"
_DEFAULT_HEAVY_MODEL = "moonshotai/kimi-k2-thinking"
_DEFAULT_LIGHT_MODEL = "mistralai/mistral-small-4-119b-2603"
_TOKEN_ESCALATION_DEFAULT = 3000
_MAX_PREVIEW_CHARS = 180

_LIGHT_EXACT_PREFIXES = (
    "reply with exactly",
    "say exactly",
    "repeat exactly",
)

_LIGHT_CHAT_PREFIXES = (
    "hi",
    "hello",
    "hey",
    "yo",
    "sup",
    "who are you",
    "what are you",
    "what can you do",
    "help",
    "thanks",
    "thank you",
    "good morning",
    "good night",
    "wyd",
)

_BROWSER_MARKERS = (
    "browser",
    "browser-use",
    "website",
    "web site",
    "webpage",
    "web page",
    "page title",
    "title of",
    "open ",
    "navigate",
    "go to ",
    "visit ",
    "click",
    "scroll",
    "tab",
    "selector",
    "xpath",
    "dom",
    "html",
    "screenshot",
    "screen shot",
    "capture page",
    "extract from page",
    "extract visible",
    "page content",
    "download",
    "upload",
    "form",
    "link",
    "links",
    "image result",
    "search result",
    "search results",
    "playwright",
)

_TOOL_OR_ENV_MARKERS = (
    "shell command",
    "bash",
    "sh ",
    "python ",
    "workspace/",
    "/tmp/",
    "file named",
    "create a file",
    "markdown file",
    "save a file",
    "send it here",
    "send this file",
    "full path",
    "install ",
    "command line",
    "cli",
)

_HEAVY_KEYWORDS = (
    "debug",
    "error",
    "traceback",
    "exception",
    "stack trace",
    "bug",
    "write code",
    "implement",
    "build",
    "architecture",
    "plan",
    "optimize",
    "analyze",
    "analysis",
    "compare",
    "benchmark",
    "research",
    "investigate",
    "scrape",
    "search the web",
    "read docs",
    "summarize these pages",
    "api",
    "integrate",
    "fastapi",
    "docker",
    "sql",
    "javascript",
    "typescript",
    "refactor",
    "fix",
    "deploy",
    "gpu",
    "modal",
    "kaggle",
    "multi-step",
    "step by step",
)

_MULTI_STAGE_MARKERS = (
    " and then ",
    " then ",
    " after that ",
    " first ",
    " second ",
    " finally ",
    " also ",
    " plus ",
    "\n- ",
    "\n1.",
    "\n2.",
)

_ROUTE_LOCAL = threading.local()
_ROUTE_CACHE_LOCK = threading.Lock()
_ROUTE_STATE_LOCK = threading.Lock()


def _normalize_pin(raw: str | None) -> str | None:
    if raw is None:
        return None
    v = str(raw).strip().lower()
    if v in {"heavy", "h"}:
        return "heavy"
    if v in {"light", "l"}:
        return "light"
    if v in {"none", "null", "default", "auto", "reset", "clear", ""}:
        return None
    return None


def _is_truthy(raw: str | None, *, default: bool = True) -> bool:
    if raw is None:
        return default
    v = str(raw).strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _coerce_str_list(obj: Any, *, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(obj, list):
        return fallback
    out = [str(x).strip().lower() for x in obj if str(x).strip()]
    return tuple(out) if out else fallback


def _clean_preview(text: str) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= _MAX_PREVIEW_CHARS:
        return compact
    return compact[:_MAX_PREVIEW_CHARS] + "..."


def _flatten_content(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        out: list[str] = []
        for item in obj:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    out.append(str(item.get("text")))
                elif isinstance(item.get("content"), str):
                    out.append(str(item.get("content")))
                elif isinstance(item.get("text"), str):
                    out.append(str(item.get("text")))
        return "\n".join(out)
    if isinstance(obj, dict):
        if isinstance(obj.get("content"), str):
            return str(obj.get("content"))
        if isinstance(obj.get("text"), str):
            return str(obj.get("text"))
    return ""


def _last_user_text(payload: dict[str, Any]) -> str:
    try:
        msgs = payload.get("messages")
        if not isinstance(msgs, list):
            return ""
        for m in reversed(msgs):
            if isinstance(m, dict) and str(m.get("role") or "").strip().lower() == "user":
                return _flatten_content(m.get("content"))
        if msgs and isinstance(msgs[-1], dict):
            return _flatten_content(msgs[-1].get("content"))
    except Exception:
        return ""
    return ""


def _payload_text(payload: dict[str, Any]) -> str:
    chunks: list[str] = []
    msgs = payload.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "").strip().lower()
            content = _flatten_content(m.get("content"))
            if content:
                chunks.append(f"[{role}] {content}")
    tools = payload.get("tools")
    if isinstance(tools, list):
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function")
            if not isinstance(fn, dict):
                continue
            nm = fn.get("name")
            desc = fn.get("description")
            if isinstance(nm, str) and nm.strip():
                chunks.append(f"[tool:{nm.strip()}]")
            if isinstance(desc, str) and desc.strip():
                chunks.append(desc.strip())
    return "\n".join(chunks).strip()


def estimate_prompt_tokens(payload: dict[str, Any]) -> int:
    text = _payload_text(payload)
    if not text:
        return 0
    # Lightweight estimate (works across mixed natural/code prompts).
    rough = len(text) // 4
    lexical = len(re.findall(r"\w+|[^\w\s]", text))
    return max(1, rough, lexical)


def _looks_like_url_prompt(text: str) -> bool:
    t = str(text or "").lower()
    if "http://" in t or "https://" in t:
        return True
    return bool(re.search(r"\[[^\]]+\]\((https?://[^)]+)\)", t))


@dataclass(frozen=True)
class RoutingConfig:
    enabled: bool
    heavy_model: str
    light_model: str
    default_pin: str | None
    token_escalation_threshold: int
    route_log_path: str
    light_exact_prefixes: tuple[str, ...]
    light_chat_prefixes: tuple[str, ...]
    browser_markers: tuple[str, ...]
    tool_or_env_markers: tuple[str, ...]
    heavy_keywords: tuple[str, ...]
    multi_stage_markers: tuple[str, ...]
    short_chat_max_chars: int
    short_light_max_chars: int
    long_prompt_heavy_chars: int
    structured_heavy_chars: int


@dataclass(frozen=True)
class RouteDecision:
    model: str
    reason: str
    token_estimate: int
    pin: str | None
    prompt_preview: str
    logged_at: str


@dataclass
class _RouteCache:
    signature: tuple[tuple[str, int], ...] | None = None
    config: RoutingConfig | None = None


_ROUTE_CACHE = _RouteCache()
_LAST_ROUTE_DECISION: dict[str, Any] = {}


def _routing_base_path() -> Path:
    override = os.getenv("FREECLAW_ROUTING_CONFIG_PATH")
    if override and override.strip():
        return Path(override).expanduser().resolve()
    return (config_dir() / "routing.json").resolve()


def _routing_agent_path() -> Path | None:
    # Prefer explicit per-agent config path first.
    agent_cfg = os.getenv("FREECLAW_AGENT_CONFIG")
    if agent_cfg and agent_cfg.strip():
        p = Path(agent_cfg).expanduser().resolve()
        return p.parent / "routing.json"
    agent_name = (os.getenv("FREECLAW_AGENT_NAME") or "").strip()
    if not agent_name:
        return None
    return (config_dir() / "agents" / agent_name / "routing.json").resolve()


def _default_routing_doc() -> dict[str, Any]:
    return {
        "enabled": True,
        "heavy_model": os.getenv("FREECLAW_HEAVY_MODEL", _DEFAULT_HEAVY_MODEL),
        "light_model": os.getenv("FREECLAW_LIGHT_MODEL", _DEFAULT_LIGHT_MODEL),
        "default_pin": _normalize_pin(os.getenv("FREECLAW_ROUTING_DEFAULT_PIN")),
        "token_escalation_threshold": _TOKEN_ESCALATION_DEFAULT,
        "route_log_path": _ROUTE_LOG_DEFAULT,
        "light_exact_prefixes": list(_LIGHT_EXACT_PREFIXES),
        "light_chat_prefixes": list(_LIGHT_CHAT_PREFIXES),
        "browser_markers": list(_BROWSER_MARKERS),
        "tool_or_env_markers": list(_TOOL_OR_ENV_MARKERS),
        "heavy_keywords": list(_HEAVY_KEYWORDS),
        "multi_stage_markers": list(_MULTI_STAGE_MARKERS),
        "short_chat_max_chars": 80,
        "short_light_max_chars": 120,
        "long_prompt_heavy_chars": 220,
        "structured_heavy_chars": 120,
    }


def _ensure_base_routing_file(path: Path) -> None:
    if path.exists():
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_default_routing_doc(), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    except Exception:
        # Runtime should continue with in-memory defaults even when file writes fail.
        return


def _load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return raw
    except Exception:
        return {}
    return {}


def _merge_docs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        out[k] = v
    return out


def _signature(paths: list[Path]) -> tuple[tuple[str, int], ...]:
    sig: list[tuple[str, int]] = []
    for p in paths:
        try:
            mt = int(p.stat().st_mtime_ns) if p.exists() else -1
        except Exception:
            mt = -1
        sig.append((str(p), mt))
    return tuple(sig)


def _config_from_doc(doc: dict[str, Any]) -> RoutingConfig:
    return RoutingConfig(
        enabled=_is_truthy(str(doc.get("enabled")) if "enabled" in doc else None, default=True),
        heavy_model=(
            str(doc.get("heavy_model") or os.getenv("FREECLAW_HEAVY_MODEL") or _DEFAULT_HEAVY_MODEL).strip()
            or _DEFAULT_HEAVY_MODEL
        ),
        light_model=(
            str(doc.get("light_model") or os.getenv("FREECLAW_LIGHT_MODEL") or _DEFAULT_LIGHT_MODEL).strip()
            or _DEFAULT_LIGHT_MODEL
        ),
        default_pin=_normalize_pin(
            str(doc.get("default_pin"))
            if doc.get("default_pin") is not None
            else os.getenv("FREECLAW_ROUTING_DEFAULT_PIN")
        ),
        token_escalation_threshold=max(
            1,
            int(
                doc.get("token_escalation_threshold")
                or (os.getenv("FREECLAW_ROUTING_TOKEN_THRESHOLD") or _TOKEN_ESCALATION_DEFAULT)
            ),
        ),
        route_log_path=(
            str(doc.get("route_log_path") or os.getenv("FREECLAW_ROUTE_LOG_PATH") or _ROUTE_LOG_DEFAULT).strip()
            or _ROUTE_LOG_DEFAULT
        ),
        light_exact_prefixes=_coerce_str_list(doc.get("light_exact_prefixes"), fallback=_LIGHT_EXACT_PREFIXES),
        light_chat_prefixes=_coerce_str_list(doc.get("light_chat_prefixes"), fallback=_LIGHT_CHAT_PREFIXES),
        browser_markers=_coerce_str_list(doc.get("browser_markers"), fallback=_BROWSER_MARKERS),
        tool_or_env_markers=_coerce_str_list(doc.get("tool_or_env_markers"), fallback=_TOOL_OR_ENV_MARKERS),
        heavy_keywords=_coerce_str_list(doc.get("heavy_keywords"), fallback=_HEAVY_KEYWORDS),
        multi_stage_markers=_coerce_str_list(doc.get("multi_stage_markers"), fallback=_MULTI_STAGE_MARKERS),
        short_chat_max_chars=max(1, int(doc.get("short_chat_max_chars", 80))),
        short_light_max_chars=max(1, int(doc.get("short_light_max_chars", 120))),
        long_prompt_heavy_chars=max(1, int(doc.get("long_prompt_heavy_chars", 220))),
        structured_heavy_chars=max(1, int(doc.get("structured_heavy_chars", 120))),
    )


def current_routing_config() -> RoutingConfig:
    base_path = _routing_base_path()
    _ensure_base_routing_file(base_path)
    paths = [base_path]
    agent_path = _routing_agent_path()
    if agent_path is not None:
        paths.append(agent_path)

    sig = _signature(paths)
    with _ROUTE_CACHE_LOCK:
        if _ROUTE_CACHE.config is not None and _ROUTE_CACHE.signature == sig:
            return _ROUTE_CACHE.config

        merged = _default_routing_doc()
        for p in paths:
            merged = _merge_docs(merged, _load_json_file(p))
        cfg = _config_from_doc(merged)
        _ROUTE_CACHE.signature = sig
        _ROUTE_CACHE.config = cfg
        return cfg


def is_nim_chat_url(url: str) -> bool:
    try:
        u = urlsplit(str(url))
    except Exception:
        return False
    host = (u.netloc or "").lower()
    path = (u.path or "").rstrip("/").lower()
    return ("integrate.api.nvidia.com" in host) and path.endswith("/chat/completions")


def should_route_request(*, url: str, payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    if not is_nim_chat_url(url):
        return False
    cfg = current_routing_config()
    return bool(cfg.enabled)


def _pin_model(pin: str, cfg: RoutingConfig) -> str:
    return cfg.heavy_model if pin == "heavy" else cfg.light_model


def _heuristic_route(prompt_text: str, *, cfg: RoutingConfig, token_estimate: int) -> tuple[str, str]:
    t = (prompt_text or "").strip().lower()
    if not t:
        return cfg.light_model, "empty"

    if token_estimate >= int(cfg.token_escalation_threshold):
        return cfg.heavy_model, "token-budget"

    if _looks_like_url_prompt(t):
        return cfg.heavy_model, "url"
    if any(m in t for m in cfg.browser_markers):
        return cfg.heavy_model, "browser"
    if any(m in t for m in cfg.tool_or_env_markers):
        return cfg.heavy_model, "tool-env"
    if any(k in t for k in cfg.heavy_keywords):
        return cfg.heavy_model, "keyword"
    if sum(1 for m in cfg.multi_stage_markers if m in t) >= 2:
        return cfg.heavy_model, "multistage"
    if len(t) > int(cfg.long_prompt_heavy_chars):
        return cfg.heavy_model, "long"
    if "\n" in t and len(t) > int(cfg.structured_heavy_chars):
        return cfg.heavy_model, "structured"
    if len(t) <= int(cfg.short_chat_max_chars) and any(t.startswith(x) for x in cfg.light_chat_prefixes):
        return cfg.light_model, "short-chat"
    if len(t) <= int(cfg.short_light_max_chars) and any(t.startswith(x) for x in cfg.light_exact_prefixes):
        return cfg.light_model, "exact"
    if len(t) <= int(cfg.short_light_max_chars) and not any(ch in t for ch in ("/", "\\", "`")):
        return cfg.light_model, "short-simple"
    return cfg.heavy_model, "default-heavy"


def _append_route_log(path: str, line: str) -> None:
    try:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")
    except Exception:
        return


def _remember_route(decision: RouteDecision) -> None:
    with _ROUTE_STATE_LOCK:
        _LAST_ROUTE_DECISION.clear()
        _LAST_ROUTE_DECISION.update(
            {
                "model": decision.model,
                "reason": decision.reason,
                "token_estimate": int(decision.token_estimate),
                "pin": decision.pin,
                "prompt_preview": decision.prompt_preview,
                "logged_at": decision.logged_at,
            }
        )


def get_last_route_decision() -> dict[str, Any]:
    with _ROUTE_STATE_LOCK:
        return dict(_LAST_ROUTE_DECISION)


def current_route_pin() -> str | None:
    return _normalize_pin(getattr(_ROUTE_LOCAL, "pin", None))


@contextmanager
def route_pin_context(pin: str | None):
    prev = getattr(_ROUTE_LOCAL, "pin", None)
    _ROUTE_LOCAL.pin = _normalize_pin(pin)
    try:
        yield
    finally:
        _ROUTE_LOCAL.pin = prev


def choose_route_model(
    *,
    url: str,
    payload: dict[str, Any],
    default_pin: str | None = None,
    explicit_pin: str | None = None,
) -> RouteDecision:
    cfg = current_routing_config()
    pin = _normalize_pin(explicit_pin) or current_route_pin() or _normalize_pin(default_pin) or cfg.default_pin
    prompt_text = _last_user_text(payload)
    token_est = estimate_prompt_tokens(payload)
    if pin in {"heavy", "light"}:
        model = _pin_model(pin, cfg)
        reason = f"pin:{pin}"
    else:
        model, reason = _heuristic_route(prompt_text, cfg=cfg, token_estimate=token_est)

    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    preview = _clean_preview(prompt_text)
    log_line = f"{now} route: {reason} -> {model} tokens={token_est} preview={preview}"
    _append_route_log(cfg.route_log_path, log_line)
    decision = RouteDecision(
        model=model,
        reason=reason,
        token_estimate=int(token_est),
        pin=pin,
        prompt_preview=preview,
        logged_at=now,
    )
    _remember_route(decision)
    return decision

