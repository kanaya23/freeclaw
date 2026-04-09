from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


_DEFAULT_MAX_CHARS = 12_000


@dataclass(frozen=True)
class _Schema:
    required: dict[str, type[Any] | tuple[type[Any], ...]]


_EXACT_SCHEMAS: dict[str, _Schema] = {
    "text_search": _Schema(required={"ok": bool, "tool": str, "results": list}),
    "web_search": _Schema(required={"ok": bool, "tool": str, "results": list}),
    "web_fetch": _Schema(required={"ok": bool, "tool": str, "url": str}),
    "http_request_json": _Schema(required={"ok": bool, "tool": str, "url": str, "status": int}),
    "timer_api_get": _Schema(required={"ok": bool, "tool": str, "endpoint": str, "status": int}),
    "sh_exec": _Schema(required={"ok": bool, "tool": str}),
    "discord_send_file": _Schema(required={"ok": bool, "tool": str}),
}

_PREFIX_SCHEMAS: list[tuple[str, _Schema]] = [
    ("fs_", _Schema(required={"ok": bool, "tool": str})),
    ("memory_", _Schema(required={"ok": bool, "tool": str})),
    ("task_", _Schema(required={"ok": bool, "tool": str})),
    ("doc_", _Schema(required={"ok": bool, "tool": str})),
    ("google_", _Schema(required={"ok": bool, "tool": str})),
]


def _schema_for_tool(name: str) -> _Schema:
    exact = _EXACT_SCHEMAS.get(name)
    if exact is not None:
        return exact
    for prefix, schema in _PREFIX_SCHEMAS:
        if name.startswith(prefix):
            return schema
    return _Schema(required={"ok": bool, "tool": str})


def _matches_type(value: Any, expected: type[Any] | tuple[type[Any], ...]) -> bool:
    if isinstance(expected, tuple):
        return isinstance(value, expected)
    return isinstance(value, expected)


def _normalize_tool_result(name: str, result: Any) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if not isinstance(result, dict):
        errors.append(f"result must be a JSON object (got {type(result).__name__})")
        base = {
            "ok": False,
            "tool": name,
            "error": "invalid tool result shape",
            "raw_preview": str(result)[:1000],
        }
        return base, errors

    out = dict(result)
    out["tool"] = name
    if not isinstance(out.get("ok"), bool):
        errors.append("missing/invalid `ok` boolean")
        out["ok"] = False

    if out.get("ok") is False:
        if not isinstance(out.get("error"), str) or not str(out.get("error")).strip():
            errors.append("error result is missing `error` string")
            out["error"] = "tool execution failed"
        return out, errors

    schema = _schema_for_tool(name)
    for key, expected_type in schema.required.items():
        if key not in out:
            errors.append(f"missing required key `{key}`")
            continue
        if not _matches_type(out.get(key), expected_type):
            errors.append(f"key `{key}` has invalid type ({type(out.get(key)).__name__})")
    return out, errors


def serialize_tool_result(name: str, result: Any, *, max_chars: int | None = None) -> str:
    normalized, schema_errors = _normalize_tool_result(name, result)
    if schema_errors:
        normalized = {
            "ok": False,
            "tool": name,
            "error": "tool result schema validation failed",
            "schema_errors": schema_errors,
            "raw_result_preview": json.dumps(result, ensure_ascii=True)[:2000]
            if not isinstance(result, str)
            else result[:2000],
        }

    resolved_max = int(max_chars or int(os.getenv("FREECLAW_TOOL_RESULT_MAX_CHARS") or _DEFAULT_MAX_CHARS))
    if resolved_max < 200:
        resolved_max = 200

    blob = json.dumps(normalized, ensure_ascii=True)
    if len(blob) <= resolved_max:
        return blob

    overflow = len(blob) - resolved_max
    truncated = {
        "ok": bool(normalized.get("ok", False)),
        "tool": name,
        "annotation": f"[truncated: {overflow} chars]",
        "content_preview": blob[:resolved_max],
    }
    out = json.dumps(truncated, ensure_ascii=True)
    if len(out) <= resolved_max:
        return out
    return out[:resolved_max]

