import json
import logging
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
import threading
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .routing import choose_route_model, is_nim_chat_url, should_route_request


log = logging.getLogger(__name__)

_NIM_LAT_LOCK = threading.Lock()
_NIM_LAT_SAMPLES_MS: deque[float] = deque(maxlen=10)
_NIM_LAT_LAST_MS: float | None = None


def _safe_url(url: str) -> str:
    try:
        u = urlsplit(url)
        # Avoid logging query strings that might include secrets.
        return urlunsplit((u.scheme, u.netloc, u.path, "", ""))
    except Exception:
        return url


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: dict[str, str]
    json: Any


def _record_nim_latency(elapsed_ms: float) -> None:
    global _NIM_LAT_LAST_MS
    with _NIM_LAT_LOCK:
        _NIM_LAT_LAST_MS = float(elapsed_ms)
        _NIM_LAT_SAMPLES_MS.append(float(elapsed_ms))


def nim_latency_snapshot() -> dict[str, Any]:
    with _NIM_LAT_LOCK:
        samples = list(_NIM_LAT_SAMPLES_MS)
        last = _NIM_LAT_LAST_MS
    avg = (sum(samples) / len(samples)) if samples else None
    return {
        "last_ms": (None if last is None else float(last)),
        "avg_ms_10": (None if avg is None else float(avg)),
        "samples": len(samples),
    }


def post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_s: float,
    *,
    route_hint: dict[str, Any] | None = None,
) -> HttpResponse:
    t0 = time.perf_counter()
    safe_url = _safe_url(url)
    is_nim = is_nim_chat_url(url)
    routing_enabled = bool(is_nim and should_route_request(url=url, payload=payload))
    req_payload = payload
    route_decision: dict[str, Any] | None = None
    if routing_enabled:
        default_pin = None
        explicit_pin = None
        if isinstance(route_hint, dict):
            default_pin = route_hint.get("default_pin")
            explicit_pin = route_hint.get("pin")
        try:
            decision = choose_route_model(
                url=url,
                payload=payload,
                default_pin=(None if default_pin is None else str(default_pin)),
                explicit_pin=(None if explicit_pin is None else str(explicit_pin)),
            )
            req_payload = dict(payload)
            req_payload["model"] = decision.model
            route_decision = {
                "model": decision.model,
                "reason": decision.reason,
                "token_estimate": decision.token_estimate,
            }
        except Exception as e:
            log.warning("route selection failed; preserving request model: %s", e)
            req_payload = payload

    log.debug(
        "http request method=POST url=%s timeout_s=%.2f payload_bytes=%d",
        safe_url,
        float(timeout_s),
        len(json.dumps(req_payload)),
    )
    if route_decision is not None:
        log.info(
            "route decision reason=%s model=%s tokens=%s",
            str(route_decision.get("reason") or "unknown"),
            str(route_decision.get("model") or "unknown"),
            str(route_decision.get("token_estimate") or "n/a"),
        )
    body = json.dumps(req_payload).encode("utf-8")
    req = urllib.request.Request(url, method="POST", data=body)
    for k, v in headers.items():
        req.add_header(k, v)
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
            content_type = resp.headers.get("Content-Type", "")
            if "application/json" not in content_type:
                raise RuntimeError(f"Expected JSON response, got Content-Type={content_type!r}")
            data = json.loads(raw.decode("utf-8"))
            log.debug(
                "http response method=POST url=%s status=%d elapsed_ms=%.1f bytes=%d",
                safe_url,
                int(resp.status),
                (time.perf_counter() - t0) * 1000.0,
                len(raw),
            )
            if is_nim:
                _record_nim_latency((time.perf_counter() - t0) * 1000.0)
            return HttpResponse(
                status=int(resp.status),
                headers={k: v for k, v in resp.headers.items()},
                json=data,
            )
    except urllib.error.HTTPError as e:
        raw = e.read()
        msg = raw.decode("utf-8", errors="replace")
        log.warning(
            "http error method=POST url=%s status=%d elapsed_ms=%.1f body_preview=%r",
            safe_url,
            int(e.code),
            (time.perf_counter() - t0) * 1000.0,
            msg[:300],
        )
        if is_nim:
            _record_nim_latency((time.perf_counter() - t0) * 1000.0)
        raise RuntimeError(f"HTTP {e.code} from {url}: {msg}") from None


def get_json(url: str, headers: dict[str, str], timeout_s: float) -> HttpResponse:
    t0 = time.time()
    safe_url = _safe_url(url)
    log.debug("http request method=GET url=%s timeout_s=%.2f", safe_url, float(timeout_s))
    req = urllib.request.Request(url, method="GET")
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
            content_type = resp.headers.get("Content-Type", "")
            if "application/json" not in content_type:
                raise RuntimeError(f"Expected JSON response, got Content-Type={content_type!r}")
            data = json.loads(raw.decode("utf-8"))
            log.debug(
                "http response method=GET url=%s status=%d elapsed_ms=%.1f bytes=%d",
                safe_url,
                int(resp.status),
                (time.time() - t0) * 1000.0,
                len(raw),
            )
            return HttpResponse(
                status=int(resp.status),
                headers={k: v for k, v in resp.headers.items()},
                json=data,
            )
    except urllib.error.HTTPError as e:
        raw = e.read()
        msg = raw.decode("utf-8", errors="replace")
        log.warning(
            "http error method=GET url=%s status=%d elapsed_ms=%.1f body_preview=%r",
            safe_url,
            int(e.code),
            (time.time() - t0) * 1000.0,
            msg[:300],
        )
        raise RuntimeError(f"HTTP {e.code} from {url}: {msg}") from None
