"""
tools/cache.py
===============
File-based response cache behind the data tools, matching the design doc's
data-sources table: every tool response is cached to ``data/cache/`` so a
notebook re-run doesn't need network access, and a live call that fails
falls back to the last cached response instead of erroring out.

Usage: decorate a tool's underlying function, innermost of the two
decorators, so ``@tool`` wraps the cached callable rather than the other
way around::

    @tool(name="get_ticker_details", description="...")
    @cached("get_ticker_details", ttl_seconds=ONE_HOUR)
    def get_ticker_details(ticker: str) -> dict[str, Any]:
        ...
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from research_agent.config import CACHE_DIR, logger

ONE_HOUR = 3600
SIX_HOURS = 6 * ONE_HOUR
ONE_DAY = 24 * ONE_HOUR

DEFAULT_TTL_SECONDS = SIX_HOURS


def _cache_key(prefix: str, call_kwargs: dict[str, Any]) -> str:
    """Stable cache key from a tool name and its bound call arguments."""
    payload = json.dumps({"prefix": prefix, "kwargs": call_kwargs}, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.json"


def _read(key: str) -> dict[str, Any] | None:
    """Return ``{"cached_at": float, "value": Any}``, or ``None`` if missing/corrupt."""
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write(key: str, value: Any) -> None:
    entry = {"cached_at": time.time(), "value": value}
    try:
        _cache_path(key).write_text(json.dumps(entry, default=str), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not write cache entry %s: %s", key, exc)


def cached(prefix: str, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> Callable:
    """Cache a function's return value to ``data/cache/``, keyed by its arguments.

    If the live call raises, a stale cache entry (of any age) is served
    instead of propagating the error - the "fallback: cached file" behaviour
    in the design doc - and the exception is only raised when there is no
    cache to fall back to. Callers may pass ``force_refresh=True`` to skip a
    fresh cache entry and re-fetch.

    Args:
        prefix: Cache-key prefix; use the tool's name so entries are easy to
            find in ``data/cache/``.
        ttl_seconds: How long a cached entry is served without re-fetching.

    Returns:
        The decorator.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            force_refresh = kwargs.pop("force_refresh", False)

            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            key = _cache_key(prefix, dict(bound.arguments))

            entry = _read(key)
            is_fresh = entry is not None and (time.time() - entry["cached_at"]) <= ttl_seconds
            if is_fresh and not force_refresh:
                logger.info("Cache hit for %s", key)
                return entry["value"]

            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                if entry is not None:
                    logger.warning(
                        "Live call failed for %s (%s); serving stale cache", prefix, exc
                    )
                    return entry["value"]
                raise

            _write(key, result)
            return result

        return wrapper

    return decorator
