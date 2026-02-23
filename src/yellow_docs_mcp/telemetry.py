"""Lightweight structured telemetry helpers."""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Iterator


def _format_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return json.dumps(str(value), ensure_ascii=True)


def log_metric(logger: logging.Logger, event: str, **fields: Any) -> None:
    parts = [f"event={event}"]
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={_format_value(value)}")
    logger.info("metric %s", " ".join(parts))


@contextmanager
def timed_metric(logger: logging.Logger, event: str, **fields: Any) -> Iterator[None]:
    started = perf_counter()
    status = "ok"
    error = None
    try:
        yield
    except Exception as exc:
        status = "error"
        error = exc.__class__.__name__
        raise
    finally:
        duration_ms = (perf_counter() - started) * 1000
        log_metric(
            logger,
            event,
            status=status,
            duration_ms=duration_ms,
            error=error,
            **fields,
        )
