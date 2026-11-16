# SPDX-License-Identifier: Apache-2.0
"""Default-off host-side trace for the layerwise v3 warm batch path.

The sink is intentionally tiny and core-neutral: it has no imports from
LMCache-Ascend or vLLM, so the request server and the ZMQ transport can record
their own spans without depending on a client implementation.

Environment
-----------
``DSV41_LW_V2_MASK_TRACE``
    Truthy (``1``/``true``/``yes``/``on``) enables the sink. Default off. The
    name is a legacy v2 name; v3 reuses it so one B01 trace directory can hold
    both the Ascend client spans and these server spans.

``DSV41_LW_V2_MASK_TRACE_DIR``
    Output directory. Defaults to ``/tmp/dsv41-lw-v2-mask-trace``. The sink
    writes ``mask-trace-<pid>.jsonl`` so processes sharing a directory never
    interleave lines.

Off behaviour is "zero extra clock, zero file": every helper returns before
reading the clock or touching the filesystem, so disabled runs keep the same
byte stream and call order as an uninstrumented build.

Record shape::

    {
      "ts_ns": ..., "seq": ..., "pid": ..., "tid": ...,
      "trace_source": "server" | "client",
      "event": "server_hash",
      "elapsed_ns": ...,
      "request_id": ..., "batch_id": ..., "operation_id": ...,
      ...
      "field_status": {"group_id": "unavailable"}
    }

Fields that the instrumentation point cannot obtain are written as JSON
``null`` and named in ``field_status`` as ``"unavailable"``; payload bytes and
prompt text are never recorded.
"""

# Standard
from pathlib import Path
import json
import os
import threading
import time
from typing import IO, Any

_ENV_ENABLED = "DSV41_LW_V2_MASK_TRACE"
_ENV_DIR = "DSV41_LW_V2_MASK_TRACE_DIR"
_DEFAULT_DIR = "/tmp/dsv41-lw-v2-mask-trace"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {
    "enabled": None,
    "directory": None,
    "handle": None,
    "seq": 0,
    "warned": False,
}


def enabled() -> bool:
    """Return whether tracing is enabled (cached after the first call)."""
    value = _STATE["enabled"]
    if value is None:
        raw = os.environ.get(_ENV_ENABLED, "")
        value = raw.strip().lower() in _TRUTHY
        _STATE["enabled"] = value
    return bool(value)


def reset_for_tests() -> None:
    """Forget the cached configuration and close the current file."""
    with _LOCK:
        handle = _STATE["handle"]
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
        _STATE["enabled"] = None
        _STATE["directory"] = None
        _STATE["handle"] = None
        _STATE["seq"] = 0
        _STATE["warned"] = False


def start() -> int | None:
    """Return a monotonic start timestamp, or ``None`` when tracing is off."""
    if not enabled():
        return None
    return time.monotonic_ns()


def now_ns() -> int:
    """Return the current monotonic timestamp (callers only use it when on)."""
    return time.monotonic_ns()


def elapsed_since(start_ns: int | None) -> int | None:
    """Return ``now - start_ns``, or ``None`` when there is no start stamp."""
    if start_ns is None:
        return None
    return time.monotonic_ns() - start_ns


def output_path() -> Path | None:
    """Return the per-process trace file path, or ``None`` when disabled."""
    if not enabled():
        return None
    directory = _STATE["directory"]
    if directory is None:
        directory = os.environ.get(_ENV_DIR, _DEFAULT_DIR)
        _STATE["directory"] = directory
    return Path(directory) / f"mask-trace-{os.getpid()}.jsonl"


def _handle() -> IO[str] | None:
    handle = _STATE["handle"]
    if handle is not None:
        return handle
    path = output_path()
    if path is None:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a", encoding="utf-8")
    except OSError:
        if not _STATE["warned"]:
            _STATE["warned"] = True
            # Imported lazily: the trace module must stay importable from
            # minimal contexts (tests build fake server objects).
            from lmcache.logging import init_logger

            init_logger(__name__).warning(
                "cannot open layerwise v3 trace file %s; tracing disabled", path
            )
        _STATE["enabled"] = False
        return None
    _STATE["handle"] = handle
    return handle


def emit(
    trace_source: str,
    event: str,
    *,
    elapsed_ns: int | None = None,
    **fields: Any,
) -> None:
    """Append one trace record; a no-op when tracing is disabled.

    Args:
        trace_source: ``"server"`` or ``"client"``.
        event: Stable snake-case span name.
        elapsed_ns: Optional span duration in monotonic nanoseconds.
        **fields: Flat scalar fields; ``None`` is written as JSON null and
            should be listed in ``field_status`` by the caller.
    """
    if not enabled():
        return
    with _LOCK:
        handle = _handle()
        if handle is None:
            return
        _STATE["seq"] += 1
        record: dict[str, Any] = {
            "ts_ns": time.monotonic_ns(),
            "seq": _STATE["seq"],
            "pid": os.getpid(),
            "tid": threading.get_ident(),
            "trace_source": trace_source,
            "event": event,
        }
        if elapsed_ns is not None:
            record["elapsed_ns"] = int(elapsed_ns)
        record.update(fields)
        try:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            handle.flush()
        except OSError:
            if not _STATE["warned"]:
                _STATE["warned"] = True
                _STATE["enabled"] = False
