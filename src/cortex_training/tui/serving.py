# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""A deployment's serving numbers: the ``drain_metrics`` summary the zone head writes to the
console every 30 s (requests, latency, queue, KV cache, replicas), read from the saved console."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
import math
from pathlib import Path
from typing import Any
from typing import Optional

from cortex_training.tui import views

# Cheap substring checks before json.loads: most console lines are neither.
_DRAIN_MARK = '"drain_metrics"'
_STARTUP_MARK = '"job_metrics"'


@dataclass(frozen=True)
class Drain:
    """One 30 s drain: what completed since the previous one, and the replicas' state at its end."""

    drained_at: float
    completed: int
    latency_p50: Optional[float] = None
    latency_p99: Optional[float] = None
    ttft_p50: Optional[float] = None
    ttft_p99: Optional[float] = None
    queue_p99: Optional[float] = None
    running: Optional[float] = None
    waiting: Optional[float] = None
    kv_max: Optional[float] = None
    replicas: Optional[int] = None
    replica_errors: Optional[int] = None


@dataclass
class ServingData:
    drains: list = field(default_factory=list)  # oldest first
    startup_s: Optional[float] = None  # create-job's duration, when the console has it

    @property
    def latest(self) -> Optional[Drain]:
        return self.drains[-1] if self.drains else None


def _num(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if math.isfinite(f) else None


def _int(value: Any) -> Optional[int]:
    f = _num(value)
    return int(f) if f is not None else None


def _row(line: str, mark: str, msg: str) -> Optional[dict]:
    if mark not in line:
        return None
    try:
        row = json.loads(line)
    except ValueError:
        return None
    return row if isinstance(row, dict) and row.get("msg") == msg else None


def parse_line(line: str) -> Optional[Drain]:
    """The drain a console line holds; None for any other line, or a drain without its time or count."""
    row = _row(line, _DRAIN_MARK, "drain_metrics")
    if row is None:
        return None
    at, done = _num(row.get("drained_at")), _int(row.get("completed_requests"))
    if at is None or done is None:
        return None
    return Drain(
        drained_at=at, completed=done,
        latency_p50=_num(row.get("latency_s_p50")), latency_p99=_num(row.get("latency_s_p99")),
        ttft_p50=_num(row.get("ttft_s_p50")), ttft_p99=_num(row.get("ttft_s_p99")),
        queue_p99=_num(row.get("queue_s_p99")),
        running=_num(row.get("num_running_reqs_latest_sum")), waiting=_num(row.get("num_waiting_reqs_latest_sum")),
        kv_max=_num(row.get("kv_cache_usage_max")),
        replicas=_int(row.get("replica_count")), replica_errors=_int(row.get("replica_errors")),
    )


def _startup(line: str) -> Optional[float]:
    row = _row(line, _STARTUP_MARK, "job_metrics")
    return _num(row.get("duration_s")) if row is not None and row.get("op") == "create-job" else None


class ConsoleReader:
    """A saved console (``stdout.log``) that a resumed download only appends to: each read parses
    the whole lines added since the last one, so a deployment's days of console are parsed once."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._offset = 0
        self._data = ServingData()

    def read(self) -> ServingData:
        try:
            size = self._path.stat().st_size
        except FileNotFoundError:
            return self._copy()
        if size < self._offset:  # rewritten, not appended to: start again
            self._offset, self._data = 0, ServingData()
        with self._path.open("rb") as f:
            f.seek(self._offset)
            chunk = f.read()
        end = chunk.rfind(b"\n") + 1  # a line cut short waits for the rest of it
        added = False
        for raw in chunk[:end].splitlines():
            line = raw.decode("utf-8", "replace")
            drain = parse_line(line)
            if drain is not None:
                self._data.drains.append(drain)
                added = True
            elif self._data.startup_s is None:
                self._data.startup_s = _startup(line)
        self._offset += end
        if added:
            self._data.drains.sort(key=lambda d: d.drained_at)  # chunks can land out of order
        return self._copy()

    def _copy(self) -> ServingData:
        return ServingData(list(self._data.drains), self._data.startup_s)


def rate(prev: Optional[Drain], cur: Drain) -> Optional[float]:
    """Requests per second over the gap since ``prev``; None for the first drain."""
    if prev is None:
        return None
    gap = cur.drained_at - prev.drained_at
    return cur.completed / gap if gap > 0 else None


def serving_state(job: dict, data: ServingData) -> str:
    """Where a deployment stands, from its status and whether it has drained yet."""
    if not views.is_live(job):
        return "ended"
    if views.status_word(job.get("status")) != "running":
        return "starting"
    return "serving" if data.drains else "starting (no metrics yet)"
