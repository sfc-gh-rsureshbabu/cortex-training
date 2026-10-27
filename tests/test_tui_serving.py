# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""A deployment's serving numbers, read from its console's drain_metrics lines (``tui/serving.py``)."""

from __future__ import annotations

import json

from cortex_training.tui import serving


def _drain(at, done, **extra) -> str:
    row = {"ts": "2026-10-26T19:46:24+00:00", "level": "INFO", "msg": "drain_metrics", "op": "drain-metrics",
           "drained_at": at, "completed_requests": done, "latency_s_p50": 0.05, "latency_s_p99": 0.2,
           "ttft_s_p50": 0.01, "ttft_s_p99": 0.03, "queue_s_p99": 0.0, "num_running_reqs_latest_sum": 1.0,
           "num_waiting_reqs_latest_sum": 0.0, "kv_cache_usage_max": 0.25, "replica_count": 1,
           "replica_errors": 0, "stdout_summarized": True, **extra}
    return json.dumps(row)


STARTUP = json.dumps({"msg": "job_metrics", "op": "create-job", "duration_s": 135.9, "error": None})


def test_parse_line_reads_a_drain_and_skips_everything_else():
    d = serving.parse_line(_drain(100.0, 16))
    assert d.drained_at == 100.0 and d.completed == 16 and d.latency_p99 == 0.2 and d.replicas == 1
    assert serving.parse_line("plain text with drain_metrics in it") is None
    assert serving.parse_line(json.dumps({"msg": "something_else", "drained_at": 1})) is None
    assert serving.parse_line(_drain(None, 3)) is None  # no time: no row
    idle = serving.parse_line(_drain(100.0, 0, latency_s_p50=None, latency_s_p99=None, kv_cache_usage_max=None))
    assert idle.completed == 0 and idle.latency_p50 is None and idle.kv_max is None
    assert serving.parse_line(_drain(100.0, 1, latency_s_p99="NaN")).latency_p99 is None


def test_rate_divides_by_the_actual_gap():
    a, b, c = (serving.parse_line(_drain(t, n)) for t, n in ((0.0, 0), (30.0, 15), (90.0, 30)))
    assert serving.rate(None, a) is None
    assert serving.rate(a, b) == 0.5
    assert serving.rate(b, c) == 0.5  # a drain missing in between: 30 over 60 s
    assert serving.rate(b, b) is None


def test_reader_reads_only_what_was_appended(tmp_path):
    path = tmp_path / "stdout.log"
    reader = serving.ConsoleReader(path)
    assert reader.read().drains == []  # no file yet
    path.write_text(STARTUP + "\n" + _drain(0.0, 1) + "\n")
    first = reader.read()
    assert [d.completed for d in first.drains] == [1] and first.startup_s == 135.9
    with path.open("a") as f:
        f.write("INFO not json\n" + _drain(30.0, 2) + "\n")
    assert [d.completed for d in reader.read().drains] == [1, 2]
    assert reader._offset == path.stat().st_size


def test_a_partial_last_line_waits_for_the_next_read(tmp_path):
    path = tmp_path / "stdout.log"
    line = _drain(0.0, 4)
    path.write_text(line[:20])
    reader = serving.ConsoleReader(path)
    assert reader.read().drains == []
    path.write_text(line + "\n")
    assert [d.completed for d in reader.read().drains] == [4]


def test_a_rewritten_file_is_read_again_from_its_start(tmp_path):
    path = tmp_path / "stdout.log"
    path.write_text(_drain(0.0, 1) + "\n" + _drain(30.0, 2) + "\n")
    reader = serving.ConsoleReader(path)
    reader.read()
    path.write_text(_drain(0.0, 9) + "\n")
    assert [d.completed for d in reader.read().drains] == [9]


def test_serving_state():
    none, some = serving.ServingData(), serving.ServingData([serving.parse_line(_drain(0.0, 1))])
    assert serving.serving_state({"status": "PLACING"}, none) == "starting"
    assert serving.serving_state({"status": "RUNNING"}, none) == "starting (no metrics yet)"
    assert serving.serving_state({"status": "RUNNING"}, some) == "serving"
    assert serving.serving_state({"status": "CANCELLED"}, some) == "ended"
