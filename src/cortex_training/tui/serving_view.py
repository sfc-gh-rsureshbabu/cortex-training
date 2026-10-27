# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""The Serving tab of a deployment's job screen: its state and one row per 30 s drain."""

from __future__ import annotations

from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Callable

from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Static
from textual.worker import get_current_worker

from cortex_training.client import NoChunksError
from cortex_training.tui import serving
from cortex_training.tui import theme
from cortex_training.tui import views
from cortex_training.tui.detail_tabs import _Pane
from cortex_training.tui.format import safe_text
from cortex_training.tui.format import short_seconds
from cortex_training.tui.widgets import _warning_text

_MAX_ROWS = 240  # two hours of drains


def _seconds(value: float | None) -> str:
    return "—" if value is None else short_seconds(value)


def _count(value: float | None) -> str:
    return "—" if value is None else f"{int(value)}"


def _when(drained_at: float, tz=None) -> datetime | None:
    try:
        return datetime.fromtimestamp(drained_at, tz=tz)
    except (OverflowError, OSError, ValueError):  # finite, but past what a date holds
        return None


class ServingPane(_Pane):
    BINDINGS = [Binding("r", "refresh", "Refresh")]

    def __init__(self, client, job_id: str, root: Path, *, job: Callable[[], dict],
                 download: Callable[..., list | None], id: str = "serving-tab") -> None:
        super().__init__(client, job_id, job=job, id=id)
        self._root = Path(root)
        self._download = download
        self._reader: serving.ConsoleReader | None = None
        self._data = serving.ServingData()
        self._warnings: list = []  # the last download's, shown in the header
        self._reload_pending = False  # asked for while a load ran: one more once it finishes

    def compose(self) -> ComposeResult:
        yield Static("", id="serving-status")
        with VerticalScroll(id="serving-pane"):
            yield Static("", id="serving-table")

    def focus_target(self):
        return self.query_one("#serving-pane")

    # ── reading ──
    def reload(self) -> None:
        if self._loading:
            # The load running may have downloaded before what this one is for (the job's end).
            self._reload_pending = True
            return
        self._loading = True
        self.redraw()
        self._refresh()

    @work(thread=True, exclusive=True, group="serving")
    def _refresh(self) -> None:
        worker = get_current_worker()
        sid = views.deployment_sub_job_id(self._job())
        try:
            if sid is None:
                self._post(self._finish, None)
                return
            if self._reader is None:
                self._reader = serving.ConsoleReader(self._root / sid / "stdout.log")
            self._post(self._accept, self._reader.read())  # what is cached, while the download runs
            warnings = self._download(sub_job_id=sid, cancelled=lambda: worker.is_cancelled)
            if warnings is None or worker.is_cancelled:
                return
            self._post(self._accept, self._reader.read(), warnings)
            self._post(self._finish, None)
        except NoChunksError:  # nothing uploaded yet: a deployment that just started
            self._post(self._finish, None)
        except Exception as exc:  # noqa: BLE001 - said in the tab's header
            if not worker.is_cancelled:
                self._post(self._finish, views.access_hint(exc, "the serving metrics") or views.describe_error(exc))
        finally:
            if worker.is_cancelled:
                self._post(self._cancelled)

    def _accept(self, data: serving.ServingData, warnings: list | None = None) -> None:
        self._data = data
        if warnings is not None:
            self._warnings = warnings
        self.redraw()

    def _finish(self, error: str | None) -> None:
        self._loading = False
        self._error = error  # rows already shown stay
        if self._reload_pending:
            self._reload_pending = False
            self.reload()
        else:
            self.redraw()

    def _cancelled(self) -> None:
        self._loading = False
        self._reload_pending = False

    # ── drawing ──
    def redraw(self) -> None:
        if not self.is_attached:
            return
        self.query_one("#serving-status", Static).update(self._status_text())
        self.query_one("#serving-table", Static).update(self._table())

    def _status_text(self) -> Text:
        data, latest = self._data, self._data.latest
        text = Text(serving.serving_state(self._job(), data, reading=self._loading), style="bold")
        if latest is not None:
            if latest.replicas is not None:
                text.append(f"  ·  {latest.replicas} replica{'' if latest.replicas == 1 else 's'}", style=theme.MUTED)
            at = _when(latest.drained_at, timezone.utc)
            if at is not None:
                text.append(f"  ·  last drain {views.fmt_ago(at, datetime.now(timezone.utc))}", style=theme.MUTED)
        if data.startup_s is not None:
            text.append(f"  ·  started in {views.fmt_duration(data.startup_s)}", style=theme.MUTED)
        if self._loading:
            text.append("  ·  loading…", style=theme.MUTED)
        if self._warnings:
            text.append("\n" + " ".join(safe_text(_warning_text(self._warnings)).split()), style=theme.WARN)
        if self._error:
            text.append(f"\n[error] {self._error}  (r retries)", style=theme.ERR)
        return text

    def _table(self):
        drains = self._data.drains
        if not drains and self._loading:
            return Text("Reading the console…", style=theme.MUTED)
        if not drains:
            return Text("No drain_metrics yet: a deployment writes one every 30 s once it serves.", style=theme.MUTED)
        table = Table(box=None, pad_edge=False, header_style="bold")
        for name in ("Time", "req/s", "Requests", "Latency p50", "p99", "TTFT p50", "p99", "Queue p99",
                     "Run/Wait", "KV max", "Replicas"):
            table.add_column(name, justify="left" if name == "Time" else "right")
        start = max(0, len(drains) - _MAX_ROWS)
        for i in range(len(drains) - 1, start - 1, -1):
            d, prev = drains[i], drains[i - 1] if i else None
            r = serving.rate(prev, d)
            errors = d.replica_errors or 0
            replicas = Text(_count(d.replicas))
            if errors:
                replicas.append(f" ({errors} err)", style=theme.ERR)
            at = _when(d.drained_at)
            table.add_row(
                "—" if at is None else at.strftime("%H:%M:%S"),
                "—" if r is None else f"{r:.2f}",
                str(d.completed),
                _seconds(d.latency_p50), _seconds(d.latency_p99),
                _seconds(d.ttft_p50), _seconds(d.ttft_p99), _seconds(d.queue_p99),
                f"{_count(d.running)}/{_count(d.waiting)}",
                "—" if d.kv_max is None else f"{d.kv_max:.1%}",  # one decimal: a light load sits well under 1%
                replicas,
            )
        return table
