# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for plot_metrics.py — parser, dispatch window math, and the data pipeline.

``plot_metrics.py`` lives under ``examples/`` (not an importable package) and is
normally only exercised via the matplotlib-rendering ``main()``. These tests
cover the matplotlib-free data path (parse → collect → derive → prepare) so the
dispatch panels (dispatched / completed / cumul-completed / avg-turn / RPM /
avg-prompt-len) and the existing evidence panels stay correct in CI without a
display. We load the module from its file path with ``importlib``.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta
from pathlib import Path

import pytest

_PLOT_PATH = Path(__file__).resolve().parents[4] / "examples" / "kvc_aware_router" / "plot_metrics.py"


def _load_plot_module():
    spec = importlib.util.spec_from_file_location("_plot_metrics_under_test", _PLOT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


pm = _load_plot_module()

pytestmark = [pytest.mark.ut, pytest.mark.cpu]

T0 = datetime(2026, 1, 1, 0, 0, 0)


def _td(seconds: int) -> timedelta:
    return timedelta(seconds=seconds)


def _disp_line(rep: str, d: int, c: int, s: int, offset_s: int, p: int = 0) -> str:
    ts = (T0 + _td(offset_s)).strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"{ts} INFO collector router-dispatch replica={rep} dispatched={d} "
        f"completed={c} turn_sum={s} prompt_len_sum={p} [dispatch #{offset_s}]"
    )


# ── LogParser: router-dispatch anchor + regex ──────────────────────────────


class TestDispatchParsing:
    def test_is_signal_matches_dispatch(self):
        assert pm.LogParser.is_signal(
            "2026-01-01 00:00:00 router-dispatch replica=s0 dispatched=1 completed=0 turn_sum=1"
        )

    def test_parse_dispatch_fields(self):
        line = _disp_line("s0", 10, 8, 12, 0, p=100)
        ts, kind, g = pm.LogParser.parse(line)
        assert kind == "dispatch"
        assert ts == T0
        assert g == {
            "rep": "s0",
            "dispatched": "10",
            "completed": "8",
            "turn_sum": "12",
            "prompt_len_sum": "100",
        }

    def test_parse_dispatch_fields_without_prompt_len(self):
        # Older logs (pre-request-length tracking) omit prompt_len_sum — the
        # regex group is optional and defaults to None (→ 0 in collect).
        line = (
            f"{T0.strftime('%Y-%m-%d %H:%M:%S')} INFO collector router-dispatch "
            "replica=s0 dispatched=10 completed=8 turn_sum=12 [dispatch #0]"
        )
        _, kind, g = pm.LogParser.parse(line)
        assert kind == "dispatch"
        assert g["prompt_len_sum"] is None  # absent in the regex match

    def test_parse_ignores_trailing_fields(self):
        line = (
            "2026-01-01 00:00:00 INFO router-dispatch replica=10.0.0.1:8000 dispatched=42 "
            "completed=40 turn_sum=77 prompt_len_sum=900 [dispatch #9]"
        )
        _, kind, g = pm.LogParser.parse(line)
        assert kind == "dispatch"
        assert g["rep"] == "10.0.0.1:8000" and g["turn_sum"] == "77"
        assert g["prompt_len_sum"] == "900"

    def test_evidence_still_parses(self):
        # Regression: the new anchor must not break existing evidence parsing.
        line = (
            "2026-01-01 00:00:00 vllm-evidence replica=127.0.0.1:8000 kv=0.02 usage=0.4 "
            "run=4 wait=0 | x | prefill=10 cached=90 (hit=90.0%) decode=5 external=0 flops=100"
        )
        _, kind, g = pm.LogParser.parse(line)
        assert kind == "evidence"
        assert g["rep"] == "127.0.0.1:8000"


# ── _trailing_deltas + the 4 dispatch panels (per-replica) ─────────────────


def _sample_dispatch():
    """One replica's cumulative snapshots spaced 60 s apart; window is 300 s (5 min).

    At t+300 the trailing-300s baseline is the t+0 point; at t+360 it is t+60.
    The 5th element is prompt_len_sum (cumul. dispatched prompt length).
    """
    return [
        (T0 + _td(0), 10, 8, 12, 100),
        (T0 + _td(60), 20, 18, 30, 200),
        (T0 + _td(120), 30, 28, 50, 300),
        (T0 + _td(300), 40, 38, 75, 400),
        (T0 + _td(360), 50, 48, 95, 500),
    ]


class TestTrailingDeltas:
    def test_baseline_before_start_is_zero(self):
        deltas = pm._trailing_deltas(_sample_dispatch(), 300.0)
        assert deltas[0] == (T0 + _td(0), 10, 8, 12, 100)
        assert deltas[1] == (T0 + _td(60), 20, 18, 30, 200)
        assert deltas[2] == (T0 + _td(120), 30, 28, 50, 300)

    def test_trailing_window_subtracts_baseline(self):
        deltas = pm._trailing_deltas(_sample_dispatch(), 300.0)
        assert deltas[3] == (T0 + _td(300), 30, 30, 63, 300)  # baseline = t+0 point (400-100)
        assert deltas[4] == (T0 + _td(360), 30, 30, 65, 300)  # baseline = t+60 point (500-200)

    def test_empty_buffer(self):
        assert pm._trailing_deltas([], 300.0) == []


class TestDispatchPanels:
    def panels_by_key(self):
        return {p.key: p for p in pm.build_panels(560.0)}

    def test_dispatched_panel_values(self):
        series = self.panels_by_key()["dispatched"].derive_dispatch({"s0": _sample_dispatch()})
        pts = series["s0"]
        assert pts[-2][1] == 30.0
        assert pts[-1][1] == 30.0

    def test_completed_panel_values(self):
        series = self.panels_by_key()["completed"].derive_dispatch({"s0": _sample_dispatch()})
        assert series["s0"][-1][1] == 30.0

    def test_cumulative_completed_panel_values(self):
        # Reads the raw cumulative ``completed`` counter verbatim (no window delta).
        series = self.panels_by_key()["completed_total"].derive_dispatch({"s0": _sample_dispatch()})
        pts = series["s0"]
        assert [p[1] for p in pts] == [8.0, 18.0, 28.0, 38.0, 48.0]
        assert pts[-1][1] == 48.0  # lifetime total, not the trailing-window delta

    def test_cumulative_completed_panel_is_monotonic_per_replica(self):
        # Lifetime totals only climb; verify non-decreasing across both replicas.
        buf = {
            "s0": _sample_dispatch(),
            "s1": [(T0 + _td(0), 3, 3, 3, 30), (T0 + _td(360), 7, 7, 9, 70)],
        }
        series = self.panels_by_key()["completed_total"].derive_dispatch(buf)
        for rep in ("s0", "s1"):
            vals = [p[1] for p in series[rep]]
            assert all(vals[i + 1] >= vals[i] for i in range(len(vals) - 1))

    def test_avg_turn_panel_values(self):
        series = self.panels_by_key()["avg_turn"].derive_dispatch({"s0": _sample_dispatch()})
        pts = series["s0"]
        assert pts[-2][1] == pytest.approx(2.1)  # 63/30
        assert pts[-1][1] == pytest.approx(65 / 30)

    def test_rpm_panel_values(self):
        series = self.panels_by_key()["rpm"].derive_dispatch({"s0": _sample_dispatch()})
        assert series["s0"][-1][1] == pytest.approx(6.0)  # 30 completed / 5 min

    def test_avg_prompt_len_panel_values(self):
        series = self.panels_by_key()["avg_prompt_len"].derive_dispatch({"s0": _sample_dispatch()})
        pts = series["s0"]
        # last two deltas: prompt_len_sum_delta=300, dispatched_delta=30 → 10.0.
        assert pts[-2][1] == pytest.approx(10.0)
        assert pts[-1][1] == pytest.approx(10.0)

    def test_empty_dispatch_returns_empty(self):
        for key in ("dispatched", "completed", "completed_total", "avg_turn", "rpm", "avg_prompt_len"):
            assert self.panels_by_key()[key].derive_dispatch({}) == {}

    def test_avg_turn_nan_when_no_dispatches_in_window(self):
        flat = [(T0 + _td(i * 60), 5, 5, 9, 9) for i in range(6)]
        series = self.panels_by_key()["avg_turn"].derive_dispatch({"s0": flat})
        v = series["s0"][-1][1]
        assert v != v  # NaN check

    def test_avg_prompt_len_nan_when_no_dispatches_in_window(self):
        flat = [(T0 + _td(i * 60), 5, 5, 9, 9) for i in range(6)]
        series = self.panels_by_key()["avg_prompt_len"].derive_dispatch({"s0": flat})
        v = series["s0"][-1][1]
        assert v != v  # NaN check

    def test_panels_are_per_replica(self):
        # Two replicas produce two independent series.
        buf = {
            "s0": _sample_dispatch(),
            "s1": [(T0 + _td(0), 3, 3, 3, 30), (T0 + _td(360), 7, 7, 9, 70)],
        }
        series = self.panels_by_key()["dispatched"].derive_dispatch(buf)
        assert set(series) == {"s0", "s1"}
        assert series["s1"][-1][1] == 4.0  # 7 - 3


# ── End-to-end pipeline: log lines → collect → derive → prepare ─────────────


class TestPipelineEndToEnd:
    # (dispatched, completed, turn_sum, prompt_len_sum) per snapshot.
    _S0 = [(10, 8, 12, 100), (20, 18, 30, 200), (30, 28, 50, 300), (40, 38, 75, 400), (50, 48, 95, 500)]
    _S1 = [(5, 4, 6, 50), (9, 8, 11, 90), (14, 13, 17, 140), (20, 19, 24, 200), (26, 25, 31, 260)]
    _OFFSETS = [0, 60, 120, 300, 360]

    def _run(self, tmp_path):
        log = tmp_path / "router.log"
        lines = []
        # _S0/_S1 tuples are (dispatched, completed, turn_sum, prompt_len_sum);
        # spread positionally so offset_s and p land in the right slots.
        for off, (d, c, s, p) in zip(self._OFFSETS, self._S0, strict=False):
            lines.append(_disp_line("s0", d, c, s, off, p))
        for off, (d, c, s, p) in zip(self._OFFSETS, self._S1, strict=False):
            lines.append(_disp_line("s1", d, c, s, off, p))
        log.write_text("\n".join(lines) + "\n")

        panels = pm.build_panels(560.0)
        bundle = pm.collect([str(log)], panels)
        for p in panels:
            d = p.derive_dispatch(bundle.dispatch)
            if d is not None:
                bundle.series[p.key] = d
        data = pm.prepare(panels, bundle.series, None, 0, 560.0 * 1e12)
        return bundle, data

    def test_collect_dispatch_buffer_per_replica(self, tmp_path):
        bundle, _ = self._run(tmp_path)
        assert bundle.n_dispatch == 10
        assert set(bundle.dispatch) == {"s0", "s1"}
        assert len(bundle.dispatch["s0"]) == 5

    def test_prepare_dispatch_panels_per_replica(self, tmp_path):
        _, data = self._run(tmp_path)
        # Both replicas appear in every dispatch panel.
        for key in ("dispatched", "completed", "completed_total", "avg_turn", "rpm", "avg_prompt_len"):
            assert set(data[key]) == {"s0", "s1"}
        # s0 last dispatched delta = 50 - 20 (baseline at t+60) = 30.
        assert data["dispatched"]["s0"][-1][1] == 30.0
        # s1 last dispatched delta = 26 - 9 = 17.
        assert data["dispatched"]["s1"][-1][1] == 17.0
        # s0 rpm = 30 completed / 5 min = 6.0.
        assert data["rpm"]["s0"][-1][1] == pytest.approx(6.0)
        # s0 avg prompt len = (500-200) prompt_len_sum / 30 dispatched = 10.0.
        assert data["avg_prompt_len"]["s0"][-1][1] == pytest.approx(10.0)
        # s1 avg prompt len = (260-90) / 17 ≈ 10.0.
        assert data["avg_prompt_len"]["s1"][-1][1] == pytest.approx(10.0)
        # Cumulative completed (raw counter, lifetime): s0 ends at 48, s1 at 25.
        assert data["completed_total"]["s0"][-1][1] == 48.0
        assert data["completed_total"]["s1"][-1][1] == 25.0


# ── walltime: first/last timestamp across ALL log lines + subtitle format ──


def _noise_line(offset_s: int) -> str:
    """A non-signal line (no router anchor) that still carries a loguru ts.

    Mirrors warmup / teardown chatter around the signal window — the walltime
    scan must include these, while t_min/t_max (signal window) must not.
    """
    ts = (T0 + _td(offset_s)).strftime("%Y-%m-%d %H:%M:%S")
    return f"{ts} INFO vllm.engine warming up some unrelated thing"


class TestWalltime:
    def test_fmt_walltime_formats_hms(self):
        assert pm._fmt_walltime(T0, T0 + _td(0)) == "0h 0m 0s"
        assert pm._fmt_walltime(T0, T0 + _td(3661)) == "1h 1m 1s"
        assert pm._fmt_walltime(T0, T0 + _td(6 * 3600 + 19)) == "6h 0m 19s"

    def test_fmt_walltime_missing_bounds(self):
        assert pm._fmt_walltime(None, T0) == "-"
        assert pm._fmt_walltime(T0, None) == "-"

    def test_fmt_walltime_negative_is_dash(self):
        # Out-of-order bounds (t_max < t_min) should not render a negative span.
        assert pm._fmt_walltime(T0 + _td(10), T0) == "-"

    def test_collect_log_window_includes_non_signal_lines(self, tmp_path):
        # Signal lines span t+0..t+360; noise lines extend to t-30 and t+400.
        # log_t_min/log_t_max (all lines) must reach the noise bounds; the
        # signal window t_min/t_max must NOT — that's the whole point.
        log = tmp_path / "router.log"
        lines = [
            _noise_line(-30),  # before the first signal
            _disp_line("s0", 10, 8, 12, 0),
            _disp_line("s0", 50, 48, 95, 360),
            _noise_line(400),  # after the last signal
        ]
        log.write_text("\n".join(lines) + "\n")
        bundle = pm.collect([str(log)], pm.build_panels(560.0))
        assert bundle.log_t_min == T0 + _td(-30)
        assert bundle.log_t_max == T0 + _td(400)
        # signal window is narrower — bounded by the dispatch lines.
        assert bundle.t_min == T0 + _td(0)
        assert bundle.t_max == T0 + _td(360)

    def test_collect_log_window_empty_when_no_ts(self, tmp_path):
        # Log with no parseable timestamp at all → both bounds None → subtitle "-".
        log = tmp_path / "no_ts.log"
        log.write_text("noise with no timestamp\nmore noise\n")
        bundle = pm.collect([str(log)], pm.build_panels(560.0))
        assert bundle.log_t_min is None
        assert bundle.log_t_max is None
        assert pm._fmt_walltime(bundle.log_t_min, bundle.log_t_max) == "-"


# ── LogParser: route= latency anchor + regex ───────────────────────────────


def _route_line(req: str, srv: str, ms: float, offset_s: int) -> str:
    ts = (T0 + _td(offset_s)).strftime("%Y-%m-%d %H:%M:%S")
    return f"{ts} INFO balancer request={req} routed to server={srv} (ranking=['{srv}'], route={ms:.2f}ms, strategy=[])"


class TestRouteParsing:
    def test_is_signal_matches_route(self):
        assert pm.LogParser.is_signal(_route_line("abc", "s0", 0.30, 0))

    def test_parse_route_fields(self):
        line = _route_line("abc-123", "8.92.9.147:3997", 0.30, 0)
        ts, kind, g = pm.LogParser.parse(line)
        assert kind == "route"
        assert ts == T0
        assert g["route"] == "0.30"
        assert g["srv"] == "8.92.9.147:3997"

    def test_route_stats_line_not_matched_as_route(self):
        # The aggregated route-stats line carries mean=/max= (ms) but must NOT be
        # captured by the route parser — its anchor is "routed to server".
        line = (
            f"{T0.strftime('%Y-%m-%d %H:%M:%S')} INFO balancer route-stats: "
            "calls=64 total=0.012s mean=0.19ms max=1.65ms (flushed every 64 calls)"
        )
        assert not pm.LogParser.is_signal(line)
        assert pm.LogParser.parse(line) is None


# ── collect(): route lines → global flat buffer ────────────────────────────


class TestRouteCollect:
    def test_collect_gathers_route_points_globally(self, tmp_path):
        log = tmp_path / "r.log"
        log.write_text(_route_line("a", "s1", 0.30, 0) + "\n" + _route_line("b", "s2", 0.45, 1) + "\n")
        bundle = pm.collect([str(log)], pm.build_panels(560.0))
        assert len(bundle.route_lat) == 2
        assert bundle.route_lat[0] == (T0, 0.30)
        assert bundle.route_lat[1] == (T0 + _td(1), 0.45)
        assert bundle.n_route == 2
        # route lines must NOT create fake replicas (route latency is global).
        assert "__global__" not in bundle.replicas
        assert bundle.replicas == set()


# ── RouteLatencyPanel: derive_route + build_panels wiring ───────────────────


class TestRoutePanel:
    def test_route_panel_derives_global_series(self):
        panel = pm.RouteLatencyPanel()
        lat = [(T0 + _td(i), round(i * 0.1, 2)) for i in range(3)]
        assert panel.derive_route(lat) == {"__global__": lat}
        assert panel.derive_route([]) is None

    def test_route_panel_in_build_panels(self):
        panels = pm.build_panels(560.0)
        route_panels = [p for p in panels if isinstance(p, pm.RouteLatencyPanel)]
        assert len(route_panels) == 1
        # RouteLoadPanel retired — the per-replica load now comes from the
        # strategy ``score():`` line as a ScoreFieldPanel keyed route_load_scored.
        load_panel = next(p for p in panels if p.key == "route_load_scored")
        assert isinstance(load_panel, pm.ScoreFieldPanel)
        assert load_panel.hline == 0.9  # default load_threshold mirrors the old RouteLoadPanel
        # the full score-component set is wired (load + s_cache + inflight + capacity 4)
        assert {
            "route_load_scored",
            "s_cache",
            "inflight",
            "avail",
            "need",
            "inflight_tokens",
            "remaining",
        } <= {p.key for p in panels}


# ── LogParser: strategy score(): line (prefix-load-aware + capacity-token-aware) ─


def _score_pla_line(
    rep: str,
    *,
    kv: float,
    run: int,
    wait: int,
    inflight: int | None,
    load: float,
    s_load: float,
    gpu_hit: float,
    s_cache: float,
    score: float = 0.2,
    offset_s: int = 0,
) -> str:
    """A prefix-load-aware ``score(): replica=...`` line.

    ``inflight=None`` omits the fragment (older ``uni_agent`` logs predate the
    inflight term); the regex group is optional either way.
    """
    ts = (T0 + _td(offset_s)).strftime("%Y-%m-%d %H:%M:%S")
    inflight_frag = f" inflight={inflight}" if inflight is not None else ""
    return (
        f"{ts} INFO strategy score(): replica={rep} kv={kv} running={run} waiting={wait}"
        f"{inflight_frag} → load={load:.4f} s_load={s_load:.4f} | gpu_hit={gpu_hit:.2f}"
        f" s_cache={s_cache:.4f} (0.30·cache + 0.70·load) → score={score:.4f}"
    )


def _score_cap_line(
    rep: str,
    *,
    kv_perc: float,
    gpu_hit: float,
    inflight: int,
    avail: float,
    need: float,
    mbt,
    inflight_tokens: int,
    remaining: float,
    offset_s: int = 0,
    winner: bool = False,
) -> str:
    """A capacity-token-aware ``score(): replica=...`` line (winner tag optional)."""
    ts = (T0 + _td(offset_s)).strftime("%Y-%m-%d %H:%M:%S")
    tag = " ← WINNER" if winner else ""
    return (
        f"{ts} INFO strategy score(): replica={rep} kv_perc={kv_perc:.3f} gpu_hit={gpu_hit:.3f}"
        f" inflight={inflight} avail={avail:.0f} need={need:.0f} max_num_batched_tokens={mbt}"
        f" inflight_tokens={inflight_tokens} remaining={remaining:.0f}{tag}"
    )


class TestScoreParsing:
    def test_is_signal_matches_score_line(self):
        assert pm.LogParser.is_signal(
            _score_pla_line("s0", kv=0.1, run=1, wait=0, inflight=2, load=0.25, s_load=0.75, gpu_hit=0.5, s_cache=0.35)
        )
        assert pm.LogParser.is_signal(
            _score_cap_line(
                "s0",
                kv_perc=0.1,
                gpu_hit=0.5,
                inflight=2,
                avail=1000,
                need=100,
                mbt=8192,
                inflight_tokens=200,
                remaining=800,
            )
        )

    def test_parse_score_pla_fields(self):
        line = _score_pla_line(
            "8.92.9.147:41425",
            kv=0.12,
            run=4,
            wait=1,
            inflight=3,
            load=0.25,
            s_load=0.75,
            gpu_hit=0.50,
            s_cache=0.35,
            offset_s=5,
        )
        ts, kind, g = pm.LogParser.parse(line)
        assert kind == "score_pla"
        assert ts == T0 + _td(5)
        assert g["rep"] == "8.92.9.147:41425"
        assert g["load"] == "0.2500"
        assert g["s_cache"] == "0.3500"
        assert g["inflight"] == "3"

    def test_parse_score_pla_without_inflight(self):
        # Older uni_agent logs omit inflight — the group must be optional.
        line = _score_pla_line(
            "s0", kv=0.0, run=0, wait=0, inflight=None, load=0.0, s_load=1.0, gpu_hit=0.0, s_cache=0.0
        )
        _, kind, g = pm.LogParser.parse(line)
        assert kind == "score_pla"
        assert g["inflight"] is None
        assert g["load"] == "0.0000"

    def test_parse_score_cap_fields(self):
        line = _score_cap_line(
            "s1",
            kv_perc=0.123,
            gpu_hit=0.450,
            inflight=3,
            avail=12345,
            need=678,
            mbt=8192,
            inflight_tokens=2048,
            remaining=9999,
            offset_s=2,
            winner=True,
        )
        ts, kind, g = pm.LogParser.parse(line)
        assert kind == "score_cap"
        assert ts == T0 + _td(2)
        assert g["rep"] == "s1"
        assert g["avail"] == "12345"
        assert g["need"] == "678"
        assert g["inflight_tokens"] == "2048"
        assert g["remaining"] == "9999"

    def test_combined_scores_line_not_matched_as_score(self):
        # The aggregate "score(): COMBINED scores:" line carries no replica= field
        # and must NOT be captured by the score parser.
        line = f"{T0.strftime('%Y-%m-%d %H:%M:%S')} INFO strategy score(): COMBINED scores: s0=0.2, s1=0.3"
        assert pm.LogParser.parse(line) is None


# ── collect(): score() lines → per-replica ScoreFieldPanel series ───────────


class TestScoreCollect:
    def test_collect_feeds_score_pla_panels(self, tmp_path):
        log = tmp_path / "s.log"
        log.write_text(
            _score_pla_line(
                "s0", kv=0.1, run=1, wait=0, inflight=2, load=0.25, s_load=0.75, gpu_hit=0.5, s_cache=0.35, offset_s=0
            )
            + "\n"
            + _score_pla_line(
                "s0", kv=0.2, run=2, wait=1, inflight=3, load=0.40, s_load=0.60, gpu_hit=0.6, s_cache=0.42, offset_s=1
            )
            + "\n"
        )
        bundle = pm.collect([str(log)], pm.build_panels(560.0))
        assert [p[1] for p in bundle.series["route_load_scored"]["s0"]] == [0.25, 0.40]
        assert [p[1] for p in bundle.series["s_cache"]["s0"]] == [0.35, 0.42]
        assert [p[1] for p in bundle.series["inflight"]["s0"]] == [2.0, 3.0]
        assert bundle.n_score == 2
        assert bundle.replicas == {"s0"}

    def test_collect_feeds_score_cap_panels(self, tmp_path):
        log = tmp_path / "c.log"
        log.write_text(
            _score_cap_line(
                "s1",
                kv_perc=0.1,
                gpu_hit=0.5,
                inflight=3,
                avail=1000,
                need=100,
                mbt=8192,
                inflight_tokens=200,
                remaining=800,
                offset_s=0,
            )
            + "\n"
        )
        bundle = pm.collect([str(log)], pm.build_panels(560.0))
        assert [p[1] for p in bundle.series["avail"]["s1"]] == [1000.0]
        assert [p[1] for p in bundle.series["need"]["s1"]] == [100.0]
        assert [p[1] for p in bundle.series["inflight_tokens"]["s1"]] == [200.0]
        assert [p[1] for p in bundle.series["remaining"]["s1"]] == [800.0]

    def test_evidence_panels_not_contaminated_by_score_line(self, tmp_path):
        # CRITICAL: evidence and prefix score lines BOTH carry kv/run/wait. The
        # evidence-derived panels (run/wait/kv-load) must come ONLY from evidence
        # lines — the score-line run/wait must NOT double-count into them.
        log = tmp_path / "m.log"
        log.write_text(
            _score_pla_line(
                "s0", kv=0.1, run=9, wait=9, inflight=2, load=0.25, s_load=0.75, gpu_hit=0.5, s_cache=0.35, offset_s=0
            )
            + "\n"
        )
        bundle = pm.collect([str(log)], pm.build_panels(560.0))
        assert not bundle.series.get("run", {}).get("s0")  # score-line run=9 leaked?
        assert not bundle.series.get("wait", {}).get("s0")  # score-line wait=9 leaked?
        # the score-line load lands in route_load_scored, NOT the evidence KV-load panel.
        assert not bundle.series.get("load", {}).get("s0")
        assert [p[1] for p in bundle.series["route_load_scored"]["s0"]] == [0.25]


# ── StickyOverloadPanel: derive_loads (is_overload buffer only) ─────────────


class TestStickyOverloadPanel:
    def test_derive_loads_single_arg_is_overload(self):
        # After RouteLoadPanel retired, derive_loads takes only the is_overload
        # buffer (route_load arg dropped) and copies it verbatim per replica.
        panel = pm.StickyOverloadPanel(load_threshold=0.9)
        is_overload = {"s0": [(T0, 0.3), (T0 + _td(1), 0.4)], "s1": [(T0, 0.95)]}
        assert panel.derive_loads(is_overload) == {
            "s0": [(T0, 0.3), (T0 + _td(1), 0.4)],
            "s1": [(T0, 0.95)],
        }
        assert panel.derive_loads({}) is None  # empty → None (panel hidden)
