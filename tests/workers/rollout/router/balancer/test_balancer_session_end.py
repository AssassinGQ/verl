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

"""Tests for KVCAwareBalancer.on_session_end — the gateway terminal-state entry.

The router's request_id IS the gateway session_id, so ``on_session_end`` just
drops the sticky binding; the binding write paths own the ACTIVE_SESSIONS
count (see store/test_active_sessions.py). These tests pin the balancer-level
behavior: idempotency, callback fan-out, and no interference with routing.
"""

from __future__ import annotations

import pytest

from verl.workers.rollout.router.kvcaware.types import MetricKey

from ._helpers import (
    _make_balancer,
)

pytestmark = [pytest.mark.ut, pytest.mark.cpu]


class TestOnSessionEnd:
    def test_releases_binding_and_count(self):
        """acquire → bind; on_session_end → binding gone, ACTIVE_SESSIONS back to 0."""
        balancer = _make_balancer()
        server_id, _ = balancer.acquire_server("session-0-0-abc", [1, 2, 3])
        assert balancer._store.get_metric(server_id, MetricKey.ACTIVE_SESSIONS) == 1

        balancer.on_session_end("session-0-0-abc")

        assert balancer._store.get_sticky_binding("session-0-0-abc") is None
        assert balancer._store.get_metric(server_id, MetricKey.ACTIVE_SESSIONS) == 0

    def test_idempotent_on_unknown_session(self):
        """A session the router never saw (or a duplicate notify) is a no-op."""
        balancer = _make_balancer()
        balancer.on_session_end("never-bound")  # must not raise
        balancer.on_session_end("never-bound")

    def test_duplicate_notify_does_not_go_negative(self):
        """abort-after-finalize delivers the end event twice; count stays 0."""
        balancer = _make_balancer()
        server_id, _ = balancer.acquire_server("session-1-2-xyz", [1, 2, 3])
        balancer.on_session_end("session-1-2-xyz")
        balancer.on_session_end("session-1-2-xyz")
        assert balancer._store.get_metric(server_id, MetricKey.ACTIVE_SESSIONS) == 0

    def test_fires_on_session_end_callback(self):
        """The ``on_session_end`` hook fires with the session id (reserved for P4 observers)."""
        fired: list[str] = []
        balancer = _make_balancer()
        balancer.register_call_back("on_session_end", fired.append)
        balancer.on_session_end("session-9-9-9")
        assert fired == ["session-9-9-9"]

    def test_ended_session_reroutes_away(self):
        """After session end, the same id is cold again — next acquire rebinds by
        min(active_sessions) instead of short-circuiting to the old replica."""
        balancer = _make_balancer()
        first, _ = balancer.acquire_server("session-0-0-abc", [1, 2, 3])
        balancer.on_session_end("session-0-0-abc")

        second, _ = balancer.acquire_server("session-0-0-abc", [1, 2, 3])
        # Cold again; with all replicas at active_sessions equal, ranking may
        # legitimately return either server — but the binding must exist again.
        assert balancer._store.get_sticky_binding("session-0-0-abc") == second
        assert balancer._store.get_metric(second, MetricKey.ACTIVE_SESSIONS) == 1
        assert balancer._store.get_metric(first, MetricKey.ACTIVE_SESSIONS) == 0 or first == second
