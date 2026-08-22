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

"""Tests for ACTIVE_SESSIONS — the sticky binding table's live-count projection.

The binding table is the single source of truth for session→replica (router
request_id == gateway session_id); ACTIVE_SESSIONS must move exactly with the
three binding write paths (put / rebind / invalidate) and with LRU eviction —
no more, no less, no second ledger.
"""

from __future__ import annotations

import pytest

from verl.workers.rollout.router.kvcaware.store.data_store import DataStore
from verl.workers.rollout.router.kvcaware.store.per_replica_store import PerReplicaStore
from verl.workers.rollout.router.kvcaware.store.per_request_store import PerRequestStore
from verl.workers.rollout.router.kvcaware.types import MetricKey

pytestmark = [pytest.mark.ut, pytest.mark.cpu]


@pytest.fixture()
def store() -> DataStore:
    """Fresh DataStore over isolated (non-singleton) backing stores."""
    PerReplicaStore._instance = PerReplicaStore()
    PerRequestStore._instance = PerRequestStore()
    return DataStore()


class TestActiveSessions:
    def test_first_bind_counts_up(self, store):
        store.put_sticky_binding("s1", "rep_a")
        store.put_sticky_binding("s2", "rep_a")
        store.put_sticky_binding("s3", "rep_b")
        assert store.get_metric("rep_a", MetricKey.ACTIVE_SESSIONS) == 2
        assert store.get_metric("rep_b", MetricKey.ACTIVE_SESSIONS) == 1

    def test_refresh_same_replica_is_noop(self, store):
        store.put_sticky_binding("s1", "rep_a")
        store.put_sticky_binding("s1", "rep_a")  # same-replica refresh
        assert store.get_metric("rep_a", MetricKey.ACTIVE_SESSIONS) == 1

    def test_rebind_moves_the_count(self, store):
        store.put_sticky_binding("s1", "rep_a")
        store.put_sticky_binding("s1", "rep_b")  # rebind (overload fallback path)
        assert store.get_metric("rep_a", MetricKey.ACTIVE_SESSIONS) == 0
        assert store.get_metric("rep_b", MetricKey.ACTIVE_SESSIONS) == 1

    def test_session_end_releases(self, store):
        store.put_sticky_binding("s1", "rep_a")
        released = store.invalidate_sticky_binding("s1")
        assert released == "rep_a"
        assert store.get_metric("rep_a", MetricKey.ACTIVE_SESSIONS) == 0

    def test_invalidate_is_idempotent(self, store):
        """Duplicate session-end notifications (retry / abort after finalize) must not double-decrement."""
        store.put_sticky_binding("s1", "rep_a")
        assert store.invalidate_sticky_binding("s1") == "rep_a"
        assert store.invalidate_sticky_binding("s1") is None  # second notify: no-op
        assert store.get_metric("rep_a", MetricKey.ACTIVE_SESSIONS) == 0  # never negative

    def test_unknown_session_end_is_noop(self, store):
        assert store.invalidate_sticky_binding("never-bound") is None

    def test_bulk_replica_drop_releases_all(self, store):
        store.put_sticky_binding("s1", "rep_a")
        store.put_sticky_binding("s2", "rep_a")
        store.put_sticky_binding("s3", "rep_b")
        store.invalidate_sticky_replica("rep_a")
        assert store.get_metric("rep_a", MetricKey.ACTIVE_SESSIONS) == 0
        assert store.get_metric("rep_b", MetricKey.ACTIVE_SESSIONS) == 1

    def test_lru_eviction_releases(self, store):
        """A binding row dropping off the LRU tail must settle its count."""
        PerRequestStore._instance = PerRequestStore(max_size=2)
        PerReplicaStore._instance = PerReplicaStore()
        store = DataStore()
        store.put_sticky_binding("s1", "rep_a")
        store.put_sticky_binding("s2", "rep_b")
        store.put_sticky_binding("s3", "rep_b")  # evicts s1 (LRU tail)
        assert store.get_metric("rep_a", MetricKey.ACTIVE_SESSIONS) == 0
        assert store.get_metric("rep_b", MetricKey.ACTIVE_SESSIONS) == 2

    def test_gauge_matches_binding_table(self, store):
        """Invariant: ACTIVE_SESSIONS(replica) == that replica's live bindings."""
        store.put_sticky_binding("s1", "rep_a")
        store.put_sticky_binding("s2", "rep_a")
        store.put_sticky_binding("s3", "rep_a")
        store.put_sticky_binding("s4", "rep_b")
        store.put_sticky_binding("s3", "rep_b")  # rebind s3
        store.invalidate_sticky_binding("s2")  # end s2
        assert store.get_metric("rep_a", MetricKey.ACTIVE_SESSIONS) == 1  # s1
        assert store.get_metric("rep_b", MetricKey.ACTIVE_SESSIONS) == 2  # s3, s4
