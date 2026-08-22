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

"""PerRequestStore — bounded per-request state (``request_id → {key: value}``).

Generic per-request key/value storage, LRU-bounded and thread-safe. The store
owns only storage + locking; callers own the keys and their semantics. A full
row is LRU-evicted as a unit when ``request_id`` stops recurring.
``singleton()`` returns the shared instance; tests reset ``_instance``.
"""

from __future__ import annotations

import threading
from typing import Any

from cachetools import LRUCache

from ..logging import get_router_logger

logger = get_router_logger("per-request")

# Max request_ids retained; least-recently-used evicted past this.
DEFAULT_PER_REQUEST_MAX_SIZE = 10000


class PerRequestStore:
    """Singleton per-request state store — ``request_id → {key: value}``, LRU-bounded."""

    _instance: PerRequestStore | None = None

    def __init__(self, max_size: int = DEFAULT_PER_REQUEST_MAX_SIZE) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be > 0, got {max_size}")
        self._max_size = int(max_size)
        self._lock = threading.Lock()
        self._data: LRUCache[str, dict[str, Any]] = LRUCache(maxsize=self._max_size)
        # Eviction observer: called as ``fn(request_id, row)`` with the evicted
        # row (dict of keys) under this store's lock — must be cheap and
        # exception-free. Lets derived counters (ACTIVE_SESSIONS) settle when a
        # binding row drops off the LRU tail, so the count never leaks. None by
        # default; wired by DataStore.
        self.on_row_evicted = None
        logger.info(f"PerRequestStore created: max_size={self._max_size}")

    @classmethod
    def singleton(cls) -> PerRequestStore:
        """Return the shared singleton (tests construct a fresh instance / reset _instance)."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def max_size(self) -> int:
        """Configured per-request table capacity."""
        return self._max_size

    def get(self, request_id: str, key: str, default: Any = None) -> Any:
        """Return the per-request value for ``key`` (``default`` if unset/evicted)."""
        with self._lock:
            row = self._data.get(request_id)
            return default if row is None else row.get(key, default)

    def set(self, request_id: str, key: str, value: Any) -> None:
        """Set ``request_id``'s ``key`` to ``value`` (creates the row if new)."""
        with self._lock:
            row = self._data.get(request_id, {})
            row[key] = value
            self._insert(request_id, row)

    def _insert(self, request_id: str, row: dict[str, Any]) -> None:
        """Insert/touch a row, settling any LRU eviction via ``on_row_evicted``.

        Caller must hold ``self._lock``. cachetools expels the tail row as a
        side effect of ``__setitem__`` on a full cache; catching it keeps
        derived counters (sticky ACTIVE_SESSIONS) from leaking upward.
        """
        if self.on_row_evicted is not None and len(self._data) >= self._data.maxsize:
            evicted_id, evicted_row = next(iter(self._data.items()))
            self._on_row_evicted_safe(evicted_id, evicted_row)
        self._data[request_id] = row  # insert (new) or touch LRU recency (existing)

    def _on_row_evicted_safe(self, request_id: str, row: dict[str, Any]) -> None:
        """Invoke the eviction observer; observer errors are logged, never raised."""
        try:
            self.on_row_evicted(request_id, dict(row))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"on_row_evicted callback failed for {request_id}: {type(exc).__name__}: {exc}")

    def incr(self, request_id: str, key: str, delta: int | float = 1) -> int | float:
        """Add ``delta`` to ``request_id``'s numeric ``key``; return the new value."""
        with self._lock:
            row = self._data.get(request_id, {})
            value = row.get(key, 0) + delta
            row[key] = value
            self._insert(request_id, row)
            return value

    def delete(self, request_id: str, key: str) -> None:
        """Drop ``key`` from ``request_id``'s row (no-op if absent)."""
        with self._lock:
            row = self._data.get(request_id)
            if row is None:
                return
            row.pop(key, None)
            if not row:
                del self._data[request_id]

    def delete_where(self, key: str, value: Any) -> int:
        """Drop ``key`` from every request whose value for it equals ``value``.

        Returns the number of dropped entries so callers can settle derived
        counters (e.g. ACTIVE_SESSIONS on bulk replica invalidation).
        """
        with self._lock:
            dropped = 0
            for request_id, row in [(rid, r) for rid, r in self._data.items() if r.get(key) == value]:
                row.pop(key, None)
                if not row:
                    del self._data[request_id]
                dropped += 1
            return dropped

    def count(self, key: str) -> int:
        """Number of requests that currently have ``key`` set."""
        with self._lock:
            return sum(1 for row in self._data.values() if key in row)

    def reset(self) -> None:
        """Clear all per-request state (test helper; not used on the hot path)."""
        with self._lock:
            self._data.clear()
