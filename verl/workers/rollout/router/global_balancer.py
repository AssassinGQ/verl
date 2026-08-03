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

import logging
import os
from typing import Optional

import ray
from cachetools import LRUCache

from .base import LoadBalancerRegistry

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_ROUTING_CACHE_SIZE = 10000


@LoadBalancerRegistry.register("global_sticky_inflight")
@ray.remote
class GlobalRequestLoadBalancer:
    """Global sticky-session + least-inflight load balancer.

    ``request_id`` → server via LRU so multi-turn routes to the same server;
    no sticky hit → least-inflight server. A sticky entry pointing at a removed
    server is invalidated and re-selected. ``full_determinism=True`` makes
    tie-breaking ``hash(request_id)``-based for reproducible runs.
    """

    def __init__(
        self,
        servers: dict[str, ray.actor.ActorHandle],
        config: Optional[dict] = None,
    ):
        """Construct; ``servers`` may be empty (added later via ``add_servers``).

        Reads ``max_cache_size`` and ``full_determinism`` from ``config``.
        """

        config = config or {}
        max_cache_size = config.get("max_cache_size", DEFAULT_ROUTING_CACHE_SIZE)
        full_determinism = config.get("full_determinism", False)

        self._servers: dict[str, ray.actor.ActorHandle] = dict(servers)
        self._inflight_requests: dict[str, int] = {sid: 0 for sid in servers}
        self._request_id_to_server: LRUCache = LRUCache(maxsize=max_cache_size)
        self._full_determinism = full_determinism

    def acquire_server(self, request_id: str, prompt_ids: list[int] | None = None) -> tuple[str, ray.actor.ActorHandle]:
        """Sticky-first (``request_id``→server via LRU), else least-inflight;
        ``full_determinism`` → ``hash(request_id)`` tie-break."""
        # Try sticky session first
        if request_id in self._request_id_to_server:
            server_id = self._request_id_to_server[request_id]
            # Check if server is still in the active pool
            if server_id in self._inflight_requests:
                self._inflight_requests[server_id] += 1
                return server_id, self._servers[server_id]
            # Server was removed, clear stale cache entry and re-select
            del self._request_id_to_server[request_id]

        # Select new server (least-loaded among available)
        if not self._inflight_requests:
            raise RuntimeError("No available servers in load balancer")

        min_count = min(self._inflight_requests.values())
        candidates = [sid for sid, count in self._inflight_requests.items() if count == min_count]
        if len(candidates) == 1:
            server_id = candidates[0]
        elif self._full_determinism:
            # Deterministic tie-breaking: same request_id → same server across runs
            server_id = candidates[hash(request_id) % len(candidates)]
        else:
            server_id = candidates[0]
        self._request_id_to_server[request_id] = server_id
        self._inflight_requests[server_id] += 1
        return server_id, self._servers[server_id]

    def release_server(self, server_id: str, prompt_len: int = 0, request_id: str | None = None) -> None:
        """Release after a request completes (decrement in-flight; ignores ``prompt_len``/``request_id``)."""
        if server_id not in self._inflight_requests:
            return
        if self._inflight_requests[server_id] > 0:
            self._inflight_requests[server_id] -= 1

    def add_servers(self, servers: dict[str, ray.actor.ActorHandle]) -> None:
        """Register ``servers`` into the pool (in-flight counts start at 0)."""
        for sid, handle in servers.items():
            self._inflight_requests[sid] = 0
            self._servers[sid] = handle
        logger.info(f"[GlobalLoadBalancer] added {len(servers)} servers")

    def remove_servers(self, server_ids: list[str]) -> None:
        """Drop ``server_ids`` from the pool (clears in-flight + handle)."""
        for sid in server_ids:
            self._inflight_requests.pop(sid, None)
            self._servers.pop(sid, None)
        logger.info(f"[GlobalLoadBalancer] removed {len(server_ids)} servers")

    def get_inflight_count(self, server_id: str) -> int:
        """Get number of in-flight requests for a server."""
        return self._inflight_requests.get(server_id, 0)

    def get_all_servers(self) -> list[str]:
        """Get list of all active server IDs."""
        return list(self._inflight_requests.keys())

    def clear_sticky_cache(self) -> dict:
        """Clear sticky-session cache so subsequent ``acquire_server()`` calls
        re-select least-inflight. Returns ``{cleared_entries, server_loads}``."""
        cleared = len(self._request_id_to_server)
        self._request_id_to_server.clear()
        logger.info(
            f"[GlobalLoadBalancer] Sticky cache cleared: {cleared} entries dropped. "
            f"Server loads: {dict(self._inflight_requests)}"
        )
        return {
            "cleared_entries": cleared,
            "server_loads": dict(self._inflight_requests),
        }

    def get_status(self) -> dict:
        """Return current load balancer state for debugging."""
        return {
            "servers": dict(self._inflight_requests),
            "total_inflight": sum(self._inflight_requests.values()),
            "active_servers": len(self._inflight_requests),
            "registered_handles": list(self._servers.keys()),
        }
