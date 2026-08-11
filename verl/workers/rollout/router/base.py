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

"""Base class and registry for rollout load-balancer strategies.

Mirrors the pattern in ``verl/workers/engine/base.py`` — the Protocol (like
``BaseEngine``) and ``LoadBalancerRegistry`` (like ``EngineRegistry``) live
together in one zero-dependency module.
"""

import logging
import os
from typing import Any, Optional, Protocol, runtime_checkable

import ray
from omegaconf import OmegaConf

from verl.workers.config import RolloutConfig

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class RequestLoadBalancer(Protocol):
    """Protocol for rollout inference load balancers (structural subtyping)."""

    def __init__(self, servers: dict[str, Any], config: Optional[dict] = None) -> None:
        """Construct the balancer.

        Args:
            servers: ``{server_id: handle}`` of replicas to route across.
            config: Optional implementation config (e.g. cache size, determinism).
        """

    def acquire_server(self, request_id: str, prompt_ids: list[int] | None = None) -> tuple[str, Any]:
        """Acquire a server for the given request.

        Args:
            request_id: Caller-supplied request id; sticky implementations reuse
                the server bound to it when available.
            prompt_ids: Prompt token ids (used by prefix-aware implementations).

        Returns:
            A tuple of ``(server_id, handle)`` in a single atomic call.
        """

    def release_server(self, server_id: str, prompt_len: int = 0, request_id: str | None = None) -> None:
        """Release a server after a request completes.

        Args:
            server_id: The server returned by a prior ``acquire_server``.
            prompt_len: Prompt length (used by some implementations for token accounting).
            request_id: The request id (used by some implementations for turn accounting).
        """

    def add_servers(self, servers: dict[str, Any]) -> None:
        """Atomically add multiple servers to the load balancer pool.

        Args:
            servers: Dict mapping ``server_id`` → ``handle`` for all servers to register.
        """

    def remove_servers(self, server_ids: list[str]) -> None:
        """Atomically remove multiple servers from the load balancer pool.

        Args:
            server_ids: List of server identifiers to remove.
        """

    def get_all_servers(self) -> list[str]:
        """Get list of all active server IDs."""

    def get_status(self) -> dict:
        """Return current load balancer state for debugging."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class LoadBalancerRegistry:
    """Registry for load-balancer strategy classes.

    Strategies are registered by name via the :meth:`register` decorator and
    instantiated through ``get_router_handle``.
    """

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator that registers a balancer class under ``name``.

        Usage::

            @LoadBalancerRegistry.register("global_sticky_inflight")
            @ray.remote
            class GlobalRequestLoadBalancer:
                ...

            @LoadBalancerRegistry.register("kvcaware")
            class KVCAwareBalancer:
                ...
        """

        def decorator(balancer_cls):
            if name in cls._registry:
                raise ValueError(f"Load balancer '{name}' is already registered. Existing: {cls._registry[name]}")
            cls._registry[name] = balancer_cls
            logger.info("Registered load balancer strategy: %s", name)
            return balancer_cls

        return decorator

    @classmethod
    def get_cls(cls, name: str) -> type:
        """Look up a registered balancer class by name."""
        if name not in cls._registry:
            raise ValueError(f"Unknown load balancer strategy: '{name}'. Available strategies: {cls.list_strategies()}")
        return cls._registry[name]

    @classmethod
    def list_strategies(cls) -> list[str]:
        """List all registered strategy names."""
        return sorted(cls._registry.keys())


def _is_ray_actor_class(cls: type) -> bool:
    """Return True if *cls* is a ``@ray.remote`` actor class."""
    return hasattr(cls, "remote") and hasattr(cls, "__ray_metadata__")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _resolve_router_strategy(rollout_config: RolloutConfig) -> str:
    """Return the ``router_strategy`` field, defaulting to ``global_sticky_inflight``."""
    return rollout_config.get("router_strategy", "global_sticky_inflight")


def _router_actor_options() -> dict[str, Any]:
    """Actor options pinning the router to the driver's node.

    Unpinned, a multi-node cluster places the router on a random node each job,
    so its polling loops, logs, and ray-start env snapshot vary run to run.
    Pinning keeps all of that on one deterministic host (the head, where the
    driver lives). soft=False: if this node is gone the run fails loudly
    instead of drifting — the driver on the same node is already a single
    point of failure, so no availability is lost.
    """
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    return {
        "scheduling_strategy": NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(),
            soft=False,
        )
    }


def get_router_handle(servers: dict[str, Any], rollout_config: RolloutConfig) -> Any:
    """Create a load balancer instance from router configuration."""
    strategy = _resolve_router_strategy(rollout_config)
    cls = LoadBalancerRegistry.get_cls(strategy)
    # Materialize the router_config node into a plain dict before mutating it:
    # Hydra-composed nodes are struct-mode, so adding ``full_determinism`` raises
    # ConfigKeyError. from_config accepts a plain dict either way.
    router_cfg = rollout_config.get("router_config", None) or {}
    config = OmegaConf.to_container(OmegaConf.create(router_cfg), resolve=True)
    config["full_determinism"] = getattr(rollout_config, "full_determinism", False)

    options = _router_actor_options()
    if _is_ray_actor_class(cls):
        return cls.options(**options).remote(servers=servers, config=config)
    return ray.remote(cls).options(**options).remote(servers=servers, config=config)
