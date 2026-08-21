# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Unit tests for verl.workers.rollout.router"""

from typing import Any

import pytest
import ray
import yaml

from verl.workers.rollout.router import get_router_handle


@ray.remote
class _MockPluginLoadBalancer:
    """Ray actor implementing RequestLoadBalancer Protocol.

    Used as the ``router_class`` for plugin tests via ``importlib`` dynamic
    loading, and directly for Protocol structural checks."""

    def __init__(self, servers: dict[str, Any], router_kwargs: dict):
        self._servers = dict(servers)
        self._inflight: dict[str, int] = {sid: 0 for sid in self._servers}
        self._router_kwargs = dict(router_kwargs)
        self.releases: list[tuple] = []

    def get_router_kwargs(self) -> dict:
        """Return the kwargs dict passed to the constructor."""
        return dict(self._router_kwargs)

    def get_releases(self) -> list[tuple]:
        """Return recorded release_server calls (server_id, prompt_len, request_id)."""
        return list(self.releases)

    def acquire_server(self, request_id: str, prompt_ids: list[int] | None = None) -> tuple[str, Any]:
        if not prompt_ids:
            raise RuntimeError("No available prompt_ids")
        if not self._inflight:
            raise RuntimeError("No available servers")

        sid = min(self._inflight, key=self._inflight.get)
        self._inflight[sid] += 1
        return sid, self._servers[sid]

    def release_server(self, server_id: str, prompt_len: int = 0, request_id: str | None = None) -> None:
        self.releases.append((server_id, prompt_len, request_id))
        if server_id in self._inflight and self._inflight[server_id] > 0:
            self._inflight[server_id] -= 1

    def add_servers(self, servers: dict[str, Any]) -> None:
        for sid, handle in servers.items():
            self._servers[sid] = handle
            self._inflight[sid] = 0

    def remove_servers(self, server_ids: list[str]) -> None:
        for sid in server_ids:
            self._inflight.pop(sid, None)
            self._servers.pop(sid, None)

    def get_all_servers(self) -> list[str]:
        return list(self._inflight.keys())

    def get_status(self) -> dict:
        return {
            "servers": dict(self._inflight),
            "total_inflight": sum(self._inflight.values()),
            "active_servers": len(self._inflight),
        }


class TestRequestLoadBalancer:
    def test_protocol_methods_present(self):
        """All six Protocol methods are callable on _MockPluginLoadBalancer."""
        for name in (
            "acquire_server",
            "release_server",
            "add_servers",
            "remove_servers",
            "get_all_servers",
            "get_status",
        ):
            assert callable(getattr(_MockPluginLoadBalancer, name, None)), f"'{name}' missing or not callable"


@pytest.fixture(scope="module")
def ray_session():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


class TestGetRouterHandleDefault:
    def test_none_config_defaults_to_sticky_inflight(self, ray_session):
        lb = get_router_handle(servers={"s0": None, "s1": None}, router_config_path=None)
        status = ray.get(lb.get_status.remote())
        assert status["active_servers"] == 2
        assert status["total_inflight"] == 0

    def test_empty_router_config_defaults_to_sticky_inflight(self, ray_session):
        lb = get_router_handle(servers={"a": None, "b": None}, router_config_path=None)
        status = ray.get(lb.get_status.remote())
        assert status["active_servers"] == 2


class TestGetRouterHandlePluginExtensionYaml:
    """Plugin router via external YAML (router_config_path), with Hydra
    ``defaults`` composition and pkg:// package-relative resolution."""

    @staticmethod
    def _write_router_yaml(tmp_path, router_class, **kwargs):
        """Write a temporary router YAML and return its path."""
        content = {"router_class": router_class, **kwargs}
        yaml_path = tmp_path / "router.yaml"
        yaml_path.write_text(yaml.dump(content))
        return str(yaml_path)

    def test_missing_yaml_file_raises(self, ray_session):
        config = "/nonexistent/path/router.yaml"
        with pytest.raises(FileNotFoundError, match="Router config file not found"):
            get_router_handle(servers={"s0": None}, router_config_path=config)

    def test_yaml_missing_router_class_raises(self, ray_session, tmp_path):
        yaml_path = tmp_path / "no_class.yaml"
        yaml_path.write_text(yaml.dump({"some_key": "value"}))
        config = str(yaml_path)
        with pytest.raises(ValueError, match="must contain 'router_class'"):
            get_router_handle(servers={"s0": None}, router_config_path=config)

    def test_acquire_least_loaded(self, ray_session, tmp_path):
        yaml_path = self._write_router_yaml(tmp_path, __name__ + "._MockPluginLoadBalancer")
        config = yaml_path
        lb = get_router_handle(servers={"s0": None, "s1": None, "s2": None}, router_config_path=config)
        s_a, _ = ray.get(lb.acquire_server.remote("a", prompt_ids=[1]))
        s_b, _ = ray.get(lb.acquire_server.remote("b", prompt_ids=[1]))
        s_c, _ = ray.get(lb.acquire_server.remote("c", prompt_ids=[1]))
        assert len({s_a, s_b, s_c}) == 3

    def test_add_remove_get_all_servers(self, ray_session, tmp_path):
        yaml_path = self._write_router_yaml(tmp_path, __name__ + "._MockPluginLoadBalancer")
        config = yaml_path
        lb = get_router_handle(servers={"s0": None}, router_config_path=config)
        ray.get(lb.add_servers.remote({"s1": None, "s2": None}))
        assert sorted(ray.get(lb.get_all_servers.remote())) == ["s0", "s1", "s2"]
        ray.get(lb.remove_servers.remote(["s0"]))
        assert ray.get(lb.get_all_servers.remote()) == ["s1", "s2"]

    def test_release_and_get_status(self, ray_session, tmp_path):
        yaml_path = self._write_router_yaml(tmp_path, __name__ + "._MockPluginLoadBalancer")
        config = yaml_path
        lb = get_router_handle(servers={"s0": None, "s1": None}, router_config_path=config)
        ray.get(lb.acquire_server.remote("a", prompt_ids=[1]))  # s0: 1
        ray.get(lb.acquire_server.remote("a", prompt_ids=[1]))  # s0: 2
        ray.get(lb.acquire_server.remote("b", prompt_ids=[1]))  # s1: 1
        assert ray.get(lb.get_status.remote())["total_inflight"] == 3
        ray.get(lb.release_server.remote("s0"))
        assert ray.get(lb.get_status.remote())["total_inflight"] == 2

    def test_empty_pool_raises(self, ray_session, tmp_path):
        yaml_path = self._write_router_yaml(tmp_path, __name__ + "._MockPluginLoadBalancer")
        config = yaml_path
        lb = get_router_handle(servers={"s0": None}, router_config_path=config)
        ray.get(lb.remove_servers.remote(["s0"]))
        with pytest.raises(ray.exceptions.RayTaskError, match="No available servers"):
            ray.get(lb.acquire_server.remote("req", prompt_ids=[1]))

    def test_yaml_forwards_composed_dict_to_constructor(self, ray_session, tmp_path):
        """The whole composed YAML dict (router_class included) is passed as kwargs."""
        yaml_path = self._write_router_yaml(tmp_path, __name__ + "._MockPluginLoadBalancer", extra_param="hello")
        config = yaml_path
        lb = get_router_handle(servers={"s0": None}, router_config_path=config)
        kwargs = ray.get(lb.get_router_kwargs.remote())
        assert kwargs.get("extra_param") == "hello"
        assert kwargs.get("router_class") == __name__ + "._MockPluginLoadBalancer"

    def test_yaml_defaults_block_is_composed(self, ray_session, tmp_path):
        """Hydra ``defaults`` referencing a sibling YAML merges it into the config."""
        (tmp_path / "defaults_group").mkdir()
        (tmp_path / "defaults_group" / "base.yaml").write_text(yaml.dump({"from_group": 42}))
        main = tmp_path / "composed.yaml"
        main.write_text(
            yaml.dump({"defaults": [{"defaults_group": "base"}], "router_class": __name__ + "._MockPluginLoadBalancer"})
        )
        config = str(main)
        lb = get_router_handle(servers={"s0": None}, router_config_path=config)
        kwargs = ray.get(lb.get_router_kwargs.remote())
        # Hydra group defaults nest under the group name ({group: name} form)
        assert kwargs["defaults_group"]["from_group"] == 42


class TestResolveConfigPath:
    """Tests for ``_resolve_config_path`` (pkg:// package-relative URIs).

    Uses the ``verl`` package itself (always importable in tests) instead of
    hardcoding ``uni_agent`` (the production use case is
    ``pkg://uni_agent.llm_router.configs/...``).
    """

    def test_pkg_resolves_to_abs_path(self):
        import os

        from verl.workers.rollout.router import _resolve_config_path

        resolved = _resolve_config_path("pkg://verl/__init__.py")
        assert os.path.isabs(resolved)
        assert resolved.endswith(os.path.join("verl", "__init__.py"))
        assert os.path.isfile(resolved)

    def test_pkg_missing_rel_path_raises(self):
        from verl.workers.rollout.router import _resolve_config_path

        with pytest.raises(ValueError, match="pkg://"):
            _resolve_config_path("pkg://verl")

    def test_pkg_not_found_raises(self):
        from verl.workers.rollout.router import _resolve_config_path

        with pytest.raises(ImportError, match="Package 'no_such_pkg' not found"):
            _resolve_config_path("pkg://no_such_pkg/x.yaml")

    def test_filesystem_path_handling(self):
        import os

        from verl.workers.rollout.router import _resolve_config_path

        assert _resolve_config_path("/abs/path/router.yaml") == "/abs/path/router.yaml"
        rel = _resolve_config_path("relative/router.yaml")
        assert os.path.isabs(rel)
        assert rel == os.path.abspath("relative/router.yaml")

    def test_pkg_file_not_exist(self, ray_session):
        """pkg:// package exists but file missing → FileNotFoundError."""
        config = "pkg://verl/nonexistent_router.yaml"
        with pytest.raises(FileNotFoundError, match="Router config file not found"):
            get_router_handle(servers={"s0": None}, router_config_path=config)


class TestReleaseServerSignature:
    """release_server carries prompt_len + request_id for token/turn-aware balancers."""

    def test_release_accepts_prompt_len_and_request_id(self, ray_session, tmp_path):
        yaml_path = TestGetRouterHandlePluginExtensionYaml._write_router_yaml(
            tmp_path, __name__ + "._MockPluginLoadBalancer"
        )
        config = yaml_path
        lb = get_router_handle(servers={"s0": None, "s1": None}, router_config_path=config)
        sid, _ = ray.get(lb.acquire_server.remote("req-1", prompt_ids=[1, 2, 3]))
        ray.get(lb.release_server.remote(sid, prompt_len=3, request_id="req-1"))
        recs = ray.get(lb.get_releases.remote())
        assert recs == [(sid, 3, "req-1")]
        # In-flight decremented alongside the recording
        assert ray.get(lb.get_status.remote())["total_inflight"] == 0

    def test_default_balancer_release_ignores_extra_args(self, ray_session):
        from verl.workers.rollout.router import GlobalRequestLoadBalancer

        lb = ray.remote(GlobalRequestLoadBalancer).remote(servers={"s0": None, "s1": None})
        sid, _ = ray.get(lb.acquire_server.remote("req-1", prompt_ids=[1, 2, 3]))
        ray.get(lb.release_server.remote(sid, prompt_len=3, request_id="req-1"))
        assert ray.get(lb.get_status.remote())["total_inflight"] == 0
