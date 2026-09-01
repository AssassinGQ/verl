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

"""Shared types, imported by many internal modules."""

from .cache_group import (
    FULL_ATTENTION,
    KNOWN_SPEC_KINDS,
    MAMBA,
    SLIDING_WINDOW,
    CacheScope,
    KVCacheEventObservation,
    KVCacheGroupMetadata,
    KVCacheRegistryMetadata,
    PrefixHashChain,
    VLLMCacheCapabilities,
    get_vllm_cache_capabilities,
    normalize_spec_kind,
    resolve_registry_metadata,
    validate_registry,
)
from .emit_spec import EMIT_SPECS, EmitKey
from .layer import Layer
from .metric_spec import METRIC_SPECS, MetricKey
from .overload_mode import OverloadMode
from .slow_cut import SlowCut

__all__ = [
    "CacheScope",
    "EmitKey",
    "EMIT_SPECS",
    "FULL_ATTENTION",
    "KNOWN_SPEC_KINDS",
    "KVCacheEventObservation",
    "KVCacheGroupMetadata",
    "KVCacheRegistryMetadata",
    "Layer",
    "MAMBA",
    "MetricKey",
    "METRIC_SPECS",
    "OverloadMode",
    "PrefixHashChain",
    "SLIDING_WINDOW",
    "SlowCut",
    "VLLMCacheCapabilities",
    "get_vllm_cache_capabilities",
    "normalize_spec_kind",
    "resolve_registry_metadata",
    "validate_registry",
]
