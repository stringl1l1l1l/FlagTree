"""Deferred TLE raw materialization for the NVIDIA CUDA backend."""

from __future__ import annotations

from typing import Any

from triton._C.libtriton import nvidia
from triton.experimental.tle.raw.source_store import (
    clear_pending_sources,
    list_pending_sources,
)


def deferred_raw_materialize(pm: Any) -> None:
    pending = list_pending_sources()
    if not pending:
        return
    nvidia.passes.tle_raw.deferred_raw_materialize(pending, pm)


def finish_deferred_raw_materialize() -> None:
    if not list_pending_sources():
        return
    clear_pending_sources()
