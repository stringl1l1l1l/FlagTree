"""Benchmark TLE mega linear+RMSNorm against normal non-mega Triton kernels.

Example:
  conda run -n flagtree python python/tutorials/tle/mega/bench/bench_linear_rmsnorm.py \
      --shape 8:128 --warmup 10 --iters 50
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Callable

import torch
import triton
import triton.experimental.tle.mega as tlem
import triton.language as tl

MEGA_ROOT = Path(__file__).resolve().parents[1]
if str(MEGA_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGA_ROOT))

from kernels.linear_rmsnorm import (  # noqa: E402
    _TASK_PAYLOAD_SIGNATURE,
    _linear_rmsnorm_baseline_linear_kernel,
    _launch_linear_rmsnorm_baseline,
    _materialize_linear_rmsnorm_scheduler,
    _triton_task_body_ttir,
    _validate_linear_rmsnorm_mega_bounds,
    linear_rmsnorm_reference,
)
from kernels._common import require_cuda_contiguous  # noqa: E402
from kernels.norm import _rms_norm_kernel  # noqa: E402
from triton._internal_testing import default_alloc_fn  # noqa: E402
from triton._C.libtriton import ir, tle as tle_ir  # noqa: E402
from triton.runtime import driver  # noqa: E402
from triton.runtime._allocation import NullAllocator  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Linear+RMSNorm mega scheduler vs non-mega Triton baseline")
    parser.add_argument("--shape", action="append", default=[],
                        help="Benchmark shape as rows:hidden. Can be repeated.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--eps", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mega-workers", type=int, default=None,
                        help="Override cooperative scheduler worker CTA count. Defaults to min(SMs, task instances).")
    parser.add_argument("--skip-noop", action="store_true",
                        help="Skip the same-topology no-op scheduler diagnostic.")
    parser.add_argument("--output-dir", default="build/mega/linear_rmsnorm")
    return parser


def _parse_shapes(values: list[str]) -> list[tuple[int, int]]:
    if not values:
        values = ["1:2", "2:4", "8:16", "16:64", "8:128"]
    shapes = []
    for value in values:
        try:
            rows, hidden = value.split(":", 1)
            shapes.append((int(rows), int(hidden)))
        except ValueError as exc:
            raise ValueError(f"invalid --shape {value!r}; expected rows:hidden") from exc
    return shapes


def _time_cuda(fn: Callable[[], None], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / max(iters, 1))


def _task_graph_stats(rows: int, hidden: int) -> dict[str, int | float]:
    num_instances = rows * hidden * 2 + rows
    num_edges = rows * hidden * 3
    dispatch_checks = num_instances * (num_instances + 1) // 2
    return {
        "num_instances": num_instances,
        "num_edges": num_edges,
        "avg_static_dispatch_checks": (num_instances + 1) / 2.0,
        "total_static_dispatch_checks": dispatch_checks,
    }


@triton.jit(repr=lambda _: "noop_linear_body")
def _noop_linear_body(
    row,
    h,
    x,
    linear_weight,
    rms_weight,
    scratch,
    partial,
    stat,
    out,
    HIDDEN: tl.constexpr,
):
    return


@triton.jit(repr=lambda _: "noop_reduce_body")
def _noop_reduce_body(
    row,
    x,
    linear_weight,
    rms_weight,
    scratch,
    partial,
    stat,
    out,
    HIDDEN: tl.constexpr,
):
    return


@triton.jit(repr=lambda _: "noop_apply_body")
def _noop_apply_body(
    row,
    h,
    x,
    linear_weight,
    rms_weight,
    scratch,
    partial,
    stat,
    out,
    HIDDEN: tl.constexpr,
):
    return


def _noop_compute_kernels_ttir(hidden: int, target) -> str:
    return "\n\n".join(
        [
            _triton_task_body_ttir(
                _noop_linear_body,
                "noop_linear_body",
                {
                    "row": "i32",
                    "h": "i32",
                    **_TASK_PAYLOAD_SIGNATURE,
                    "HIDDEN": "constexpr",
                },
                {"HIDDEN": hidden},
                target,
            ),
            _triton_task_body_ttir(
                _noop_reduce_body,
                "noop_reduce_body",
                {
                    "row": "i32",
                    **_TASK_PAYLOAD_SIGNATURE,
                    "HIDDEN": "constexpr",
                },
                {"HIDDEN": hidden},
                target,
            ),
            _triton_task_body_ttir(
                _noop_apply_body,
                "noop_apply_body",
                {
                    "row": "i32",
                    "h": "i32",
                    **_TASK_PAYLOAD_SIGNATURE,
                    "HIDDEN": "constexpr",
                },
                {"HIDDEN": hidden},
                target,
            ),
        ]
    )


def _noop_mega_graph_ttir(rows: int, hidden: int) -> str:
    graph = tlem.mega_graph("linear_rmsnorm_noop_scheduler_kernel")
    graph.arg("x", "!tt.ptr<f32>")
    graph.arg("linear_weight", "!tt.ptr<f32>")
    graph.arg("rms_weight", "!tt.ptr<f32>")
    linear_grid = graph.grid("linear_to_rms", shape=(rows, hidden),
                             fields={"scratch": "!tt.ptr<f32>", "partial": "!tt.ptr<f32>"})
    stat_grid = graph.grid("rms_stat", shape=(rows, ), fields={"stat": "!tt.ptr<f32>"})
    out_grid = graph.grid("rms_out", shape=(rows, hidden), fields={"out": "!tt.ptr<f32>"})
    graph.task(_noop_linear_body, name="linear_tile", domain=(rows, hidden), reads={},
               writes={linear_grid: tlem.affine_map(2, 0, 1)})
    graph.task(_noop_reduce_body, name="rms_reduce", domain=(rows, ),
               reads={linear_grid: tlem.affine_map(1, 0, tlem.ALL)},
               writes={stat_grid: tlem.affine_map(1, 0)})
    graph.task(_noop_apply_body, name="rms_apply", domain=(rows, hidden),
               reads={
                   linear_grid: tlem.affine_map(2, 0, 1),
                   stat_grid: tlem.affine_map(2, 0),
               },
               writes={out_grid: tlem.affine_map(2, 0, 1)})
    return graph.to_mlir_function().rstrip()


def _materialize_noop_scheduler(rows: int, hidden: int, root: Path, target):
    from triton.compiler import make_backend

    backend = make_backend(target)
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    ttir_path = root / f"linear_rmsnorm_noop_scheduler_{rows}x{hidden}.ttir"
    ttir_path.write_text("\n".join([
        'module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {',
        _noop_compute_kernels_ttir(hidden, target),
        "",
        _noop_mega_graph_ttir(rows, hidden),
        "}",
    ]))
    module = ir.parse_mlir_module(str(ttir_path), context)
    module.context = context

    pm = ir.pass_manager(context)
    pm.enable_debug()
    tle_ir.passes.add_verify_task_graph(pm)
    tle_ir.passes.add_analyze_task_graph(pm)
    tle_ir.passes.add_materialize_task_scheduler(pm)
    tle_ir.passes.add_materialize_task_runtime_state(pm)
    pm.run(module, "materialize_linear_rmsnorm_noop_scheduler")

    ttgir_path = root / f"linear_rmsnorm_noop_scheduler_{rows}x{hidden}.ttgir"
    materialized = str(module)
    ttgir_path.write_text(materialized)
    return ttgir_path, materialized


def _prepare_mega_runner(
    x: torch.Tensor,
    linear_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    *,
    eps: float,
    mega_workers: int | None,
) -> tuple[Callable[[], None], torch.Tensor, int]:
    require_cuda_contiguous("x", x)
    require_cuda_contiguous("linear_weight", linear_weight)
    require_cuda_contiguous("rms_weight", rms_weight)
    rows = x.shape[0]
    hidden = linear_weight.shape[0]
    total_instances = _validate_linear_rmsnorm_mega_bounds(rows, hidden)

    target = driver.active.get_current_target()
    if target.backend != "cuda":
        raise RuntimeError(f"linear+RMSNorm mega benchmark requires CUDA target, got {target.backend!r}")

    out = torch.empty((rows, hidden), device=x.device, dtype=x.dtype)
    scratch = torch.empty_like(out)
    partial = torch.empty_like(out)
    stat = torch.empty((rows, ), device=x.device, dtype=torch.float32)
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    default_workers = max(1, min(num_sms, total_instances))
    num_workers = default_workers if mega_workers is None else mega_workers
    if num_workers <= 0:
        raise ValueError("--mega-workers must be positive")

    with tempfile.TemporaryDirectory() as tmp_dir:
        ttgir_path, _ = _materialize_linear_rmsnorm_scheduler(rows, hidden, eps, Path(tmp_dir), target)
        scheduler = triton.compile(str(ttgir_path), target=target)
    if not scheduler.metadata.launch_cooperative_grid:
        raise RuntimeError("generated TLE scheduler did not request cooperative-grid launch")
    if "membar.gl;" not in scheduler.asm.get("ptx", ""):
        raise RuntimeError("generated TLE scheduler is missing device-scope publish fence")

    def run() -> None:
        scheduler[(num_workers, 1, 1)](x, linear_weight, rms_weight, scratch, partial, stat, out)

    return run, out, num_workers


def _prepare_noop_runner(
    x: torch.Tensor,
    linear_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    *,
    mega_workers: int | None,
) -> tuple[Callable[[], None], int]:
    rows = x.shape[0]
    hidden = linear_weight.shape[0]
    total_instances = _validate_linear_rmsnorm_mega_bounds(rows, hidden)

    target = driver.active.get_current_target()
    if target.backend != "cuda":
        raise RuntimeError(f"linear+RMSNorm noop benchmark requires CUDA target, got {target.backend!r}")

    out = torch.empty((rows, hidden), device=x.device, dtype=x.dtype)
    scratch = torch.empty_like(out)
    partial = torch.empty_like(out)
    stat = torch.empty((rows, ), device=x.device, dtype=torch.float32)
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    default_workers = max(1, min(num_sms, total_instances))
    num_workers = default_workers if mega_workers is None else mega_workers
    if num_workers <= 0:
        raise ValueError("--mega-workers must be positive")

    with tempfile.TemporaryDirectory() as tmp_dir:
        ttgir_path, _ = _materialize_noop_scheduler(rows, hidden, Path(tmp_dir), target)
        scheduler = triton.compile(str(ttgir_path), target=target)
    if not scheduler.metadata.launch_cooperative_grid:
        raise RuntimeError("generated no-op TLE scheduler did not request cooperative-grid launch")

    def run() -> None:
        scheduler[(num_workers, 1, 1)](x, linear_weight, rms_weight, scratch, partial, stat, out)

    return run, num_workers


def _prepare_baseline_runner(
    x: torch.Tensor,
    linear_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    *,
    eps: float,
) -> tuple[Callable[[], None], torch.Tensor, Callable[[], None], Callable[[], None]]:
    rows = x.shape[0]
    hidden = linear_weight.shape[0]
    linear_out = torch.empty((rows, hidden), device=x.device, dtype=x.dtype)
    out = torch.empty_like(linear_out)

    def run() -> None:
        _launch_linear_rmsnorm_baseline(x, linear_weight, rms_weight, linear_out, out, eps=eps)

    def run_linear() -> None:
        _linear_rmsnorm_baseline_linear_kernel[(rows, 1)](
            x,
            linear_weight,
            linear_out,
            hidden,
            triton.next_power_of_2(hidden),
        )

    def run_rms() -> None:
        _rms_norm_kernel[(rows, )](
            out,
            linear_out,
            rms_weight,
            out.stride(-2),
            out.stride(-1),
            linear_out.stride(-2),
            linear_out.stride(-1),
            rms_weight.stride(-1),
            hidden,
            eps,
            triton.next_power_of_2(hidden),
        )

    return run, out, run_linear, run_rms


def _bench_shape(rows: int, hidden: int, *, eps: float, warmup: int, iters: int, seed: int,
                 mega_workers: int | None, skip_noop: bool) -> dict:
    torch.manual_seed(seed + rows * 1000 + hidden)
    x = torch.randn((rows, 1), device="cuda", dtype=torch.float32)
    linear_weight = torch.randn((hidden, 1), device="cuda", dtype=torch.float32)
    rms_weight = torch.randn((hidden, ), device="cuda", dtype=torch.float32)

    ref = linear_rmsnorm_reference(x, linear_weight, rms_weight, eps=eps)
    mega_run, mega_out, actual_workers = _prepare_mega_runner(x, linear_weight, rms_weight, eps=eps,
                                                              mega_workers=mega_workers)
    baseline_run, baseline_out, baseline_linear_run, baseline_rms_run = _prepare_baseline_runner(
        x,
        linear_weight,
        rms_weight,
        eps=eps,
    )
    noop_run = None
    if not skip_noop:
        noop_run, noop_workers = _prepare_noop_runner(x, linear_weight, rms_weight, mega_workers=mega_workers)
        if noop_workers != actual_workers:
            raise RuntimeError("no-op scheduler worker count diverged from mega scheduler worker count")

    triton.set_allocator(default_alloc_fn)
    try:
        mega_run()
    finally:
        triton.set_allocator(NullAllocator())
    baseline_run()
    torch.cuda.synchronize()
    torch.testing.assert_close(mega_out, ref, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(baseline_out, ref, rtol=1.0e-5, atol=1.0e-5)

    triton.set_allocator(default_alloc_fn)
    try:
        mega_ms = _time_cuda(mega_run, warmup=warmup, iters=iters)
        noop_ms = _time_cuda(noop_run, warmup=warmup, iters=iters) if noop_run is not None else None
    finally:
        triton.set_allocator(NullAllocator())
    baseline_ms = _time_cuda(baseline_run, warmup=warmup, iters=iters)
    baseline_linear_run()
    baseline_linear_ms = _time_cuda(baseline_linear_run, warmup=warmup, iters=iters)
    baseline_rms_ms = _time_cuda(baseline_rms_run, warmup=warmup, iters=iters)

    result = {
        "rows": rows,
        "hidden": hidden,
        "mega_workers": actual_workers,
        **_task_graph_stats(rows, hidden),
        "mega_ms": mega_ms,
        "mega_noop_ms": noop_ms if noop_ms is not None else "",
        "mega_noop_fraction": (noop_ms / mega_ms) if noop_ms is not None and mega_ms > 0.0 else "",
        "baseline_ms": baseline_ms,
        "baseline_linear_ms": baseline_linear_ms,
        "baseline_rms_ms": baseline_rms_ms,
        "speedup_baseline_over_mega": baseline_ms / mega_ms if mega_ms > 0.0 else float("inf"),
        "mega_abs_err": float((mega_out - ref).abs().max().item()),
        "baseline_abs_err": float((baseline_out - ref).abs().max().item()),
        "warmup": warmup,
        "iters": iters,
    }
    return result


def _write_results(rows: list[dict], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "linear_rmsnorm_mega_vs_baseline.csv"
    jsonl_path = output_dir / "linear_rmsnorm_mega_vs_baseline.jsonl"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return csv_path, jsonl_path


def main() -> None:
    args = _build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("bench_linear_rmsnorm requires CUDA")

    rows = []
    for shape_rows, hidden in _parse_shapes(args.shape):
        result = _bench_shape(
            shape_rows,
            hidden,
            eps=args.eps,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed,
            mega_workers=args.mega_workers,
            skip_noop=args.skip_noop,
        )
        rows.append(result)
        noop_text = "" if result["mega_noop_ms"] == "" else f"{result['mega_noop_ms']:.6f} ms"
        print(
            f"rows={shape_rows:4d} hidden={hidden:4d} "
            f"mega={result['mega_ms']:.6f} ms baseline={result['baseline_ms']:.6f} ms "
            f"noop={noop_text} "
            f"baseline/mega={result['speedup_baseline_over_mega']:.3f}x "
            f"checks={int(result['total_static_dispatch_checks'])}"
        )

    csv_path, jsonl_path = _write_results(rows, Path(args.output_dir))
    print(f"Wrote {csv_path}")
    print(f"Wrote {jsonl_path}")


if __name__ == "__main__":
    main()
