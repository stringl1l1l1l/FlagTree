"""Mega-style linear + RMSNorm example using the TLE task scheduler."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import triton
import triton.experimental.tle.mega as tlem
import triton.language as tl

from ._common import next_power_of_2, require_cuda_contiguous, row_stride
from .norm import _rms_norm_kernel, rms_norm


_TASK_PAYLOAD_SIGNATURE = {
    "x": "*fp32",
    "linear_weight": "*fp32",
    "rms_weight": "*fp32",
    "scratch": "*fp32",
    "partial": "*fp32",
    "stat": "*fp32",
    "out": "*fp32",
}


@triton.jit
def _linear_rmsnorm_baseline_linear_kernel(
    x,
    linear_weight,
    linear_out,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    block_h = tl.program_id(1)
    h = block_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = h < HIDDEN
    x_value = tl.load(x + row)
    weight = tl.load(linear_weight + h, mask=mask, other=0.0)
    tl.store(linear_out + row * HIDDEN + h, x_value * weight, mask=mask)


@triton.jit(repr=lambda _: "linear_tile_body")
def _linear_tile_body(
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
    idx = row * HIDDEN + h
    xv = tl.load(x + row)
    wv = tl.load(linear_weight + h)
    lin = xv * wv
    tl.store(scratch + idx, lin)
    tl.store(partial + idx, lin * lin)


@triton.jit(repr=lambda _: "rms_reduce_body")
def _rms_reduce_body(
    row,
    x,
    linear_weight,
    rms_weight,
    scratch,
    partial,
    stat,
    out,
    HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
):
    row_off = row * HIDDEN
    acc = tl.full((), 0.0, tl.float32)
    for h in tl.static_range(0, HIDDEN):
        acc += tl.load(partial + row_off + h)
    mean = acc / HIDDEN
    tl.store(stat + row, tl.rsqrt(mean + EPS))


@triton.jit(repr=lambda _: "rms_apply_body")
def _rms_apply_body(
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
    idx = row * HIDDEN + h
    lin = tl.load(scratch + idx)
    inv = tl.load(stat + row)
    weight = tl.load(rms_weight + h)
    tl.store(out + idx, lin * inv * weight)


def _extract_private_task_body(module_text: str, symbol_name: str) -> str:
    lines = module_text.splitlines()
    start = None
    needle = f"tt.func public @{symbol_name}"
    for idx, line in enumerate(lines):
        if needle in line:
            if start is not None:
                raise RuntimeError(f"Triton task body @{symbol_name} was emitted more than once")
            start = idx
    if start is None:
        raise RuntimeError(f"Triton task body @{symbol_name} was not emitted")

    depth = 0
    end = None
    for idx in range(start, len(lines)):
        depth += lines[idx].count("{")
        depth -= lines[idx].count("}")
        if idx > start and depth == 0:
            end = idx
            break
    if end is None:
        raise RuntimeError(f"Triton task body @{symbol_name} has an unterminated MLIR function body")

    func_text = "\n".join(lines[start:end + 1])
    return func_text.replace(needle, f"tt.func private @{symbol_name}", 1)


def _triton_task_body_ttir(fn, symbol_name: str, signature: dict[str, str], constexprs: dict[str, object],
                           target) -> str:
    from triton._C.libtriton import ir
    from triton.compiler import ASTSource, make_backend

    backend = make_backend(target)
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    options = backend.parse_options({"num_warps": 4, "num_ctas": 1})
    source = ASTSource(fn=fn, signature=signature, constexprs=constexprs, attrs={})
    module = source.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    return _extract_private_task_body(module.str_nodebug(), symbol_name)


def _linear_rmsnorm_compute_kernels_ttir(hidden: int, eps: float, target) -> str:
    """Lower Triton task-body functions to private TTIR callees for the scheduler."""
    return "\n\n".join(
        [
            _triton_task_body_ttir(
                _linear_tile_body,
                "linear_tile_body",
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
                _rms_reduce_body,
                "rms_reduce_body",
                {
                    "row": "i32",
                    **_TASK_PAYLOAD_SIGNATURE,
                    "HIDDEN": "constexpr",
                    "EPS": "constexpr",
                },
                {"HIDDEN": hidden, "EPS": eps},
                target,
            ),
            _triton_task_body_ttir(
                _rms_apply_body,
                "rms_apply_body",
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


def _linear_rmsnorm_mega_graph_ttir(rows: int, hidden: int) -> str:
    """Return the top-level mega graph that binds task bodies to dependencies."""
    graph = tlem.mega_graph("linear_rmsnorm_mega_scheduler_kernel")
    graph.arg("x", "!tt.ptr<f32>")
    graph.arg("linear_weight", "!tt.ptr<f32>")
    graph.arg("rms_weight", "!tt.ptr<f32>")
    linear_grid = graph.grid("linear_to_rms", shape=(rows, hidden),
                             fields={"scratch": "!tt.ptr<f32>", "partial": "!tt.ptr<f32>"})
    stat_grid = graph.grid("rms_stat", shape=(rows, ), fields={"stat": "!tt.ptr<f32>"})
    out_grid = graph.grid("rms_out", shape=(rows, hidden), fields={"out": "!tt.ptr<f32>"})
    graph.task(_linear_tile_body, name="linear_tile", domain=(rows, hidden), reads={},
               writes={linear_grid: tlem.affine_map(2, 0, 1)})
    graph.task(_rms_reduce_body, name="rms_reduce", domain=(rows, ),
               reads={linear_grid: tlem.affine_map(1, 0, tlem.ALL)},
               writes={stat_grid: tlem.affine_map(1, 0)})
    graph.task(_rms_apply_body, name="rms_apply", domain=(rows, hidden),
               reads={
                   linear_grid: tlem.affine_map(2, 0, 1),
                   stat_grid: tlem.affine_map(2, 0),
               },
               writes={out_grid: tlem.affine_map(2, 0, 1)})
    return graph.to_mlir_function().rstrip()


def _linear_rmsnorm_scheduler_ttir(rows: int, hidden: int, eps: float, target) -> str:
    """Return a full TTIR module with compute task bodies and mega graph entry."""
    lines = [
        'module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {',
        _linear_rmsnorm_compute_kernels_ttir(hidden, eps, target),
        "",
        _linear_rmsnorm_mega_graph_ttir(rows, hidden),
        "}",
    ]
    return "\n".join(lines)


def _validate_linear_rmsnorm_inputs(
    x: torch.Tensor,
    linear_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    *,
    caller: str,
) -> tuple[int, int]:
    require_cuda_contiguous("x", x)
    require_cuda_contiguous("linear_weight", linear_weight)
    require_cuda_contiguous("rms_weight", rms_weight)
    if x.dim() != 2:
        raise ValueError(f"x must be 2D, got shape={tuple(x.shape)}")
    if linear_weight.dim() != 2 or linear_weight.shape[1] != x.shape[1]:
        raise ValueError("linear_weight must be [hidden, in_features]")
    if rms_weight.dim() != 1 or rms_weight.shape[0] != linear_weight.shape[0]:
        raise ValueError("rms_weight must be 1D with length matching linear_weight.shape[0]")
    if x.dtype != torch.float32 or linear_weight.dtype != torch.float32 or rms_weight.dtype != torch.float32:
        raise ValueError(f"{caller} currently requires float32 tensors")
    rows, in_features = x.shape
    hidden = linear_weight.shape[0]
    if in_features != 1:
        raise ValueError(f"{caller} currently requires in_features == 1")
    if rows <= 0 or hidden <= 0:
        raise ValueError("rows and hidden must be positive")
    return rows, hidden


def _validate_linear_rmsnorm_mega_bounds(rows: int, hidden: int) -> int:
    if hidden > 128:
        raise ValueError("linear_rmsnorm_mega_scheduler currently limits hidden <= 128 to keep generated IR bounded")
    total_instances = rows * hidden * 2 + rows
    if total_instances > 4096:
        raise ValueError("linear_rmsnorm_mega_scheduler currently limits total task instances <= 4096")
    return total_instances


def _launch_linear_rmsnorm_baseline(
    x: torch.Tensor,
    linear_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    linear_out: torch.Tensor,
    out: torch.Tensor,
    *,
    eps: float,
) -> None:
    rows, hidden = x.shape[0], linear_weight.shape[0]
    block_h = next_power_of_2(hidden)
    _linear_rmsnorm_baseline_linear_kernel[(rows, triton.cdiv(hidden, block_h))](
        x,
        linear_weight,
        linear_out,
        hidden,
        block_h,
    )
    _rms_norm_kernel[(rows, )](
        out,
        linear_out,
        rms_weight,
        row_stride(out),
        out.stride(-1),
        row_stride(linear_out),
        linear_out.stride(-1),
        rms_weight.stride(-1),
        hidden,
        eps,
        block_h,
    )


def linear_rmsnorm_triton_baseline(
    x: torch.Tensor,
    linear_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    *,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    """Compute ``rmsnorm(linear(x))`` with normal non-mega Triton kernels."""
    rows, hidden = _validate_linear_rmsnorm_inputs(
        x,
        linear_weight,
        rms_weight,
        caller="linear_rmsnorm_triton_baseline",
    )
    linear_out = torch.empty((rows, hidden), device=x.device, dtype=x.dtype)
    _linear_rmsnorm_baseline_linear_kernel[(rows, 1)](
        x,
        linear_weight,
        linear_out,
        hidden,
        next_power_of_2(hidden),
    )
    return rms_norm(linear_out, (hidden, ), rms_weight, eps)


def _materialize_linear_rmsnorm_scheduler(rows: int, hidden: int, eps: float, root: Path, target):
    from triton._C.libtriton import ir, tle as tle_ir
    from triton.compiler import make_backend

    backend = make_backend(target)
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    ttir_path = root / f"linear_rmsnorm_scheduler_{rows}x{hidden}.ttir"
    ttir_path.write_text(_linear_rmsnorm_scheduler_ttir(rows, hidden, eps, target))
    module = ir.parse_mlir_module(str(ttir_path), context)
    module.context = context

    pm = ir.pass_manager(context)
    pm.enable_debug()
    tle_ir.passes.add_verify_task_graph(pm)
    tle_ir.passes.add_analyze_task_graph(pm)
    tle_ir.passes.add_materialize_task_scheduler(pm)
    tle_ir.passes.add_materialize_task_runtime_state(pm)
    pm.run(module, "materialize_linear_rmsnorm_mega_scheduler")

    ttgir_path = root / f"linear_rmsnorm_scheduler_{rows}x{hidden}.ttgir"
    materialized = str(module)
    ttgir_path.write_text(materialized)
    return ttgir_path, materialized


def linear_rmsnorm_mega_scheduler(
    x: torch.Tensor,
    linear_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    *,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    """Compute ``rmsnorm(linear(x))`` with a generated TLE task scheduler."""
    from triton.runtime import driver
    from triton.runtime._allocation import NullAllocator
    from triton._internal_testing import default_alloc_fn

    rows, hidden = _validate_linear_rmsnorm_inputs(
        x,
        linear_weight,
        rms_weight,
        caller="linear_rmsnorm_mega_scheduler",
    )
    total_instances = _validate_linear_rmsnorm_mega_bounds(rows, hidden)

    target = driver.active.get_current_target()
    if target.backend != "cuda":
        raise RuntimeError(f"linear_rmsnorm_mega_scheduler requires CUDA target, got {target.backend!r}")

    out = torch.empty((rows, hidden), device=x.device, dtype=x.dtype)
    scratch = torch.empty_like(out)
    partial = torch.empty_like(out)
    stat = torch.empty((rows, ), device=x.device, dtype=torch.float32)
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    num_workers = max(1, min(num_sms, total_instances))

    with tempfile.TemporaryDirectory() as tmp_dir:
        ttgir_path, _ = _materialize_linear_rmsnorm_scheduler(rows, hidden, eps, Path(tmp_dir), target)
        scheduler = triton.compile(str(ttgir_path), target=target)
        if not scheduler.metadata.launch_cooperative_grid:
            raise RuntimeError("generated TLE scheduler did not request cooperative-grid launch")
        if "membar.gl;" not in scheduler.asm.get("ptx", ""):
            raise RuntimeError("generated TLE scheduler is missing device-scope publish fence")
        triton.set_allocator(default_alloc_fn)
        try:
            scheduler[(num_workers, 1, 1)](x, linear_weight, rms_weight, scratch, partial, stat, out)
        finally:
            triton.set_allocator(NullAllocator())
    return out


def linear_rmsnorm_reference(
    x: torch.Tensor,
    linear_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    *,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    """Torch reference for the linear + RMSNorm tutorial paths."""
    linear = (x.to(torch.float32) @ linear_weight.to(torch.float32).t()).to(x.dtype)
    linear_f32 = linear.to(torch.float32)
    inv_rms = torch.rsqrt(torch.mean(linear_f32 * linear_f32, dim=-1, keepdim=True) + eps)
    return (linear_f32 * inv_rms * rms_weight.to(torch.float32)).to(x.dtype)


def linear_rmsnorm_mega_task_grid(
    x: torch.Tensor,
    linear_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    *,
    eps: float = 1.0e-5,
    block_h: int = 64,
    block_k: int = 128,
) -> torch.Tensor:
    """Compatibility wrapper for the final generated scheduler path."""
    if block_h != 64 or block_k != 128:
        raise ValueError("linear_rmsnorm_mega_task_grid is now a scheduler wrapper and no longer accepts tile knobs")
    return linear_rmsnorm_mega_scheduler(x, linear_weight, rms_weight, eps=eps)


def validate_linear_rmsnorm_mega_task_grid() -> None:
    """Compatibility validation wrapper for the generated scheduler path."""
    if not torch.cuda.is_available():
        raise RuntimeError("validate_linear_rmsnorm_mega_task_grid requires CUDA")

    validate_linear_rmsnorm_mega_scheduler()


def validate_linear_rmsnorm_mega_scheduler() -> None:
    """Run CUDA correctness checks for the generated task-graph scheduler path."""
    if not torch.cuda.is_available():
        raise RuntimeError("validate_linear_rmsnorm_mega_scheduler requires CUDA")

    torch.manual_seed(0)
    for rows, hidden in [(1, 2), (2, 4), (3, 5)]:
        x = torch.randn((rows, 1), device="cuda", dtype=torch.float32)
        linear_weight = torch.randn((hidden, 1), device="cuda", dtype=torch.float32)
        rms_weight = torch.randn((hidden, ), device="cuda", dtype=torch.float32)

        out = linear_rmsnorm_mega_scheduler(x, linear_weight, rms_weight, eps=1.0e-5)
        ref = linear_rmsnorm_reference(x, linear_weight, rms_weight, eps=1.0e-5)
        torch.testing.assert_close(out, ref, rtol=1.0e-5, atol=1.0e-5)
