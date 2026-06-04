"""
GLU with Triton (TLE Tutorial style) — autotune fixed
======================================================

This is test_122_v1 with the baseline BLOCK_D properly autotuned per
(N, D, dtype). Two key fixes vs the previous attempts:

  1. PRE-COMPILE pass: every BLOCK_D candidate is warmed/compiled BEFORE
     do_bench is called for timing. This avoids the trap where the first
     compile latency of a large BLOCK_D (like 1024) is counted into the
     do_bench result, making it look slower than BLOCK_D=256 and forcing
     a fallback to the initial value.

  2. NO swallowed exceptions. If a BLOCK_D truly cannot compile (e.g.
     out-of-shared-memory), the error is printed so it's visible.

GLU(x) = x[:, :D] * sigmoid(x[:, D:])
where x has shape (N, 2D).
"""

# %%
# Setup
# -----

import argparse
import math
import sys

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 2 ** math.ceil(math.log2(x))


# %%
# Kernels (Triton baseline — test_122_v1 style: chunk + a/b pointers)
# -------------------------------------------------------------------


@triton.jit
def glu_baseline(
    a_ptr, b_ptr, out_ptr, D,
    stride_an, stride_bn, stride_outn,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    a = tl.load(a_ptr + pid_n * stride_an + offs_d,
                mask=mask_d, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + pid_n * stride_bn + offs_d,
                mask=mask_d, other=0.0).to(tl.float32)

    result = a * (1.0 / (1.0 + tl.exp(-b)))

    tl.store(out_ptr + pid_n * stride_outn + offs_d,
             result.to(out_ptr.dtype.element_ty), mask=mask_d)


# %%
# Kernels (TLE static extract_tile)
# ---------------------------------


@triton.jit
def glu_extract_static(
    x_ptr, out_ptr, D,
    stride_xn, stride_outn,
    D_P2:  tl.constexpr,
    D2_P2: tl.constexpr,
):
    pid_n = tl.program_id(0)

    offs = tl.arange(0, D2_P2)
    mask = offs < (D * 2)
    halo = tl.load(x_ptr + pid_n * stride_xn + offs,
                   mask=mask, other=0.0)

    a_tile = tle.extract_tile(halo, index=[0],
                              tile_shape=[D_P2], strides=[D_P2])
    b_tile = tle.extract_tile(halo, index=[1],
                              tile_shape=[D_P2], strides=[D_P2])

    a_f32     = a_tile.to(tl.float32)
    b_f32     = b_tile.to(tl.float32)
    sigmoid_b = 1.0 / (1.0 + tl.exp(-b_f32))
    result    = a_f32 * sigmoid_b

    offs_d = tl.arange(0, D_P2)
    mask_d = offs_d < D
    tl.store(out_ptr + pid_n * stride_outn + offs_d,
             result.to(out_ptr.dtype.element_ty), mask=mask_d)


# %%
# Python wrappers
# ---------------


def triton_glu(x, BLOCK_D):
    N, D2 = x.shape
    D     = D2 // 2
    out   = torch.empty((N, D), device=x.device, dtype=x.dtype)
    a, b  = torch.chunk(x, 2, dim=-1)
    grid  = (N, triton.cdiv(D, BLOCK_D))
    glu_baseline[grid](
        a, b, out, D,
        a.stride(0), b.stride(0), out.stride(0),
        BLOCK_D=BLOCK_D,
    )
    return out


def tle_glu(x):
    N, D2 = x.shape
    D     = D2 // 2
    out   = torch.empty((N, D), device=x.device, dtype=x.dtype)
    d_p2  = _next_pow2(D)
    d2_p2 = _next_pow2(D2)
    glu_extract_static[(N,)](
        x, out, D,
        x.stride(0), out.stride(0),
        D_P2=d_p2,
        D2_P2=d2_p2,
    )
    return out


def torch_glu(x):
    return torch.nn.functional.glu(x, dim=-1)


# %%
# Baseline BLOCK_D autotune  ★ FIXED ★
# ------------------------------------
# Two-phase: (1) warm-compile all candidates, (2) timing pass.
# Without phase 1, the first-compile time of BLOCK_D=1024 gets folded into
# do_bench and the autotuner mis-elects BLOCK_D=256 as "fastest".


_BLOCK_D_CANDIDATES = [64, 128, 256, 512, 1024]
_block_d_cache: dict = {}


def best_block_d(N: int, D: int, dtype: torch.dtype,
                 verbose: bool = False) -> int:
    """Pick the fastest BLOCK_D for the baseline on (N, D, dtype). Cached."""
    key = (N, D, dtype)
    if key in _block_d_cache:
        return _block_d_cache[key]

    x = _make_input(N, D, dtype)

    # ── Phase 1: warm-compile every candidate ────────────────────────────
    # This is the critical fix — drag every BLOCK_D through the JIT once
    # so the timing phase below measures pure execution, not compilation.
    valid_bds = []
    for bd in _BLOCK_D_CANDIDATES:
        try:
            triton_glu(x, bd)
            torch.cuda.synchronize()
            valid_bds.append(bd)
        except Exception as e:
            if verbose:
                print(f"  [autotune] N={N} D={D} {dtype} BLOCK_D={bd} "
                      f"COMPILE FAIL ({type(e).__name__}: {e})")

    if not valid_bds:
        raise RuntimeError(f"no BLOCK_D worked for N={N} D={D} {dtype}")

    # ── Phase 2: time each candidate fairly ──────────────────────────────
    # NOTE: do_bench's return type depends on `quantiles`:
    #   * quantiles=None  -> float (median ms)
    #   * quantiles=[..]  -> list of floats  (some triton builds return
    #                        a float when len==1; do not index)
    # We use the no-quantiles form and let it return a single float.
    best_bd, best_ms = valid_bds[0], float("inf")
    for bd in valid_bds:
        ms = triton.testing.do_bench(
            lambda b=bd: triton_glu(x, b),
            warmup=25, rep=100)
        if verbose:
            print(f"  [autotune] N={N} D={D} {dtype} BLOCK_D={bd:>4}  "
                  f"{ms*1000:.3f} us")
        if ms < best_ms:
            best_ms, best_bd = ms, bd

    if verbose:
        print(f"  [autotune] -> picked BLOCK_D={best_bd}  "
              f"({best_ms*1000:.3f} us)")

    _block_d_cache[key] = best_bd
    return best_bd


# %%
# Correctness check
# -----------------


def _get_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("float16", "fp16", "half"):
        return torch.float16
    if name in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _make_input(N, D, dtype):
    torch.manual_seed(0)
    return torch.randn(N, 2 * D, device=DEVICE, dtype=dtype)


def _check_static_feasibility(D: int) -> None:
    d_p2 = _next_pow2(D)
    if d_p2 != D:
        raise ValueError(
            f"D={D} is not pow2 (next_pow2={d_p2}); static extract_tile "
            f"requires D == next_pow2(D)")
    if d_p2 < 256:
        raise ValueError(
            f"D_P2={d_p2} < 256; cannot guarantee divisibility by "
            f"ctaTile(<=128)")


def run_correctness(N, D, dtype, BLOCK_D=None):
    _check_static_feasibility(D)
    x = _make_input(N, D, dtype)
    if BLOCK_D is None:
        BLOCK_D = best_block_d(N, D, dtype)

    ref      = torch_glu(x.float()).to(dtype)
    y_triton = triton_glu(x, BLOCK_D)
    y_tle    = tle_glu(x)

    atol = 1e-2 if dtype == torch.float16 else 1e-4
    rtol = 1e-3 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(y_triton.float(), ref.float(),
                               rtol=rtol, atol=atol)
    torch.testing.assert_close(y_tle.float(), ref.float(),
                               rtol=rtol, atol=atol)
    dt = "fp16" if dtype == torch.float16 else "fp32"
    print(f"  [OK]  N={N} D={D} {dt}  BLOCK_D={BLOCK_D}")


if "--only_unit_test" in sys.argv:
    for _dtype in (torch.float32, torch.float16):
        for _N, _D in [(1024, 256), (4096, 1024), (8192, 4096)]:
            run_correctness(_N, _D, _dtype)
    sys.exit(0)


# %%
# Benchmark (test_122_v1 perf_report style, three providers)
# ----------------------------------------------------------


_BENCH_PROVIDERS = ["triton", "tle", "torch"]
_BENCH_NAMES     = ["Triton (baseline, autotuned)",
                    "TLE (extract_tile static)",
                    "Torch (F.glu)"]
_BENCH_STYLES    = [("blue", "-"), ("orange", "-"), ("green", "-")]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["D"],
        x_vals=[256, 512, 1024, 2048, 4096],
        x_log=False,
        line_arg="provider",
        line_vals=_BENCH_PROVIDERS,
        line_names=_BENCH_NAMES,
        styles=_BENCH_STYLES,
        ylabel="ms",
        plot_name="tle-glu-performance",
        args={"N": 4096},
    ))
def benchmark(N, D, provider, dtype_str):
    dtype = _get_dtype(dtype_str)
    _check_static_feasibility(D)

    x = _make_input(N, D, dtype)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch_glu(x), quantiles=quantiles)
    elif provider == "tle":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: tle_glu(x), quantiles=quantiles)
    else:  # triton baseline with autotuned BLOCK_D
        BLOCK_D = best_block_d(N, D, dtype)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_glu(x, BLOCK_D), quantiles=quantiles)

    return ms, max_ms, min_ms


# %%
# Main
# ----


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=4096,
                        help="rows for correctness")
    parser.add_argument("--D", type=int, default=1024,
                        help="last-dim half-size for correctness "
                             "(pow2, >=256)")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float16", "float32", "fp16", "fp32"])
    parser.add_argument("--show_plots", action="store_true",
                        help="show plots in benchmark")
    parser.add_argument("--verbose_autotune", action="store_true",
                        help="print autotune timing for each BLOCK_D")
    args = parser.parse_args(argv)

    dtype = _get_dtype(args.dtype)

    # Optionally pre-warm autotune for the benchmark sizes with verbose
    # output so the user can see what BLOCK_D actually got picked.
    if args.verbose_autotune:
        print("--- autotune (verbose) ---")
        for D in [256, 512, 1024, 2048, 4096]:
            best_block_d(4096, D, dtype, verbose=True)
        print()

    print("--- correctness ---")
    run_correctness(args.N, args.D, dtype)
    print()

    benchmark.run(print_data=True, show_plots=args.show_plots,
                  dtype_str=args.dtype.replace("float", "fp"))


if __name__ == "__main__":
    main()
