# Qwen3-32B TLE Tutorial

This directory contains a first end-to-end Qwen3-32B inference path whose
transformer compute is implemented through TLE/Triton tutorial kernels.

## Layout

- `serving/`: interactive chat CLI.
- `bench/`: prefill and decode benchmark entry points.
- `models/`: Qwen3 config, HF weight extraction, KV cache, and engine logic.
- `kernels/`: TLE-backed embedding, RMS inv-scale, linear/fused gate-up-SiLU,
  RoPE/KV cache, and attention kernels.

## Chat

```bash
conda run -n flagtree python python/tutorials/tle/mega/serving/qwen3_32b_chat.py \
  --model-path Qwen/Qwen3-32B \
  --max-new-tokens 256
```

## Benchmark

```bash
conda run -n flagtree python python/tutorials/tle/mega/bench/bench_qwen3_32b.py \
  --model-path Qwen/Qwen3-32B \
  --batch-size 1 \
  --prompt-len 1024 \
  --decode-steps 16
```

Results are appended to `bench/results/qwen3_32b_tle_prefill_decode.jsonl`.

## Compatibility Entry Point

```bash
conda run -n flagtree python python/tutorials/tle/mega/task_grid_linear_rmsnorm.py
```

This legacy script name now dispatches the generated scheduler validation. The
old explicit-counter `partial_count` path has been removed from the tutorial.

## Task Graph Scheduler Test

```bash
conda run -n flagtree python python/tutorials/tle/mega/task_graph_linear_rmsnorm.py
```

This runs the generated scheduler path for a static dense
`linear_tile -> rms_reduce -> rms_apply` graph. The current tutorial entry point
is intentionally narrow: it supports `float32`, `in_features == 1`, bounded
static `rows`/`hidden`, and fails fast for unsupported shapes instead of
falling back to host launches or row counters. The graph envelope is built with
`triton.experimental.tle.mega as tlem`: ordinary kernel arguments use
`graph.arg(...)`, task grids use `graph.grid(...)`, and task callees are bound
through `graph.task(..., name=...)`.

The compute bodies are written as Triton `@triton.jit` tile functions in
`kernels/linear_rmsnorm.py`. The tutorial lowers those Triton functions to the
private MLIR task-body callees `linear_tile_body`, `rms_reduce_body`, and
`rms_apply_body` required by the scheduler ABI. They are not separate
host-launched Triton kernels; the generated cooperative scheduler dispatches
those callees inside one persistent kernel according to the task graph
dependencies.

## Linear + RMSNorm Microbenchmark

```bash
conda run -n flagtree python python/tutorials/tle/mega/bench/bench_linear_rmsnorm.py \
  --warmup 10 --iters 50
```

This compares the generated mega scheduler against a normal non-mega Triton
baseline that launches a linear stage and an RMSNorm stage. The benchmark
precompiles the mega scheduler and preallocates intermediate buffers before
timing.

## Mega Scheduler IR

The host graph API can lower a dense static task graph to MLIR metadata. The
compiler path verifies the graph, analyzes dependencies, materializes a
scheduler plan/runtime-state pair, and can lower the scheduler boundary to LLVM
IR:

```bash
build/cmake.linux-x86_64-cpython-3.10/bin/triton-opt graph.mlir \
  --triton-tle-analyze-task-graph \
  --triton-tle-materialize-task-scheduler \
  --triton-tle-materialize-task-runtime-state \
  --convert-triton-gpu-to-llvm=compute-capability=90 \
  -reconcile-unrealized-casts
```

The generated `tle.task_graph.scheduler` op carries task dispatch order,
per-instance dependency counts, reverse dependency edges, and the initial ready
queue. It also materializes dense integer tables for task ids, dependency
counts, initial ready ids, per-instance coordinates, and producer-to-consumer
CSR edges. The generated `tle.task_graph.runtime_state` op records the i32
device-state layout for init, queue metadata, dependency counters, and ready
queue storage. The LLVM lowering emits a cooperative-grid persistent scheduler
loop with a device global ready queue, dependency-counter atomics, task-body
dispatch, and task commit logic. Task bodies publish global writes with a
device-scope `membar.gl` before dependency commits wake consumers. A minimal
producer/consumer scheduler and the generated `linear+rmsnorm` scheduler are
covered by `python/test/tle/integration/test_tle_mega_scheduler.py`. The
current scheduler MVP also rejects task-body callees that require non-zero
global scratch memory.

## Current Scope

The first version is intentionally a TLE end-to-end engine, not yet a single
persistent CUDA megakernel. The Python engine dispatches TLE/Triton kernels for
each Qwen3 operation and uses one shared KV cache across prefill and decode.
The next step is to fold selected layer stages into persistent `tle.pipe` +
`tle.gpu.warp_specialize` kernels, starting from attention and MLP tiles.
