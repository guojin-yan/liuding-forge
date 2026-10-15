# Benchmark Results

The benchmark programs emit CSV and need no database service:

```text
tests/benchmarks/bench_dataloader
tests/benchmarks/bench_trainer_throughput
tests/benchmarks/bench_optimizer_convergence
tests/benchmarks/bench_memory_usage
```

Run `tools/Run-Baseline.ps1` from a quiet machine. The runner performs one
warmup process and five independent sample processes by default, then records
the environment identity, raw samples, median, nearest-rank p95, sample
standard deviation, coefficient of variation (CV), and regression allowance.
DataLoader latency and peak memory are lower-is-better; Trainer throughput is
higher-is-better. Optimizer convergence remains deterministic correctness
evidence and is not treated as a timing metric.

A run is unstable when any metric exceeds 10% CV. A stable reference stores a
per-metric regression allowance of `max(10%, 3 * CV)`. References are valid
only when their exact environment key matches the candidate run, so CPU,
operating system, toolchain, LibTorch, and build-configuration changes cannot
be compared accidentally. `tools/baseline_stats.py` rejects unstable samples,
environment mismatches, and threshold regressions with a nonzero exit code.

| Date | Platform | LibTorch | Batch | Workers | Throughput | Notes |
|---|---|---|---:|---:|---:|---|
| 2026-10-12 | Windows 11 | 2.14.1 | 32 | 0 | 3,311 samples/s | Ryzen 7 5800H; local reference, PyTorch comparison pending |

Values are regression references and are never thresholds across different
CPU, compiler, LibTorch, or BLAS implementations. For a PyTorch comparison,
use identical generated tensors, dimensions, dtype, seed, batch sizes, and
optimizer steps.

Checked-in JSON records live under `docs/benchmarks/results/`. The initial
Windows reference is a local measurement and does not claim PyTorch parity
until a matched PyTorch run is collected.
