# Benchmark Results

The benchmark programs emit CSV and need no database service:

```text
tests/benchmarks/bench_trainer_throughput
tests/benchmarks/bench_optimizer_convergence
tests/benchmarks/bench_memory_usage
```

Run them from a Release build on a quiet machine. Record CPU model, core
count, operating system, compiler, LibTorch version, CUDA availability, and
the exact command line. Throughput is samples/second, convergence is one loss
row per optimizer step, and memory is resident bytes before and after an
epoch. CI may attach these CSV files as artifacts.

| Date | Platform | LibTorch | Batch | Workers | Throughput | Notes |
|---|---|---|---:|---:|---:|---|
| 2026-10-12 | Windows 11 | 2.14.1 | 32 | 0 | 3,311 samples/s | Ryzen 7 5800H; local reference, PyTorch comparison pending |

Values are regression references and are not thresholds across different CPU,
compiler, or BLAS implementations. For a PyTorch comparison, use identical
generated tensors, dimensions, dtype, seed, batch sizes, and optimizer steps.

Checked-in JSON records live under `docs/benchmarks/results/`. The initial
Windows reference is a local measurement and does not claim PyTorch parity
until a matched PyTorch run is collected.
