# Benchmark Results

The benchmark programs emit CSV and need no database service:

```text
tests/benchmarks/bench_data_pipeline
tests/benchmarks/bench_trainer_throughput
tests/benchmarks/bench_optimizer_convergence
tests/benchmarks/bench_sgd_adapter_overhead
tests/benchmarks/bench_adam_adapter_overhead
tests/benchmarks/bench_adamw_adapter_overhead
tests/benchmarks/bench_rmsprop_adapter_overhead
tests/benchmarks/bench_adagrad_adapter_overhead
tests/benchmarks/bench_adadelta_native_throughput
tests/benchmarks/bench_lamb_native_throughput
tests/benchmarks/bench_nadam_native_throughput
tests/benchmarks/bench_memory_usage
tests/benchmarks/bench_cuda_training
```

Run `tools/Run-Baseline.ps1` from a quiet machine. Pass `-LibTorchDir` when
LibTorch is installed outside `third_party/install/libtorch`. The runner
performs one warmup process and five independent sample processes by default,
then records the environment identity, raw samples, median, nearest-rank p95,
sample standard deviation, coefficient of variation (CV), and regression
allowance.
DataPipeline latency and peak memory are lower-is-better; Trainer throughput is
higher-is-better. Optimizer convergence remains deterministic correctness
evidence and is not treated as a timing metric.

`bench_sgd_adapter_overhead` alternates five direct LibTorch SGD and delegated
adapter samples after two warmups. It reports nanoseconds per step and the
median adapter/native ratio for the exact host and build. Treat the ratio as a
local regression observation, not a threshold that transfers across machines.

`bench_adam_adapter_overhead` applies the same paired methodology to AMSGrad
Adam so adapter validation cost is tracked independently from SGD.

`bench_adamw_adapter_overhead` applies the same paired methodology to AMSGrad
AdamW, including LibTorch's decoupled weight-decay path.

`bench_rmsprop_adapter_overhead` applies it to centered RMSprop with momentum
and coupled weight decay, so validation overhead is measured with all native
state branches enabled.

`bench_adagrad_adapter_overhead` applies it to Adagrad with learning-rate
decay, coupled weight decay, and a nonzero initial accumulator.

`bench_adadelta_native_throughput` records absolute Adadelta median
nanoseconds per step. The candidate LibTorch C++ APIs do not expose Adadelta,
so numerical behavior is gated by the independent reference and convergence
tests instead of an incomparable adapter/native ratio.

`bench_lamb_native_throughput` records absolute LAMB median nanoseconds per
step. LibTorch does not expose a comparable public C++ LAMB optimizer, so no
adapter/native ratio is reported. Numerical behavior is gated by the
independent tensor reference and checkpoint-resume tests.

`bench_nadam_native_throughput` records absolute NAdam median nanoseconds per
step. LibTorch does not expose a public C++ NAdam optimizer, so this benchmark
does not manufacture an incomparable adapter/native ratio. Numerical parity
is covered separately by the independent reference and convergence tests.

`bench_cuda_training` is built only when CMake detects CUDA LibTorch. Run it
through `tools/run_cuda_baseline.py`, which collects the frozen FP32/single-SGD
and FP16/two-SGD/AMP modes in separate processes. Each structured JSON sample
contains the workload digest, exact BuildFingerprint, CUDA device and compile
toolkit identity, Trainer/optimizer/scaler correctness counters, throughput,
mean step time, and CUDA allocator peaks. The runner adds `nvidia-smi` driver,
UUID, power-limit, persistence-mode and noise observations, then delegates CV
and regression enforcement to `tools/baseline_stats.py`.

The 2026-10-18 RMSprop Windows Release sample (MSVC 19.44, LibTorch 2.14.1,
CPU, 4096 float32 parameters, 2000 steps, two warmups, five paired samples)
produced an adapter/native median ratio of `1.014278332`.

The 2026-10-18 Adagrad Windows Release sample (MSVC 19.51, LibTorch 2.14.1,
CPU, 4096 float32 parameters, 2000 steps, two warmups, five paired samples)
produced an adapter/native median ratio of `1.040400845`. These measurements
are local evidence only; the environment-keyed repeated-sampling gate remains
authoritative.

The 2026-10-18 LAMB Windows Release sample (MSVC 19.51, LibTorch 2.5.1 CPU,
4096 float32 parameters, 2000 steps, two warmups, five samples) produced an
absolute median of `58003 ns/step`. LibTorch has no comparable public C++ LAMB
optimizer, so this is a local regression observation rather than a parity
ratio or a cross-machine threshold.

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

CUDA references are additionally invalid across GPU, compute capability,
driver, power limit, persistence mode, official LibTorch package build tag,
host CUDA compile toolkit, or either frozen workload mode. Keep raw reports in
CI artifacts or the ignored `out/` tree; commit only reviewed aggregate evidence
when a compatibility lane is established.

Checked-in JSON records live under `docs/benchmarks/results/`. The initial
Windows reference is a local measurement and does not claim PyTorch parity
until a matched PyTorch run is collected.
