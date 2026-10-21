# Benchmark Results

The benchmark programs emit CSV and need no database service:

```text
tests/benchmarks/bench_data_pipeline
tests/benchmarks/bench_trainer_throughput
tests/benchmarks/bench_optimizer_convergence
tests/benchmarks/bench_loss_vertical_slice
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

Run `scripts/benchmarks/Run-Baseline.ps1` from a quiet machine. Pass `-LibTorchDir` when
LibTorch is installed outside `third_party/install/libtorch`. The runner
performs one warmup process and five independent sample processes by default,
then records the environment identity, raw samples, median, nearest-rank p95,
sample standard deviation, coefficient of variation (CV), and regression
allowance.
The persistent build directory is reused by default so unchanged targets remain
cached; pass `-FreshBuild` only when an intentionally isolated clean-build
measurement is required. Output reports remain replaceable run artifacts, but
the configured build tree is not deleted after a run.
DataPipeline latency and peak memory are lower-is-better; Trainer throughput is
higher-is-better. Optimizer convergence remains deterministic correctness
evidence and is not treated as a timing metric.

`bench_data_pipeline` emits throughput, p50/p95 successful batch latency,
sampled mean/peak scheduled-but-uncommitted queue depth, aggregate factory busy
time divided by available execution-lane time, peak active workers, and sampled
resident memory. Use `--workers` to isolate each worker count in its own process;
the baseline runner does this automatically so peak memory from earlier
configurations cannot contaminate later rows. Queue depth and peak active worker
counts are recorded as diagnostic observations rather than monotonic regression
metrics.

`bench_loss_vertical_slice` uses one CPU thread, fixed seeds and frozen tensor
shapes to report five post-warmup forward/backward samples for label smoothing,
binary focal, multilabel Dice and contrastive loss. It links only `jyppx::nn`
so the same target can also measure component-level clean build time. Compare
results only on the same host, toolchain, LibTorch build and workload revision;
the executable records observations and does not define a portable threshold.

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
through `scripts/benchmarks/run_cuda_baseline.py`, which collects the frozen FP32/single-SGD
and FP16/two-SGD/AMP modes in separate processes. Each structured JSON sample
contains the workload digest, exact BuildFingerprint, CUDA device and compile
toolkit identity, Trainer/optimizer/scaler correctness counters, throughput,
mean step time, and CUDA allocator peaks. The runner adds `nvidia-smi` driver,
UUID, power-limit, persistence-mode and noise observations, then delegates CV
and regression enforcement to `scripts/benchmarks/baseline_stats.py`.

`bench_cuda_optimizer` is also CUDA-only. Invoke the same runner with
`--suite optimizer` to collect isolated FP32 update-path measurements for SGD,
Adam, AdamW, RMSprop, Adagrad, NAdam, Adadelta, and LAMB. The frozen workload
uses eight equal-size CUDA tensors, fixed dense gradients, a stateful option
profile, 200 steps, and synchronized timing boundaries. A sample is rejected
unless the adapter commits every step, changes at least one parameter, and
leaves every parameter finite. Throughput and allocator results are regression
references for the same algorithm and exact environment, never a ranking
between algorithms.

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
be compared accidentally. `scripts/benchmarks/baseline_stats.py` rejects unstable samples,
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
