# Linux CPU ThreadSanitizer lane

Run `bash scripts/validation/run-tsan.sh` from a clean, committed Linux checkout.
The lane requires Python 3.10+, Clang, CMake/CTest with JSON discovery and JUnit
support, Git, `ldd`, and an official shared CPU LibTorch distribution. Set
`CMAKE_PREFIX_PATH` to its installation prefix for the first invocation, or use
`JYPPX_TSAN_TORCH_DIR=<prefix>/share/cmake/Torch` explicitly.

The incremental build directory is always `out/build/tsan-linux`. Build and test
parallelism default to **2**; `JYPPX_LIUDING_BUILD_PARALLEL_LEVEL` overrides both. Serving
and all Core components are enabled explicitly. Models, quantization, tools,
examples, benchmarks and fuzzers are disabled. Coverage, ASan and UBSan are
incompatible with this lane. An existing cache with a different source,
compiler, CPU Torch package or instrumentation is rejected for review instead
of silently replacing that configuration. `JYPPX_TSAN_CXX` can select a Clang
executable on the first invocation; later invocations retain the cached compiler.

Cached CPU `Torch_DIR` takes precedence over ambient `CMAKE_PREFIX_PATH`, so a
CUDA lane's environment cannot redirect the fixed cache. `Torch_DIR`,
`Caffe2_DIR`, `TORCH_LIBRARY`, package version, CPU library hashes and executable
runtime linkage are checked. A package with CUDA libraries or a non-CPU version
is rejected. The selected CPU library directory is first in `LD_LIBRARY_PATH`.
`LD_PRELOAD`, `GTEST_*`, sanitizer suppressions and options that suppress TSan
reports are rejected. `TSAN_OPTIONS` always includes `halt_on_error=1`,
`report_bugs=1` and nonzero `exitcode=66`.

## Selected scope

The lane builds these seven targets together:

- `jyppx_test_data`
- `jyppx_test_concurrent_stress`
- `jyppx_test_serving`
- `jyppx_test_experiment_run_store`
- `jyppx_test_experiment_run_events`
- `jyppx_test_experiment_telemetry`
- `jyppx_test_training_event_sink`

Its 15 required suites cover data pipeline workers/checkpoints/cancellation,
iterable pipelines, batch memory and schema integration, concurrent stress,
public serving leases/drain/batching/queue cancellation, SQLite run/event writer
competition and cancellation, telemetry exporters, event outbox and the training
telemetry adapter. `tsan_lane.py --policy` prints the complete source-derived
case list and target mapping without running CMake, a compiler or CTest.

Structured CTest discovery must match every literal test declaration in each
required suite, including the exact executable and GoogleTest filter. Empty,
disabled, missing, duplicate or unexpected tests fail **before test execution**.
PRE_TEST discovery itself runs instrumented binaries to list GoogleTest cases;
it can therefore expose a sanitizer runtime failure before test bodies run.
If supported, `setarch -R` wraps both discovery and execution. The probe result
and chosen address layout are recorded; unsupported shadow-memory layouts fail
the lane and are not reported as skipped coverage.

The actual CTest exit code and a fresh JUnit file must both show complete
execution. Missing results, conditional skips, failures, disabled/not-run tests
and TSan diagnostics fail acceptance even if CTest returns zero. CUDA suites and
deliberate death/crash subprocess tests are excluded. LibTorch and SQLite
third-party internals are not rebuilt with TSan instrumentation. This is selected
CPU concurrency validation, not full project coverage or CUDA qualification.

## Evidence and lightweight maintenance checks

Every invocation retains its logs and metadata in a unique directory beneath
`out/sanitizer/linux/tsan/runs/`. The top-level `metadata.json` points to the most
recently updated run by its `run_dir`. Metadata records source commit/tree and
clean status before/after, dependency manifest and actual dependency hashes,
compiler/tool versions, fixed cache identity, options, command argv/exit codes,
selected tests, binary hashes, actual CTest exit and JUnit results. Preflight,
build, discovery and execution failures retain metadata; `ctest_exit_code: null`
means test execution never completed. Source/cache changes invalidate acceptance.
Historical run directories are preserved. A process lock prevents concurrent
lane invocations from sharing the incremental cache.

Each configure, build, discovery and test command starts in its own POSIX
process session. SIGINT/SIGTERM first records an interrupted lane request and
terminates that command's process group, escalates only that group if needed,
and waits until the group is confirmed gone before the cache lock is released.
This also covers CTest's JSON discovery subprocess path and prevents orphaned
grandchildren from continuing after a cancelled run. Windows keeps a direct
process fallback and is rejected during the real Linux preflight.

The portable maintenance checks do not build or execute instrumented binaries:

```text
python -m unittest discover -s tests/scripts -p test_tsan_lane.py
bash -n scripts/validation/run-tsan.sh
python scripts/validation/tsan_lane.py --policy
```
