# Resource benchmark sampling

`run_resource_baseline.py` samples public production APIs in **one warmup process
and five measured processes per profile** by default. It never configures,
builds, runs CTest, or contacts a remote host. Run it after a complete development
chapter, using the existing Windows and Ubuntu/CUDA Release build directories.
Keep build parallelism at `2` and retain those directories for incremental builds.

Enable `JYPPX_BUILD_BENCHMARKS=ON` in the existing configuration. Models,
quantization and serving targets additionally require their corresponding
`JYPPX_BUILD_MODELS`, `JYPPX_BUILD_QUANTIZATION` and `JYPPX_BUILD_SERVING` options.
Preview benchmarks link their explicit owner libraries; enabling a benchmark
does not enable a Preview or change the SDK's capability claims.

After committing, reconfigure that same directory with its existing options so
the generated build fingerprint records the exact clean commit, then build the
resource targets together. A stale fingerprint or `+dirty` binary is rejected.
No benchmark source/output is silently relabelled as a different commit.

```powershell
# Windows: retain the established Release cache and its dependencies/options.
cmake --build out/build/config-windows --config Release --parallel 2 --target jyppx_bench_resource_stable jyppx_bench_resource_models jyppx_bench_resource_quantization jyppx_bench_resource_serving
python scripts/benchmarks/run_resource_baseline.py --build-dir out/build/config-windows --output out/baselines/resources/windows-current
```

```sh
# Ubuntu: use the established CUDA Release cache, not a second benchmark cache.
cmake --build out/build/cuda-release --config Release --parallel 2 --target jyppx_bench_resource_stable jyppx_bench_resource_models jyppx_bench_resource_quantization jyppx_bench_resource_serving
python3 scripts/benchmarks/run_resource_baseline.py --build-dir out/build/cuda-release --device cuda:0 --output out/baselines/resources/cuda-current
```

The default selection requires all four executables. For a Core-only build,
use `--profiles storage checkpoint telemetry`. `--bin-dir` selects a known
custom executable directory. Output directories must be new and should be under
an ignored evidence directory such as `out/` to preserve a clean checkout.

| Profile | Fixed work and resource evidence |
| --- | --- |
| storage | Eight distinct 256 KiB CAS blobs, SQLite FULL publication, metadata reopen, exact reads and integrity check; publish/read time, payload and total disk bytes |
| checkpoint | Four distinct 1 MiB coordinated checkpoints with synchronization enabled; mean save pause including serialization/hash/publication, restore time, exact restored bytes |
| telemetry | 256 fixed canonical log envelopes, 16 encoding rounds, JSONL emit/stream flush, verified reopen and duplicate replay; encoding/emit/recovery time and output bytes |
| resnet18, resnet50, mobilenet_v3_small, mobilenet_v3_large | Real randomly initialized FP32 architectures, seven classes, `[2,3,32,32]`, eval mode, one warmup and 32 forwards |
| transformer_encoder | Two layers, dimension 32, four heads, FFN 64, dropout 0, GELU/final norm, `[2,16,32]`, one warmup and 256 forwards |
| quantization | Calibrated 64-to-32 Linear storage artifact, canonical serialization/decode, FP32 reconstruction and 512 batches of 16 verified against an exact arithmetic vector |
| serving | Existing generated linear bundle fixture, verified NativeStateBackend load/warmup, 128 Predictor calls, 32 rounds of eight DynamicBatcher requests, exact output and actual batch/row counters |

All workloads are synthetic and bounded. Model `load_seconds` means architecture
construction, random initialization and device transfer; it does not mean
pretrained weight loading. Model inputs and initialization seed are fixed, and
all retained results are verified for shape, dtype, device, finite values and
eval repeatability. Model output tensors remain alive through the measured
forward loop; memory includes those outputs. Serving measures the public
in-process boundary and batch scheduler, including validation and snapshots.
It reports real coalescing counters; no HTTP transport is involved.

`--device cuda:0` runs models and serving on the actual visible device zero,
synchronizing before and after each stage. Reports include its UUID, name,
compute capability, CUDA toolkit/runtime/driver versions, and allocator
allocated/reserved peaks since before load. The runner verifies the UUID and
driver with `nvidia-smi`; `CUDA_VISIBLE_DEVICES` mappings use the reported UUID,
not an assumed physical GPU index. CPU-only binaries reject CUDA requests.
Persistence, telemetry and quantization execute on CPU in either suite.
Quantization declares **storage-quantized / FP32 fallback**, with no compute
quantization, QAT or speedup comparison.

RSS is the process lifetime peak working set on Windows or Linux `VmHWM`,
including LibTorch/runtime overhead. CUDA peaks cover the LibTorch caching
allocator, excluding driver/context allocations. Unavailable measurements carry
an explicit status and reason with no numeric value. CPU CUDA metrics are
unavailable; their absence never counts as a CUDA pass. JSONL promises stream
flush rather than filesystem fsync; checkpoint and SQLite timings retain their
real synchronization semantics. Disk metrics describe the sample workload,
not machine-wide I/O or filesystem cache eviction. Processes are independent;
the OS file cache and GPU driver/context startup are not forcibly reset.

Each process has a 120-second timeout and a 1 MiB bound on each output stream
by default. Every process retains `stdout.log`, `stderr.log`, `process.json`,
`sample.json`, `validation.json` and its dedicated workload directory. Failed
processes retain available logs and failure metadata. The serving example owns
and removes its own temporary generated bundle. Runs preserve `run.json` and
`statistics.json`; invalid protocol/source samples fail before aggregation.
The runner never recursively deletes evidence.

`statistics.json` includes median, nearest-rank p95, mean, raw samples and sample
CV. Required repeated resource metrics use a **10% CV gate**. Single short
startup/warmup/conversion measurements remain explicit diagnostics, with
`gate:false`; the runner still validates their positive finite values.
There are no silent retries or discarded measured outliers. An unstable run
fails and cannot create a reference. Stabilize the host and retain the failed
evidence before collecting a new run.

For a first stable reference, add
`--write-reference out/baselines/resources/windows-reference.json` (or a distinct
Ubuntu/CUDA reference). Later use `--reference` with that unchanged file.
References are created exclusively; existing files are never overwritten.
Comparison allows at most `max(10%, 3 * reference CV)` regression per gated
metric, with the same metric contract and environment.

Environment identity binds host/CPU, OS, CMake performance/configuration flags,
compiler, LibTorch ABI/version, dependency lock, actual device/driver, runtime
thread settings, relevant environment variables, fixed workload and metric
availability. Source provenance separately retains the exact Git commit/tree,
clean state, binary hashes, full build fingerprints, runner hash and cache hash.
Changing only the source commit does not change the environment key. A legitimate
new commit can compare against old evidence while retaining both original
identities. A changed environment or workload rejects comparison; copying a
new commit into historical provenance is never a substitute for new evidence.

Focused runner verification (no C++ compilation or hardware required):

```sh
python3 -m unittest discover -s tests/scripts -p test_resource_baseline.py
```
