# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- An independently exported `jyppx::interop` target with a schema-v1
  checkpoint artifact manifest codec, deterministic JSON, strict field and
  digest validation, portable path checks, resource limits, capability
  metadata, and an installed-package consumer.
- An installed-package component matrix that configures, builds, links, and
  runs a consumer against each exported target independently.
- A legacy-surface gate that rejects retired headers, sources, package names,
  and build target names in both the source tree and staged installation.

### Changed
- The remaining in-tree SDK target and project identifiers now use canonical
  `jyppx` names; the retired package name remains only in enforcement checks
  and historical release notes.
- The installed CMake package now uses the canonical `jyppx` package name,
  validates requested components, and rejects unknown components. Legacy
  package and target entry points are not retained.

### Removed
- The legacy global text `Logger` and hand-encoded TensorBoard protobuf writer.
  Structured event sinks remain available; a future TensorBoard adapter must
  live in its own component and use reviewed generated schemas.
- The v1.0 single-process distributed-process-group and DDP placeholders. The
  framework will not publish distributed APIs until they provide real
  multi-process collectives, rank-safe training, and checkpoint semantics.

### Added
- A real-GPU CUDA Trainer benchmark and repeated-sampling runner covering
  FP32/single-SGD and FP16/two-SGD/AMP execution, explicit synchronization,
  correctness counters, peak allocated/reserved VRAM, environment identity,
  stability checks, and environment-bound regression references.
- Foundation-owned schema-v1 capability manifests with bounded validation,
  deterministic JSON, exact build fingerprints, and configured data-pipeline
  capability descriptors.
- Explicit `IterableDataPipeline` schema-v2 to schema-v3 checkpoint migration
  with historical exact, bounded, and best-effort fixtures, strict manifest
  compatibility, and failure-atomic source cursor replacement.
- Checkpoint-stable logical data-worker shard identities and an explicit
  caller-current CUDA transfer-stream policy with two-phase background
  preparation and consumer-thread publication.
- Current-only multi-optimizer Trainer execution with disjoint complete
  parameter ownership, canonical update order, independent scheduler cursors,
  exact coordinated recovery, and all-or-none AMP overflow handling.
- A current-only transactional `Trainer` that directly composes map/iterable
  pipelines, model/optimizer/RNG checkpoint participants, scheduler events,
  callbacks, the bounded event outbox, and a training strategy. It supports
  accumulation tail policies, validation, gradient clipping, cancellation,
  and safe-boundary checkpoint save/restore.
- A shared `BatchPipeline` contract implemented by both map-style and iterable
  data pipelines, removing data-style branching from the training loop.
- Descriptor-first `SequentialLrScheduler` and `ChainedLrScheduler` with unique
  child ownership, ordered typed descriptors, pure full-tree preflight, one
  atomic optimizer update, and exact recursive checkpoint recovery.
- A common `LrScheduler` transaction surface and generic scheduler checkpoint
  participant shared by leaf and composite schedulers.
- A descriptor-first `StepLrScheduler` with typed immutable configuration,
  stable optimizer/group identity, atomic rate control, and bounded exact
  checkpoint recovery without legacy scheduler maps or native archives.
- A narrow `OptimizerRateControl` contract implemented by the delegated SGD
  adapter so schedulers do not depend on algorithm-specific native objects.
- An independently exported `jyppx::optim` target with a LibTorch-delegated
  SGD adapter, stable group/parameter identities, fixed versioned raw state
  fields, bounded preflight, and atomic exact checkpoint restore.
- A LibTorch-delegated Adam adapter with stable groups and parameter IDs,
  long-run exact differential coverage, per-parameter moment/step recovery,
  AMSGrad state, and atomic storage-neutral checkpoint restore.
- A LibTorch-delegated AdamW adapter with decoupled weight-decay semantics,
  stable groups and parameter IDs, complete moment/step recovery, and atomic
  storage-neutral checkpoint restore.
- A LibTorch-delegated RMSprop adapter with typed centered/momentum topology,
  stable groups and parameter IDs, complete conditional state recovery, and
  atomic storage-neutral checkpoint restore.
- A LibTorch-delegated Adagrad adapter with typed decay and accumulator
  topology, stable identities, independent per-parameter step recovery, and
  atomic storage-neutral checkpoint restore.
- A current-only native NAdam/NAdamW adapter with the complete scheduled
  Nesterov update, typed stable topology, per-parameter momentum-product and
  step recovery, independent numerical coverage, and bounded atomic restore.
- A current-only native Adadelta adapter with typed rho/epsilon/coupled-decay
  semantics, two per-parameter accumulators, independent numerical coverage,
  and bounded atomic checkpoint restore.
- An exact `jyppx::nn` model checkpoint participant with versioned topology,
  bounded non-pickle tensor blobs, strict state-key/type/shape/alias preflight,
  and rollback-protected parameter and buffer restore.
- An independently exported `jyppx::training` checkpoint coordinator with a
  deterministic participant registry, canonical aggregate descriptors,
  transactional sink/source boundaries, restore preflight, bounded rollback
  snapshots, and reverse-order recovery after participant failure.
- A storage-neutral checkpoint participant contract in `jyppx::foundation`,
  including stable restore guarantees, validated state manifests, bounded named
  blob I/O, and an atomic per-participant load requirement.
- A schema-checked `ModelStateView`/`ModelAdapter` contract that separates
  shared LibTorch module state from explicit invocation, including typed
  tensor helpers and structured multi-input/output validation.
- A versioned `compatibility.json` source of truth and dependency-free
  validator. CI and benchmark workflows now read the candidate LibTorch version
  from this source instead of carrying an independent hard-coded version.
- A draft `dependency-manifest.json` with SHA-256-verified local archives and a
  validator that keeps unverified candidate downloads explicit.
- A reproducible local Release baseline runner covering clean configure/build,
  the full CTest suite, artifact sizes, throughput, optimizer convergence, and
  isolated process peak memory.
- A Linux coverage runner and unified GCC/Clang instrumentation helper covering
  framework libraries and test/benchmark executables, including UBSan support.
- An Ubuntu 22.04/GCC 11.4/LibTorch 2.5.1 CPU Debug evidence lane covering the
  full test suite, installed consumer, line/branch coverage, and ASan+UBSan.
- A strict foundation quality gate covering Google-style formatting,
  clang-tidy diagnostics, complete Doxygen documentation (including private
  declarations and enum values), and an installed-package consumer test.
- An independently exported `jyppx::nn` target, initially owning the audited
  label-smoothed cross-entropy adapter and its focused public header.
- An independently exported contrastive-loss target with strict pair-shape,
  device, dtype, finite-margin and binary-label validation; the duplicate SDK
  implementation and its compatibility surface were removed.
- An independently exported focal-loss target with explicit binary/multiclass
  shape semantics, finite alpha/gamma validation, class-index checks, and no
  duplicate SDK implementation.
- Exported `jyppx::foundation` and `jyppx::sdk` CMake targets for focused and
  full-framework consumers.
- Versioned DataLoader/sampler state for deterministic epoch-boundary resume.
- Transactional Trainer checkpoint restore covering model, optimizer,
  scheduler, AMP scaler, callbacks, RNG, loaders, and loop counters.
- Optimizer parameter-group insertion with topology validation.

### Changed
- Build fingerprints now derive CUDA LibTorch package identity from the
  official `build-version` tag when available, preventing CUDA distributions
  that omit `TORCH_CUDA_VERSION` from being mislabeled as CPU builds. Dirty
  state detection now includes untracked files.
- Dataset utilities, transforms, stress tests, examples, documentation, and
  throughput benchmarks now use the explicit-epoch `DataPipeline` contract.
  The baseline runner accepts an explicit external `LibTorchDir` so durable
  dependencies do not need to be copied into the source tree.
- Delegated SGD now preflights every gradient after an optional closure and
  rejects sparse layouts before any parameter update. CPU bfloat16, long-run
  LibTorch differential, CUDA capability, and adapter-overhead coverage are
  part of the optimizer quality matrix.
- Adam consumers now use `AdamOptimizerAdapter`; sparse gradients and invalid
  descriptors are rejected before mutation, while learning-rate control is
  exposed through the same narrow scheduler contract as delegated SGD.
- AdamW consumers now use `AdamWOptimizerAdapter`; immutable decoupled decay
  policy and complete state are represented without exposing mutable native
  options or pointer-keyed state.
- RMSprop consumers now use `RmspropOptimizerAdapter`; alpha, epsilon, coupled
  decay, momentum, and centered mode are immutable topology, and only learning
  rates remain mutable.
- Adagrad consumers now use `AdagradOptimizerAdapter`; learning-rate decay,
  coupled weight decay, initial accumulator value, and epsilon are immutable
  topology, while eager accumulators and per-parameter steps are recovered
  exactly.
- NAdam consumers now use `NAdamOptimizerAdapter`; coupled and decoupled
  weight decay, maximize mode, momentum decay, and all scheduled state are
  represented without mutable option maps or exposed pointer-keyed state.
- Adadelta consumers now use `AdadeltaOptimizerAdapter`; rho, epsilon, coupled
  weight decay, maximize mode, lazy steps, and both accumulator tensors are
  represented without mutable option maps or exposed pointer-keyed state.
- Delegated optimizers now synchronize model-owned dtype/device placement once
  during strategy preparation, then reject every placement change after native
  state or a successful step exists. This keeps CUDA Trainer checkpoints valid
  without exposing mutable parameter groups.
- Training examples, tutorials, the throughput benchmark, and the installed
  consumer now use explicit `ModelAdapter`, task step, participant, and
  `prepare()`/`fit()` contracts. Precision remains explicit task/strategy work;
  the current Trainer does not retain the legacy implicit `use_amp` switch.
- Removed the replaced legacy composite scheduler implementation and its two
  forwarding headers; composite scheduling now has one direct public API.
- Windows tests, benchmarks, and examples now copy LibTorch runtime DLLs once
  per output directory, preventing parallel MSBuild copy races.
- Coverage and sanitizer options now fail clearly on unsupported compilers and
  reject incompatible instrumentation combinations.
- Linux coverage and sanitizer builds now bound parallelism to avoid exhausting
  memory on constrained runners; coverage exposes `JYPPX_BUILD_PARALLEL_LEVEL`
  for hosts that need a different budget.

### Removed
- The legacy iterator-style `DataLoader` and its private-purpose
  `ThreadSafeQueue` public header. Current consumers use `DataPipeline`
  directly; no forwarding header, target alias, or compatibility type remains.
- The hand-written `SGD` implementation and the old `optim/schedulers`
  hierarchy. Current consumers use `SgdOptimizerAdapter` and descriptor-first
  schedulers directly; no aliases or forwarding APIs remain.
- The hand-written `Adam` implementation and mutable options surface. Current
  consumers use the delegated adapter directly; no compatibility header,
  forwarding type, or legacy map checkpoint reader remains.
- The hand-written `AdamW` implementation and mutable options surface. Current
  consumers use the delegated adapter directly with no forwarding API.
- The hand-written `RMSprop` implementation and mutable options surface.
  Current consumers use the delegated adapter directly with no compatibility
  header, forwarding type, or legacy map checkpoint reader.
- The hand-written `Adagrad` implementation and mutable options surface.
  Current consumers use the delegated adapter directly with no compatibility
  header, forwarding type, or legacy map checkpoint reader.
- The incomplete hand-written `NAdam` implementation and mutable options
  surface. The native adapter is a direct replacement with no forwarding
  header, legacy map checkpoint reader, or compatibility alias.
- The legacy Trainer, mutable callback aggregate, callback implementations,
  compatibility headers, old callback tests, and the duplicate complete
  training example. No forwarding API or legacy state reader is retained.
- Foundation headers now define fully documented error, RNG, version, and
  build-fingerprint contracts; backend failures encountered while collecting
  the fingerprint are translated into the stable JYPPX error taxonomy with
  their original cause preserved.
- Removed the legacy exported CMake target names instead of retaining aliases
  or forwarding compatibility targets.
- Project licensing changed from MIT to Apache-2.0-only for the current and
  future source tree. Previously published MIT-licensed versions retain their
  original license.
- Checkpoint, optimizer, and DataLoader state readers now require their current
  explicit format instead of retaining unversioned compatibility branches.
- `AutocastGuard` validates device/dtype pairs and restores nested dtype state;
  Trainer supports CPU BFloat16 autocast without a loss scaler.
- Weighted sampling without replacement now uses a bounded algorithm and
  rejects input sets that cannot produce the requested draw.
- `Trainer` now steps metric-driven schedulers with the validation loss and
  rejects them when no validation loader is configured.
- Trainer checkpoints now preserve scheduler state, AMP `GradScaler` state,
  and the last train/validation loss values used by callbacks.
- Label-smoothed cross entropy now delegates to LibTorch for unbatched and
  spatial inputs, accepts the documented `smoothing = 1` boundary, reports
  structured JYPPX configuration errors, and has no duplicate SDK-owned
  implementation.
- The duplicate JYPPX triplet-margin wrapper was removed; callers use the
  complete LibTorch API directly, including its `p`, `eps` and `swap` options.
- The legacy `nn/losses.hpp` aggregate, public `apply_reduction` helper and
  policy-heavy `distillation_loss` were removed. The distillation example now
  states its hard/soft composition, teacher-gradient, reduction and dtype
  policy directly as a local training recipe.
- The duplicate `jyppx::nn::init` wrappers were removed; examples now use the
  complete typed LibTorch initialization API directly.
- The legacy `nn/quantization.hpp` storage-only prototype was removed from the
  default SDK. A future optional `jyppx::quantization` component must start
  from versioned formats, safe resource limits and separately proven storage,
  training and compute capabilities.
- `LRSchedulerCallback` follows the same metric-aware stepping rules as the
  Trainer loop.
- Release test coverage increased to 249 passing tests.

## [0.3.0] - 2026-10-11

Plan-completion release: closes every remaining Phase 1–4 item from the
project plan and lays the Phase 5 groundwork. 240 unit/integration tests
pass in Release and Debug configurations.

### Added
- **Plan-completion sweep** (closing every remaining Phase 1–4 item):
  - Callbacks: `LoggingCallback` (per-batch/epoch logging through the
    framework logger with optional log file), `LRMonitor` (per-epoch learning
    rate log), `TensorBoardCallback` (batch loss, epoch losses, learning
    rates). `on_train_batch_end` now receives the batch loss.
  - `utils::ProgressBar` (single-line console bar with description/postfix)
    and `utils::ProfilerReport`/`ScopedTimer` (named-scope wall-clock
    aggregation).
  - Metrics: `Precision`, `Recall`, `F1Score` (macro-averaged over the
    accumulated confusion matrix, lazily growing class count).
  - Samplers: `WeightedRandomSampler` (with/without replacement),
    `SubsetRandomSampler`, `DistributedSampler` (epoch-based shuffling,
    even padding, drop_last). Datasets: `ChainDataset`, `StackDataset`
    (variadic tuple zipping).
  - Transforms: `Pad`, `RandomRotation`, `RandomAffine` (grid_sample-based
    rotate/scale/translate), `Grayscale`/`RandomGrayscale` (ITU-R 601-2
    luma), `ColorJitter` (brightness/contrast/saturation/hue in random
    order).
  - Optimizers: `NAdam` (Nesterov momentum with the official mu schedule;
    `momentum_decay > 0` not supported) and `Lamb` (trust-ratio update with
    the reference weight-norm clamp).
  - TensorBoard: `utils::TensorBoardWriter` writes real event files
    (TFRecord framing with masked CRC32C, hand-encoded Event/Summary
    protobuf messages; scalars + text).
  - Model analysis: `utils::calculate_model_stats`/`model_summary`
    (Keras-style table) — plus a torch 2.14 API note: `named_children()`
    returns by value and must be kept alive while iterating `items()`.
  - Phase 5 groundwork: `jyppx::distributed` single-process process group
    (init/rank/world_size/barrier + `average_gradients`), a hook-based
    `DistributedDataParallel<ModuleT>` wrapper (no-op with world_size 1;
    cross-process all-reduce requires a c10d backend build),
    `functional::distillation_loss` (temperature-scaled KL), and
    `nn::quantization` weight-only int8 helpers (`quantize_symmetric`,
    `dequantize`, `QuantizedLinear`).
- **Examples**: `examples/mnist/mnist.cpp` — the plan's MNIST acceptance
  example with a self-contained gzip (RFC 1952) decoder and IDX reader
  (auto-download), CNN training through the Trainer with a 2-worker loader
  and TensorBoard logging (reaches 98.6% test accuracy in 3 epochs), and
  `examples/complete_training.cpp` — the full-featured flow (random split,
  AdamW + cosine schedule, all callbacks, metrics, checkpoint best-model
  restore; 99.5% validation accuracy).

## [0.2.0] - 2026-10-11

First feature-complete release of the training framework: data loading
(single- and multi-threaded), optimizers and schedulers, the Trainer with
callbacks and checkpointing, mixed precision, transforms, losses,
initialization, documentation and benchmarks. 195 unit/integration tests
pass in Release and Debug configurations.

### Added
- **Loss functions** (`include/jyppx/nn/losses.hpp`): functional
  implementations in `jyppx::nn::functional` — `focal_loss` (multiclass
  cross-entropy path and binary sigmoid path, alpha weighting, gamma
  focusing), `dice_loss` (soft Dice with one-hot multiclass and binary
  paths, fully differentiable tensor ops), `label_smoothing_cross_entropy`
  (verified equal to torch's `label_smoothing` option), `contrastive_loss`
  (Euclidean pairs with margin hinge) and `triplet_margin_loss` (delegates
  to ATen for exact parity). Criterion modules `nn::FocalLoss`, `nn::DiceLoss`,
  `nn::LabelSmoothingCrossEntropy`, `nn::ContrastiveLoss`, `nn::TripletLoss`
  wrap the functions for forward-style use. 20 tests cover hand-computed
  values (1e-12), torch parity anchors, gradient flow, reduction semantics
  and argument validation.
- **Data transforms** (`jyppx::data::transforms`, header-only):
  `Transform<I, O>` base + `TensorTransform` alias, `Compose` (ordered
  chaining with descriptive `name()`), `RandomChoice`, and six vision
  transforms — `Normalize` (per-channel, promotes integral inputs),
  `Resize` (bilinear/nearest, square or `{h, w}`), `CenterCrop`,
  `RandomCrop`, `RandomHorizontalFlip`, `RandomVerticalFlip`. Random
  transforms draw from the global torch RNG (reproducible via
  `torch::manual_seed`, safe under the multi-worker DataLoader). 15 tests
  cover ordering, exact values, fan-free crop geometry, seeded
  determinism, probability bounds and DataLoader integration.
- **DataLoader benchmark** (`tests/benchmarks/bench_dataloader.cpp`, built
  with `JYPPX_BUILD_BENCHMARKS=ON`): a self-contained harness (no Google
  Benchmark dependency) measuring per-epoch loading time for 0/1/2/4/8
  workers over a synthetic dataset whose `get()` burns a configurable
  amount of CPU per sample (simulating decode/preprocessing cost), plus a
  one-epoch training-throughput reference line. Measured on the
  development machine (16 hardware threads, 2 ms/sample, 1024 samples):
  2 workers 2.0x, 4 workers 3.9x, 8 workers 7.1x speedup over the
  single-threaded loader — exceeding the plan's 2.5x/3.5x targets with
  near-linear scaling.
- **Documentation**: a docs tree (`docs/index.md`) with four API reference
  pages (`data`, `optim`, `training`, `utils`) and five tutorials (basic
  training, data loading, optimizers and schedulers, callbacks and
  checkpoints, mixed precision). Every code snippet was extracted and
  compile/run-checked, which fixed three reader-facing issues in the
  process: the README example fed int64 targets to the default MSE loss and
  used an invalid `make_shared<torch::nn::Linear>` construction, the
  ReduceLROnPlateau snippet relied on braced-init-list forwarding that
  `make_shared` does not support, and a TrainerOptions snippet used
  out-of-order designated initializers (rejected by MSVC).
- **`nn::init` parameter initialization** (all 9 functions declared in the
  header are now implemented, delegating to `torch::nn::init` for exact
  PyTorch semantics): `uniform_`, `normal_`, `constant_`, `zeros_`, `ones_`,
  `xavier_uniform_`, `xavier_normal_`, `kaiming_uniform_`,
  `kaiming_normal_` (string-based `mode` = "fan_in"/"fan_out"; leaky_relu
  gain). 13 tests cover seeded RNG parity with `torch::nn::init`, fan-based
  bounds/statistics for 2D and 4D weights, gain handling and argument
  validation. Platform notes: torch 2.14 C++ requires dim >= 2 for
  fan-based init (1D tensors are rejected, unlike Python) and the
  `NonlinearityType`/`FanModeType` parameters are `std::variant` enums.
- **Dataset utilities**:
  - `data::Subset<T>`: index-restricted view over an existing dataset.
  - `data::ConcatDataset<T>`: contiguous concatenation of same-typed
    datasets with cumulative-size indexing.
  - `data::random_split(dataset, lengths, seed)`: seeded train/validation
    splitting (mirrors `torch.utils.data.random_split`), returning
    heap-allocated subsets ready for DataLoader consumption.
- **Optimizers** (all validated numerically against `torch::optim`):
  - `optim::RMSprop` (alpha, eps, weight decay, momentum, centered).
    NOTE: momentum follows the `torch::optim` C++ semantics — the buffer
    accumulates the normalized gradient `grad / avg` and the step is
    `-lr * buffer`; this differs from `torch.optim.rmsprop` in Python
    (raw gradient in the buffer, divided by avg at update time) starting
    from the second step.
  - `optim::Adagrad` (lr_decay, weight_decay, per-parameter step counting).
- **Schedulers**:
  - `MultiStepLR`: decays by gamma at each milestone.
  - `ReduceLROnPlateau`: metric-driven (`step(metric)`) with mode, factor,
    patience, rel/abs threshold and min_lr; metric-less schedulers reject
    `step(metric)` and vice versa.
- **Metrics**:
  - `TopKAccuracy` (top-k hit rate over {batch, classes} scores),
    `MSE`, `MAE`, `R2Score` (exact whole-set coefficient of determination),
    `AverageMeter` (weighted running average), `MetricCollection`
    (named bundle with joint update/compute/reset).
- **Multi-threaded DataLoader** (`num_workers > 0`):
  - `jyppx::data::ThreadSafeQueue<T>`: bounded blocking queue for
    producer/consumer pipelines (blocking `put`/`get`, non-blocking
    `try_get`, close semantics that keep already-queued items retrievable).
  - Worker pool inside `DataLoader`: workers pull index batches, load and
    collate them, and push results through a sequence-tagged pipeline.
    Results are returned in submission order (identical to single-threaded
    loading), worker exceptions are rethrown on the consuming thread when
    their sequence is reached, and prefetching bounds the number of
    in-flight batches (`num_workers * prefetch_factor`).
  - Workers restart per epoch so a partially consumed epoch never leaks
    into the next one; the destructor drains in-flight work.
  - `DataLoaderOptions::prefetch_factor` (default 2).
  - Note: the seeded first-epoch shuffle no longer consumes an extra
    generator draw at construction, so a seeded loader's first epoch now
    matches its seed directly.
  - 16 new tests: queue semantics (ordering, blocking, close, move-only
    elements), multi-worker output equivalence, permutations, custom
    collate, exception propagation, early-break reuse, 20-epoch stress,
    and Trainer fit integration.
- **AMP (mixed-precision training)**:
  - `jyppx::training::amp::GradScaler` mirroring `torch.cuda.amp.GradScaler`
    semantics: loss scaling, gradient unscaling with inf/NaN detection, step
    skipping with scale backoff on overflow, scale growth after
    `growth_interval` clean steps, and a serializable state dict.
  - `jyppx::training::amp::AutocastGuard`, an RAII wrapper around the ATen
    autocast thread-local state (device-type/dtype scoped, restores prior
    state on destruction).
  - `Trainer` integration: `use_amp = true` (CUDA only, validated at
    construction) runs forward passes under autocast, scales the loss before
    backward, unscales before gradient clipping, and routes optimizer steps
    through the scaler (skipping steps that overflowed). Works together with
    gradient accumulation and gradient clipping.
- GoogleTest/CTest unit test infrastructure. `tests/CMakeLists.txt` fetches
  GoogleTest 1.14.0 from the local `third_party/downloads` cache (network
  fallback included) and registers 240 tests via `gtest_discover_tests`
  (PRE_TEST discovery). Suites cover:
  - `core`: device parsing, NaN/Inf helpers, tensor clone/device moves
  - `utils`: Logger levels, file output, unwritable path handling
  - `data`: TensorDataset, Sequential/Random/BatchSampler, DataLoader
    (ordering, seeded shuffling, drop_last, custom collate, tensor batching)
  - `optim`: SGD/Adam/AdamW validated numerically against `torch::optim`
    (momentum, dampening, weight decay, Nesterov, AMSGrad), state_dict
    round trips, gradient zeroing and learning-rate accessors, StepLR /
    ExponentialLR / CosineAnnealingLR trajectories
  - `training`: Trainer end-to-end (fit/validate/test, max_steps, gradient
    accumulation equivalence, gradient clipping, checkpoint round trip and
    resume), callback lifecycle ordering, EarlyStopping, ModelCheckpoint,
    Accuracy
- Shared test helpers in `tests/test_utils/jyppx_test_helpers.hpp`
  (deterministic linear model/data, leaf-tensor clones, torch::optim parity
  harness, callback event recorder).

### Fixed
- Adam / AdamW now match PyTorch's update exactly: eps is added to the
  bias-corrected denominator (`sqrt(v) / sqrt(bias_correction2) + eps` with
  `step_size = lr / bias_correction1`). The previous paper-style formula
  (`sqrt(v) + eps` before bias correction) diverged from `torch::optim` by
  ~1e-7 per step.
- `Optimizer::state_dict()` serializes parameter-group options as float64.
  `torch::tensor(double)` defaults to float32, which corrupted lr/momentum
  values in checkpoint round trips.
- Trainer: the delegating (concrete-module) constructor no longer throws before
  it can install the type-erased forward function. The `forward_fn` check moved
  from `validate_configuration()` to the `fit`/`validate`/`test` entry points,
  so a Trainer built from a base `torch::nn::Module` plus `set_forward_fn()`
  is now a valid workflow. Previously this surfaced as an uncaught
  `std::invalid_argument` (Windows: silent crash `0xC0000409`).
- MSVC Debug builds: use the release CRT (`/MD` via
  `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL`) and
  `_ITERATOR_DEBUG_LEVEL=0` in every configuration. LibTorch's official
  Windows binaries are Release-built with `/MD`; mixing them with a Debug
  runtime corrupted the heap across the DLL boundary (access violations,
  `STATUS_HEAP_CORRUPTION`) and mismatched STL layouts (segfaults inside
  `register_module`).
- Fixed MSVC Debug/Release flag handling for multi-config generators: per-config
  compiler flags in `cmake/CompilerOptions.cmake` now use generator expressions
  instead of `CMAKE_BUILD_TYPE` checks (previously building the Debug config
  failed with D8016 `/O2` vs `/RTC1`).
- Examples: the LibTorch POST_BUILD copy now copies only `*.dll`
  (`copy_if_different`) instead of the whole `lib` directory, which also
  contained large import libraries.

### Verified
- `ctest -C Release` and `ctest -C Debug`: 240/240 tests pass in both
  configurations. The CUDA AMP integration test skips gracefully when
  `torch::cuda::is_available()` is false; on this workstation the installed
  driver (576.02, up to CUDA 12.9) cannot run the cu132 LibTorch binaries, so
  the GPU path is compile-checked and unit-covered but not yet exercised
  end-to-end.
- `examples/basic_trainer`, `examples/basic_dataloader`,
  `examples/complete_training` (99.5% val accuracy),
  `examples/mnist` (98.6% test accuracy on the real dataset) and
  `examples/distillation` (teacher transfer into the label-scarce student)
  run cleanly in both Release and Debug configurations.
- The installed CMake package was exercised by an out-of-tree consumer via
  `find_package(jyppx_training_framework CONFIG REQUIRED)` (build, link, run).
- DataLoader multi-worker scaling measured with the bundled benchmark:
  2 workers 2.0x, 4 workers 3.9x, 8 workers 7.1x over the single-threaded
  loader on the development machine.

## [0.1.0] - 2025-10-10

### Added
- Project initialization
- Basic architecture and build system
