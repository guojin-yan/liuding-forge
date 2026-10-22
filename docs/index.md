# 六丁炼算（Liuding Forge）Documentation

Liuding Forge (historical name: JYPPX Training Framework) is a C++20
deep-learning training framework on top of LibTorch. It
fills the gap between LibTorch's tensor primitives and the training
experience PyTorch users know: data loaders, optimizers, schedulers,
metrics, callbacks, checkpointing and mixed precision.

The `jyppx` namespace, `include/jyppx` headers, `jyppx::` CMake targets,
`find_package(jyppx)`, and protocol fields such as `jyppx_version` and
`jyppx.model_bundle` are compatibility contracts. They remain unchanged while
the public brand and generated artifact prefix use `Liuding Forge` and
`liuding-forge`.

- Slogan: *Bridge PyTorch to C++, Empower Your Training*
- License: Apache-2.0

## Getting started

- Installation and build instructions: the [README](../README.md)
- [Beta scope and migration limits](user-guide/beta_limits.md)
- [Tutorial 1: Basic Training](tutorials/01_basic_training.md) — the
  smallest complete training program

## Tutorials

1. [Basic Training](tutorials/01_basic_training.md) — models, datasets and
   the Trainer loop
2. [Data Loading](tutorials/02_data_loading.md) — custom datasets, splits,
   collate functions, multi-worker loading
3. [Optimizers and Schedulers](tutorials/03_optimizers.md) — choosing and
   pairing them, manual loops
4. [Callbacks, Checkpoints and Metrics](tutorials/04_callbacks.md) — the
   production setup
5. [Mixed Precision (AMP)](tutorials/05_mixed_precision.md) — float16
   training on CUDA

## API reference

- [Capability Manifest](api/capability_manifest.md) — bounded cross-domain
  feature descriptors, exact build identity, validation, and deterministic JSON
- [Checkpoint Participant](api/checkpoint_participant.md) — storage-neutral
  recovery levels, state descriptors, bounded blob I/O, and participant
  ownership boundaries
- [Checkpoint Artifact Manifest](api/checkpoint_artifact_manifest.md) — strict
  bounded format metadata, portable blob paths, digests, and deterministic JSON
- [Checkpoint Coordinator](api/checkpoint_coordinator.md) — deterministic
  participant registration, transactional capture, restore preflight, and
  cross-participant rollback
- [Durable Checkpoint Files](api/durable_checkpoint.md) — atomic publication,
  integrity checks, writer coordination, resource limits, and crash recovery
- [Local Run Storage](api/local_run_storage.md) — immutable CAS artifacts,
  SQLite run lineage, lifecycle transactions, integrity, and retention
- [Training telemetry](api/training_telemetry.md) — typed neutral events,
  shared durable outbox, replay and checkpoint migration
- [Evaluation evidence](api/evaluation.md) — versioned suites, provenance,
  local JSON/HTML reports, baseline comparisons and basic data checks
- [Library CLI](user-guide/cli.md) — text/JSON commands and installation
- [Configuration migration](api/config_migration.md) — explicit schema chains,
  preview, new-file apply and verified rollback
- [API migration diagnostics](api/api_migration.md) — bounded source scanning
- [Sidecar process transport](api/telemetry_sidecar_process.md) — bounded
  supervised optional telemetry processes
- [Model weight import](api/model_weight_import.md) — verified safetensors,
  complete mapping reports and staged model state application
- [Interop formats and backends](api/interop_backends.md) — raw safetensors
  and explicit compatibility/ONNX capability boundaries
- [Model bundles](api/model_bundle.md) — immutable inventory, sharding,
  safe publication and explicit signature trust
- [TorchScript compatibility](api/torchscript_compatibility.md) — explicitly
  trusted existing artifacts, actual JIT execution and bounded value contracts
- [Models Preview](api/models.md) — gated reference architectures, explicit
  weight provenance and independent reference comparisons
- [Quantization Preview](api/quantization.md) — portable int8 storage,
  calibration, bounded artifacts and explicit FP32 runtime fallback
- [Serving Preview](api/serving.md) — verified bundle loading, bounded requests,
  dynamic batching, cancellation and typed runtime failures
- [Training Scheduler Adapter](api/scheduler_adapter.md) — ordered lifecycle
  events with exact descriptor-first scheduler checkpoint delegation
- [Classification Metrics](api/classification_metrics.md) — fixed-topology,
  mergeable Precision, Recall, and F1 with exact checkpoint state
- [Model Checkpoint Participant](api/model_checkpoint_participant.md) — exact
  model topology, bounded raw tensor state, strict preflight, and atomic restore
- [Data](api/data.md) — datasets, subset/concat/split, map/iterable pipelines,
  samplers, structured batches, workers, and capability descriptors
- [Optimizers and Schedulers](api/optim.md) — SGD, Adam, AdamW, RMSprop,
  Adagrad, NAdam, Adadelta and LAMB; descriptor-first schedulers including
  warmup and composite schedulers
- [Delegated SGD Adapter](api/sgd_adapter.md) — LibTorch-backed SGD with
  stable group and parameter identities plus exact bounded checkpoint state
- [Delegated Adam Adapter](api/adam_adapter.md) — LibTorch-backed Adam with
  stable identities and exact moment, step-counter, and AMSGrad recovery
- [Delegated AdamW Adapter](api/adamw_adapter.md) — decoupled LibTorch AdamW
  with stable identities and exact bounded optimizer-state recovery
- [Delegated RMSprop Adapter](api/rmsprop_adapter.md) — LibTorch RMSprop with
  typed centered/momentum topology and exact bounded state recovery
- [Delegated Adagrad Adapter](api/adagrad_adapter.md) — LibTorch Adagrad with
  eager accumulators, per-parameter steps, and exact bounded state recovery
- [Native NAdam Adapter](api/nadam_adapter.md) — current-only NAdam/NAdamW
  with the complete momentum schedule and exact per-parameter recovery
- [Native Adadelta Adapter](api/adadelta_adapter.md) — current-only two-state
  Adadelta with independent reference coverage and exact bounded recovery
- [Native LAMB Adapter](api/lamb_adapter.md) — current-only layer-wise trust
  ratio updates with independent reference coverage and exact recovery
- [Descriptor-First StepLR](api/step_lr_scheduler.md) — typed scheduler
  semantics, stable optimizer linkage, and exact bounded checkpoint state
- [Transactional Composite Schedulers](api/composite_lr_scheduler.md) —
  ordered child descriptors, pure transition preflight, and atomic tree restore
- [Descriptor-First Epoch Schedulers](api/epoch_lr_scheduler.md) —
  exponential, multi-step, cosine, and warm-restart scheduling with exact
  recovery
- [Descriptor-First Event Schedulers](api/event_lr_scheduler.md) —
  metric plateau, one-cycle, cyclic, and identity-bearing callable schedules
- [Descriptor-First Linear-family Schedulers](api/linear_lr_scheduler.md) —
  linear interpolation, warmup, constant prefix, and polynomial decay
- [Training](api/training.md) — Trainer, callbacks, metrics, AMP
- [Model Adapter](api/model_adapter.md) — shared model state, typed adapter
  helpers, structured invocation schemas, and multi-input/output boundaries
- [Label-smoothed cross entropy](api/losses/label_smoothing_cross_entropy.md)
  — the first independently packaged `jyppx::nn` loss adapter
- [Contrastive loss](api/losses/contrastive_loss.md) — a native pairwise
  metric-learning loss with explicit tensor and label contracts
- [Focal loss](api/losses/focal_loss.md) — binary and class-index focal loss
  with explicit alpha/gamma and tensor contracts
- [Utilities](api/utils.md) — devices, tensor checks, parameter initialization,
  version

## Project status

The [component dependency contract](development/component_dependencies.md)
documents the machine-enforced target, link-interface, and include graph.

See [CHANGELOG](../CHANGELOG.md) for the running list of changes. The current
framework scope includes loss, optimizer, scheduler, hooks, augmentation,
analysis, metric, checkpoint and AMP extensions. Optional telemetry adapters
use the official TensorBoard event writer and OpenTelemetry libraries through
the supervised sidecar. Models, storage quantization and serving are separate
Preview components, each disabled by default; see their capability and
qualification limits above.
Planning records are maintained outside the source repository and are not
part of the installed documentation.
