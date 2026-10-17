# JYPPX Training Framework Documentation

JYPPX is a C++20 deep-learning training framework on top of LibTorch. It
fills the gap between LibTorch's tensor primitives and the training
experience PyTorch users know: data loaders, optimizers, schedulers,
metrics, callbacks, checkpointing and mixed precision.

- Slogan: *Bridge PyTorch to C++, Empower Your Training*
- License: Apache-2.0

## Getting started

- Installation and build instructions: the [README](../README.md)
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

- [Checkpoint Participant](api/checkpoint_participant.md) — storage-neutral
  recovery levels, state descriptors, bounded blob I/O, and participant
  ownership boundaries
- [Checkpoint Coordinator](api/checkpoint_coordinator.md) — deterministic
  participant registration, transactional capture, restore preflight, and
  cross-participant rollback
- [Training Scheduler Adapter](api/scheduler_adapter.md) — ordered lifecycle
  events with exact descriptor-first scheduler checkpoint delegation
- [Classification Metrics](api/classification_metrics.md) — fixed-topology,
  mergeable Precision, Recall, and F1 with exact checkpoint state
- [Model Checkpoint Participant](api/model_checkpoint_participant.md) — exact
  model topology, bounded raw tensor state, strict preflight, and atomic restore
- [Data](api/data.md) — datasets, subset/concat/split, DataLoader,
  samplers, thread-safe queue
- [Optimizers and Schedulers](api/optim.md) — SGD, Adam, AdamW, RMSprop,
  Adagrad, NAdam, Lamb, RAdam, Adamax, ASGD, Rprop, Adadelta, SparseAdam,
  LBFGS, LARS and Lookahead; the complete scheduler family including warmup
  and composite schedulers
- [Delegated SGD Adapter](api/sgd_adapter.md) — LibTorch-backed SGD with
  stable group and parameter identities plus exact bounded checkpoint state
- [Delegated Adam Adapter](api/adam_adapter.md) — LibTorch-backed Adam with
  stable identities and exact moment, step-counter, and AMSGrad recovery
- [Delegated AdamW Adapter](api/adamw_adapter.md) — decoupled LibTorch AdamW
  with stable identities and exact bounded optimizer-state recovery
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
- [Utilities](api/utils.md) — devices, tensor checks, logger, parameter
  initialization, version
- [Distributed](api/distributed.md) — process group, DDP wrapper,
  DistributedSampler

## Project status

See [CHANGELOG](../CHANGELOG.md) for the running list of changes. The current
framework scope includes loss, optimizer, scheduler, hooks, TensorBoard,
augmentation, analysis, metric, checkpoint and AMP extensions. Quantization is
planned as a separately gated optional component and is not part of the current
default SDK surface.
Planning records are maintained outside the source repository and are not
part of the installed documentation.
