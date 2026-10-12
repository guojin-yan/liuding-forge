# JYPPX Training Framework Documentation

JYPPX is a C++20 deep-learning training framework on top of LibTorch. It
fills the gap between LibTorch's tensor primitives and the training
experience PyTorch users know: data loaders, optimizers, schedulers,
metrics, callbacks, checkpointing and mixed precision.

- Slogan: *Bridge PyTorch to C++, Empower Your Training*
- License: MIT

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

- [Data](api/data.md) — datasets, subset/concat/split, DataLoader,
  samplers, thread-safe queue
- [Optimizers and Schedulers](api/optim.md) — SGD, Adam, AdamW, RMSprop,
  Adagrad; StepLR, ExponentialLR, CosineAnnealingLR, MultiStepLR,
  ReduceLROnPlateau
- [Training](api/training.md) — Trainer, callbacks, metrics, AMP
- [Utilities](api/utils.md) — devices, tensor checks, logger, parameter
  initialization, version
- [Distributed](api/distributed.md) — process group, DDP wrapper,
  DistributedSampler

## Project status

See [CHANGELOG](../CHANGELOG.md) for the running list of changes and
`project-management/CURRENT_STATUS.md` for the phase-by-phase progress
against the implementation plan. In short: the core framework (data,
optim, training, utils, nn init, AMP, multi-threaded DataLoader) is
implemented with 161 passing unit tests; data transforms and additional
losses are next.
