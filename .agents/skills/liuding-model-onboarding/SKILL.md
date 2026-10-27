---
name: liuding-model-onboarding
description: Add or extend a vision model in Liuding Forge's optional models component, including registry metadata, feature and head contracts, fixed-reference tests, resource gates, benchmark exposure, and user documentation. Use for new backbones, capacities, model factories, or weight mappings in this repository.
---

# Onboard a Liuding Forge Model

## Reuse the existing model surface

Inspect `include/jyppx/liuding/models/`, `src/jyppx/liuding/models/registry.cpp`, `tests/unit/models/test_models.cpp`, `tests/install/models_consumer/`, and `docs/api/models.md`. Extend an existing family when its blocks and state layout match; add a new family only when ownership or architecture differs materially.

Public code stays under `jyppx::liuding::models` and the optional `jyppx::liuding::models` CMake target. Register each public factory with a stable name, family, input contract, feature dimensions, parameter count, and capability status. `create_model()` must retain model state through `nn::ModelAdapter` and must not perform network access.

Keep architecture, pretrained weights, preprocessing, and task qualification separate. Random initialization never implies weight availability or accuracy. A weight entry needs an independently verified upstream revision, source, SHA-256, license, complete key mapping, and task evidence before it becomes available.

Use Windows-safe public identifiers. In particular, capacity enums use `small_capacity` instead of `small` because the Windows SDK defines `small` as a macro.

## Required evidence

- Pin an authoritative upstream revision and generate independent fixtures without copying upstream implementation into production code.
- Compare exact state inventory, parameter count, output shape, logits or embeddings, and family-specific intermediate features. Cover backward and an optimizer update for at least one practical capacity.
- Test invalid shapes/options and registry duplication. Update the installed models consumer and verify `models=OFF` isolation when the public surface changes.
- Gate high-resource capacities separately with explicit memory limits; do not construct them in the default test selection.
- Add CPU and declared CUDA evidence independently. Benchmark exposure requires actual parameter updates and report identity checks; a micro fixture qualifies the execution path only.

Update `docs/api/models.md`, the relevant benchmark instructions, and `CHANGELOG.md` only with user-visible facts. Keep validation reports and raw measurements outside the source repository.
