---
name: liuding-build-and-test
description: Configure, build, test, or validate Liuding Forge on Windows or Linux with the repository's CMake profiles, optional components, and focused quality gates. Use for local build failures, CTest selection, installed-consumer checks, or pre-commit validation in this repository.
---

# Build and Test Liuding Forge

## Establish the profile

Implement every shipped product feature in C++. Python is limited to development tooling and independent reference/checker tests; never use it as an exporter, converter, preparation tool, installed runtime or user API. Existing Python product paths require native migration and do not justify new exceptions.

Read `compatibility.json`, `CONTRIBUTING.md`, the affected `CMakeLists.txt`, and `component-dependencies.json` before changing build options. Use an exact declared LibTorch/compiler/CUDA combination; do not infer support from a version range.

Reuse a compatible build tree under `out/build/`. Prefer incremental configuration and builds, preserve its generator and dependency paths, and use at most two compile jobs unless the user provides a different resource budget. Configure a new tree only when no compatible cache exists or a build-system change invalidates it.

Enable only the required options:

```text
JYPPX_LIUDING_BUILD_TESTS=ON
JYPPX_LIUDING_BUILD_BENCHMARKS=ON       # benchmark work only
JYPPX_LIUDING_BUILD_MODELS=ON           # models work only
```

Optional components are independent gates. Do not enable every preview component to make an unrelated target link.

## Validate the change

1. Build the smallest affected library, test, tool, or benchmark target.
2. Run the matching CTest selection with `--output-on-failure`; pass `-C Release` for multi-config Windows trees.
3. For public headers, targets, or component dependencies, also run `ComponentDependencyGraph`, `RepositoryHygiene`, and the relevant installed consumer.
4. Run formatting or static checks only for changed files first. Run the full repository gate at a review or release boundary, or when the change affects shared build behavior.
5. Treat skipped hardware tests as unsupported in that environment. A CPU result does not qualify CUDA, and a retained dirty validation tree must be identified separately from a clean committed tree.

Do not download dependencies during configure, silently change cached Torch paths, clean an unrelated shared build tree, or dispatch GitHub Actions. The repository workflows are manual during development.

Report the source commit or dirty state, exact build tree/profile, targets and test counts, conditional skips, and any gate that was not run or failed for a known external reason.
