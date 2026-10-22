# Contributing to 六丁炼算（Liuding Forge）

The repository's historical and technical name is JYPPX Training Framework.
Keep the `jyppx` namespace, include paths, CMake options and package names
stable when making brand-facing changes.

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/liuding-forge.git`
3. Create a feature branch: `git checkout -b feature/your-feature-name`

## Development Setup

### Prerequisites

- C++20 compatible compiler
- CMake 3.18+
- Python 3 (required by configure-time repository gates)
- An exact LibTorch combination listed in [`compatibility.json`](compatibility.json)

### Building
```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build . --parallel
```

### Running Tests
```bash
ctest --output-on-failure
```

## Coding Standards

### C++ Style

- Follow the Google C++ Style Guide and compile as C++20
- Run clang-format with the repository configuration before submitting
- Run clang-tidy on every target that is inside the strict migration scope
- Follow naming conventions:
  - Classes: `PascalCase`
  - Functions: `snake_case`
  - Variables: `snake_case`
  - Constants: `UPPER_CASE`
  - Member variables: `snake_case_` (trailing underscore)

### Documentation

- Add complete Doxygen comments to every declaration in a header, including
  private members, constructors, methods, structs and their fields, enums and
  every enum value
- Document ownership, lifetime, thread-safety, error behavior, state mutation,
  units, valid ranges, and determinism whenever those constraints apply
- Treat undocumented declarations and malformed Doxygen as build failures
- Include usage examples for complex features
- Update relevant tutorials in `docs/tutorials/`

### Local Quality Gates

The strict gates currently cover the migrated foundation and core targets and
expand with each migrated component:

```bash
cmake -S . -B build-quality \
  -DJYPPX_BUILD_TESTS=ON \
  -DJYPPX_BUILD_DOCS=ON \
  -DJYPPX_ENABLE_CLANG_TIDY=ON \
  -DJYPPX_ENABLE_FORMAT_CHECK=ON \
  -DJYPPX_ENABLE_WARNINGS_AS_ERRORS=ON \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build build-quality --target jyppx_migrated_api_docs --parallel
ctest --test-dir build-quality --output-on-failure
```

The documentation option requires Doxygen. The clang options require
`clang-format` and `clang-tidy` to be discoverable on `PATH`, unless their
corresponding `JYPPX_*_EXECUTABLE` cache entries are supplied explicitly.

Before configuring a build, validate the repository compatibility source:

```bash
python scripts/validation/compatibility.py --file compatibility.json --validate
python scripts/validation/dependency_manifest.py --file dependency-manifest.json
```

Do not infer support from a version range. Add a verified lane with reproducible
configure, build, test, and installed-consumer evidence before changing a
candidate lane to `supported`.

Generate the local Release timing, artifact-size, throughput, and peak-memory
baseline with `pwsh -NoProfile -File scripts/benchmarks/Run-Baseline.ps1`. The default clean
build is required for comparable compile-time evidence.

Changes affecting CUDA data transfer, Trainer execution, optimizer orchestration,
AMP, or allocator behavior also require `scripts/benchmarks/run_cuda_baseline.py` on CUDA
LibTorch and a real GPU. Keep its default one warmup plus five independent
sample processes for both frozen modes. A reference is valid only for its exact
environment key; never compare or copy thresholds across GPUs, drivers, power
limits, LibTorch builds, host CUDA toolkits, workloads, or compiler builds.

The Linux coverage lane is `bash scripts/validation/run-coverage.sh`; it requires `gcovr 5.0`
or newer and
`CMAKE_PREFIX_PATH` pointing to a CPU LibTorch installation. Coverage and
sanitizer options are rejected on unsupported compilers, and sanitizer flags are
applied to test executables as well as framework libraries. The coverage runner
uses two build jobs by default; set `JYPPX_BUILD_PARALLEL_LEVEL` explicitly when
running on a host with a different memory budget.

### Testing

- Write unit tests for new features
- Ensure all tests pass before submitting
- Aim for >80% code coverage
- Follow the [repository hygiene and test-data rules](docs/development/repository_hygiene.md)
- Keep logs, downloaded data and runtime outputs out of the source root

### Exception safety checklist
- [ ] Public APIs validate null pointers, empty paths, and invalid numeric
      options before starting worker threads or mutating state.
- [ ] Worker-thread exceptions are captured and rethrown on the consumer
      thread; no exception escapes a `std::thread` entry point.
- [ ] Queues and worker threads are closed and joined on early loop exit and
      destruction.
- [ ] Tests cover the exception type and an informative message.
- [ ] Error paths leave checkpoints, optimizers, and models in a documented
      state so callers can log or retry safely.

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Tests
- `chore`: Maintenance

Example:
```
feat(optim): add AdamW optimizer

Implement AdamW optimizer with decoupled weight decay.
Includes unit tests and documentation.

Closes #123
```

## Pull Request Process

1. **Update documentation**: Add/update relevant docs
2. **Add tests**: Include unit tests for new features
3. **Run tests**: Ensure all tests pass
4. **Update CHANGELOG.md**: Add entry under "Unreleased"
5. **Create PR**: Use descriptive title and description

### PR Checklist
- [ ] Code follows project style
- [ ] Header declarations and enum values have complete Doxygen comments
- [ ] clang-format, clang-tidy, and Doxygen gates pass for migrated targets
- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Commit messages follow conventions
- [ ] `RepositoryHygiene` passes and obsolete build/output directories are removed

## Code Review

- Be respectful and constructive
- Address all review comments
- Update PR based on feedback

## Questions?

- Open an issue for bugs or feature requests
- Use discussions for questions
- Contact maintainers via email

## License

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this project is licensed exclusively under the
[Apache License 2.0](LICENSE), without additional terms or conditions. Submit
only work that you own or are authorized to contribute, and preserve all
applicable third-party copyright, patent and attribution notices.
