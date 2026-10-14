# Contributing to JYPPX Training Framework

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/jyppx-training-framework.git`
3. Create a feature branch: `git checkout -b feature/your-feature-name`

## Development Setup

### Prerequisites
- C++17 compatible compiler (GCC 9+, Clang 10+, MSVC 2019+)
- CMake 3.18+
- LibTorch 2.0.1+

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
- Follow C++17 standard
- Use clang-format for formatting (config provided)
- Follow naming conventions:
  - Classes: `PascalCase`
  - Functions: `snake_case`
  - Variables: `snake_case`
  - Constants: `UPPER_CASE`
  - Member variables: `snake_case_` (trailing underscore)

### Documentation
- Add Doxygen comments for all public APIs
- Include usage examples for complex features
- Update relevant tutorials in `docs/tutorials/`

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

By contributing, you agree that your contributions will be licensed under the MIT License.
