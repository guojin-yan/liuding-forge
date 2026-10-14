# Third-Party Notices

JYPPX Training Framework is licensed under the Apache License, Version 2.0.
Third-party software remains subject to its own license terms. A dependency's
presence here does not mean its source or binaries are redistributed with every
JYPPX package.

## Runtime Dependency

### LibTorch / PyTorch

- Project: https://github.com/pytorch/pytorch
- Role: tensor, autograd, device and runtime backend
- Upstream license: BSD-style license with separately licensed bundled
  third-party components
- Distribution rule: when a JYPPX artifact redistributes LibTorch binaries,
  it must also redistribute the exact license and notice files supplied with
  that LibTorch build. This summary does not replace those files.

## Test Dependency

### GoogleTest

- Project: https://github.com/google/googletest
- Role: build-time unit and integration testing only
- Version selected by the current build: 1.14.0
- Upstream license: BSD 3-Clause
- Distribution rule: GoogleTest is fetched for tests and is not installed as
  part of the JYPPX SDK. Any future redistribution must include its upstream
  copyright notice, license conditions and disclaimer.

## Reference Projects

Projects stored outside the Git repository under the local `refer/` directory
are research inputs, not JYPPX dependencies or redistributed source. Their
licenses do not change the JYPPX project license. Any future source reuse,
linking or binary redistribution requires a separate provenance and license
review before integration.
