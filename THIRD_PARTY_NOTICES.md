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

### SQLite

- Project: https://sqlite.org/
- Role: embedded local run metadata, transactions, and CAS coordination inside
  `jyppx::experiment`
- Version selected by the current build: 3.54.1
- Source: https://sqlite.org/2026/sqlite-amalgamation-3540100.zip
- Upstream status: public domain (https://sqlite.org/copyright.html)
- Distribution rule: the amalgamation is compiled into the experiment static
  library; public JYPPX headers do not expose SQLite types. The upstream source
  disclaims copyright and provides a blessing rather than a license requirement.
  Preserve this provenance and the pinned digest in source distributions.

## Test Dependency

### GoogleTest

- Project: https://github.com/google/googletest
- Role: build-time unit and integration testing only
- Version selected by the current build: 1.14.0
- Upstream license: BSD 3-Clause
- Distribution rule: GoogleTest is fetched for tests and is not installed as
  part of the JYPPX SDK. Any future redistribution must include its upstream
  copyright notice, license conditions and disclaimer.

## Build Dependency

### JSON for Modern C++ (nlohmann/json)

- Project: https://github.com/nlohmann/json
- Role: strict JSON parsing and serialization inside `jyppx::experiment`
- Version selected by the current build: 3.12.0
- Upstream license: MIT
- Distribution rule: the header is compiled into the experiment config static
  library and is not exposed by JYPPX public headers. Source or binary
  redistribution must retain the upstream copyright notice and MIT license.

## Reference Projects

Projects stored outside the Git repository under the local `refer/` directory
are research inputs, not JYPPX dependencies or redistributed source. Their
licenses do not change the JYPPX project license. Any future source reuse,
linking or binary redistribution requires a separate provenance and license
review before integration.
