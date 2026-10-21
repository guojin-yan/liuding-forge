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
- The locked Windows `2.14.1+cu126` archive was inspected on 2026-10-21 and
  contains no license-like entry. Its PyTorch source `LICENSE` at build commit
  `5c4886908584029761b579af026dcfb627c84070` is retained in
  `third_party/licenses/`, but does not close the bundled-component notice
  audit. The candidate `2.5.1+cu124` installation has the same unresolved
  notice boundary. Both manifest entries remain pending.
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
- The pinned amalgamation archive contains no license file; the upstream
  copyright page and the exact copyright/blessing excerpt from pinned
  `sqlite3.h` are retained in `third_party/licenses/` with their audit digests.
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
- The exact `googletest-1.14.0/LICENSE` entry is retained in
  `third_party/licenses/` and the dependency manifest marks its notice audit
  recorded. GoogleTest is test-only.
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

## Optional Python Sidecar

The optional sidecar runtime is separately deployed and is not installed by the
C++ SDK. Its exact wheel allowlist is
`tools/telemetry_sidecar/requirements.lock`. The 2026-10-21 security remediation
pins Werkzeug 3.1.9 (BSD-3-Clause), protobuf 6.33.5 (BSD-3-Clause), Pillow
13.0.0 (HPND and separately licensed bundled components), and setuptools 83.0.0
(MIT and separately licensed bundled vendors). Exact original license/notice
entries for the selected Windows/Linux and universal wheels are retained
under `third_party/licenses/optional-sidecar/`, with artifact and file digests
in `third_party/licenses/manifest.json`. Preserve the full originals when
redistributing those wheels; this list does not establish a complete transitive
license review of the entire optional Python runtime.

## Reference Projects

Projects stored outside the Git repository under the local `refer/` directory
are research inputs, not JYPPX dependencies or redistributed source. Their
licenses do not change the JYPPX project license. Any future source reuse,
linking or binary redistribution requires a separate provenance and license
review before integration.
