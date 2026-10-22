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
- Role: strict JSON parsing and serialization inside interop, experiment and the
  optional `jyppx::data_image` annotation readers
- Version selected by the current build: 3.12.0
- Upstream license: MIT
- Distribution rule: the header is compiled into the experiment config static
  library and is not exposed by JYPPX public headers. Source or binary
  redistribution must retain the upstream copyright notice and MIT license.
  SDK installations with interop, experiment or data_image include the original notice at
  `share/doc/jyppx/licenses/nlohmann-json-3.12.0-LICENSE.MIT`. Experiment SDK
  installations also retain the original SQLite copyright/blessing provenance
  in that directory.

### TinyXML2

- Project: https://github.com/leethomason/tinyxml2
- Version: 11.0.0; license: Zlib
- Role: private Pascal VOC XML parsing in optional `jyppx::data_image`
- The pinned source archive and the exact original `LICENSE.txt` are recorded in
  the dependency and retained-license manifests. The original is installed at
  `share/doc/jyppx/licenses/tinyxml2-11.0.0-LICENSE.txt` with this profile.
- Preserve the upstream copyright and license text in redistributions. Parser
  objects are embedded in the image static library; public headers expose no
  TinyXML2 types. Release security review is separate from this notice record.

### yaml-cpp

- Project: https://github.com/jbeder/yaml-cpp
- Version: 0.9.0; license: MIT
- Role: private data-only YOLO dataset YAML parsing in optional `jyppx::data_image`
- The pinned source archive and exact original `LICENSE` are recorded in the
  dependency and retained-license manifests. The original is installed at
  `share/doc/jyppx/licenses/yaml-cpp-0.9.0-LICENSE.txt` with this profile.
- Preserve the original copyright notice and MIT license in source and binary
  redistributions. No YAML parser types enter public headers. JYPPX rejects
  aliases, anchors and explicit tags and never executes download instructions;
  release security review remains separate from this notice record.

### stb_image

- Project: https://github.com/nothings/stb
- Version: 2.30 (2024-05-31); license: MIT or Unlicense (public domain dedication)
- Role: private JPEG, PNG, BMP, GIF, TGA, PNM and related still-image decoding
  inside optional `jyppx::data_image`
- The vendored header is pinned by SHA-256 in `dependency-manifest.json`; its
  upstream license block is retained in
  `third_party/licenses/stb_image-2.30-LICENSE.txt` and installed with the
  data-image profile. No stb_image types enter public headers.
- Decoder limits are enforced before third-party allocation. HDR/PIC and other
  formats that need a different scalar contract remain outside this profile;
  callers receive a structured decode error rather than an implicit conversion.

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
