# Dependency and license audit

This document records the dependency audit performed on 2026-10-21. The
machine-readable lock remains `dependency-manifest.json`; this document
explains which claims are supported by exact local artifacts and which remain
release blockers. The audit does not turn a draft manifest into a GA approval.

The complete local evidence index is
`out/deps/dependency-audit-20261021/audit-index.json`. It records archive
sizes, SHA-256 values, exact archive entries, downloaded upstream documents,
and the raw OSV query responses. The directory is generated evidence under
`out/`; it is intentionally excluded from the source archive and must be
recreated or copied into a release evidence bundle by the publisher.

## Locked dependencies

| Manifest ID | Exact evidence | License evidence | Audit state |
|---|---|---|---|
| `libtorch-primary-windows-cu126` | `libtorch-win-shared-with-deps-2.14.1+cu126.zip`, 5,158,600,319 bytes, SHA-256 `7e65f208302ebd8501fe6beba69ce2a8b582a3d1caf17b3f00fc0b7ecd20d6e8`; archive `build-version` is `2.14.1+cu126`, `build-hash` is `5c4886908584029761b579af026dcfb627c84070`. | PyTorch source `LICENSE` at that build commit is retained as `third_party/licenses/pytorch-2.14.1-build-5c488690-LICENSE.txt` (SHA-256 `bd018feef8825e88181c84eb7e3aa4eafb8f08a20d9fd6ef948569610c4a3e43`). The binary archive contains no `LICENSE`, `LICENCE`, `NOTICE`, or `COPYRIGHT` entry. | **Pending.** A PyTorch root license does not satisfy the exact notice obligation for bundled binary components. |
| `googletest` | `googletest-1.14.0.zip`, 1,090,859 bytes, SHA-256 `1f357c27ca988c3f7c6b4bf68a9395005ac6761f034046e9dde0896e3aba00e4`. | Archive entry `googletest-1.14.0/LICENSE`, extracted with SHA-256 `9702de7e4117a8e2b20dafab11ffda58c198aede066406496bef670d40a22138`; BSD-3-Clause. | Recorded. This is test-only and is not installed by the SDK. |
| `nlohmann-json` | `nlohmann-json-3.12.0.tar.xz`, 114,576 bytes, SHA-256 `42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa`. | Archive entry `json/LICENSE.MIT`, extracted with SHA-256 `46a65cffd1ea955132d95a8dd921640714a8d6b537d2e4e482d31145ae95b603`; MIT. | Recorded. It is a private implementation dependency and is not exposed by installed public headers. |
| `sqlite` | `sqlite-amalgamation-3540100.zip`, 2,990,966 bytes, SHA-256 `0a9c8e34352dad46006aa82836d5383bff9dc6bc9ce5a60553c05aa30e6ad1f9`. | The five-entry amalgamation archive contains no separate license-like file. The exact first 294 bytes of pinned `sqlite3.h` retain its copyright disclaimer and blessing with SHA-256 `d18a5a730e4776a0487cdec51ab38139573711c6387d96745898be4c779bb16e`. The exact upstream `copyright.html` response is retained with SHA-256 `44ca9f793055c8e32fc65f65a4f5bcf813a33f5bdaaa084067dd617a4ed3cc70`; it documents the public-domain dedication and blessing. | Recorded with upstream evidence. Preserve the provenance page in source and release notices. |
| `libtorch-candidate-cpu` | No exact cache archive or digest is recorded. The manifest source is a URL template and lists CPU, Linux, Windows, and macOS candidates. | The PyTorch `v2.5.1` root `LICENSE` is retained (SHA-256 `47a26beb94e3f6b333a3677fc85d546f1fdfd2f0b3686c26d2fb5b10e0134165`), but no candidate package notice set was inspected. | **Pending.** It is not a release-supported runtime. Do not promote it by filling in a guessed digest. |
| `libtorch-candidate-cuda-cu124` | The manifest records the exact official URL and archive digest independently rechecked with `sha256sum` on the remote build host on 2026-10-21: `08552b1d13de1389d01c64aaa5e2c6c5092f922618b0d32ccff8ed9099d75735`; the archive is retained on the remote build host, not in this checkout. The remote installation reports `2.5.1+cu124`, build hash `a8d6afb511a69687bbb2b7e88a3cf67917e1697e`. | The package installation contains no license-like file. The `v2.5.1` source license above is provenance only; bundled component notices remain unavailable. | **Pending.** The archive digest was reverified, but exact bundled notices and security qualification are still required before redistribution or GA support. |

`THIRD_PARTY_NOTICES.md` is a concise policy and provenance summary. The
individual evidence files in `third_party/licenses/` are the retained source
documents. Neither one substitutes for the exact bundled notice set missing
from a LibTorch binary archive.

The cu126 archive's `libtorch/bin` inventory also makes the unresolved scope
concrete: it contains eight PyTorch/ATen DLLs, twenty-four CUDA/NVIDIA DLLs,
two Intel OpenMP DLLs, and bundled `uv.dll`, `zlibwapi.dll`, and `shm.dll`.
The inventory and exact filenames are retained in the audit index. A root
PyTorch license alone cannot establish the license terms for cuDNN, CUDA
redistributables, OpenMP, zlib, libuv, or the other bundled runtime pieces.

## Optional Python and sidecar audit

The Python telemetry and safetensors environments are optional and are not
installed by the C++ SDK. Their exact wheel locks and package metadata are
maintained separately in `tools/telemetry_sidecar/requirements.lock`,
`tools/telemetry_sidecar/dependency-metadata.json`, and
`scripts/interop/requirements-safetensors.lock`. The recorded wheel hashes and
Apache-2.0 safetensors license evidence are retained under `out/deps/` and in
`scripts/interop/dependency-metadata.json`.

The prepared Windows CPython 3.11 environment yielded license-file paths and
SHA-256 values for all 14 top-level sidecar lock packages in
`out/deps/dependency-audit-20261021/optional-python-license-index.json`.
This is platform-specific evidence for that prepared environment; it does not
replace exact wheel archives, transitive SBOM review, or the notices for a
future redistributed sidecar bundle.

An OSV API snapshot was taken for selected locked packages. The raw responses
are stored in `out/deps/dependency-audit-20261021/` and bound by SHA-256 in the
audit index. The query found advisories affecting the previous optional lock versions
retained in historical environment snapshots:

| Package/version | Advisory IDs observed | Release implication |
|---|---|---|
| Werkzeug 3.1.3 | `GHSA-29vq-49wr-vm6x`, `GHSA-87hc-h4r5-73f7`, `GHSA-g6x2-hccm-hh4m`, `GHSA-hgf8-39gv-g3f2`, `PYSEC-2026-2044`, `PYSEC-2026-2046`, `PYSEC-2026-2320`, `PYSEC-2026-4312`; fixed versions reported by OSV are 3.1.4, 3.1.5, 3.1.6, and 3.1.9 depending on the advisory. | **Remediated to 3.1.9.** Both actual environments and official sidecar acceptance passed. |
| protobuf 6.32.1 | `GHSA-7gcm-g887-7qv7` (HIGH) and `PYSEC-2026-1805`; OSV reports the relevant 6.x fix as 6.33.5. | **Remediated to 6.33.5.** Exact versions, official messages and sidecar acceptance passed on both platforms. |
| Pillow 11.3.0 | 45 OSV records before GHSA/PYSEC alias deduplication, including CRITICAL `GHSA-8g5g-8vmr-9782` (JPEG heap overflow) and HIGH image/font/PDF issues. Reported fixes range through 13.0.0. | **Remediated to 13.0.0.** Both actual environments and official sidecar acceptance passed; raw findings remain retained. |
| setuptools 80.9.0 | `GHSA-h35f-9h28-mq5c` and `PYSEC-2026-3447`, covering Unicode normalization collisions in sdist exclusion; OSV reports fix 83.0.0. | **Remediated to 83.0.0.** Event writer/readback passed on both platforms. TensorBoard UI/server uses retired `pkg_resources` and remains unqualified. |
| GoogleTest 1.14.0 and SQLite 3.54.1 | No advisories returned for the queried OSV package coordinates. | This is a point-in-time query result, not proof of absence; continue normal monitoring. |

An initial nlohmann/json GitHub-coordinate query returned HTTP 400 and an
initial PyTorch query incorrectly used the PyPI package name `pytorch`. Both
are retained as inconclusive evidence, not clean security results. Follow-up
queries used OSS-Fuzz `nlohmann_json` 3.12.0 and PyPI `torch`; the former and
`torch` 2.14.1 returned empty results. These ecosystems have limited binary
and transitive coverage; empty responses do not substitute for upstream
security review. No transitive graph or CVE result is inferred from
`pip check`, wheel metadata, or a clean build.

All fourteen top-level sidecar lock packages were queried by exact PyPI
package/version. Ten returned empty results; Werkzeug, protobuf, Pillow, and
setuptools returned the findings above. The counts are raw advisory records,
not a deduplicated CVE count or a claim that every issue is exposed by this
sidecar's runtime path.

OSV exact-version queries returned empty results for all four remediation
pins on the same date: Werkzeug 3.1.9, protobuf 6.33.5, Pillow 13.0.0 and
setuptools 83.0.0. The seven selected wheel artifacts were downloaded from
URLs in official PyPI metadata, checked against both declared sizes and
SHA-256, and inspected for `METADATA`, wheel tags and license/notice entries.
Their 24 exact notice originals are retained under
`third_party/licenses/optional-sidecar/`. The checked-in license index binds
these and the six C++ provenance documents to their exact upstream artifacts.
The sidecar lock, metadata and protobuf runtime pin have been updated together.
The Windows offline, hash-enforced pip dry run succeeded, followed by actual
upgrade of the four packages, a fresh pip report, `pip check`, exact-version
probes for all fourteen pins, and EventFileWriter/EventAccumulator/OTLP schema
imports. The new report and exact lock snapshot are bound in
`scripts/interop/optional-runtime-provenance.json` under `security_remediation`.
Ubuntu also completed the actual hash-only upgrade, `pip check` and the exact
fourteen-package probe. The current Windows and Ubuntu CTest runs both passed
`OfficialTelemetryAcceptance`, which requires the product CLI, real Collector,
TensorBoard readback and no skipped subtests. Separate pip reports, platform
input-lock byte snapshots and acceptance-log hashes are bound under
`security_remediation`. Windows CRLF and Linux LF snapshots have different exact
digests and are recorded independently. This acceptance covers the runtime
adapter, not TensorBoard UI/server or a complete transitive security audit.

The corrected PyPI `torch` 2.5.1 query returned 22 raw advisory records,
including CRITICAL `GHSA-53q9-r3pm-6pq6` for `torch.load(weights_only=True)`
(fixed in 2.6), as well as packed-sequence, LSTM, JIT, quantization and memory
issues. The raw response is retained as `torch-2.5.1.json` with SHA-256
`63c55df61d5d9a136cf2f14741428f748d1c17af871aea2f3ef21df067708da3`.
The Python `torch.load` exploit does not by itself establish an exploit in
JYPPX's C++ checkpoint path; neither does that distinction establish that all
LibTorch C++ operators are unaffected. The remote 2.5.1 runtime remains an
explicit qualification/security audit item, with exact API exposure and
fixed-runtime validation required for release support.

The current release preparation targets an SDK-only candidate requiring a
separately obtained LibTorch runtime, excluding LibTorch and optional Python
binaries from the candidate. This resolves the candidate's binary bundling
choice; it does not silently promote an affected runtime to GA support or
close the missing bundled notice and transitive review records.

## GA blockers and required follow-up

The release scripts intentionally keep the candidate ineligible while any of
these conditions remain:

1. `dependency-manifest.json` is `draft` or any entry has pending integrity or
   notice evidence.
2. The exact LibTorch binary notice set is absent, or the publisher cannot
   state that the release requires a separately obtained LibTorch runtime and
   excludes those binaries.
3. Optional sidecar packages remain on versions with applicable OSV advisories
   when the sidecar is part of the delivered release.
4. The complete dependency inventory, transitive license review, and trusted
   publisher evidence have not been bound to the release candidate.

Before changing a pending field to `verified` or `recorded`, retain the exact
archive/source bytes, recompute their digest independently, update this audit
and `THIRD_PARTY_NOTICES.md`, and run
`scripts/validation/dependency_manifest.py`. The release preparation and
verification tools must then be run against a fresh candidate and their GA
blocker output reviewed. A lock or manifest string alone cannot close a
license, vulnerability, or source-integrity finding.
