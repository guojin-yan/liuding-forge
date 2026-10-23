# Preparing and verifying Liuding Forge SDK releases

The public brand is 六丁炼算（Liuding Forge）. Release candidates use
`liuding-forge-<version>-<platform>.zip`; CPack packages also use the
`liuding-forge` prefix. Installed headers, `find_package(jyppx)`, exported
`jyppx::liuding::` targets, and existing release/SBOM protocol identities remain stable.

`scripts/release/prepare_release.py` prepares a reviewable candidate from an
explicit installation prefix. `verify_release.py` independently checks its declarations, hashes, evidence bindings, and safely extracted contents. These are offline maintainer tools, separate from the installed product CLI. They never build code, fetch dependencies, run installed binaries, perform a CVE scan, invent a test result, generate a project key, sign, or publish a release.

The current dependency manifest is a **draft**, including unresolved integrity and third-party notice audits. This state is preserved in candidate metadata and prevents GA approval. Existing notices are a summary: redistributing a particular LibTorch build still requires that build's actual bundled license and notice files. The explicit dependency audit must cover these obligations, the resolved/transitive graph, and the precise files actually redistributed.

## Inputs and exact identity

Use a staged `cmake --install` prefix, kept unchanged during packaging. The caller exclusively owns the prefix, input documents and output parent for the operation. Provide the exact version, portable platform/build label, **full** 40/64-character lowercase source commit, explicit clean/dirty state and a fixed
`SOURCE_DATE_EPOCH`. These are publisher declarations; the scripts do not
derive build identity by inspecting an unrelated current checkout. The
installed `include/jyppx/liuding/foundation/build_config.hpp` must agree with the base
semantic version, preventing relabeling a 0.3.0 SDK as 1.0.0. An `-rc.1` label
may describe a candidate compiled with that base version; it blocks GA.

The packaging operation has an in-toto Statement v1 / SLSA provenance v1
document that explicitly describes **install-prefix packaging**. It is not an
attestation that this script built or tested the original binary. Original
build provenance, toolchain, flags, LibTorch/ABI/CRT/CUDA facts and acceptance
results must be present in publisher-supplied evidence artifacts.

First calculate the installation inventory independently of release evidence:

```sh
python scripts/release/prepare_release.py inventory \ --prefix /work/install --output /work/install-inventory.json
```

The response includes the exact SHA-256 of canonical inventory bytes. The
inventory contains sorted relative paths, lengths, SHA-256 hashes, portable
0644/0755 modes and aggregate bytes. Use that digest in the evidence index.
Hash the **exact original dependency-manifest bytes** separately; formatting
changes also change its binding.

## Evidence index schema 1

Supply a UTF-8 JSON object with exactly these fields:

```json
{ "schema_version": 1, "release": {
    "version": "1.0.0-rc.1",
    "platform": "linux-x86_64-cuda-cu124-cxx11-abi1",
    "source": {"commit": "FULL_SOURCE_COMMIT", "dirty": false}
}, "installation_inventory_sha256": "EXACT_INVENTORY_SHA256", "dependency_lock_sha256": "EXACT_LOCK_SHA256", "checks": [
    {
      "id": "windows",
      "status": "pending",
      "details": "Awaiting validation of this exact source and dependency lock",
      "artifacts": []
    }
], "dependency_audit": {
    "status": "pending",
    "audited_dependency_ids": [],
    "blocking_licenses": [],
    "blocking_vulnerabilities": [],
    "artifacts": []
} }
```

Placeholders are not accepted as actual commits/digests. `release` must match
the preparation inputs exactly. A null inventory digest permits early
candidate review and adds a pending GA blocker. A non-null digest must match.
Every referenced artifact is an exact `{ "path": "logs/ctest.log", "size": 123,
"sha256": "64_lowercase_hex_characters" }` record, relative to the index's
parent. Preparation reads, hashes and includes those actual files in the
candidate. Passed checks require at least one artifact. A textual statement
alone cannot satisfy a gate. Duplicate checks, unknown fields, conflicting
artifact records, missing files and hash mismatches are rejected.

All check states are `passed`, `pending` or `failed`. Missing/non-passed checks
block GA. Required check IDs are:

| Check ID | Evidence responsibility |
| --- | --- |
| `windows` | Exact Windows configure/build/tests and runtime/toolchain facts |
| `ubuntu-cuda` | Exact Ubuntu build/tests and real CUDA device execution |
| `compatibility-matrix` | Every declared supported compatibility combination |
| `install-consumer` | Clean installed package and optional ON/OFF consumers |
| `sanitizers` | Supported sanitizer lanes and reviewed exclusions |
| `fuzz` | Current parser/codec/input-boundary fuzz evidence |
| `static-analysis` | Source, secret and static-analysis gates |
| `coverage-risk-review` | Current denominator, coverage and risk decisions |
| `performance-memory-build-size` | Reviewed performance/resource/build/package budgets |
| `documentation-api-examples` | API, examples, migration and troubleshooting review |
| `upgrade-rollback` | Beta/RC upgrade and rollback drill |
| `release-blockers` | No open P0/P1 release blockers |
| `offline-rebuild` | Clean/offline rebuild equivalence evidence |

The dependency audit must be `passed`, enumerate every exact manifest
dependency ID, include actual audit artifacts and contain no unresolved
`blocking_licenses` or `blocking_vulnerabilities`. A complete manifest is also
required: every dependency must have a verified source digest, an exact URL and recorded notice audit. The conservative release license policy allows
`Apache-2.0`, `BSD-2-Clause`, `BSD-3-Clause`, `MIT`, `ISC`, `Zlib`, `BSL-1.0`,
`CC0-1.0`, `Unlicense` and `blessing`. Other/custom licenses remain pending until
a separate reviewed policy update; a manifest string cannot extend the policy. An empty blocking-vulnerability list is a **publisher audit declaration**, not an automatically performed or inferred CVE scan. The tools cannot interpret arbitrary test/scan logs or prove their truth; the external trusted signature authenticates the publisher's reviewed assertions and exact evidence bytes.

## Candidate preparation

```sh
python scripts/release/prepare_release.py prepare \
  --prefix /work/install --out /work/candidate \
  --version 1.0.0-rc.1 --platform linux-x86_64-cuda-cu124-cxx11-abi1 \
  --source-commit FULL_SOURCE_COMMIT --source-state clean \
  --source-date-epoch 1750000000 \
  --evidence-index /work/evidence/index.json \
  --dependency-lock dependency-manifest.json \
  --license LICENSE --notices THIRD_PARTY_NOTICES.md
```

The output directory must not exist or lie inside the installation prefix. Preparation reserves a fresh directory and refuses overwrites. An interrupted publication can leave a partial output; it is invalid until independently verified and must not be treated as a completed release.

The output contains:

- `liuding-forge-VERSION-PLATFORM.zip`: deterministic, sorted, uncompressed ZIP/ZIP64 entries under `payload/`, without links, comments, timestamps from wall time, executable installer code or implicit dependency downloads.
- `inventory.json`, `dependency-manifest.json`, `LICENSE` and
  `THIRD_PARTY_NOTICES.md`: exact content and dependency/license declarations.
- `evidence-index.json` and `evidence/`: original index and referenced logs,
  reports or attestations, copied with exact size/hash checks.
- `sbom.cdx.json`: CycloneDX 1.5 inventory of **declared** locked dependencies.
  It records pending integrity/notice states and does not invent transitive
  dependencies or vulnerability results. Resolved graph completeness needs
  the dependency-audit evidence.
- `provenance.intoto.json`, `release-manifest.json` and
  `signing-payload.json`: exact identity/packaging/evidence bindings and all
unsigned GA blockers. The signing payload binds every published file except itself and the externally held signature.

JSON generated by the tools uses sorted keys, compact UTF-8 and a final LF; the exact final newline is part of the signed payload. Original dependency and evidence JSON bytes are preserved. Repeated preparation with identical inputs produces identical output bytes. This does not claim reproducibility of the upstream SDK build.

## Independent validation and external signatures

```sh
python scripts/release/verify_release.py --directory /work/candidate \
  --expected-version 1.0.0-rc.1 \
  --expected-source-commit FULL_SOURCE_COMMIT \
  --expected-platform linux-x86_64-cuda-cu124-cxx11-abi1
```

Verification checks the exact expected identity and complete folder file set, all metadata/artifact hashes, canonical declarations, dependency/evidence bindings, regenerated SBOM/provenance, and actual safely extracted file bytes. It rejects compressed/encrypted/nonregular/unknown/duplicate ZIP entries, path traversal, Windows device/stream names, case aliases, local/central header disagreement, overlapping/gapped data, archive comments, prepended/trailing bytes and size/hash/CRC failures. Contents are extracted only into a fresh private temporary directory and are never executed. Installed version checks are repeated against the extracted header.

After the candidate is reviewed, an authorized publisher signs the exact payload using an existing protected release key and an approved external OpenSSL installation. For RSA or ECDSA keys using the `dgst -sha256` interface:

```sh
/approved/bin/openssl dgst -sha256 \
  -sign /protected/release-private.pem \
  -out /work/signatures/candidate.sig /work/candidate/signing-payload.json

python scripts/release/verify_release.py --directory /work/candidate \
  --expected-version 1.0.0 --expected-source-commit FULL_SOURCE_COMMIT \
  --expected-platform linux-x86_64-cuda-cu124-cxx11-abi1 \
  --signature /work/signatures/candidate.sig \
  --trusted-public-key /trusted/project-release-public.pem \
  --trusted-public-key-sha256 INDEPENDENTLY_OBTAINED_PUBLIC_KEY_FILE_SHA256 \
  --openssl /approved/bin/openssl --require-ga
```

The GA example applies to a separately prepared final `1.0.0` release; signing an RC never changes its version or promotes it to GA. Keep signature and public key outside the candidate directory. Obtain trust and the key pin through an independent trusted channel, not from an untrusted candidate. Verification pins exact PEM file bytes, invokes the explicit executable with fixed argv/no shell and a 30-second deadline, and rejects missing/bad signatures or key-pin mismatches. It uses no implicit local/OpenSSL trust store and never generates a release key. A valid signature removes only the
`signature.pending` blocker; it cannot close dependency or acceptance gates.
Ed25519/Sigstore/PKCS11 workflows need a separately reviewed verifier adapter.

Candidate preparation and successful candidate verification exit 0 even when
GA is pending; JSON always reports `ga_ready` and `ga_blockers`. `--require-ga`
exits 1 while any blocker remains and exits 0 only after all declared gates
and the trusted signature verify. Malformed, unsafe or inconsistent input
exits 2. A missing verifier/key is never treated as signature success.

## Resource and operational bounds

Both tools reject symlinks, Windows reparse points and hard-linked input files.
Paths use portable ASCII components. Inputs must remain quiescent; file
identity, size and modification timestamps are checked around each read, but
these checks do not substitute for exclusive ownership against a hostile
concurrent process. Verification uses a private temporary extraction directory
and requires enough disk space for expanded bytes.

Default hard caps are 100,000 files/directories, 4 GiB per installed/evidence
file, 32 GiB aggregate installed bytes/archive bytes, 32 GiB aggregate evidence
bytes, 32 MiB per JSON/license/notice document or ZIP central directory, JSON
depth 24, two million JSON nodes including keys, and 8192 UTF-8 bytes per JSON
string. Callers of the Python API may select smaller limits, never larger
ones. Rejecting all compression makes the accepted expansion ratio exactly
one. No generic ZIP extraction or deserialization of executable formats occurs.

Run the script tests once at the chapter validation boundary on Windows and
Ubuntu, together with the agreed full build/test/install lanes. Reuse the
existing build directory. Optional real OpenSSL tests require an available
approved executable; a skip is visible and is not signature gate evidence.
