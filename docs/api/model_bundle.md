# Immutable model bundles

`jyppx::interop::ModelBundleManifest` describes a local, immutable directory of
model data. `verify_model_bundle_directory` verifies actual file bytes and every
safetensors tensor association. `publish_model_bundle` copies declared sources to
private staging, verifies them, flushes them, and atomically publishes a **new**
directory. These APIs belong to `jyppx::interop` and require foundation; they do
not depend on core, nn, training, experiment, Python, or a model runtime.

The public header is `jyppx/interop/model_bundle.hpp`. The reserved manifest file
is `manifest.json`. Schema version 1 is the first documented bundle format.
Other versions, unknown fields, duplicate decoded JSON keys, and unsupported
compression/encryption fail. There is no invented legacy/N−1 migration: a future
documented migration needs fixtures for the actual older format.

## What a bundle contains

Every data file appears once in the exhaustive `files` inventory. Its entry has a
unique id, relative path, closed data role, and foundation `ArtifactRef` containing
the full SHA-256 digest, exact size, media type, logical name, and producer.
References use file ids; an `ArtifactRef` remains storage neutral and never
contains a local storage path. Every file has exactly one manifest owner. A
present but unreferenced file fails validation.

| Field | Contract |
|---|---|
| `format`, `schema_version` | Exactly `jyppx.model_bundle`, integer `1` |
| `bundle_id`, `build` | Producer identity and the complete producer `BuildFingerprint`; reading never samples the reader's build |
| `compression`, `encryption` | Exactly `none`; no decompression or custom cryptography |
| `models` | Explicit variants with complete file lists; formats are safetensors or single-file ONNX |
| safetensors `tensors` | Unique name, exact dtype/shape, device intent, and owning shard id for **every** tensor |
| input/output schemas | Required owner-versioned descriptor references |
| preprocessing/postprocessing | Required owner-versioned declarative descriptor references, including an explicit identity descriptor where applicable |
| labels | Owner-versioned descriptor reference, or explicit JSON `null` |
| evaluation, SBOM, provenance | Required owner-versioned evidence/data references |
| weight import reports | Optional versioned full/partial-import report references |
| runtime | Explicit backend/version, environment-lock digest, exact dependency versions, device/dtype/OS matrices, and dynamic-shape declaration |
| signatures | Bounded detached signature metadata; the empty array means unsigned |

A descriptor reference contains its owner's `schema_id`, positive
`schema_version`, and `file_id`. The referenced file is verified by size and
digest. The schema owner must parse its own data before invocation; bundle
verification does not pretend that an unknown owner's schema is understood.
The same applies to evaluation evidence: including a report does not imply that
evaluation passed, or that it meets a particular deployment policy.

Runtime requirements are declarations. Verification never claims that the
backend is installed, an accelerator is available, or numerical results agree.
Use backend preflight and official runtime verification for those decisions.

## Safetensors sharding

`shard_safetensors_for_bundle(tensors, maximum_bytes, prefix, device_intent)`
validates names, shapes, dtypes, byte order, and exact payload sizes before
returning encoded shards. It sorts tensor names and greedily groups **whole**
tensors. A conservative JSON-header bound includes offsets, escaping, padding,
and framing. Every produced file fits the requested maximum including its
header; some space may remain unused. A single tensor that cannot fit fails
explicitly. No tensor is silently split or converted. Empty tensor sets produce
one valid empty safetensors file.

Each `EncodedBundleShard` owns its complete bytes, populated inventory entry,
and complete tensor associations. Add `file.id` to the model's `file_ids`, its
`tensors` to the model's tensor list, and its `file` to the manifest inventory.
The caller writes the bytes to a local source file for publication. All source
spans must remain alive and unchanged while sharding runs. Output is buffered in
memory and bounded by `max_total_bytes`; it is not a streaming filesystem writer.

Directory verification parses every actual safetensors shard after matching its
full-file digest. Actual names, dtypes, and shapes must match the manifest
exactly. Missing tensors, extra tensors, tensors assigned to another shard, and
duplicate names across shards of one model all fail. Zero dimensions and scalars
retain the safetensors codec's exact semantics. `device_intent` is `cpu`, `cuda`,
or `any` and never causes device allocation.

ONNX support in this envelope is **file integrity only**. It accepts one opaque
model file with media type `application/onnx`, without a safetensors tensor list.
It does not parse protobuf, follow ONNX external-data references, call a checker,
or run ONNX Runtime. `ModelBundleVerification::integrity_only_models` explicitly
counts these variants. A deployment adapter must use the official parser/checker
and numerical verification contract. Multi-file ONNX/external-data packaging is
not supported by this schema-v1 implementation. An ONNX bundle verification
result must not be presented as official model/runtime validation.

## Verify and publish

```cpp
#include <jyppx/interop/model_bundle.hpp>

namespace io = jyppx::interop;

auto verified = io::verify_model_bundle_directory("artifacts/model-v1");
// verified.manifest is a validated snapshot; verified.manifest_digest is the
// SHA-256 of the original manifest bytes, not a re-encoded approximation.
// verified.signature_status distinguishes integrity from authentication.

// Construct manifest with all owner descriptors and actual inventory hashes.
// Supply exactly one immutable local source file per declared file id.
io::ModelBundleManifest manifest = verified.manifest;
std::vector<io::BundlePublishSource> sources;
for (const auto& file : manifest.files) {
  sources.push_back({.file_id = file.id,
                     .source_path = std::filesystem::path("artifacts/model-v1") /
                                    file.relative_path});
}
auto published = io::publish_model_bundle("artifacts/model-v2", manifest, sources);
```

Publication requires an existing local parent directory. It exclusively creates
a sibling staging directory, opens each source without following links, copies
and hashes it, writes the canonical manifest, verifies the complete staging
directory, and flushes every file. Linux additionally fsyncs the child directories
and parent. Windows uses `FlushFileBuffers` and `MOVEFILE_WRITE_THROUGH`.

The final operation is atomic **no-replace** rename: Linux `renameat2` with
`RENAME_NOREPLACE`, or Windows `MoveFileExW` without replace flags. An existing or
racing destination is never modified, including an existing empty directory.
An OS/filesystem without the required primitive fails closed. Directory
publication is supported on Windows local drive paths and Linux; UNC/device
paths and fallback non-atomic copy publication are unsupported. The local
filesystem must provide the documented rename/flush semantics.

On a precommit failure, cleanup of private staging is attempted and any older
valid bundle remains intact. A crash or cleanup I/O failure may leave a private
staging directory. The function never recursively clears a destination. If the
rename succeeds but the later parent-directory flush fails, the function may
throw while the complete new directory is already visible. Verify that target
before retrying. Atomic visibility is distinct from the platform/filesystem's
power-loss guarantees. Crash-left staging directories are not adopted or
automatically deleted during verification.

Immutable names are deliberate: publish the next version under a new name and
let the owning application select its active artifact reference. This API does
not claim atomic replacement of a populated directory or manage application
release pointers.

## Paths and resources

All manifest paths use lowercase ASCII letters, digits, `_`, `-`, `.`, and `/`.
Components are nonempty, at most 128 bytes, have no trailing dot, and cannot be
`.`/`..` or Windows device names. There are at most 16 components. Absolute,
drive, backslash, alternate stream, traversal, case-alias, duplicate, reserved
manifest, and file/directory-prefix collisions fail. The trusted **outer** local
directory and source paths may contain Unicode and spaces.

Native directory handles traverse every ancestor without following symlinks or
Windows reparse points. Native file handles supply size and all hashed/read
bytes; files with multiple hard links are rejected. Windows denies write/delete
sharing during reads. Linux checks inode size, link count, modification time,
and change time before/after reading. Bundle membership is checked before and
after file reads, and undeclared files or empty directories fail.

The owner must keep the outer source/destination namespace and source content
stable, with trusted parent-directory permissions. Verification returns a
snapshot, not a filesystem lease. It does not authorize reopening an untrusted
path later without checking it again or loading different bytes. This is not an
OS security sandbox and makes no claim against privileged concurrent mutation,
mount replacement, or compromised filesystem/kernel behavior.

The default limits are 16 MiB of manifest bytes, 1,024 files, 32 model variants,
65,536 tensors, rank 32, dimension/element count up to 2³², 512 MiB per file, and
4 GiB aggregate data files. `max_total_bytes` excludes the separately bounded
manifest. Decoded strings are at most 4,096 bytes; paths default to 1,024 bytes.
Signature limits are 16 entries and 2,048 decoded bytes per signature. All limits
are explicit positive values and can be reduced for the application.

JSON depth and nodes are bounded independently of declared array counts. File
sizes are read from the opened native handle and checked before buffering.
Only one safetensors shard is buffered during directory verification; descriptor
and opaque model bytes are streamed. Encoded byte/count limits are not a process
RSS guarantee. Resource exhaustion uses `ResourceExhausted`; malformed encoded
contracts use `SerializationError`; unsupported versions/encodings use
`CompatibilityError`; filesystem/no-follow failures use `IoError`.

## Signature trust and data-only policy

`model_bundle_signing_payload` returns the exact bytes to sign:

```text
JYPPX-MODEL-BUNDLE-SIGNATURE-V1\n
<canonical compact JSON with the entire signatures field omitted>
```

The displayed `\n` is one newline byte. There is no extra newline before JSON.
All inventory digests/sizes, producer build, model/tensor mappings, descriptors,
and runtime requirements are covered. Canonicalization sorts object keys and
identity-based arrays; semantically identical inventory ordering yields the same
payload. Detached signature metadata uses an algorithm token, trusted key id,
and nonempty lowercase hex. No private key or implicit key discovery is stored.

The caller supplies `ModelBundleVerifyOptions::signature_verifier`. Its callback
receives each signature and the **actual full canonical payload** and must use a
mature cryptographic implementation plus the application's trusted-key policy.
Every declared signature must be accepted. Returning false fails verification;
throwing produces a redacted `BackendError`. No algorithm is implemented by
inventing a hash-based signature substitute.

| Situation | Result |
|---|---|
| No signatures, default policy | `unsigned_bundle` |
| Signatures present, no verifier, default policy | `present_unverified` |
| Every signature accepted by explicit verifier | `trusted` |
| `require_trusted_signature=true` with no signatures/verifier | Reject |
| Any supplied verifier rejects or throws | Reject |

SHA-256 identity alone does not authenticate a producer. Applications accepting
untrusted distributions should require a trusted signature or independently
trusted manifest digest. Signature presence alone is never called valid.

The v1 role/format set has no executable/plugin entry. Known script, dynamic
library, executable, pickle, Torch archive, and zip filenames are rejected;
common executable/shebang magic is rejected even under a data filename. These
checks are intentionally not a universal malware classifier. No bundle file is
ever imported, loaded as a library, unpickled, or executed. Owner descriptor
parsers and downstream runtimes must maintain that policy and apply their own
untrusted-data checks.

Unit coverage includes canonical round trips, version/unknown/duplicate keys,
resource bounds, path aliases/traversal, file/tensor coverage, real hashes/sizes,
no-follow and hard-link checks, stage cleanup, preservation of existing bundles,
and explicit signature-policy callbacks. Synthetic callbacks in tests verify the
trust interface only and are not cryptographic signing implementations.
