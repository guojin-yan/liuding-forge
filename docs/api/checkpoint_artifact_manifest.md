# Checkpoint Artifact Manifest

Header: `jyppx/interop/checkpoint_manifest.hpp`<br>
Target: `jyppx::liuding::interop`

`CheckpointArtifactManifest` is the format-level bridge between storage-neutral
`CheckpointParticipant` state and a future checkpoint container. It records the
exact build fingerprint, participant schema, concrete blob size, portable
relative path, and SHA-256 identity without performing filesystem publication.

The schema-v1 codec provides:

- deterministic compact JSON with participants, fields, degradation reasons,
  and blobs in stable order;
- strict parsing that rejects duplicate and unknown fields;
- UTF-8, JSON depth/node/string, participant/blob, path, and total-byte limits;
- required-field completeness, declared-field size, unique path, and digest
  syntax validation;
- portable path validation that rejects absolute paths, traversal segments,
  backslashes, drive prefixes, and unsafe characters;
- an experimental capability descriptor for machine-readable preflight.

The current slice validates digest declarations but does not read blobs or
compute their hashes. Atomic file publication, digest computation, sharding,
checkpoint migration, model bundles, and safetensors are separate interop
capabilities and must not be inferred from successful manifest parsing.

```cpp
jyppx::liuding::interop::CheckpointArtifactManifest manifest{/* bounded fields */};
auto json = jyppx::liuding::interop::checkpoint_artifact_manifest_json(manifest);
auto decoded =
    jyppx::liuding::interop::parse_checkpoint_artifact_manifest_json(json);
```

Callers should parse and validate the manifest before opening any referenced
path. A later container reader must independently enforce the declared blob
size, compute the digest over stored bytes, and publish restored state only
after all participants pass preflight.
