# Durable checkpoint files

`jyppx::liuding::training::FileCheckpointStore` adapts the training coordinator to
`jyppx::liuding::interop::FileStateArchive`. Link `jyppx::liuding::training` with interop enabled;
the file adapter is absent from the installed training headers when interop is disabled. Custom loops can use `FileStateArchive` directly with foundation participant manifests and readers/writers.

```cpp
#include <jyppx/liuding/training/file_checkpoint_store.hpp>

jyppx::liuding::training::FileCheckpointStore destination("checkpoints/last.bin");
coordinator.save(destination);  // Capture at a supported committed boundary.

// Rebuild the complete training domain before restoring it.
jyppx::liuding::training::FileCheckpointStore source("checkpoints/last.bin");
new_coordinator.restore(source, jyppx::liuding::RestorePolicy::exact_only);
```

Register every state owner before capture and reconstruction: model, optimizers, schedulers, precision scaler, RNG, data pipelines, metrics, callbacks, dispatcher, event outbox, strategy, and trainer. The coordinator validates the complete domain before mutation and rolls back participant changes on a failed restore. The file store does not discover missing state owners. Batch midpoint and uncommitted gradient recovery remain unsupported.

The format-v1 file contains an eight-byte magic, a little-endian version and manifest length, the canonical checkpoint manifest JSON, the declared blobs, and a SHA-256 trailer over the preceding bytes. The manifest includes a build fingerprint, participant schema and restore levels, and each blob's exact size and SHA-256. Digest identities belong to foundation (`ArtifactDigest` and
`ArtifactDigestAlgorithm`), shared with the local CAS. Field order is not
identity: the canonical codec sorts fields by name, and participants must
validate fields by their names and schema properties.

Readers verify framing, bounds, complete-artifact and per-blob hashes before
exposing state. SHA-256 detects byte changes; producer authentication requires
a separate trusted signature. The first released container format is v1.
Historical participant schemas can be migrated by their participant readers;
there is no invented v0 file container migration.

Writes use a sibling temporary file, flush data, and atomically replace the
final file on the same filesystem. A per-path operating-system lock rejects
concurrent writers. Read handles retain one opened file generation and allow
concurrent replacement. A reader cannot remove an active writer's temporary
file. Stale temporary files can be discarded when their writer lock is free.
The lock sidecar remains as a reusable synchronization object.

For inspection tools, set `FileStateArchiveOptions::read_only=true`. This
requires an existing valid snapshot, rejects capture, and preserves all sibling
temporary/lock files without creating directories or acquiring a writer lock.
The CLI's `checkpoint inspect/verify/diff` commands use this mode.

On Windows, publication retries transient sharing, lock, and access conflicts
for at most 500 milliseconds. A persistent conflict fails without deleting the
prior destination. Readers open the snapshot directly and treat only a missing
file/path as a new archive; other open failures remain errors.

`synchronize=true` requests file and directory durability (Windows uses
`FlushFileBuffers` and `MOVEFILE_WRITE_THROUGH`; POSIX uses `fsync` and
`rename`). Newly created parent directory entries are synchronized as well.
A failure before replacement preserves the prior file. If directory sync fails after replacement, either complete generation may survive a power loss: the call fails, the current handle keeps its prior descriptor and readers together, and reopening inspects the complete generation at the final path. Atomicity and durability depend on the local filesystem's guarantees.

All blobs and the encoded file are held in memory. Set `max_field_bytes`,
`max_artifact_bytes`, manifest parser limits, and a free-storage reserve for
the workload; these bounds are not a streaming-memory guarantee. Truncation,
hash mismatch, unknown format version, incomplete required fields, lock
contention, storage exhaustion, and filesystem errors produce structured
foundation errors. Objects are not thread-safe; quiesce training state during
capture and use separate handles for independent readers.
