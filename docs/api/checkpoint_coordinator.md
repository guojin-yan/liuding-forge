# Checkpoint Coordinator

Header: `#include <jyppx/liuding/training/checkpoint_coordinator.hpp>`

Link target: `jyppx::liuding::training`

The training checkpoint coordinator combines the storage-neutral participant
contracts from `jyppx::liuding::foundation`. It does not define a file format, hash,
codec, path layout, or artifact publication mechanism.

## Participant registry

`CheckpointParticipantRegistry` owns registered participants through shared
pointers. IDs must be stable and unique. The registry always traverses them in
ascending ID order, so registration order cannot change checkpoint output or
restore order.

Registration validates that `participant_id()` matches
`describe_state().participant_id`. Null, duplicate, or invalid participants are
rejected before they can enter the registry. Registry mutation and coordinator
operations are not thread-safe and must be externally serialized. The caller
must also quiesce training mutations while save or restore runs; the sink
transaction makes publication atomic but cannot freeze model or optimizer
updates by itself.

## Coordinated descriptor

`CheckpointDescriptor` contains the canonically ordered participant manifests
and the weakest `RestoreLevel` among them. It intentionally omits physical blob
lengths and digests. An interop container may embed this descriptor beside its
own version, length, digest, codec, and publication metadata.

`validate_checkpoint_descriptor()` rejects empty, unsorted, duplicated, or
invalid participant sets and rejects an aggregate restore level that is
stronger or weaker than the participant manifests actually permit.

## Save transaction

`CheckpointStateSink` is a transaction boundary implemented by a memory store,
artifact writer, database adapter, or future interop container. A save follows
this sequence:

1. Validate and freeze the complete descriptor.
2. Call `begin()` once.
3. Ask every participant to write through its assigned bounded `StateWriter`.
4. Call `commit()` once after every required field is complete.
5. Call non-throwing `rollback()` after any failure from `begin()` through
   `commit()`.

The sink must copy the descriptor, reject undeclared or duplicate writes, check
required fields, and publish atomically. Filesystem and hash policy remain an
interop responsibility.

## Restore transaction

`CheckpointStateSource` exposes one immutable descriptor and stable readers.
The source verifies physical integrity before exposing bytes. The coordinator
then performs logical preflight before any participant mutation:

- validates the stored descriptor and caller restore policy;
- requires the exact registered participant ID set;
- requires each reader manifest to equal its descriptor entry;
- validates every current participant and the configured rollback budget.

After preflight, the coordinator captures each current participant in bounded
memory. It then restores stored state in canonical order. If any participant
fails, all attempted participants, including the failing participant, are
restored from snapshots in reverse order. A rollback failure becomes an
`InternalError` with the original restore failure nested; callers must then
treat live training state as indeterminate.

## Resource limits

`CheckpointCoordinatorOptions` bounds participant count and the sum of declared
field maxima used for rollback snapshots. The defaults are 1024 participants
and 1 GiB. Projects with larger states must raise the byte limit explicitly;
the coordinator never silently disables rollback.

Participant field limits are security and reliability boundaries rather than
capacity hints. Interop readers must additionally enforce actual artifact
lengths, aggregate archive limits, decompression ratios, and strong digests.

## Current boundary

This API establishes coordination mechanics only. Exact training recovery is
not supported until model, optimizer, scheduler, scaler, RNG, DataPipeline,
metric, callback, and strategy participants have their own tested schemas and
the interop checkpoint container has passed corruption and interruption tests.
The legacy Trainer checkpoint format is not adapted or forwarded by this API.
