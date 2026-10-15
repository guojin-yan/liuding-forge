# Checkpoint Participant Contract

Header: `jyppx/foundation/checkpoint.hpp`  
Target: `jyppx::foundation`

The checkpoint participant API is the storage-neutral boundary shared by model,
data, optimizer, scheduler, scaler, RNG, callback, metric, and strategy state.
It deliberately does not define a checkpoint file, directory layout, hash
algorithm, compression method, or publication mechanism.

## Ownership

- `jyppx::foundation` owns restore levels and policies, participant state
  descriptors, bounded blob reader/writer interfaces, and
  `CheckpointParticipant`.
- `jyppx::training` owns participant registration, consistency boundaries,
  snapshot ordering, and cross-participant rollback.
- `jyppx::interop` owns checkpoint container formats, actual blob lengths and
  digests, codecs, and safe artifact loading.

This split lets lower-level components implement one protocol without depending
on the training engine or a concrete storage backend.

## Restore Guarantees

`RestoreLevel` describes what a participant can promise:

- `exact`: equivalent continuation under the declared compatibility matrix.
- `bounded`: equivalence within documented numeric or data-order bounds.
- `best_effort`: only the explicitly listed state is restored.

`RestorePolicy` forms a matching acceptance lattice. A caller can require exact
state, allow bounded state, or explicitly opt into best-effort restoration.
Every non-exact manifest must include at least one degradation reason.

## State Descriptors

Each participant publishes a stable ID, a positive schema version, a restore
level, and a complete list of logical state fields. Every field has a media
type, a positive maximum byte count, and required/optional presence. Logical
field names are not filesystem paths.

The participant manifest is a schema descriptor. Actual byte counts, hashes,
producer identity, and artifact lineage belong to the concrete interop
manifest created by a checkpoint coordinator.

## Atomicity And Limits

`StateReader` exposes the validated manifest stored with the artifact and must
enforce the supplied maximum before returning bytes. Participants use the
stored manifest, rather than their current capabilities, when checking schema
compatibility and restore policy. `StateWriter` must reject undeclared,
duplicate, and oversized fields. A participant must parse and validate all of
its state before mutation so a failed `load_state()` leaves that participant
unchanged. The training coordinator remains responsible for rollback across
multiple participants.

This first slice does not change the legacy Trainer checkpoint container. It
establishes the common protocol that later model, data, optimizer, callback,
metric, scaler, and RNG adapters will implement before the coordinator is
switched over.
