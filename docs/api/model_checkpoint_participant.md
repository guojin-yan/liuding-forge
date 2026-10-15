# Model Checkpoint Participant

Header: `#include <jyppx/nn/model_checkpoint_participant.hpp>`

Link target: `jyppx::nn`

`ModelCheckpointParticipant` exposes the parameters and buffers owned by a
`ModelStateView` as exact checkpoint state. It implements the storage-neutral
foundation participant contract and can be registered directly with the
training checkpoint coordinator.

## State contract

Schema version 1 writes one `topology` field followed by deterministic numeric
tensor fields. The topology records each stable state key, parameter or buffer
role, scalar type, shape, stride, storage alias group, storage offset, and
`requires_grad` flag. A restore requires this complete topology to match the
currently constructed model.

Tensor values use little-endian raw bytes in logical contiguous order. The
format does not use `torch::save`, pickle, executable code, filesystem paths,
or a LibTorch module archive. It currently accepts dense, strided,
non-overlapping tensors with common training scalar types. Sparse, quantized,
meta, overlapping, unsupported-dtype, and stateless modules fail explicitly.

## Restore transaction

Load validates the stored manifest, policy, required fields, topology encoding,
state keys, roles, dtypes, shapes, strides, aliases, offsets, byte counts, and
boolean values before changing the model. All candidate tensors are allocated
on their target devices and all current tensors are cloned before commit.

Commit copies values under a no-grad guard. If a backend copy fails, every
attempted tensor, including the failing tensor, is restored in reverse order.
An incomplete rollback reports `InternalError` with the original failure
nested; the caller must stop training because model state is then
indeterminate.

## Resource limits

`ModelCheckpointParticipantOptions` defines the participant ID and positive
limits for tensor count, state-key length, topology bytes, per-tensor bytes,
and total declared bytes. Limits are enforced during describe, save, metadata
parse, and load allocation. The coordinator has an additional aggregate
rollback budget which may need to be raised for large models.

## Current boundary

The participant preserves tensor values and the model's existing topology. It
does not restore train/eval mode, optimizer state, gradients, hooks, arbitrary
module attributes, device strategy, or invocation schemas. Those states belong
to their respective participants or Trainer control state. Schema migration,
state-key remapping, permissive partial load, interop hashes, compression, and
artifact publication are not supported by this schema version.

Callers must quiesce model mutation during describe, save, and load. The class
retains shared ownership of the model but does not provide internal locking.
