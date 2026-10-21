# Verified model weight import

Header: `jyppx/interop/model_weight_import.hpp`; target: `jyppx::interop` with
the optional `nn` adapter enabled. The header is not installed when `nn` is OFF.
The storage-neutral safetensors codec and weight mapping planner remain usable
without nn.

`import_model_weights` checks a trusted whole-artifact SHA-256, parses the
complete bounded safetensors file, inventories named parameters and buffers,
and creates a complete `WeightImportPlan`. A rejected mapping returns its full
diagnostics with `applied=false`; no tensor changes. Exact keys, shapes and
dtypes are required by default. Explicit partial mode reports every missing,
unexpected, incompatible and retained entry, with the requested count coverage.

Accepted mappings first stage all decoded tensors and device transfers, check
the entire destination topology again, then replace the assigned data handles
under `NoGradGuard`. Original data handles are retained until all assignments
complete. If committing throws, restoration attempts cover every attempted
assignment in reverse order, including the assignment that threw after a possible
partial mutation. A failed restoration does not prevent the remaining attempts.
When all restorations succeed, the original failure is rethrown through the
framework error boundary. If any restoration fails, `InternalError` reports the
failed restoration count and retains the original commit exception as its nested
cause. The model is then unusable: discard it and reconstruct a fresh instance.
Successful imports preserve parameter TensorImpl identity, dtype, device, mode
and requires-grad.

The caller must provide exclusive, quiescent access for the whole call: finish
prior CPU/CUDA work, exclude state readers and writers, forward/backward
execution and optimizer steps, and release live autograd graphs involving the
state. The final topology check detects some accidental state changes; it is
not a lock and cannot make concurrent tensor access safe. Import must run
outside inference mode. Existing gradients, non-leaf tensors, autograd views,
inference tensors and lazy negative/conjugate state are rejected. Alias checks
cover repeated TensorImpl handles, shared nonempty storage and overlapping
actual byte ranges on the same device, including separate `from_blob` storage
wrappers. External storage views cannot be inventoried; discard them and rebuild
optimizer state before subsequent training. This operation imports model
weights, not a resumable training checkpoint.

Supported destination types are BOOL, U8, I8, I16, I32, I64, F16, BF16, F32 and
F64 on dense contiguous CPU/CUDA tensors. The format codec also supports opaque
U16/U32/U64 bytes; this nn adapter does not materialize those types. BOOL bytes
must be 0 or 1 before materialization. Unsupported sparse, quantized, complex,
aliased, special autograd or noncontiguous state fails before mutation.

Partial dtype conversion requires an exact per-key `jyppx.float-cast.v1` rule
between F16/BF16/F32/F64 types. It uses LibTorch's cast semantics including
rounding and overflow; no implicit precision or finite-value guarantee is
invented. Unknown conversion rules are rejected before staging. Strict mode
never casts. The result retains the versioned mapping report and separately
states whether tensors were actually applied.

Cancellation and deadline checks run during chunked hashing and between staged
entries, and immediately before committing. The commit is not interrupted
midway, and rollback runs to completion regardless of cancellation/deadline.
Device staging synchronizes the affected CUDA devices before mutation; this
does not replace the caller's obligation to finish prior work before entry.
The staging budget conservatively charges source decode plus two destination
buffers per assigned entry; it is not a process RSS cap. Producer authentication
requires a separate trusted signature; SHA-256 alone establishes byte identity.

LibTorch failures are translated to framework errors with the original exception
retained as a nested cause. Typed backend out-of-memory and host `bad_alloc`
failures become `ResourceExhausted`; CUDA synchronization failures become
`DeviceError`; other LibTorch failures become `BackendError`. Classification
never depends on backend error text. Top-level messages and details do not copy
raw backend diagnostics; inspect the nested cause only in an appropriate
diagnostic context.

This chapter is awaiting consolidated Windows and Ubuntu/CUDA qualification.
