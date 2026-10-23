# Producer process recovery acceptance

`recovery_child.cpp` is a test-only executable. It uses the production
`EventOutbox`, typed training telemetry adapter, `SQLiteRunTelemetrySink`,
`SQLiteRunStore`, checkpoint coordinator, and `FileCheckpointStore`. Its sink
wrapper delegates all acceptance and persistence to the SQLite sink and exposes
the identical sink descriptor. It only captures evidence and provides a
supervisor synchronization fence.

`test_process_recovery.py` launches the child with a Unicode fixture path, waits
for a bounded JSON readiness line, and calls `subprocess.Popen.kill()` while the
child is blocked inside `emit()`. This is `SIGKILL` on POSIX and
`TerminateProcess` on Windows. Stdin stays open; no exception, EOF, destructor, or
normal process shutdown creates the tested crash boundary. A cleanup marker
would expose unexpected stack unwinding.

Both scenarios save a synchronized production checkpoint before delivery:

1. Kill before SQLite acceptance, reopen the store in another process, and
   recover the pending event from the file checkpoint. The first receipt must be
   `accepted`.
2. Kill after a real SQLite FULL synchronous commit but before returning its
   acknowledgment to the outbox. The reopened store must retain exactly one
   event and metric; recovery of the still-pending checkpoint must return
   `duplicate`, preserving all original facts.

Each scenario then replays the original pending checkpoint in another process,
restores a delivered checkpoint to append the next event, and reopens the final
delivered checkpoint without emitting again. Assertions compare canonical bytes
and independently computed SHA-256, full projections including captured clocks
and callback decisions, producer identity and sequence, stable run commit order,
typed metric indices, and fresh process clock IDs for new events. An independent
read-only Python SQLite connection checks canonical BLOBs, stream/cursor state,
batch receipts/items, definitions, counts, integrity, and foreign keys. The
original checkpoint hash must remain unchanged throughout.

Build the executable with C++20 and link `jyppx::liuding::training`, `jyppx::liuding::experiment`,
`jyppx::liuding::interop`, and `nlohmann_json::nlohmann_json` (or use `jyppx::liuding::sdk` plus the
JSON target). Interop must be enabled for `FileCheckpointStore`. Apply the same
instrumentation and LibTorch runtime dependencies as the other integration
executables. Register the stdlib Python driver as a CTest entry using
`--child $<TARGET_FILE:...>`. No test is skipped and no optional Python package is
required.

Example after the chapter build:

```text
python tests/integration/experiment/test_process_recovery.py --child <built-recovery-child> --work-dir out/acceptance/process-recovery
```

`--timeout` bounds each child operation (default 30 seconds). `--work-dir` keeps a
unique directory containing the databases, checkpoints, stderr, and each
scenario's `evidence.json`; omitting it uses an automatically removed temporary
directory. Use the existing incremental build directory for Windows and Ubuntu
chapter validation.

This covers producer/outbox/store recovery across forced process termination.
It does not establish recovery of the entire offline Trainer process, an
interrupted artifact upload/finalize/migration transaction, operating-system
reboot, or power loss. Those require their own acceptance evidence.
