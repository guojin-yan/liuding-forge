# SQLite storage commit boundaries

`storage_crash_child.cpp` runs four actual production operations:

- a two-event `SQLiteRunStore::append` batch after previously confirmed metrics;
- `SQLiteRunStore::put_artifact`, including real CAS publication and DB relation;
- `SQLiteRunStore::transition` to the terminal completed status;
- the constructor's actual v1-to-v2 database migration.

Each operation is interrupted twice. `sqlite3_auto_extension` installs SQLite's
official `sqlite3_commit_hook` and `sqlite3_wal_hook` only in this single-threaded
test child. The first fence is inside the real transaction before commit; the
second is after a real WAL commit, before the production operation returns.
Hooks never modify the database or execute substitute writes. They emit one
bounded readiness record, then block with stdin open until the Python supervisor
force-kills the process. The extension is scoped and unregistered on normal
exit. Installing the WAL callback replaces SQLite's automatic checkpoint hook
in this test process; SQLite persistence and FULL synchronous commits remain
real, and the tiny fixture does not require a checkpoint.

For the migration case, setup uses normal public APIs to produce metadata and
CAS, then removes only the empty v2 event tables to create a v1 INPUT fixture,
as the existing migration unit tests do. The killed/retried operation is the
production constructor, which executes its own validation, DDL and COMMIT. The
test does not replace migration with hand-written SQL. Executed-statement trace
proves that the production migration ran and set FULL synchronous before its
commit; the seed database is verified to use WAL.

`test_storage_crash_boundaries.py` verifies eight scenarios with independent
read-only SQLite queries and CAS hashes. A killed uncommitted operation must
preserve the entire previous logical schema and rows. Committed event batches
must be complete; committed terminal status, artifact relationships and migrated
schema must survive. Repeated real retries preserve all logical facts. For an
artifact published before an uncommitted DB relation, production GC removes only
the legitimate orphan and preserves all referenced confirmed bytes; retry then
republishes and attaches it. CAS temp files must already be absent at these
database commit fences. A separate CAS publication test is needed to cover the
rename itself.

Link the test executable to `jyppx::experiment`, `nlohmann_json::nlohmann_json`
and the project's existing SQLite header target (`SQLite::SQLite3`). The
experiment archive already contains the same official SQLite amalgamation;
keep one SQLite implementation and do not substitute a different library.
Apply the normal test instrumentation and LibTorch runtime-copy dependencies.
No GTest main or optional Python package is needed.

```text
python tests/integration/experiment/test_storage_crash_boundaries.py --child <storage-crash-child> --work-dir out/acceptance/storage-crash
```

The driver runs four named tests with before/after subcases, rejects skipped or
missing cases, and retains each scenario's `evidence.json`, executed SQL trace,
database, CAS, and child stderr when `--work-dir` is supplied. `--timeout` bounds
each child operation (default 30 seconds). Build and run it in the next chapter
gate using the existing Windows and Ubuntu incremental build directories.

This is process termination evidence. It makes no OS reboot, power-loss,
network-filesystem or hardware storage-cache guarantee.
