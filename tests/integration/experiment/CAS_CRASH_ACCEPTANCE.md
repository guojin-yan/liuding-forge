# Native CAS publication boundaries

`cas_crash_child.cpp` uses real `FileArtifactStore` upload, hashing, platform
publication, synchronization, verified reads, recovery, duplicate upload and GC.
The target is a deterministic binary object of 2 MiB plus 257 bytes, written in
several chunks. A separate confirmed object predates every interrupted upload.

The private `detail::publish_artifact_file` forwarding definition is a separate
static archive member. This test executable supplies that one forwarding symbol
and delegates to the unchanged production `native_publish_artifact_file`. It
blocks immediately before or after the real native rename, and also checks a
third fence after the complete production `commit()` returns. The production
wrapper has no callback state, environment switch or test branch. Its private
header is not installed and does not alter public product APIs.

The Python supervisor waits for readiness and force-kills the blocked process,
with stdin still open and no stack unwinding. Before publication, only the
complete synchronized temporary upload may exist. After native publication,
the entire new object must exist at its content address and the temporary name
must be gone. The POSIX after-rename fence precedes directory fsync; this is
process termination evidence and does not claim power-loss durability. The
after-commit fence follows all production sync and committed-session updates.

Production recovery must retain the recent actual abandoned upload. The driver
then ages that same crash-created file and verifies stale recovery removes it
without touching confirmed content. The after-rename case supplies the exact
standalone live set to production GC and checks that only the unconfirmed
published orphan is removed. After full commit, the live set includes both
confirmed digests and GC must preserve both. Every scenario retries the real
upload twice, verifies exact binary content through production readers and
independent Python SHA-256, and checks no extra objects or temporary files.

## Build and run

Add `src/jyppx/liuding/experiment/detail/artifact_publish.cpp` and
`src/jyppx/liuding/experiment/detail/native_artifact_publish.cpp` as separate sources of
the **static** experiment target. Keep them in separate archive members: unity builds or whole-archive linking must not merge/force-load the default forwarder. The native implementation retains the original `MoveFileExW` flags on Windows and `std::filesystem::rename` on POSIX. Original directory sync, error mapping and session-commit ordering remain in `artifact_store.cpp`.

Link this helper to `jyppx::liuding::experiment` and `nlohmann_json::nlohmann_json`, with
`${PROJECT_SOURCE_DIR}/src` as a private include directory for the internal
header. Apply normal test instrumentation and LibTorch runtime dependencies.
The test requires static archive linkage; configure an explicit failure for a
shared experiment target rather than claiming this seam works there. Every
crash mode also explicitly requires its private seam/native-call evidence, so
missing interposition cannot silently pass.

```text
python tests/integration/experiment/test_cas_crash_boundaries.py --child <cas-crash-child> --work-dir out/acceptance/cas-crash
```

Three tests are mandatory. `--timeout` bounds every child operation (default
30 seconds). Supplying `--work-dir` retains unique scenario directories with
CAS files, readiness evidence, independent hashes and child stderr. Build and
run as part of the next Windows and Ubuntu chapter gate in the existing
incremental build directories.

These are local process-crash checks. OS reboot, storage-device power loss,
network filesystems and concurrent unsynchronized standalone CAS ownership are
outside this evidence.
