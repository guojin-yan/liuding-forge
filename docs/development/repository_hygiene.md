# Repository hygiene

The source tree contains code and durable project metadata. Runtime data,
logs, build products, checkpoints, TensorBoard events, profiler output and
downloaded datasets must not be written to the repository root.

## Layout

- Public APIs: `include/jyppx/<component>/`
- Implementations: `src/jyppx/<component>/`
- Tests: `tests/<level>/<component>/`
- Build helpers: `cmake/` and `scripts/`
- Generated files: the active build tree only

Do not create top-level numeric or temporary directories. A new top-level
directory requires a documented owner, purpose and retention policy.

## Test data and outputs

Unit tests should generate deterministic data in memory or under the operating
system temporary directory. Small reviewed fixtures belong in `tests/fixtures/`;
reviewed reference outputs belong in `tests/reference_data/`. Downloaded or
large datasets live outside the repository under `JYPPX_DATA_ROOT` and require
a manifest and checksum.

Test output belongs in the build tree's `Testing/` directory or a unique OS
temporary directory. Tests must clean temporary files they own.

## Builds and cleanup

Use `out/build/<preset>/` for new build trees. Keep the active build and, when
needed, one recent rollback build. Delete older trees after preserving a small
result summary. Always resolve and verify an absolute cleanup target before a
recursive delete; never delete through an unresolved variable or wildcard.

## Git gate

Run the repository gate with:

```text
ctest --test-dir <build-dir> -C Release -R RepositoryHygiene --output-on-failure
```

The gate rejects top-level numeric directories, root logs/temp files, tracked
build/runtime paths, and unreviewed test data files larger than 5 MiB. Ignoring
a path in `.gitignore` does not make root pollution acceptable.
