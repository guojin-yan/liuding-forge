# Build metadata and incremental compilation

The installed `jyppx/foundation/build_config.hpp` contains the semantic version
and enabled component flags. These values describe the public header contract.
Git commits, working-tree state, and dependency fingerprints do not change this
header. `configure_file` retains its timestamp when these public values are
unchanged, preserving incremental compilation across source commits.

Build provenance lives in the private generated header
`generated-private/jyppx/foundation/build_metadata.hpp`. CMake gives only
`src/jyppx/foundation/version.cpp` access to that include directory; the header is
not installed or exported to consumers. It records the source commit, compiler,
operating system and architecture, LibTorch version and ABI/build variant,
accelerator build, and dependency-manifest SHA-256 digest. The build configuration
continues to come from the private `JYPPX_BUILD_TYPE` compile definition so
multi-configuration generators report the configuration actually compiled.

Read these values through `jyppx::build_info()`,
`jyppx::current_build_fingerprint()`, or `jyppx::build_fingerprint_json()` from
`jyppx/foundation/version.hpp`. Provenance macros are implementation details.
The fingerprint APIs retain their existing fields and values, including runtime
accelerator availability.

Provenance remains a snapshot taken at CMake configure time. Reconfigure the
existing build directory after changing the source commit, working-tree state,
or dependency manifest, then build incrementally. Git status still includes
untracked files; local changes or an unsuccessful status query append `+dirty`
to the commit identity. An unavailable commit is recorded as `unknown`, and a
missing dependency manifest as `unlocked`. No commit or dirty state is frozen or
discarded to avoid recompilation.

When only provenance changes, the generated dependency change recompiles
`version.cpp`, updates the foundation library, and relinks dependent binaries.
Ordinary source/header edits and public version or component changes still
rebuild their normal dependents. Reuse one build directory per configured
platform/toolchain to keep compiler dependency information and object files.
