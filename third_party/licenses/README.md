# Audited license evidence

These files are exact license or provenance documents extracted from the
verified source archives, or downloaded from the upstream revision used by a
binary dependency. Their exact byte sizes, SHA-256 values, source artifact hashes, URLs and archive
entries are recorded in the checked-in `manifest.json`. Additional raw audit
responses and preparation evidence are retained in
`out/deps/dependency-audit-20261021/audit-index.json`. License originals use
`-text` Git attributes to preserve their upstream bytes across platforms.

| File | Evidence |
|---|---|
| `googletest-1.14.0-LICENSE` | `googletest-1.14.0.zip`, entry `googletest-1.14.0/LICENSE`; BSD 3-Clause; SHA-256 `9702de7e4117a8e2b20dafab11ffda58c198aede066406496bef670d40a22138`. |
| `nlohmann-json-3.12.0-LICENSE.MIT` | `nlohmann-json-3.12.0.tar.xz`, entry `json/LICENSE.MIT`; MIT; SHA-256 `46a65cffd1ea955132d95a8dd921640714a8d6b537d2e4e482d31145ae95b603`. |
| `sqlite-3.54.1-copyright-blessing.txt` | Exact first 294 bytes of `sqlite-amalgamation-3540100/sqlite3.h` from the pinned archive; SHA-256 `d18a5a730e4776a0487cdec51ab38139573711c6387d96745898be4c779bb16e`. |
| `sqlite-copyright-page.txt` | Exact upstream `https://sqlite.org/copyright.html` response captured during the 2026-10-21 audit; SHA-256 `44ca9f793055c8e32fc65f65a4f5bcf813a33f5bdaaa084067dd617a4ed3cc70`. |
| `pytorch-2.14.1-build-5c488690-LICENSE.txt` | PyTorch `LICENSE` at build commit `5c4886908584029761b579af026dcfb627c84070`; SHA-256 `bd018feef8825e88181c84eb7e3aa4eafb8f08a20d9fd6ef948569610c4a3e43`. |
| `pytorch-2.5.1-LICENSE.txt` | PyTorch `LICENSE` at tag `v2.5.1`; SHA-256 `47a26beb94e3f6b333a3677fc85d546f1fdfd2f0b3686c26d2fb5b10e0134165`. |

The LibTorch Windows archives used by this repository were inspected as ZIP
archives. They contain no `LICENSE`, `LICENCE`, `NOTICE`, or `COPYRIGHT` file;
the package root license cannot be treated as the complete notice set for the
bundled binary components. The primary `2.14.1+cu126` and candidate `2.5.1+cu124` entries therefore
retain a pending notice audit in the dependency manifest. An additional
`2.14.1+cu132` archive was inspected with the same absence; it is audit
evidence only and is not an entry in the main dependency manifest. The exact source license
files above are provenance evidence only and do not close that blocker.

These evidence files do not change the distribution obligations of a future
release. If LibTorch is redistributed, the release must carry the exact
notices supplied for that build, or the release must clearly require a
separately obtained runtime and omit those binaries from the artifact.

The `optional-sidecar/` tree contains 24 exact license/notice entries extracted
from the seven verified remediation wheels for Werkzeug 3.1.9, protobuf 6.33.5,
Pillow 13.0.0 and setuptools 83.0.0. The setuptools entries include its bundled
vendor licenses and notices. Pillow Windows and Linux wheel notices differ,
so both originals are retained. These entries establish source evidence for
these exact wheels; they do not declare the complete optional Python runtime
or its transitive licenses audited. The index currently records 30 files
including the six C++ dependency provenance documents.

Run `python scripts/validation/dependency_manifest.py` from the repository
root to validate the dependency lock and this index together. The validator
checks exact bytes, declared sizes, portable paths and recorded-notice
coverage; it rejects duplicate JSON keys, aliases and links. It does not
authenticate an upstream publisher or grant redistribution rights.
