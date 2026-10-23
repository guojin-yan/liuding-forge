# Read-only API migration scan

`liuding-forge api scan` reports known retired public includes, namespaces, and the old
CMake package name in an application's source tree:

```text
liuding-forge api scan --root application --format json liuding-forge api scan --root application --max-files 2000 --max-bytes 8388608 --max-findings 512 --format json
```

The scan reads files only. It proposes review directions and never rewrites
source, executes CMake, or builds the application. Use the current API's
documentation to review constructor signatures, optimizer/checkpoint state,
worker ownership, and callback transactions before making a replacement.

The scanner is explicitly lexical. Comments, strings, disabled preprocessor
branches, and larger identifiers containing a retired needle can match.
Generated spellings, macros, aliases, or unlisted historical symbols can be
missed. An empty result establishes only that the configured rules found no
matching text; it does not prove AST completeness, behavioral equivalence, or
ABI compatibility. Consumers must recompile against the supported v1 headers
and targets.

## Results and review

The common `--format json|text` output contains `scanner: bounded-lexical-v1`,
`source_untouched`, scanned file/byte counts, a limitation statement, and
`findings`. Every finding contains:

| Field | Meaning |
|---|---|
| `path` | UTF-8 path relative to the scan root. |
| `line` | One-based line containing the match. |
| `feature_id` | Owning migration feature for review. |
| `retired_surface` | Fixed rule needle that matched. |
| `suggestion` | Fixed explanation of the owning replacement or retired capability. |
| `automatic_change` | Always false. |

Files are visited in sorted path order, then fixed rule order and text order.
Repeated matches for the same needle on the same line produce one finding.
Findings do not contain source snippets or configuration values.

The 28 rules cover the old `jyppx_liuding_training_framework` package, version/utilities,
data loader and queue, loss/init/quantization headers, distributed namespace
and headers, optimizer headers/base/schedulers, and old training callback
headers. Suggestions identify the owning public contract. Reserved
quantization/distributed surfaces are reported as removed capabilities rather
than working replacements. The rule table is maintained in
`tools/cli/api_migration.cpp`; it is intentionally a bounded diagnostic list,
not an exhaustive symbol database.

For example, an old `jyppx/optim/sgd.hpp` include points to the owning
`jyppx/optim/sgd_adapter.hpp` contract and requires explicit optimizer-state
review. The old general `jyppx::liuding::utils` namespace requires selecting the actual
foundation/data/training owner rather than a blanket namespace substitution.

## Scope and limits

Recognized files are `CMakeLists.txt`, `.cmake`, `.c`, `.cc`, `.cpp`, `.cxx`,
`.h`, `.hpp`, `.hxx`, `.ipp`, and `.inl`; matching is case-sensitive.
Directories named `.git`, `.hg`, `.svn`, `out`, `build`, `third_party`,
`node_modules`, or `.venv` are excluded from recursion. Other build directory
names must be excluded by choosing a suitable source root.

| Option/resource | Default and maximum |
|---|---|
| `--max-files` | 10,000 recognized source files |
| `--max-bytes` | 67,108,864 aggregate source bytes |
| `--max-findings` | 4,096 findings |
| Individual source file | 2 MiB |
| Traversal | 50,000 visited directory entries; iterator depth 32 |
| Relative path | 1,024 UTF-8 bytes |

Options can lower their limits and must be positive. Exceeding a limit rejects
the scan rather than returning a silently truncated result. The source tree
must remain quiescent and exclusively controlled for the scan. The shared
[migration filesystem rules](config_migration.md#filesystem-and-resource-bounds)
reject traversal, inspected links/reparse points, special files, and
hard-linked source inputs. No source files are written.

Exit 0 means no rule findings, exit 1 means findings are available for review,
exit 2 means invalid input, and exit 3 covers resource, filesystem, or other
runtime failures. Tests in `tests/cli/test_migration.py` check read-only behavior,
comment matches, directory exclusions, and finding limits at the chapter's
unified validation boundary.
