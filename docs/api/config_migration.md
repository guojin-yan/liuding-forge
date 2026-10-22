# Explicit configuration migration

`liuding-forge config migrate` migrates a user-owned, versioned configuration document
through an explicit plan. It uses the same typed configuration schema and
secret-reference validation as `config lint`. It does not infer historical
formats or change an application's configuration in place.

```text
liuding-forge config migrate --input config.json --plan migration.json --to-version 2 --format json
liuding-forge config migrate --mode apply --input config.json --plan migration.json --to-version 2 --output migrated-config --format json
liuding-forge config migrate --mode rollback --receipt migrated-config --plan migration.json --output restored-config --format json
```

The default mode is `preview`, which accepts no `--output`. `apply` and
`rollback` require a new output directory whose parent already exists. All
commands accept the common `--format json|text` option. The installed C++ CLI
implements these commands; Python is only used by the repository's tests.

## Document and plan contracts

The input envelope has exactly these fields:

```json
{
  "schema_version": 1,
  "schema_id": "example.training",
  "version": 1,
  "values": {"timeout_ms": 5000, "retries": 3}
}
```

`schema_version` identifies this envelope format. `schema_id` and `version`
identify the application's value schema. This envelope is an explicit tool
contract, not a claim that old JYPPX configurations used a global v1 format.

The following complete plan changes the timeout unit and supplies a new value
only when it is absent. The reverse step is independently declared:

```json
{
  "schema_version": 1,
  "plan_id": "example-timeout-v1-v2",
  "feature_id": "CONFIG-MIGRATION",
  "schema_id": "example.training",
  "schemas": [
    {
      "schema_version": 1,
      "schema_id": "example.training",
      "version": 1,
      "root": {
        "kind": "dictionary",
        "fields": {
          "timeout_ms": {"kind": "integer"},
          "retries": {"kind": "integer"}
        }
      }
    },
    {
      "schema_version": 1,
      "schema_id": "example.training",
      "version": 2,
      "root": {
        "kind": "dictionary",
        "fields": {
          "timeout_seconds": {"kind": "integer"},
          "retries": {"kind": "integer"},
          "enabled": {"kind": "boolean"}
        }
      }
    }
  ],
  "steps": [
    {
      "from_version": 1,
      "to_version": 2,
      "breaking_change": "timeout-unit-and-enable-default",
      "operations": [
        {"kind": "rename", "from": "/timeout_ms", "path": "/timeout_seconds"},
        {"kind": "scale-int", "path": "/timeout_seconds", "numerator": 1, "denominator": 1000},
        {"kind": "set-default", "path": "/enabled", "value": true}
      ]
    },
    {
      "from_version": 2,
      "to_version": 1,
      "breaking_change": "restore-timeout-unit-and-remove-default",
      "operations": [
        {"kind": "scale-int", "path": "/timeout_seconds", "numerator": 1000, "denominator": 1},
        {"kind": "rename", "from": "/timeout_seconds", "path": "/timeout_ms"},
        {"kind": "remove-checked", "path": "/enabled", "expected": true}
      ]
    }
  ]
}
```

All plan fields shown are required, and unknown fields are rejected. Identities
and `breaking_change` use 1–128 ASCII letters, digits, dots, underscores, or
hyphens. Versions are positive uint32 values. Every schema has the same ID and
a distinct version; directed steps connect adjacent declared versions only.
Duplicate edges and implicit reverse conversions are rejected. An input or
target version with no declared schema is rejected, including a future version.

Value schemas use `dictionary`, `list`, `tuple`, `integer`, `floating`,
`boolean`, and `string` nodes. A dictionary requires `fields`, a list requires
`element`, and a tuple requires ordered `elements`. Optional node metadata is
`name`, `semantic_tag`, `required` (default true), and `nullable` (default
false). Scalar nodes cannot declare children. Migration schemas must have a
dictionary root, no configured defaults, and no environment bindings; any
default change must appear as an operation. Optional `secret_paths` retain the
SDK's JSON Pointer and secret-reference rules.

The complete starting value and the complete value after every step are
validated against their declared schema, including required fields, unknown
fields, types, and secret references. Defaults and process environment values
are never injected into a migration. An application may use a sequence of
operations to reshape a value before the next complete schema check.

## Operations

Paths are non-root RFC 6901 JSON Pointers, with `~0` and `~1` escapes. Empty
tokens, controls, missing dictionary parents, and array-element editing are
rejected. Create a required dictionary explicitly with `set-default` before
moving fields into it.

| `kind` | Exact fields in addition to `kind` | Behavior |
|---|---|---|
| `rename` | `from`, `path` | Move an existing value to an absent destination. |
| `copy` | `from`, `path` | Copy an existing value to an absent destination. |
| `set-default` | `path`, `value` | Insert only when the destination is absent; preserve an existing value. |
| `scale-int` | `path`, `numerator`, `denominator` | Multiply an existing int64 by an int64 ratio; the denominator must be positive. |
| `replace` | `path`, `expected`, `value` | Replace only when the existing JSON value equals `expected`. |
| `remove-checked` | `path`, `expected` | Remove only when the existing JSON value equals `expected`. |

`rename` and `copy` reject equal paths and ancestor/descendant overlaps. Unit
conversion reduces the ratio exactly, rejects a fractional result, and checks
signed int64 overflow, including the `INT64_MIN` boundary. It never rounds.
Checked removal can remove an empty dictionary after its fields have been
moved during a reverse migration. A reverse operation may intentionally reject
values that cannot be represented in the older schema.

`tests/cli/test_migration.py` supplies a user-owned v1 → v2 → v3 fixture, explicit
v3 → v2 → v1 steps, integer boundary cases, and secret-path relocation. These
are application fixtures rather than historical framework format claims.

## Apply receipts and rollback

`preview` runs the same operations and validation as `apply` and reports plan
identity, source/output SHA-256, versions, and ordered path-only changes. It
does not print configuration values. Migrating to the current version validates
the document and reports `already_at_target: true` with no changes.

Successful `apply` writes exactly:

| File | Content |
|---|---|
| `original.json` | Exact input bytes, including formatting. |
| `migrated.json` | Migrated document serialized deterministically, followed by LF. |
| `receipt.json` | Plan/feature/schema identity, exact plan-byte hash, input/output hashes, versions, and operation paths. |

Secret references are preserved in both configuration files. Receipts,
terminal output, and migration error messages omit values and references.
Use an appropriately protected parent directory for the configuration files;
the command inherits filesystem permissions and does not create an ACL or
resolve secret material.

`rollback` requires the original plan's exact bytes and a receipt directory
containing only those three regular files. It verifies their hashes and
identities, replays the recorded migration, and compares the resulting bytes
and change list. It then writes the original bytes as `restored.json`, plus a
`rollback.json` result, into a new directory outside the receipt directory.
Rollback preserves the original source and receipt. This receipt is a local
integrity record, not a signature or an authentication boundary.

## Filesystem and resource bounds

Inputs, source trees, and output parents must remain exclusively owned and
quiescent for the command. UTF-8 paths are supported. Dot traversal, device or
stream aliases, links/junctions/reparse points in inspected ancestors, and
hard-linked input files are rejected. Windows inputs are opened without write
or delete sharing; POSIX readers check leaf identity and modification metadata
before accepting bytes. Ancestor checks do not replace exclusive directory
ownership.

Output directories are reserved without replacement. Files are staged in the
same parent and published without replacing existing files. A filesystem
failure may leave an incomplete output directory; only successful completion
establishes a complete receipt. The command does not erase that output or any
existing directory. Choose a new destination for a retry. Publication is not a
crash-durability/fsync guarantee.

| Resource | Limit |
|---|---|
| Each input, plan, receipt, or output configuration file | 1 MiB |
| Declared schemas / directed steps / total operations | 16 / 64 / 256 |
| Top-level schema fields / JSON Pointer tokens / pointer bytes | 256 / 32 / 1,024 |
| Migrating value tree, checked initially and after each operation | 4,096 nodes including keys; depth 32; 64 KiB per string/key; 1 MiB aggregate strings/keys |
| CLI schema and typed-value conversion depth | 24 |

The shared strict JSON parser also rejects duplicates, invalid UTF-8, and
excessive node, depth, or string budgets. The SDK's schema/configuration limits
continue to apply. Exit 0 means successful preview/apply/rollback, exit 2 means
invalid or incompatible input, and exit 3 covers resource, filesystem, or other
runtime failures.
