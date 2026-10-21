#!/usr/bin/env python3
"""Validate JYPPX component ownership, includes, and direct CMake edges."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import re
import sys
from collections.abc import Iterable
from typing import Any

MAX_DOCUMENT_BYTES = 1024 * 1024
MAX_SOURCE_BYTES = 4 * 1024 * 1024
COMPONENT_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
INCLUDE_PATTERN = re.compile(
    r'^\s*#\s*include\s*[<"]jyppx/([^>"]+)[>"]', re.MULTILINE
)
CODE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inc"}


class ComponentGraphError(ValueError):
    """Raised when the component dependency contract is violated."""


@dataclasses.dataclass(frozen=True, order=True)
class Edge:
    """One consumer-to-provider component edge."""

    consumer: str
    provider: str


@dataclasses.dataclass(frozen=True)
class ComponentPolicy:
    """Ownership and dependency policy for one entity component."""

    header_roots: tuple[str, ...]
    source_roots: tuple[str, ...]
    required_dependencies: tuple[str, ...]
    required_public_dependencies: tuple[str, ...]
    optional_dependencies: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Manifest:
    """Validated machine-readable component contract."""

    components: dict[str, ComponentPolicy]
    aggregate_headers: tuple[str, ...]
    ownership_assertions: dict[str, str]
    digest: str


@dataclasses.dataclass(frozen=True)
class IncludeGraphs:
    """Public, production-source, and aggregate include edges with evidence."""

    public: dict[Edge, set[str]]
    source: dict[Edge, set[str]]
    aggregate: dict[Edge, set[str]]


@dataclasses.dataclass(frozen=True)
class TargetGraphs:
    """Enabled components and CMake direct target/link-interface edges."""

    enabled_components: frozenset[str]
    target: frozenset[Edge]
    interface: frozenset[Edge]
    sources: frozenset[str] | None = None
    headers: frozenset[str] | None = None


def _read_bounded(path: pathlib.Path, limit: int) -> str:
    """Read one UTF-8 file after enforcing a byte limit."""
    try:
        size = path.stat().st_size
        if size > limit:
            raise ComponentGraphError(f"file exceeds {limit} bytes: {path}")
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ComponentGraphError(f"cannot read UTF-8 file {path}: {error}") from error


def _require_string_list(value: Any, location: str) -> tuple[str, ...]:
    """Return a sorted, duplicate-free list of non-empty strings."""
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ComponentGraphError(f"{location} must be an array of non-empty strings")
    if len(value) != len(set(value)):
        raise ComponentGraphError(f"{location} contains duplicate values")
    if value != sorted(value):
        raise ComponentGraphError(f"{location} must be sorted")
    return tuple(value)


def _find_cycle(nodes: Iterable[str], edges: Iterable[Edge]) -> list[str] | None:
    """Return one deterministic dependency cycle, if present."""
    adjacency = {node: [] for node in sorted(nodes)}
    for edge in sorted(edges):
        adjacency.setdefault(edge.consumer, []).append(edge.provider)
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        state[node] = 1
        stack.append(node)
        for provider in adjacency.get(node, []):
            if state.get(provider, 0) == 0:
                cycle = visit(provider)
                if cycle is not None:
                    return cycle
            elif state.get(provider) == 1:
                start = stack.index(provider)
                return stack[start:] + [provider]
        stack.pop()
        state[node] = 2
        return None

    for node in sorted(nodes):
        if state.get(node, 0) == 0:
            cycle = visit(node)
            if cycle is not None:
                return cycle
    return None


def load_manifest(path: pathlib.Path) -> Manifest:
    """Load and validate the canonical component dependency manifest."""
    text = _read_bounded(path, MAX_DOCUMENT_BYTES)
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise ComponentGraphError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(document, dict):
        raise ComponentGraphError("component dependency manifest must be an object")
    expected_keys = {
        "aggregate_headers",
        "components",
        "ownership_assertions",
        "schema_version",
    }
    if set(document) != expected_keys:
        raise ComponentGraphError(
            "manifest fields must be exactly: " + ", ".join(sorted(expected_keys))
        )
    if document["schema_version"] != 1:
        raise ComponentGraphError("schema_version must be 1")

    raw_components = document["components"]
    if not isinstance(raw_components, dict) or not raw_components:
        raise ComponentGraphError("components must be a non-empty object")
    if list(raw_components) != sorted(raw_components):
        raise ComponentGraphError("component keys must be sorted")

    components: dict[str, ComponentPolicy] = {}
    expected_component_fields = {
        "header_roots",
        "optional_dependencies",
        "required_dependencies",
        "required_public_dependencies",
        "source_roots",
    }
    for name, raw_policy in raw_components.items():
        if not COMPONENT_NAME.fullmatch(name):
            raise ComponentGraphError(f"invalid component name: {name!r}")
        if not isinstance(raw_policy, dict) or set(raw_policy) != expected_component_fields:
            raise ComponentGraphError(
                f"components.{name} fields must be exactly: "
                + ", ".join(sorted(expected_component_fields))
            )
        components[name] = ComponentPolicy(
            header_roots=_require_string_list(
                raw_policy["header_roots"], f"components.{name}.header_roots"
            ),
            source_roots=_require_string_list(
                raw_policy["source_roots"], f"components.{name}.source_roots"
            ),
            required_dependencies=_require_string_list(
                raw_policy["required_dependencies"],
                f"components.{name}.required_dependencies",
            ),
            required_public_dependencies=_require_string_list(
                raw_policy["required_public_dependencies"],
                f"components.{name}.required_public_dependencies",
            ),
            optional_dependencies=_require_string_list(
                raw_policy["optional_dependencies"],
                f"components.{name}.optional_dependencies",
            ),
        )

    owners: dict[str, str] = {}
    required_edges: set[Edge] = set()
    allowed_edges: set[Edge] = set()
    for name, policy in components.items():
        for root in policy.header_roots + policy.source_roots:
            if not COMPONENT_NAME.fullmatch(root):
                raise ComponentGraphError(f"invalid ownership root: {root!r}")
        for root in policy.header_roots:
            previous = owners.setdefault(f"header:{root}", name)
            if previous != name:
                raise ComponentGraphError(
                    f"header root {root!r} is owned by both {previous} and {name}"
                )
        for root in policy.source_roots:
            previous = owners.setdefault(f"source:{root}", name)
            if previous != name:
                raise ComponentGraphError(
                    f"source root {root!r} is owned by both {previous} and {name}"
                )
        required = set(policy.required_dependencies)
        public = set(policy.required_public_dependencies)
        optional = set(policy.optional_dependencies)
        if required & optional:
            raise ComponentGraphError(
                f"components.{name} repeats dependencies as required and optional"
            )
        if not public <= required | optional:
            raise ComponentGraphError(
                f"components.{name}.required_public_dependencies must be allowed"
            )
        for provider in required | optional:
            if provider not in components:
                raise ComponentGraphError(
                    f"components.{name} references unknown provider {provider!r}"
                )
            if provider == name:
                raise ComponentGraphError(f"component {name} cannot depend on itself")
            allowed_edges.add(Edge(name, provider))
        for provider in required:
            required_edges.add(Edge(name, provider))

    cycle = _find_cycle(components, allowed_edges)
    if cycle is not None:
        raise ComponentGraphError(
            "allowed dependency graph contains a cycle: " + " -> ".join(cycle)
        )

    aggregate_headers = _require_string_list(
        document["aggregate_headers"], "aggregate_headers"
    )
    for header in aggregate_headers:
        if pathlib.PurePosixPath(header).name != header or header in {".", ".."}:
            raise ComponentGraphError(f"invalid aggregate header path: {header!r}")

    raw_assertions = document["ownership_assertions"]
    if not isinstance(raw_assertions, dict) or not raw_assertions:
        raise ComponentGraphError("ownership_assertions must be a non-empty object")
    if list(raw_assertions) != sorted(raw_assertions):
        raise ComponentGraphError("ownership_assertion keys must be sorted")
    ownership_assertions: dict[str, str] = {}
    for asserted_path, owner in raw_assertions.items():
        if (
            not isinstance(asserted_path, str)
            or not asserted_path.startswith(("include/jyppx/", "src/jyppx/"))
            or ".." in pathlib.PurePosixPath(asserted_path).parts
        ):
            raise ComponentGraphError(
                f"invalid ownership assertion path: {asserted_path!r}"
            )
        if owner not in components:
            raise ComponentGraphError(
                f"ownership assertion has unknown owner {owner!r}: {asserted_path}"
            )
        ownership_assertions[asserted_path] = owner

    canonical_document = json.dumps(
        document, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return Manifest(
        components=components,
        aggregate_headers=aggregate_headers,
        ownership_assertions=ownership_assertions,
        digest=hashlib.sha256(canonical_document).hexdigest(),
    )


def _ownership_maps(manifest: Manifest) -> tuple[dict[str, str], dict[str, str]]:
    """Build unique public-header and source-root ownership maps."""
    header_owners: dict[str, str] = {}
    source_owners: dict[str, str] = {}
    for component, policy in manifest.components.items():
        for root in policy.header_roots:
            header_owners[root] = component
        for root in policy.source_roots:
            source_owners[root] = component
    return header_owners, source_owners


def _record_includes(
    path: pathlib.Path,
    repository_root: pathlib.Path,
    consumer: str,
    header_owners: dict[str, str],
    edges: dict[Edge, set[str]],
) -> None:
    """Record cross-component JYPPX includes from one source file."""
    text = _read_bounded(path, MAX_SOURCE_BYTES)
    relative = path.relative_to(repository_root).as_posix()
    for match in INCLUDE_PATTERN.finditer(text):
        include_path = match.group(1)
        provider_root = include_path.split("/", 1)[0]
        provider = header_owners.get(provider_root)
        if provider is None:
            line = text.count("\n", 0, match.start()) + 1
            raise ComponentGraphError(
                f"{relative}:{line} includes unknown JYPPX header root "
                f"{provider_root!r}"
            )
        if provider != consumer:
            line = text.count("\n", 0, match.start()) + 1
            edges.setdefault(Edge(consumer, provider), set()).add(
                f"{relative}:{line}"
            )


def _owned_code_files(root: pathlib.Path) -> list[pathlib.Path]:
    """Return deterministic code files below one directory."""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in CODE_SUFFIXES
    )


def scan_repository(manifest: Manifest, repository_root: pathlib.Path) -> IncludeGraphs:
    """Scan public headers and production sources into include graphs."""
    repository_root = repository_root.resolve()
    public_root = repository_root / "include" / "jyppx"
    source_root = repository_root / "src" / "jyppx"
    if not public_root.is_dir() or not source_root.is_dir():
        raise ComponentGraphError(
            f"source root must contain include/jyppx and src/jyppx: {repository_root}"
        )
    header_owners, source_owners = _ownership_maps(manifest)

    actual_header_roots = {
        path.name
        for path in public_root.iterdir()
        if path.is_dir() and _owned_code_files(path)
    }
    if actual_header_roots != set(header_owners):
        missing = sorted(set(header_owners) - actual_header_roots)
        unknown = sorted(actual_header_roots - set(header_owners))
        raise ComponentGraphError(
            f"public header ownership mismatch; missing={missing}, unknown={unknown}"
        )
    actual_source_roots = {
        path.name
        for path in source_root.iterdir()
        if path.is_dir() and _owned_code_files(path)
    }
    if actual_source_roots != set(source_owners):
        missing = sorted(set(source_owners) - actual_source_roots)
        unknown = sorted(actual_source_roots - set(source_owners))
        raise ComponentGraphError(
            f"production source ownership mismatch; missing={missing}, unknown={unknown}"
        )

    actual_aggregate_headers = sorted(
        path.name
        for path in public_root.iterdir()
        if path.is_file() and path.suffix.lower() in CODE_SUFFIXES
    )
    if actual_aggregate_headers != list(manifest.aggregate_headers):
        raise ComponentGraphError(
            "aggregate header mismatch; expected="
            f"{list(manifest.aggregate_headers)}, actual={actual_aggregate_headers}"
        )

    for asserted_path, expected_owner in manifest.ownership_assertions.items():
        full_path = repository_root / pathlib.PurePosixPath(asserted_path)
        if not full_path.exists():
            raise ComponentGraphError(f"asserted owned path does not exist: {asserted_path}")
        parts = pathlib.PurePosixPath(asserted_path).parts
        if parts[:2] == ("include", "jyppx") and len(parts) >= 3:
            actual_owner = header_owners.get(parts[2])
        elif parts[:2] == ("src", "jyppx") and len(parts) >= 3:
            actual_owner = source_owners.get(parts[2])
        else:
            actual_owner = None
        if actual_owner != expected_owner:
            raise ComponentGraphError(
                f"ownership assertion failed for {asserted_path}: "
                f"expected {expected_owner}, actual {actual_owner}"
            )

    public_edges: dict[Edge, set[str]] = {}
    source_edges: dict[Edge, set[str]] = {}
    aggregate_edges: dict[Edge, set[str]] = {}
    for root, consumer in sorted(header_owners.items()):
        for path in _owned_code_files(public_root / root):
            _record_includes(
                path, repository_root, consumer, header_owners, public_edges
            )
    for root, consumer in sorted(source_owners.items()):
        for path in _owned_code_files(source_root / root):
            _record_includes(
                path, repository_root, consumer, header_owners, source_edges
            )
    for aggregate_header in manifest.aggregate_headers:
        _record_includes(
            public_root / aggregate_header,
            repository_root,
            "sdk",
            header_owners,
            aggregate_edges,
        )
    return IncludeGraphs(public_edges, source_edges, aggregate_edges)


def load_target_graph(path: pathlib.Path, manifest: Manifest) -> TargetGraphs:
    """Load the direct target and link-interface graph emitted by CMake."""
    text = _read_bounded(path, MAX_DOCUMENT_BYTES)
    lines = text.splitlines()
    if not lines or lines[0] != "component\tkind\tdependency":
        raise ComponentGraphError("CMake target graph has an invalid TSV header")

    states: dict[str, bool] = {}
    target_edges: set[Edge] = set()
    interface_edges: set[Edge] = set()
    sources: set[str] = set()
    headers: set[str] = set()
    header_owners, source_owners = _ownership_maps(manifest)
    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split("\t")
        if len(fields) != 3:
            raise ComponentGraphError(
                f"target graph line {line_number} must contain three TSV fields"
            )
        component, kind, dependency = fields
        if component not in manifest.components:
            raise ComponentGraphError(
                f"target graph contains unknown entity target: {component!r}"
            )
        if kind in {"source", "header"}:
            member_path = pathlib.PurePosixPath(dependency)
            parts = member_path.parts
            owners = source_owners if kind == "source" else header_owners
            root = "src" if kind == "source" else "include"
            if (
                member_path.is_absolute()
                or ".." in parts
                or "\\" in dependency
                or member_path.as_posix() != dependency
                or len(parts) < 4
                or parts[:2] != (root, "jyppx")
                or owners.get(parts[2]) != component
                or member_path.suffix.lower() not in CODE_SUFFIXES
            ):
                raise ComponentGraphError(f"invalid active {kind} ownership/path")
            members = sources if kind == "source" else headers
            if dependency in members:
                raise ComponentGraphError(f"target graph repeats {kind}: {dependency}")
            members.add(dependency)
            continue
        if kind in {"present", "absent"}:
            if dependency:
                raise ComponentGraphError(
                    f"target graph state row {line_number} has a dependency"
                )
            if component in states:
                raise ComponentGraphError(
                    f"target graph repeats component state: {component}"
                )
            states[component] = kind == "present"
            continue
        if kind not in {"target", "interface"} or not dependency:
            raise ComponentGraphError(
                f"target graph line {line_number} has invalid kind/dependency"
            )
        if dependency not in manifest.components:
            raise ComponentGraphError(
                f"{component} links forbidden aggregate or unknown target {dependency!r}"
            )
        edge = Edge(component, dependency)
        graph = target_edges if kind == "target" else interface_edges
        if edge in graph:
            raise ComponentGraphError(
                f"target graph repeats {kind} edge: {component} -> {dependency}"
            )
        graph.add(edge)

    if set(states) != set(manifest.components):
        raise ComponentGraphError(
            "CMake target set differs from manifest; missing="
            f"{sorted(set(manifest.components) - set(states))}, unknown="
            f"{sorted(set(states) - set(manifest.components))}"
        )
    enabled = frozenset(component for component, present in states.items() if present)
    for source in sources:
        owner = source_owners[pathlib.PurePosixPath(source).parts[2]]
        if owner not in enabled:
            raise ComponentGraphError(f"disabled component has an active source: {owner}")
    for header in headers:
        owner = header_owners[pathlib.PurePosixPath(header).parts[2]]
        if owner not in enabled:
            raise ComponentGraphError(f"disabled component has an active header: {owner}")
    for edge in target_edges | interface_edges:
        if edge.consumer not in enabled or edge.provider not in enabled:
            raise ComponentGraphError(
                f"disabled component appears in CMake graph: "
                f"{edge.consumer} -> {edge.provider}"
            )
        if edge.consumer == edge.provider:
            raise ComponentGraphError(
                f"CMake graph contains self dependency: {edge.consumer}"
            )
    for graph_name, graph in (("target", target_edges), ("interface", interface_edges)):
        cycle = _find_cycle(enabled, graph)
        if cycle is not None:
            raise ComponentGraphError(
                f"CMake {graph_name} graph contains a cycle: " + " -> ".join(cycle)
            )
    return TargetGraphs(enabled, frozenset(target_edges), frozenset(interface_edges),
                        frozenset(sources), frozenset(headers))


def _allowed_edges(manifest: Manifest) -> set[Edge]:
    """Return all required and optional policy edges."""
    return {
        Edge(consumer, provider)
        for consumer, policy in manifest.components.items()
        for provider in policy.required_dependencies + policy.optional_dependencies
    }


def validate_graphs(
    manifest: Manifest, includes: IncludeGraphs, targets: TargetGraphs
) -> None:
    """Validate actual graphs against required and allowed direct edges."""
    allowed = _allowed_edges(manifest)
    for graph_name, graph in (
        ("target", targets.target),
        ("link interface", targets.interface),
        ("public include", set(includes.public)),
        ("source include", set(includes.source)),
    ):
        forbidden = sorted(set(graph) - allowed)
        if forbidden:
            edge = forbidden[0]
            raise ComponentGraphError(
                f"{graph_name} graph contains forbidden edge: "
                f"{edge.consumer} -> {edge.provider}"
            )

    for consumer in sorted(targets.enabled_components):
        policy = manifest.components[consumer]
        for provider in policy.required_dependencies:
            edge = Edge(consumer, provider)
            if provider not in targets.enabled_components:
                raise ComponentGraphError(
                    f"enabled {consumer} requires disabled component {provider}"
                )
            if edge not in targets.target:
                raise ComponentGraphError(
                    f"required direct target edge is missing: {consumer} -> {provider}"
                )
        for provider in policy.required_public_dependencies:
            edge = Edge(consumer, provider)
            if edge not in targets.interface:
                raise ComponentGraphError(
                    f"required direct link-interface edge is missing: "
                    f"{consumer} -> {provider}"
                )

    for edge, locations in sorted(includes.public.items()):
        active_locations = locations if targets.headers is None else {
            location for location in locations
            if location.rsplit(":", 1)[0] in targets.headers
        }
        if (active_locations and edge.consumer in targets.enabled_components
                and edge not in targets.interface):
            raise ComponentGraphError(
                f"public include edge lacks a direct link interface: "
                f"{edge.consumer} -> {edge.provider} at {sorted(active_locations)[0]}"
            )
    for edge, locations in sorted(includes.source.items()):
        active_locations = locations if targets.sources is None else {
            location for location in locations
            if (location.rsplit(":", 1)[0] in targets.sources or
                pathlib.PurePosixPath(location.rsplit(":", 1)[0]).suffix.lower()
                in {".h", ".hh", ".hpp", ".hxx", ".inc"})
        }
        if (active_locations and edge.consumer in targets.enabled_components
                and edge not in targets.target):
            raise ComponentGraphError(
                f"source include edge lacks a direct target link: "
                f"{edge.consumer} -> {edge.provider} at {sorted(active_locations)[0]}"
            )


def _edge_list(edges: Iterable[Edge]) -> list[dict[str, str]]:
    """Serialize a deterministic edge set."""
    return [dataclasses.asdict(edge) for edge in sorted(edges)]


def _include_edge_list(
    edges: dict[Edge, set[str]],
) -> list[dict[str, str | list[str]]]:
    """Serialize include edges with deterministic file-and-line evidence."""
    return [
        {
            "consumer": edge.consumer,
            "provider": edge.provider,
            "locations": sorted(locations),
        }
        for edge, locations in sorted(edges.items())
    ]


def build_report(
    manifest: Manifest, includes: IncludeGraphs, targets: TargetGraphs
) -> dict[str, Any]:
    """Build the deterministic machine-readable gate report."""
    required = {
        Edge(consumer, provider)
        for consumer, policy in manifest.components.items()
        for provider in policy.required_dependencies
    }
    required_public = {
        Edge(consumer, provider)
        for consumer, policy in manifest.components.items()
        for provider in policy.required_public_dependencies
    }
    optional = {
        Edge(consumer, provider)
        for consumer, policy in manifest.components.items()
        for provider in policy.optional_dependencies
    }
    header_owners, source_owners = _ownership_maps(manifest)
    return {
        "schema_version": 1,
        "status": "passed",
        "manifest_sha256": manifest.digest,
        "enabled_components": sorted(targets.enabled_components),
        "ownership": {
            "aggregate_headers": list(manifest.aggregate_headers),
            "header_roots": dict(sorted(header_owners.items())),
            "source_roots": dict(sorted(source_owners.items())),
        },
        "graphs": {
            "allowed_optional": _edge_list(optional),
            "cmake_link_interface": _edge_list(targets.interface),
            "cmake_target": _edge_list(targets.target),
            "public_include": _include_edge_list(includes.public),
            "required": _edge_list(required),
            "required_public": _edge_list(required_public),
            "sdk_aggregate_include": _include_edge_list(includes.aggregate),
            "source_include": _include_edge_list(includes.source),
            "active_sources": sorted(targets.sources) if targets.sources is not None else None,
            "active_public_headers": sorted(targets.headers) if targets.headers is not None else None,
        },
    }


def validate_repository(
    manifest_path: pathlib.Path,
    repository_root: pathlib.Path,
    target_graph_path: pathlib.Path,
) -> dict[str, Any]:
    """Validate all component graphs and return the deterministic report."""
    manifest = load_manifest(manifest_path)
    includes = scan_repository(manifest, repository_root)
    targets = load_target_graph(target_graph_path, manifest)
    validate_graphs(manifest, includes, targets)
    return build_report(manifest, includes, targets)


def main() -> int:
    """Validate one repository and persist its deterministic graph report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--source-root", type=pathlib.Path, required=True)
    parser.add_argument("--target-graph", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        report = validate_repository(
            args.manifest, args.source_root, args.target_graph
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (ComponentGraphError, OSError) as error:
        print(f"component dependency graph validation failed: {error}", file=sys.stderr)
        return 1
    required_count = len(report["graphs"]["required"])
    print(
        "Component dependency graph passed: "
        f"{len(report['enabled_components'])} enabled components, "
        f"{required_count} policy-required edges"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
