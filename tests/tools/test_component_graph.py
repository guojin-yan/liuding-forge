# Copyright 2026 JYPPX Training Framework Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for the component ownership and direct-dependency graph gate."""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

import component_graph  # noqa: E402


class ComponentGraphTest(unittest.TestCase):
    """Covers manifest, ownership, include, and direct-link invariants."""

    def setUp(self) -> None:
        """Load the repository policy and allocate isolated graph files."""
        self.manifest_path = REPOSITORY_ROOT / "component-dependencies.json"
        self.manifest = component_graph.load_manifest(self.manifest_path)
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.temp_root = pathlib.Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        """Release isolated graph files."""
        self.temporary_directory.cleanup()

    def write_graph(
        self,
        *,
        missing_target: component_graph.Edge | None = None,
        missing_interface: component_graph.Edge | None = None,
        duplicate_target: component_graph.Edge | None = None,
    ) -> pathlib.Path:
        """Write a complete synthetic CMake graph with optional corruption."""
        lines = ["component\tkind\tdependency"]
        for consumer, policy in sorted(self.manifest.components.items()):
            lines.append(f"{consumer}\tpresent\t")
            for provider in policy.required_dependencies:
                edge = component_graph.Edge(consumer, provider)
                if edge != missing_target:
                    lines.append(f"{consumer}\ttarget\t{provider}")
                    if edge == duplicate_target:
                        lines.append(f"{consumer}\ttarget\t{provider}")
            for provider in policy.required_public_dependencies:
                edge = component_graph.Edge(consumer, provider)
                if edge != missing_interface:
                    lines.append(f"{consumer}\tinterface\t{provider}")
            for provider in policy.optional_dependencies:
                edge = component_graph.Edge(consumer, provider)
                if edge != missing_target:
                    lines.append(f"{consumer}\ttarget\t{provider}")
            for source_root in policy.source_roots:
                root = REPOSITORY_ROOT / "src" / "jyppx" / source_root
                for path in sorted(root.rglob("*.cpp")):
                    relative = path.relative_to(REPOSITORY_ROOT).as_posix()
                    lines.append(f"{consumer}\tsource\t{relative}")
        graph_path = self.temp_root / "component-target-graph.tsv"
        graph_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return graph_path

    def test_repository_graph_is_valid(self) -> None:
        """Accept the checked-in ownership and include graph with direct links."""
        report = component_graph.validate_repository(
            self.manifest_path, REPOSITORY_ROOT, self.write_graph()
        )
        self.assertEqual(
            sorted(self.manifest.components), report["enabled_components"]
        )

    def test_rejects_transitive_only_required_target_edge(self) -> None:
        """Reject training-to-core when it is available only transitively."""
        missing = component_graph.Edge("training", "core")
        targets = component_graph.load_target_graph(
            self.write_graph(missing_target=missing), self.manifest
        )
        includes = component_graph.scan_repository(self.manifest, REPOSITORY_ROOT)
        with self.assertRaisesRegex(
            component_graph.ComponentGraphError,
            "required direct target edge is missing: training -> core",
        ):
            component_graph.validate_graphs(self.manifest, includes, targets)

    def test_rejects_missing_public_link_interface(self) -> None:
        """Reject a public dependency absent from the direct link interface."""
        missing = component_graph.Edge("experiment", "core")
        targets = component_graph.load_target_graph(
            self.write_graph(missing_interface=missing), self.manifest
        )
        includes = component_graph.scan_repository(self.manifest, REPOSITORY_ROOT)
        with self.assertRaisesRegex(
            component_graph.ComponentGraphError,
            "required direct link-interface edge is missing: experiment -> core",
        ):
            component_graph.validate_graphs(self.manifest, includes, targets)

    def test_rejects_forbidden_public_include(self) -> None:
        """Reject an include edge absent from the allowed matrix."""
        forbidden = component_graph.Edge("foundation", "training")
        includes = component_graph.IncludeGraphs(
            public={forbidden: {"include/jyppx/foundation/example.hpp:1"}},
            source={},
            aggregate={},
        )
        targets = component_graph.load_target_graph(self.write_graph(), self.manifest)
        with self.assertRaisesRegex(
            component_graph.ComponentGraphError,
            "public include graph contains forbidden edge: foundation -> training",
        ):
            component_graph.validate_graphs(self.manifest, includes, targets)

    def test_rejects_duplicate_direct_edge(self) -> None:
        """Reject duplicate CMake dependency declarations."""
        duplicate = component_graph.Edge("data", "core")
        with self.assertRaisesRegex(
            component_graph.ComponentGraphError,
            "target graph repeats target edge: data -> core",
        ):
            component_graph.load_target_graph(
                self.write_graph(duplicate_target=duplicate), self.manifest
            )

    def test_rejects_cycle_in_allowed_matrix(self) -> None:
        """Reject a cycle even when one of its edges is optional."""
        document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        document["components"]["foundation"]["optional_dependencies"] = ["training"]
        changed_manifest = self.temp_root / "component-dependencies.json"
        changed_manifest.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            component_graph.ComponentGraphError,
            "allowed dependency graph contains a cycle",
        ):
            component_graph.load_manifest(changed_manifest)

    def test_active_optional_source_requires_a_direct_link(self) -> None:
        """An active file adapter may not consume a transitive interop link."""
        missing = component_graph.Edge("training", "interop")
        targets = component_graph.load_target_graph(
            self.write_graph(missing_target=missing), self.manifest
        )
        includes = component_graph.scan_repository(self.manifest, REPOSITORY_ROOT)
        with self.assertRaisesRegex(component_graph.ComponentGraphError,
                                    "source include edge lacks a direct target link: training -> interop"):
            component_graph.validate_graphs(self.manifest, includes, targets)

    def test_inactive_optional_source_does_not_require_a_link(self) -> None:
        """CMake source membership permits the interop-free training build."""
        missing = component_graph.Edge("training", "interop")
        path = self.write_graph(missing_target=missing)
        lines = path.read_text(encoding="utf-8").splitlines()
        lines = [line for line in lines if not line.startswith(
            "training\tsource\tsrc/jyppx/training/file_checkpoint_store.cpp")]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        includes = component_graph.scan_repository(self.manifest, REPOSITORY_ROOT)
        targets = component_graph.load_target_graph(path, self.manifest)
        component_graph.validate_graphs(self.manifest, includes, targets)

    def test_active_source_rows_reject_wrong_owner_and_parent_traversal(self) -> None:
        """Source membership cannot move files across component ownership."""
        for source in ("src/jyppx/interop/file_state_archive.cpp",
                       "src/jyppx/training/../interop/file_state_archive.cpp"):
            path = self.write_graph()
            with path.open("a", encoding="utf-8") as output:
                output.write(f"training\tsource\t{source}\n")
            with self.assertRaises(component_graph.ComponentGraphError):
                component_graph.load_target_graph(path, self.manifest)


if __name__ == "__main__":
    unittest.main()
