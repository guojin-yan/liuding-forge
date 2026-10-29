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
from typing import Any

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "validation"))

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
                if edge != missing_interface:
                    lines.append(f"{consumer}\tinterface\t{provider}")
            for header_root in policy.header_roots:
                root = REPOSITORY_ROOT / "include" / "jyppx" / "liuding" / header_root
                for path in sorted(root.rglob("*.hpp")):
                    relative = path.relative_to(REPOSITORY_ROOT).as_posix()
                    lines.append(f"{consumer}\theader\t{relative}")
            for source_root in policy.source_roots:
                root = REPOSITORY_ROOT / "src" / "jyppx" / "liuding" / source_root
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
            public={forbidden: {"include/jyppx/liuding/foundation/example.hpp:1"}},
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
            "training\tsource\tsrc/jyppx/liuding/training/file_checkpoint_store.cpp")]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        includes = component_graph.scan_repository(self.manifest, REPOSITORY_ROOT)
        targets = component_graph.load_target_graph(path, self.manifest)
        component_graph.validate_graphs(self.manifest, includes, targets)

    def test_active_source_rows_reject_wrong_owner_and_parent_traversal(self) -> None:
        """Source membership cannot move files across component ownership."""
        for source in ("src/jyppx/liuding/interop/file_state_archive.cpp",
                       "src/jyppx/liuding/training/../interop/file_state_archive.cpp"):
            path = self.write_graph()
            with path.open("a", encoding="utf-8") as output:
                output.write(f"training\tsource\t{source}\n")
            with self.assertRaises(component_graph.ComponentGraphError):
                component_graph.load_target_graph(path, self.manifest)

    def test_active_optional_public_header_requires_direct_interface(self) -> None:
        """Installed nn adapters must expose their optional public dependency."""
        path = self.write_graph(missing_interface=component_graph.Edge("interop", "nn"))
        targets = component_graph.load_target_graph(path, self.manifest)
        includes = component_graph.scan_repository(self.manifest, REPOSITORY_ROOT)
        with self.assertRaisesRegex(component_graph.ComponentGraphError,
                                    "public include edge lacks a direct link interface: interop -> nn"):
            component_graph.validate_graphs(self.manifest, includes, targets)

    def test_uninstalled_optional_public_headers_do_not_require_interface(self) -> None:
        """An interop-only package omits both nn-dependent public adapters."""
        path = self.write_graph(missing_interface=component_graph.Edge("interop", "nn"))
        excluded = {"include/jyppx/liuding/interop/model_weight_import.hpp",
                    "include/jyppx/liuding/interop/torchscript_compatibility.hpp"}
        lines = [line for line in path.read_text(encoding="utf-8").splitlines()
                 if line.split("\t")[-1] not in excluded]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        targets = component_graph.load_target_graph(path, self.manifest)
        includes = component_graph.scan_repository(self.manifest, REPOSITORY_ROOT)
        component_graph.validate_graphs(self.manifest, includes, targets)

    def test_active_header_rows_reject_wrong_owner_traversal_and_duplicates(self) -> None:
        """CMake header membership obeys the same ownership and path contract."""
        for header in ("include/jyppx/liuding/nn/model_adapter.hpp",
                       "include/jyppx/liuding/interop/../nn/model_adapter.hpp",
                       "include/jyppx/liuding/interop/torchscript_compatibility.hpp"):
            path = self.write_graph()
            with path.open("a", encoding="utf-8") as output:
                output.write(f"interop\theader\t{header}\n")
            with self.assertRaises(component_graph.ComponentGraphError):
                component_graph.load_target_graph(path, self.manifest)


class NestedOwnershipTest(unittest.TestCase):
    """Exercise nested roots without depending on the repository's component set."""

    def setUp(self) -> None:
        """Create two independent deployment components in an isolated repository."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.temp_root = pathlib.Path(self.temporary_directory.name)
        self.repository_root = self.temp_root / "repository"
        self.manifest_path = self.temp_root / "component-dependencies.json"
        self.document: dict[str, Any] = {
            "schema_version": 1,
            "aggregate_headers": ["liuding.hpp"],
            "components": {},
            "ownership_assertions": {
                "include/jyppx/liuding/deployment/onnx_export": "onnx_export",
                "include/jyppx/liuding/deployment/onnx_runtime/session.hpp": "onnx_runtime",
                "src/jyppx/liuding/deployment/onnx_export/exporter.cpp": "onnx_export",
                "src/jyppx/liuding/deployment/onnx_runtime": "onnx_runtime",
            },
        }
        for name in ("foundation", "onnx_export", "onnx_runtime"):
            root = "foundation" if name == "foundation" else f"deployment/{name}"
            dependencies = [] if name == "foundation" else ["foundation"]
            self.document["components"][name] = {
                "header_roots": [root],
                "source_roots": [root],
                "required_dependencies": dependencies,
                "required_public_dependencies": dependencies,
                "optional_dependencies": [],
            }
        self.write_manifest(self.document, self.manifest_path)
        self.manifest = component_graph.load_manifest(self.manifest_path)
        files = {
            "include/jyppx/liuding/foundation/api.hpp": "// Foundation API\n",
            "src/jyppx/liuding/foundation/api.cpp":
                "#include <jyppx/liuding/foundation/api.hpp>\n",
            "include/jyppx/liuding/deployment/onnx_export/exporter.hpp":
                "#include <jyppx/liuding/foundation/api.hpp>\n",
            "include/jyppx/liuding/deployment/onnx_export/detail/options.hpp":
                "#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>\n",
            "src/jyppx/liuding/deployment/onnx_export/exporter.cpp":
                "#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>\n",
            "include/jyppx/liuding/deployment/onnx_runtime/session.hpp":
                "#include <jyppx/liuding/foundation/api.hpp>\n",
            "src/jyppx/liuding/deployment/onnx_runtime/session.cpp":
                "#include <jyppx/liuding/deployment/onnx_runtime/session.hpp>\n",
            "include/jyppx/liuding/liuding.hpp":
                "#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>\n"
                "#include <jyppx/liuding/deployment/onnx_runtime/session.hpp>\n",
        }
        for relative, content in files.items():
            self.write_file(relative, content)

    def tearDown(self) -> None:
        """Release the temporary repository."""
        self.temporary_directory.cleanup()

    def write_file(self, relative: str, content: str) -> pathlib.Path:
        """Write one file inside the isolated repository."""
        path = self.repository_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def write_manifest(
        self, document: dict[str, Any], path: pathlib.Path | None = None
    ) -> pathlib.Path:
        """Write a sorted policy, leaving validation to the test."""
        path = path if path is not None else self.temp_root / "changed-manifest.json"
        path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def write_graph(self, *, disabled: frozenset[str] = frozenset()) -> pathlib.Path:
        """Describe exact component membership without merging the deployment parent."""
        lines = ["component\tkind\tdependency"]
        for component, policy in sorted(self.manifest.components.items()):
            state = "absent" if component in disabled else "present"
            lines.append(f"{component}\t{state}\t")
            if component in disabled:
                continue
            for provider in policy.required_dependencies:
                lines.append(f"{component}\ttarget\t{provider}")
            for provider in policy.required_public_dependencies:
                lines.append(f"{component}\tinterface\t{provider}")
            for kind, prefix, roots in (
                ("header", "include", policy.header_roots),
                ("source", "src", policy.source_roots),
            ):
                for root in roots:
                    directory = self.repository_root / prefix / "jyppx" / "liuding" / root
                    for path in sorted(directory.rglob("*")):
                        if path.is_file() and path.suffix in {".hpp", ".cpp"}:
                            relative = path.relative_to(self.repository_root).as_posix()
                            lines.append(f"{component}\t{kind}\t{relative}")
        path = self.temp_root / "component-target-graph.tsv"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_nested_components_keep_distinct_owners_and_aggregate_edges(self) -> None:
        """A shared physical parent preserves two logical components end to end."""
        report = component_graph.validate_repository(
            self.manifest_path, self.repository_root, self.write_graph()
        )
        self.assertEqual(
            ["foundation", "onnx_export", "onnx_runtime"], report["enabled_components"]
        )
        for kind in ("header_roots", "source_roots"):
            self.assertEqual(
                "onnx_export", report["ownership"][kind]["deployment/onnx_export"]
            )
            self.assertEqual(
                "onnx_runtime", report["ownership"][kind]["deployment/onnx_runtime"]
            )
        self.assertEqual(
            {("onnx_export", "foundation"), ("onnx_runtime", "foundation")},
            {(edge["consumer"], edge["provider"])
             for edge in report["graphs"]["public_include"]},
        )
        self.assertEqual(
            {("sdk", "onnx_export"), ("sdk", "onnx_runtime")},
            {(edge["consumer"], edge["provider"])
             for edge in report["graphs"]["sdk_aggregate_include"]},
        )

    def test_sibling_includes_still_require_independent_component_edges(self) -> None:
        """Sharing deployment/ cannot hide a forbidden public or source dependency."""
        for kind, relative in (
            ("public", "include/jyppx/liuding/deployment/onnx_export/exporter.hpp"),
            ("source", "src/jyppx/liuding/deployment/onnx_export/exporter.cpp"),
        ):
            with self.subTest(kind=kind):
                path = self.repository_root / relative
                original = path.read_text(encoding="utf-8")
                path.write_text(
                    original + "#include <jyppx/liuding/deployment/onnx_runtime/session.hpp>\n",
                    encoding="utf-8",
                )
                try:
                    includes = component_graph.scan_repository(
                        self.manifest, self.repository_root
                    )
                    edges = includes.public if kind == "public" else includes.source
                    self.assertIn(component_graph.Edge("onnx_export", "onnx_runtime"), edges)
                    targets = component_graph.load_target_graph(self.write_graph(), self.manifest)
                    with self.assertRaisesRegex(
                        component_graph.ComponentGraphError,
                        f"{kind} include graph contains forbidden edge: onnx_export -> onnx_runtime",
                    ):
                        component_graph.validate_graphs(self.manifest, includes, targets)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_manifest_rejects_noncanonical_ownership_roots(self) -> None:
        """Root paths cannot escape, alias, or use invalid identifier segments."""
        for field in ("header_roots", "source_roots"):
            for root in (
                "", ".", "..", "/deployment/onnx_export", "C:/deployment/onnx_export",
                "deployment\\onnx_export", "deployment//onnx_export",
                "deployment/./onnx_export", "deployment/../onnx_export",
                "deployment/onnx_export/", "deployment/onnx_export.",
                "deployment/onnx_export ", "deployment/ONNX_export",
                "deployment/onnx-export", "deployment/onnx_export:stream",
                "deployment/onnx_export\x00", "deployment/onnx_export\n",
            ):
                with self.subTest(field=field, root=root):
                    document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                    document["components"]["onnx_export"][field] = [root]
                    with self.assertRaises(component_graph.ComponentGraphError):
                        component_graph.load_manifest(self.write_manifest(document))

    def test_manifest_rejects_overlapping_roots_within_or_between_components(self) -> None:
        """Exact duplicates and ancestor roots are ambiguous even for the same owner."""
        for field in ("header_roots", "source_roots"):
            for component, roots in (
                ("onnx_export", ["deployment"]),
                ("onnx_export", ["deployment/onnx_runtime"]),
                ("onnx_export", ["deployment", "deployment/onnx_export"]),
                ("onnx_export", ["foundation/submodule"]),
                ("foundation", ["deployment/onnx_export/detail"]),
            ):
                with self.subTest(field=field, component=component, roots=roots):
                    document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                    document["components"][component][field] = roots
                    with self.assertRaisesRegex(
                        component_graph.ComponentGraphError, "overlaps ownership root"
                    ):
                        component_graph.load_manifest(self.write_manifest(document))

    def test_scan_rejects_unowned_public_and_production_files(self) -> None:
        """No sibling, shared-parent, or top-level source file may escape ownership."""
        for relative in (
            "include/jyppx/liuding/deployment/orphan.hpp",
            "include/jyppx/liuding/deployment/unknown/api.hpp",
            "include/jyppx/liuding/deployment/onnx_export_extra/api.hpp",
            "src/jyppx/liuding/orphan.cpp",
            "src/jyppx/liuding/deployment/orphan.cpp",
            "src/jyppx/liuding/deployment/unknown/api.cpp",
            "src/jyppx/liuding/deployment/onnx_runtime_extra/api.cpp",
        ):
            with self.subTest(relative=relative):
                path = self.write_file(relative, "// Unowned production file\n")
                try:
                    with self.assertRaisesRegex(
                        component_graph.ComponentGraphError, "ownership mismatch"
                    ):
                        component_graph.scan_repository(self.manifest, self.repository_root)
                finally:
                    path.unlink()

    def test_scan_rejects_empty_declared_nested_roots(self) -> None:
        """A declared nested header or source root still needs at least one code file."""
        for prefix in ("include", "src"):
            with self.subTest(prefix=prefix):
                root = self.repository_root / prefix / "jyppx/liuding/deployment/onnx_export"
                files = {path: path.read_text(encoding="utf-8")
                         for path in root.rglob("*") if path.is_file()}
                for path in files:
                    path.unlink()
                try:
                    with self.assertRaisesRegex(
                        component_graph.ComponentGraphError,
                        r"ownership mismatch; missing=\['deployment/onnx_export'\]",
                    ):
                        component_graph.scan_repository(self.manifest, self.repository_root)
                finally:
                    for path, content in files.items():
                        path.write_text(content, encoding="utf-8")

    def test_scan_rejects_wrong_sibling_ownership_assertions(self) -> None:
        """Assertions distinguish the full nested root for both directory and file paths."""
        for asserted_path in (
            "include/jyppx/liuding/deployment/onnx_runtime/session.hpp",
            "src/jyppx/liuding/deployment/onnx_runtime",
        ):
            with self.subTest(path=asserted_path):
                document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                document["ownership_assertions"][asserted_path] = "onnx_export"
                manifest = component_graph.load_manifest(self.write_manifest(document))
                with self.assertRaisesRegex(
                    component_graph.ComponentGraphError,
                    "expected onnx_export, actual onnx_runtime",
                ):
                    component_graph.scan_repository(manifest, self.repository_root)

    def test_manifest_rejects_noncanonical_assertion_paths(self) -> None:
        """Assertions cannot conceal traversal or alternate spellings of nested roots."""
        for asserted_path in (
            "include/jyppx/liuding/deployment//onnx_export/exporter.hpp",
            "include/jyppx/liuding/deployment/./onnx_export/exporter.hpp",
            "include/jyppx/liuding/deployment/onnx_runtime/../onnx_export/exporter.hpp",
            "include/jyppx/liuding/deployment/onnx_export/exporter.hpp:stream",
            "include\\jyppx\\liuding\\deployment\\onnx_export\\exporter.hpp",
            "/include/jyppx/liuding/deployment/onnx_export/exporter.hpp",
            "src/jyppx/liuding/deployment/onnx_export/",
        ):
            with self.subTest(path=asserted_path):
                document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                document["ownership_assertions"][asserted_path] = "onnx_export"
                with self.assertRaisesRegex(
                    component_graph.ComponentGraphError, "invalid ownership assertion path"
                ):
                    component_graph.load_manifest(self.write_manifest(document))

    def test_scan_rejects_unowned_or_noncanonical_include_paths(self) -> None:
        """Include resolution uses exact directory boundaries and canonical paths."""
        for include_path in (
            "deployment/unknown/session.hpp",
            "deployment/onnx_runtime_extra/session.hpp",
            "deployment/session.hpp",
            "deployment/onnx_export/../onnx_runtime/session.hpp",
            "deployment//onnx_runtime/session.hpp",
            "deployment/./onnx_runtime/session.hpp",
            "deployment\\onnx_runtime\\session.hpp",
            "deployment/onnx_runtime/session.hpp:stream",
            "deployment/onnx_runtime/session.hpp.",
        ):
            with self.subTest(include_path=include_path):
                self.write_file(
                    "include/jyppx/liuding/deployment/onnx_export/exporter.hpp",
                    f"#include <jyppx/liuding/{include_path}>\n",
                )
                with self.assertRaisesRegex(
                    component_graph.ComponentGraphError, "unowned or invalid JYPPX header path"
                ):
                    component_graph.scan_repository(self.manifest, self.repository_root)

    def test_target_membership_rejects_wrong_sibling_and_noncanonical_paths(self) -> None:
        """CMake membership resolves complete nested paths and rejects aliases."""
        for kind, prefix, filename in (
            ("header", "include", "exporter.hpp"),
            ("source", "src", "exporter.cpp"),
        ):
            for relative in (
                f"{prefix}/jyppx/liuding/deployment/onnx_runtime/{filename}",
                f"{prefix}/jyppx/liuding/deployment/onnx_export_extra/{filename}",
                f"{prefix}/jyppx/liuding/deployment/{filename}",
                f"{prefix}/jyppx/liuding/deployment/onnx_runtime/../onnx_export/{filename}",
                f"{prefix}/jyppx/liuding/deployment//onnx_export/{filename}",
                f"{prefix}/jyppx/liuding/deployment/./onnx_export/{filename}",
                f"/{prefix}/jyppx/liuding/deployment/onnx_export/{filename}",
                f"C:/{prefix}/jyppx/liuding/deployment/onnx_export/{filename}",
                f"{prefix}\\jyppx\\liuding\\deployment\\onnx_export\\{filename}",
                f"{prefix}/jyppx/liuding/deployment/onnx_export/{filename}:stream",
                f"{prefix}/jyppx/liuding/deployment/onnx_export/{filename}\x00",
            ):
                with self.subTest(kind=kind, path=relative):
                    path = self.write_graph()
                    with path.open("a", encoding="utf-8") as output:
                        output.write(f"onnx_export\t{kind}\t{relative}\n")
                    with self.assertRaisesRegex(
                        component_graph.ComponentGraphError,
                        f"invalid active {kind} ownership/path",
                    ):
                        component_graph.load_target_graph(path, self.manifest)

    def test_disabled_nested_component_cannot_supply_active_members(self) -> None:
        """Enabled checks use the nested owner for both headers and sources."""
        for kind, relative in (
            ("header", "include/jyppx/liuding/deployment/onnx_export/exporter.hpp"),
            ("source", "src/jyppx/liuding/deployment/onnx_export/exporter.cpp"),
        ):
            with self.subTest(kind=kind):
                path = self.write_graph(disabled=frozenset({"onnx_export"}))
                with path.open("a", encoding="utf-8") as output:
                    output.write(f"onnx_export\t{kind}\t{relative}\n")
                with self.assertRaisesRegex(
                    component_graph.ComponentGraphError,
                    f"disabled component has an active {kind}: onnx_export",
                ):
                    component_graph.load_target_graph(path, self.manifest)

    def test_nested_component_membership_still_rejects_duplicates(self) -> None:
        """Nested paths do not weaken the existing duplicate membership gate."""
        for kind, relative in (
            ("header", "include/jyppx/liuding/deployment/onnx_export/exporter.hpp"),
            ("source", "src/jyppx/liuding/deployment/onnx_export/exporter.cpp"),
        ):
            with self.subTest(kind=kind):
                path = self.write_graph()
                with path.open("a", encoding="utf-8") as output:
                    output.write(f"onnx_export\t{kind}\t{relative}\n")
                with self.assertRaisesRegex(
                    component_graph.ComponentGraphError, f"target graph repeats {kind}"
                ):
                    component_graph.load_target_graph(path, self.manifest)


if __name__ == "__main__":
    unittest.main()
