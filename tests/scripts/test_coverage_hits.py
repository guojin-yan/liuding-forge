"""Exercise the read-only coverage guard without a compiler or native tests."""
from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET


SOURCE = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("coverage_hits", SOURCE / "scripts/validation/coverage_hits.py")
assert SPEC and SPEC.loader
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


def words(*values: int) -> bytes:
    return struct.pack("<" + "I" * len(values), *values)


def string(value: str) -> bytes:
    raw = value.encode("utf-8") + b"\0"
    raw += b"\0" * (-len(raw) % 4)
    return words(len(raw) // 4) + raw


def record(tag: int, raw: bytes) -> bytes:
    return words(tag, len(raw) // 4) + raw


def notes_bytes(*, stamp: int = 73, source: str = "/fixture/src/jyppx/example.cpp") -> bytes:
    function = words(11, 22, 33) + string("example") + words(0) + string(source) + words(1, 1, 3, 1)
    return (b"oncg*41B" + words(stamp) + string("/fixture/build") + words(1)
            + record(guard.FUNCTION, function) + record(guard.BLOCKS, words(3))
            + record(guard.ARCS, words(0, 2, 4)) + record(guard.ARCS, words(2, 1, 1))
            + record(guard.LINES, words(2, 0) + string(source) + words(1, 2, 0) + string("")))


def data_bytes(hits: int = 7, *, stamp: int = 73, compressed: bool = False, empty: bool = False) -> bytes:
    result = b"adcg*41B" + words(stamp) + record(guard.SUMMARY, words(1, 7))
    if empty:
        return result + record(guard.FUNCTION, b"")
    result += record(guard.FUNCTION, words(11, 22, 33))
    return result + (words(guard.COUNTERS, (-2) & 0xFFFFFFFF) if compressed else
                     record(guard.COUNTERS, struct.pack("<q", hits)))


def xml_document(*, hits: str = "7") -> ET.Element:
    root = ET.Element("coverage", {"lines-valid": "2", "lines-covered": "1"})
    classes = ET.SubElement(ET.SubElement(ET.SubElement(root, "packages"), "package", name="fixture"), "classes")
    item = ET.SubElement(classes, "class", filename="src/jyppx/example.cpp")
    method = ET.SubElement(ET.SubElement(item, "methods"), "method", name="example", signature="()")
    for parent in (method, item):
        lines = ET.SubElement(parent, "lines")
        ET.SubElement(lines, "line", number="1", hits=hits)
        ET.SubElement(lines, "line", number="2", hits="0")
    return root


class CoverageHitsTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="coverage hits ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "src/jyppx").mkdir(parents=True)
        (self.root / "include/jyppx").mkdir(parents=True)
        self.source = self.root / "src/jyppx/example.cpp"
        self.source.write_text("int example() {\n  return 7;\n}\n", encoding="utf-8")
        self.build = self.root / "build"
        self.build.mkdir()
        self.note = self.build / "example.cpp.gcno"
        self.data = self.build / "example.cpp.gcda"
        self.note.write_bytes(notes_bytes())
        self.data.write_bytes(data_bytes())
        self.receipt = self.root / "guard.json"
        self.xml = self.root / "coverage.xml"

    def receipt_file(self) -> dict:
        receipt = guard.prepare(self.root, self.build)
        guard.write_json(self.receipt, receipt)
        return receipt

    def check_xml(self, root: ET.Element) -> dict:
        ET.ElementTree(root).write(self.xml, encoding="utf-8", xml_declaration=True)
        return guard.validate_xml(self.receipt, self.xml)

    def test_real_binary_schema_and_complete_xml_pass_without_gcc(self) -> None:
        receipt = self.receipt_file()
        self.assertEqual(receipt["parser_threshold"], guard.HIT_BOUND)
        self.assertEqual(receipt["pairs"][0]["functions"], 1)
        result = self.check_xml(xml_document())
        self.assertTrue(result["passed"])
        self.assertEqual(result["line_records"], 4)
        note = guard.decode_notes(self.note.read_bytes())
        data = guard.decode_data(self.data.read_bytes(), note)
        self.assertEqual(guard.solve_flow(note["functions"][11], data["functions"][11])["entry_count"], 7)

    def test_explicit_zero_compression_comdat_and_absent_data(self) -> None:
        for raw in (data_bytes(compressed=True), data_bytes(empty=True)):
            with self.subTest(raw=raw[-12:]):
                notes = guard.decode_notes(notes_bytes())
                decoded = guard.decode_data(raw, notes)
                self.assertEqual(decoded["functions"][11]["arcs"], [0])
        self.data.unlink()
        receipt = self.receipt_file()
        self.assertTrue(receipt["pairs"][0]["unexecuted"])
        self.assertIsNone(receipt["pairs"][0]["data_identity"])
        self.data.write_bytes(data_bytes())
        with self.assertRaisesRegex(ValueError, "changed"):
            guard.read_receipt(self.receipt)

    def test_corrupt_raw_records_are_rejected(self) -> None:
        good = data_bytes()
        cases = {
            "unsupported magic": b"gcda" + good[4:],
            "unsupported ABI": good[:4] + b"*22B" + good[8:],
            "stamp": data_bytes(stamp=74),
            "negative signed64": data_bytes(-1),
            "truncated": good[:-1],
            "missing summary": good[:12] + good[28:],
            "missing counters": good[:-16],
            "unknown tag": good + record(0xABCD0000, b""),
            "duplicate function": good + record(guard.FUNCTION, words(11, 22, 33)),
            "duplicate counters": good + record(guard.COUNTERS, struct.pack("<q", 7)),
            "incorrect arc count": good[:-16] + record(guard.COUNTERS, struct.pack("<qq", 7, 0)),
            "oversized compressed count": good[:-16] + words(guard.COUNTERS, 0x80000000),
        }
        mismatch = bytearray(good)
        struct.pack_into("<I", mismatch, 40, 23)
        cases["line checksum"] = bytes(mismatch)
        for label, raw in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    guard.decode_data(raw, guard.decode_notes(notes_bytes()))

    def test_corrupt_notes_and_flow_are_rejected(self) -> None:
        good = notes_bytes()
        for raw in (good[:-4], b"xxxx" + good[4:], good + record(0xBAD00000, b"")):
            with self.subTest(raw=raw[-8:]):
                with self.assertRaises(ValueError):
                    guard.decode_notes(raw)
        note = guard.decode_notes(good)["functions"][11]
        impossible = copy.deepcopy(note)
        impossible["edges"].append({"origin": 2, "target": 0, "flags": 4})
        with self.assertRaisesRegex(ValueError, "Invalid reconstructed"):
            guard.solve_flow(impossible, {"arcs": [1, 2]})

    def test_empty_successor_record_is_valid_but_origin_still_checked(self) -> None:
        raw = notes_bytes() + record(guard.ARCS, words(1))
        self.note.write_bytes(raw)
        receipt = guard.prepare(self.root, self.build)
        self.assertEqual(receipt["pairs"][0]["flow_bounds"]["graphs_checked"], 1)
        for origin, label in ((1, "Repeated"), (3, "out-of-range")):
            with self.subTest(origin=origin):
                self.note.write_bytes(raw + record(guard.ARCS, words(origin)))
                with self.assertRaisesRegex(ValueError, "Coverage pair .*example.cpp.gcno.*" + label):
                    guard.prepare(self.root, self.build)

    def test_metadata_only_translation_unit_has_explicit_zero_inventory(self) -> None:
        header = b"oncg*41B" + words(73) + string("/fixture/build") + words(1)
        self.note.write_bytes(header)
        self.data.unlink()
        receipt = guard.prepare(self.root, self.build)
        self.assertTrue(receipt["pairs"][0]["metadata_only"])
        self.assertEqual(receipt["pairs"][0]["functions"], 0)
        self.assertEqual(receipt["pairs"][0]["flow_bounds"]["graphs_checked"], 0)
        self.data.write_bytes(b"adcg*41B" + words(73) + record(guard.SUMMARY, words(0, 0)))
        self.assertTrue(guard.prepare(self.root, self.build)["pairs"][0]["metadata_only"])
        for raw in (b"adcg*41B" + words(73) + record(guard.SUMMARY, words(1, 0)),
                    b"adcg*41B" + words(73) + record(guard.SUMMARY, words(0, 1)),
                    data_bytes()):
            with self.subTest(raw=raw[-12:]):
                self.data.write_bytes(raw)
                with self.assertRaises(ValueError):
                    guard.prepare(self.root, self.build)

    def test_only_reviewed_gcc_artificial_thunk_shape_has_implicit_exit(self) -> None:
        source = "/fixture/src/jyppx/example.cpp"
        function = (words(11, 0, 0) + string("_ZThn8_N7Example6methodEv") + words(1) + string(source)
                    + words(1, 1, 1, 1))
        self.note.write_bytes(b"oncg*41B" + words(73) + string("/fixture/build") + words(1)
            + record(guard.FUNCTION, function) + record(guard.BLOCKS, words(4))
            + record(guard.ARCS, words(0, 2, 4)) + record(guard.ARCS, words(2))
            + record(guard.LINES, words(2, 0) + string(source) + words(1, 0) + string("")))
        self.data.write_bytes(b"adcg*41B" + words(73) + record(guard.SUMMARY, words(1, 7))
            + record(guard.FUNCTION, words(11, 0, 0)) + record(guard.COUNTERS, struct.pack("<q", 7)))
        receipt = guard.prepare(self.root, self.build)
        proof = receipt["pairs"][0]["flow_bounds"]
        self.assertEqual(proof["maximum_ordinary_block"], 7)
        self.assertEqual(proof["maximum_ordinary_line_bound"], 7)
        self.assertEqual(proof["synthetic_terminal_edges"][0]["count"], 7)
        self.assertEqual((proof["synthetic_terminal_edges"][0]["origin"],
                          proof["synthetic_terminal_edges"][0]["target"]), (2, 1))
        notes = guard.decode_notes(self.note.read_bytes())
        data = guard.decode_data(self.data.read_bytes(), notes)
        note = notes["functions"][11]
        changes = (("artificial", 0), ("mangled_name", "ordinary_function"),
                   ("lineno_checksum", 1), ("cfg_checksum", 1), ("blocks", 5))
        for key, value in changes:
            with self.subTest(field=key):
                damaged = copy.deepcopy(note)
                damaged[key] = value
                with self.assertRaisesRegex(ValueError, "Unbalanced"):
                    guard.solve_flow(damaged, data["functions"][11], allow_negative=True)
        with self.assertRaisesRegex(ValueError, "ordinary bound"):
            guard.solve_flow(note, {"arcs": [guard.HIT_BOUND]}, allow_negative=True)

    def test_non_sha_raw_bound_is_not_raised(self) -> None:
        for hits in (guard.HIT_BOUND, guard.HIT_BOUND + 1, 65 * (guard.HIT_BOUND - 1)):
            with self.subTest(hits=hits):
                self.data.write_bytes(data_bytes(hits))
                with self.assertRaisesRegex(ValueError, "ordinary bound"):
                    guard.prepare(self.root, self.build)

    def test_every_original_identity_inventory_and_receipt_are_bound(self) -> None:
        self.receipt_file()
        for target in (self.data, self.note, self.source):
            with self.subTest(target=target.name):
                original = target.read_bytes()
                target.write_bytes(original + b"\0\0\0\0")
                with self.assertRaises(ValueError):
                    guard.read_receipt(self.receipt)
                target.write_bytes(original)
        # Recreate a receipt after restoring bytes, since mtime/ctime are also
        # deliberately bound and are not repaired by this test.
        self.receipt.unlink()
        self.receipt_file()
        extra = self.build / "extra.gcda"
        extra.write_bytes(data_bytes())
        with self.assertRaisesRegex(ValueError, "missing its notes"):
            guard.read_receipt(self.receipt)
        extra.unlink()
        raw = self.receipt.read_text(encoding="utf-8")
        self.receipt.write_text(raw.replace('"parser_threshold": 68719476736', '"parser_threshold": 0'), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed"):
            guard.read_receipt(self.receipt)

    def test_xml_method_and_class_records_are_each_checked(self) -> None:
        self.receipt_file()
        for value in ("-1", "+1", "1.0", "01", str(1 << 63), str(guard.HIT_BOUND)):
            for parent in ("./packages/package/classes/class/lines/line",
                           "./packages/package/classes/class/methods/method/lines/line"):
                with self.subTest(value=value, parent=parent):
                    root = xml_document()
                    root.find(parent).set("hits", value)
                    with self.assertRaises(ValueError):
                        self.check_xml(root)

    def test_xml_missing_duplicate_unknown_and_unsafe_records_reject(self) -> None:
        self.receipt_file()
        mutations = {
            "missing method lines": lambda c: c.find("methods/method").remove(c.find("methods/method/lines")),
            "missing aggregate lines": lambda c: c.remove(c.find("lines")),
            "unsafe filename": lambda c: c.set("filename", "src/jyppx/../example.cpp"),
            "absolute filename": lambda c: c.set("filename", "/src/jyppx/example.cpp"),
            "unknown filename": lambda c: c.set("filename", "src/jyppx/unknown.cpp"),
            "unknown record": lambda c: ET.SubElement(c, "unknown"),
            "duplicate line": lambda c: c.find("lines").append(copy.deepcopy(c.find("lines/line"))),
            "duplicate method": lambda c: c.find("methods").append(copy.deepcopy(c.find("methods/method"))),
            "missing aggregate record": lambda c: c.find("lines").remove(c.find("lines/line")),
            "out of range line": lambda c: c.find("lines/line").set("number", "4"),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                root = xml_document()
                mutate(root.find("./packages/package/classes/class"))
                with self.assertRaises(ValueError):
                    self.check_xml(root)
        root = xml_document()
        classes = root.find("./packages/package/classes")
        classes.append(copy.deepcopy(classes[0]))
        with self.assertRaises(ValueError):
            self.check_xml(root)

    def test_template_method_counts_need_not_equal_aggregate(self) -> None:
        self.receipt_file()
        root = xml_document()
        item = root.find("./packages/package/classes/class")
        second = copy.deepcopy(item.find("methods/method"))
        second.set("name", "example_other_template")
        second.find("lines/line").set("hits", "9")
        item.find("methods").append(second)
        for line in second.find("lines"):
            item.find("lines").append(copy.deepcopy(line))
        root.set("lines-valid", "4")
        root.set("lines-covered", "2")
        self.assertTrue(self.check_xml(root)["passed"])

    def test_class_only_records_are_counted_and_checked(self) -> None:
        self.receipt_file()
        root = xml_document()
        ET.SubElement(root.find("./packages/package/classes/class/lines"), "line", number="3", hits="1")
        root.set("lines-valid", "3")
        root.set("lines-covered", "2")
        self.assertEqual(self.check_xml(root)["class_only_records"], 1)

    def test_sha_transform_method_and_every_loop_record_are_required(self) -> None:
        self.receipt_file()
        receipt = guard.read_receipt(self.receipt)
        # A synthetic trusted receipt isolates the XML inventory contract.
        # Real SHA binary-pair admission is verified outside this unit suite;
        # no sealed profile or machine-specific path is checked into the repo.
        expected = {90: 51, **dict.fromkeys(range(91, 96), 48), 97: 147,
                    **dict.fromkeys(range(98, 105), 144), 114: 195,
                    **dict.fromkeys(range(115, 132), 192)}
        receipt["sha"] = {"expected_lines": {str(line): hits for line, hits in expected.items()}}
        receipt["sources"][guard.SHA_SOURCE] = {"lines": 200}
        root = ET.Element("coverage", {"lines-valid": "32", "lines-covered": "32"})
        classes = ET.SubElement(ET.SubElement(ET.SubElement(root, "packages"), "package"), "classes")
        item = ET.SubElement(classes, "class", filename=guard.SHA_SOURCE)
        methods = ET.SubElement(item, "methods")
        method = ET.SubElement(methods, "method", name="jyppx::liuding::Sha256::Impl::transform",
                               signature="(std::byte const*)")
        for parent in (method, item):
            lines = ET.SubElement(parent, "lines")
            for number, hits in expected.items():
                ET.SubElement(lines, "line", number=str(number), hits=str(hits))
        with patch.object(guard, "read_receipt", return_value=receipt):
            self.assertTrue(self.check_xml(root)["passed"])
            absent_method = copy.deepcopy(root)
            methods_copy = absent_method.find("./packages/package/classes/class/methods")
            methods_copy.remove(methods_copy[0])
            with self.assertRaisesRegex(ValueError, "transform-method"):
                self.check_xml(absent_method)
            absent_line = copy.deepcopy(root)
            lines_copy = absent_line.find("./packages/package/classes/class/methods/method/lines")
            lines_copy.remove(lines_copy.find("line[@number='97']"))
            with self.assertRaisesRegex(ValueError, "transform-method"):
                self.check_xml(absent_line)

    def test_negative_tree_warning_contract_and_positive_derived_limit(self) -> None:
        notes = guard.decode_notes(notes_bytes())
        note = notes["functions"][11]
        note["edges"].append({"origin": 2, "target": 0, "flags": 4})
        data = {"functions": {11: {"arcs": [1, 2]}}}
        result = guard.ordinary_flow_bound(notes, data, admitted_sha=False)
        self.assertEqual(result["negative_tree_arcs_preserved"], 2)
        self.assertEqual([edge["count"] for edge in result["negative_tree_arcs"]], [-1, -1])
        # Each original arc is below H; their inferred entry/block sum is H.
        note["blocks"] = 4
        note["edges"] = [{"origin": 0, "target": 2, "flags": 1}, {"origin": 0, "target": 3, "flags": 1},
                         {"origin": 2, "target": 1, "flags": 4}, {"origin": 3, "target": 1, "flags": 4}]
        with self.assertRaisesRegex(ValueError, "Derived non-SHA"):
            guard.ordinary_flow_bound(notes, {"functions": {11: {"arcs": [guard.HIT_BOUND - 1, 1]}}}, admitted_sha=False)

    def test_entities_and_empty_documents_reject(self) -> None:
        self.receipt_file()
        for raw in (b"<coverage/>", b'<!DOCTYPE coverage [<!ENTITY n "1">]><coverage/>'
                    , b"<coverage><packages/></coverage>"):
            with self.subTest(raw=raw):
                self.xml.write_bytes(raw)
                with self.assertRaises(ValueError):
                    guard.validate_xml(self.receipt, self.xml)

    def test_cli_threshold_and_fail_closed_exit_are_reviewable(self) -> None:
        result = subprocess.run([sys.executable, "-I", "-B", str(SOURCE / "scripts/validation/coverage_hits.py"),
                                 "prepare", "--source-root", str(self.root), "--build-dir", str(self.build),
                                 "--output", str(self.receipt)], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, str(guard.HIT_BOUND) + "\n")
        self.xml.write_text("<coverage/>", encoding="utf-8")
        rejected = subprocess.run([sys.executable, "-I", "-B", str(SOURCE / "scripts/validation/coverage_hits.py"),
                                   "validate-xml", "--receipt", str(self.receipt), "--xml", str(self.xml),
                                   "--output", str(self.root / "proof.json")], capture_output=True, text=True, timeout=20)
        self.assertEqual(rejected.returncode, 2)
        self.assertFalse((self.root / "proof.json").exists())

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink support")
    def test_counter_symlinks_reject_when_supported(self) -> None:
        link = self.build / "linked.gcno"
        try:
            link.symlink_to(self.note)
        except OSError:
            self.skipTest("symlink creation not permitted on this platform")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            guard.prepare(self.root, self.build)


if __name__ == "__main__":
    unittest.main()
