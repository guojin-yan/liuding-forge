#!/usr/bin/env python3
"""Keep the ordinary gcov hit bound while admitting reviewed SHA fixed loops.

This is a read-only GCC 11.4 (little-endian, word-length ABI) verifier, not a
counter repair tool. The parser threshold is transport only: raw counters and
the complete Cobertura line inventory are independently checked. An ignored
gcovr suspicious hit would become zero, so ignoring it is never an option.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import sys
import xml.etree.ElementTree as ET


HIT_BOUND = 1 << 36
SHA_SOURCE = "src/jyppx/liuding/foundation/sha256.cpp"
SHA_SOURCE_LF_SHA256 = "0cee80c9056728dda413a1332970edeeb37819e2780ed237a505291a3e7f37b9"
# These hashes describe the three reviewed graphs, including source line
# mappings. They exclude build-directory-dependent ids and line checksums;
# both of those must still match between the original notes and data.
SHA_GRAPHS = {
    "_ZN5jyppx7liuding6Sha2564Impl9transformEPKSt4byte":
        "87b8e8ce3e2434f072061c719c00d8d6478264adbebd2042164992e56dec8fd2",
    "_ZN5jyppx7liuding6Sha2564Impl6updateESt4spanIKSt4byteLm18446744073709551615EE":
        "2cf0996a9ab16fc62718b1c35eb3d49bff27a4c950161a504e673715b467189a",
    "_ZN5jyppx7liuding6Sha2564Impl6finishEv":
        "81fc2027520912491c2c6d94b1280baa4448186ea49a9c87e9a38ba6ee389e5d",
}
SCHEMA = "jyppx-coverage-hits/v1"
FUNCTION, BLOCKS, ARCS, LINES = 0x01000000, 0x01410000, 0x01430000, 0x01450000
COUNTERS, SUMMARY = 0x01A10000, 0xA1000000


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


class Reader:
    def __init__(self, raw: bytes, magic: bytes):
        require(len(raw) >= 12 and len(raw) % 4 == 0, "Truncated gcov header/words")
        require(raw[:8] == magic + b"*41B", "Unsupported gcov ABI (requires GCC 11.4 little-endian)")
        self.raw, self.offset = raw, 12
        self.stamp = struct.unpack_from("<I", raw, 8)[0]

    def u32(self, end: int | None = None) -> int:
        end = len(self.raw) if end is None else end
        require(self.offset + 4 <= end, "Truncated gcov word")
        result = struct.unpack_from("<I", self.raw, self.offset)[0]
        self.offset += 4
        return result

    def string(self, end: int) -> str:
        size = self.u32(end) * 4
        require(size <= end - self.offset, "Truncated gcov string")
        raw = self.raw[self.offset:self.offset + size]
        self.offset += size
        if not size:
            return ""
        text, separator, padding = raw.partition(b"\0")
        require(bool(separator) and not any(padding), "Invalid gcov string padding")
        value = text.decode("utf-8")
        require(not any(char in value for char in "\r\n\0"), "Unsafe gcov string")
        return value

    def record(self, *, compressed: bool = False) -> tuple[int, int, int]:
        tag = self.u32()
        length = self.u32()
        if length >= 1 << 31:
            require(compressed and tag == COUNTERS, "Unexpected negative gcov record length")
            length -= 1 << 32
        end = self.offset + max(0, length) * 4
        require(end <= len(self.raw), "Truncated gcov record")
        return tag, length, end


def decode_notes(raw: bytes) -> dict:
    reader = Reader(raw, b"oncg")
    cwd = reader.string(len(raw))
    require(reader.u32() in (0, 1), "Invalid notes support-unexecuted flag")
    functions: dict[int, dict] = {}
    current = None
    while reader.offset < len(raw):
        tag, words, end = reader.record()
        if tag == FUNCTION:
            ident, line, cfg = (reader.u32(end) for _ in range(3))
            require(ident not in functions, "Repeated notes function id")
            name = reader.string(end)
            artificial = reader.u32(end)
            source = reader.string(end)
            require(bool(name) and bool(source) and artificial in (0, 1), "Invalid notes function")
            location = [reader.u32(end) for _ in range(4)]
            current = {"mangled_name": name, "lineno_checksum": line, "cfg_checksum": cfg,
                       "source": source, "artificial": artificial, "source_range": location,
                       "edges": [], "block_lines": {}, "origins": set()}
            functions[ident] = current
        else:
            require(current is not None, "Notes record without function")
            if tag == BLOCKS:
                require(words == 1 and "blocks" not in current, "Repeated/invalid notes blocks")
                current["blocks"] = reader.u32(end)
                require(2 <= current["blocks"] <= len(raw) // 4, "Invalid notes block count")
            elif tag == ARCS:
                # GCC writes an origin even when this block has no successor:
                # the record length is 1 + 2*N words, including valid N=0.
                require("blocks" in current and words >= 1 and words % 2 == 1, "Invalid notes arcs")
                origin = reader.u32(end)
                require(origin < current["blocks"] and origin not in current["origins"],
                        "Repeated/out-of-range arc origin")
                current["origins"].add(origin)
                targets: set[int] = set()
                while reader.offset < end:
                    target, flags = reader.u32(end), reader.u32(end)
                    require(target < current["blocks"] and target not in targets and not flags & ~7,
                            "Repeated/out-of-range/unsupported notes arc")
                    targets.add(target)
                    current["edges"].append({"origin": origin, "target": target, "flags": flags})
            elif tag == LINES:
                require("blocks" in current, "Notes lines without blocks")
                block = reader.u32(end)
                require(block < current["blocks"] and block not in current["block_lines"],
                        "Repeated/out-of-range notes line block")
                lines, source = [], None
                while reader.offset < end:
                    number = reader.u32(end)
                    if number == 0:
                        source = reader.string(end)
                    else:
                        require(bool(source), "Notes line without source")
                        lines.append([source, number])
                current["block_lines"][block] = lines
            else:
                raise ValueError(f"Unsupported notes tag {tag:#x}")
        require(reader.offset == end, "Invalid notes record length")
    # A generated metadata-only translation unit can contain just the valid
    # header, cwd and support flag. Its data must independently be absent or
    # an explicit all-zero summary; it is not an omitted instrumented unit.
    for function in functions.values():
        require("blocks" in function and bool(function["edges"]), "Incomplete notes function")
        del function["origins"]
    return {"stamp": reader.stamp, "cwd": cwd, "functions": functions}


def decode_data(raw: bytes, notes: dict) -> dict:
    reader = Reader(raw, b"adcg")
    require(reader.stamp == notes["stamp"], "Notes/data stamp mismatch")
    expected = list(notes["functions"].items())
    functions, summary, current = {}, None, None
    while reader.offset < len(raw):
        # GCC files may finish with a null tag, or a zero-length null record.
        if len(raw) - reader.offset in (4, 8) and not any(raw[reader.offset:]):
            reader.offset = len(raw)
            break
        tag, words, end = reader.record(compressed=True)
        if tag == SUMMARY:
            require(words == 2 and summary is None and not functions, "Repeated/invalid data summary")
            summary = {"runs": reader.u32(end), "sum_max": reader.u32(end)}
        elif tag == FUNCTION:
            require(current is None or "arcs" in current, "Missing function arc counters")
            require(len(functions) < len(expected) and words in (0, 3), "Invalid/repeated data function")
            ident, note = expected[len(functions)]
            if words:
                actual, line, cfg = (reader.u32(end) for _ in range(3))
                require((actual, line, cfg) == (ident, note["lineno_checksum"], note["cfg_checksum"]),
                        "Notes/data function order or checksum mismatch")
                current = {"lineno_checksum": line, "cfg_checksum": cfg}
            else:
                # A discarded COMDAT function has an explicit empty slot.
                current = {"lineno_checksum": note["lineno_checksum"], "cfg_checksum": note["cfg_checksum"],
                           "arcs": [0] * sum(not edge["flags"] & 1 for edge in note["edges"])}
            functions[ident] = current
        elif tag == COUNTERS:
            require(current is not None and "arcs" not in current and words % 2 == 0,
                    "Repeated/unattached/odd data counters")
            note = expected[len(functions) - 1][1]
            count = sum(not edge["flags"] & 1 for edge in note["edges"])
            require(abs(words) == 2 * count, "Notes/data instrumented arc count mismatch")
            if words < 0:
                current["arcs"] = [0] * count
            else:
                values = struct.unpack_from("<" + "q" * count, raw, reader.offset)
                require(all(value >= 0 for value in values), "Negative signed64 raw arc")
                current["arcs"] = list(values)
                reader.offset = end
        else:
            raise ValueError(f"Unsupported data tag {tag:#x}")
        require(reader.offset == end, "Invalid data record length")
    require(summary is not None and len(functions) == len(expected) and
            all("arcs" in row for row in functions.values()), "Incomplete notes/data function inventory")
    if not expected:
        require(summary == {"runs": 0, "sum_max": 0}, "Nonzero metadata-only data summary")
    return {"stamp": reader.stamp, "summary": summary, "functions": functions}


def zero_data(notes: dict) -> dict:
    return {"stamp": notes["stamp"], "summary": None, "functions": {
        ident: {"lineno_checksum": function["lineno_checksum"], "cfg_checksum": function["cfg_checksum"],
                "arcs": [0] * sum(not edge["flags"] & 1 for edge in function["edges"])}
        for ident, function in notes["functions"].items()}}


def gcc_thunk_terminal(note: dict, data: dict) -> bool:
    """Recognize GCC's minimal artificial nonvirtual tail-call thunk graph.

    GCC 11.4 emits the measured entry and a source-bearing terminal block,
    without an exit edge. gcov.c:2136-2150 deliberately treats function entry
    and exit independently; its no-successor-first rule (2252-2257) reports
    the terminal block as zero. Completing this exact shape with entry-count
    flow is a conservative bound, not a replacement for GCC report counts.
    """
    location = note["source_range"]
    return (re.match(r"_ZThn[1-9][0-9]*_", note["mangled_name"]) is not None
            and note["artificial"] == 1 and note["lineno_checksum"] == note["cfg_checksum"] == 0
            and note["blocks"] == 4 and note["edges"] == [{"origin": 0, "target": 2, "flags": 4}]
            and len(data["arcs"]) == 1 and location[0] > 0 and location[1] > 0
            and location[:2] == location[2:]
            and note["block_lines"] == {2: [[note["source"], location[0]]]})


def solve_flow(note: dict, data: dict, *, allow_negative: bool = False) -> dict:
    """Reconstruct tree arcs, checking every block, without invoking GCC."""
    edges = [dict(edge) for edge in note["edges"]]
    terminal = gcc_thunk_terminal(note, data)
    if terminal:
        require(0 <= data["arcs"][0] < HIT_BOUND, "Artificial thunk entry exceeds ordinary bound")
        # Only the private proof graph receives this implicit tail-call exit.
        # Original counter/notes bytes and gcov's own output stay untouched.
        edges.append({"origin": 2, "target": 1, "flags": 1})
    values = iter(data["arcs"])
    for edge in edges:
        if not edge["flags"] & 1:
            edge["count"] = next(values)
    edges.append({"origin": 1, "target": 0, "flags": 1})
    incoming, outgoing = [[] for _ in range(note["blocks"])], [[] for _ in range(note["blocks"])]
    for index, edge in enumerate(edges):
        outgoing[edge["origin"]].append(index)
        incoming[edge["target"]].append(index)
    pending = deque(range(note["blocks"]))
    while pending:
        block = pending.popleft()
        ins, outs = incoming[block], outgoing[block]
        missing_in = [index for index in ins if "count" not in edges[index]]
        missing_out = [index for index in outs if "count" not in edges[index]]
        known_in = sum(edges[index].get("count", 0) for index in ins)
        known_out = sum(edges[index].get("count", 0) for index in outs)
        if not missing_in and len(missing_out) == 1:
            index, value = missing_out[0], known_in - known_out
        elif not missing_out and len(missing_in) == 1:
            index, value = missing_in[0], known_out - known_in
        else:
            continue
        require(-(1 << 63) <= value < 1 << 63 and (allow_negative or value >= 0),
                "Invalid reconstructed signed64 tree arc")
        edges[index]["count"] = value
        pending.extend((edges[index]["origin"], edges[index]["target"]))
    require(all("count" in edge for edge in edges), "Unresolved notes spanning tree")
    counts = []
    for ins, outs in zip(incoming, outgoing):
        total = sum(edges[index]["count"] for index in ins)
        require(total == sum(edges[index]["count"] for index in outs), "Unbalanced control-flow graph")
        require(-(1 << 63) <= total < 1 << 63 and (allow_negative or total >= 0), "Block exceeds signed64")
        counts.append(total)
    return {"entry_count": counts[0], "blocks": counts, "edges": [edge["count"] for edge in edges],
            "synthetic_terminal_edges": [{"origin": 2, "target": 1, "count": data["arcs"][0],
                                           "reason": "gcc11-artificial-nonvirtual-tailcall-thunk"}] if terminal else []}


def ordinary_flow_bound(notes: dict, data: dict, *, admitted_sha: bool) -> dict:
    """Bound function, branch, block, and line inputs to gcovr independently.

    GCC's unreachable generated paths can reconstruct negative tree counts;
    leave their existing gcovr warning behavior intact. Positive counts must
    still satisfy H. A sum of positive block visits is a conservative upper
    bound for a gcov source line, including overlapping inline definitions.
    """
    line_bounds: dict[tuple[str, int], int] = {}
    maximum_edge = maximum_block = 0
    negative_arcs = []
    synthetic_terminals = []
    for ident, note in notes["functions"].items():
        transform = admitted_sha and note["mangled_name"] == next(iter(SHA_GRAPHS))
        flow = solve_flow(note, data["functions"][ident], allow_negative=not admitted_sha)
        for edge in flow["synthetic_terminal_edges"]:
            synthetic_terminals.append({"function_id": ident, "function": note["mangled_name"], **edge})
        edge_schema = [*note["edges"], *flow["synthetic_terminal_edges"], {"origin": 1, "target": 0, "flags": 1}]
        for edge, value in zip(edge_schema, flow["edges"]):
            if value < 0:
                negative_arcs.append({"function_id": ident, "origin": edge["origin"],
                                      "target": edge["target"], "count": value})
        if not transform:
            maximum_edge = max(maximum_edge, max(flow["edges"], default=0))
            maximum_block = max(maximum_block, max(flow["blocks"], default=0))
            require(maximum_edge < HIT_BOUND and maximum_block < HIT_BOUND,
                    "Derived non-SHA edge/block exceeds ordinary bound")
        for block, lines in note["block_lines"].items():
            for source, number in set((source, number) for source, number in lines):
                if transform and number in {90, *range(91, 96), 97, *range(98, 105), 114, *range(115, 132)}:
                    continue
                key = (source, number)
                line_bounds[key] = line_bounds.get(key, 0) + max(0, flow["blocks"][block])
                require(line_bounds[key] < HIT_BOUND, "Conservative non-SHA line bound reaches ordinary limit")
    return {"maximum_ordinary_edge": maximum_edge, "maximum_ordinary_block": maximum_block,
            "maximum_ordinary_line_bound": max(line_bounds.values(), default=0),
            "negative_tree_arcs_preserved": len(negative_arcs), "negative_tree_arcs": negative_arcs,
            "synthetic_terminal_edges": synthetic_terminals,
            "graphs_checked": len(notes["functions"])}


def is_sha_source(value: str) -> bool:
    return value == SHA_SOURCE or value.endswith("/" + SHA_SOURCE)


def validate_sha(notes: dict, data: dict) -> dict:
    selected = {}
    for ident, note in notes["functions"].items():
        name = note["mangled_name"]
        if name in SHA_GRAPHS:
            require(name not in selected and is_sha_source(note["source"]), "Duplicate/misplaced SHA function")
            require(all(is_sha_source(source) for lines in note["block_lines"].values() for source, _ in lines),
                    "Unexpected SHA graph source mapping")
            shape = {key: note[key] for key in ("mangled_name", "artificial", "cfg_checksum", "source_range", "blocks", "edges")}
            shape["block_lines"] = {str(block): [line for _, line in lines]
                                    for block, lines in note["block_lines"].items()}
            require(digest(canonical(shape)) == SHA_GRAPHS[name], "Unreviewed SHA control-flow graph")
            selected[name] = (note, data["functions"][ident])
    require(set(selected) == set(SHA_GRAPHS), "Missing reviewed SHA functions")
    # The entire SHA translation unit must have a nonnegative, conserved flow.
    flows = {ident: solve_flow(note, data["functions"][ident]) for ident, note in notes["functions"].items()}
    rows = list(SHA_GRAPHS)
    t, u, f = (selected[name][1]["arcs"] for name in rows)
    require((len(t), len(u), len(f)) == (4, 13, 8), "Unexpected SHA arc schema")
    total = t[0]
    require(0 <= total < HIT_BOUND and t == [total, 16 * total, 48 * total, 64 * total],
            "Invalid SHA entry or fixed loop factors")
    require(u[0] == u[1] == u[2] + u[3] and u[3] == u[9] + u[7] and
            u[5] == u[6] == u[7] + u[8] and u[10] == u[11], "Invalid SHA update flow")
    require(f[1] == f[2] and f[3] == f[5] == 8 * f[0] and f[6] == f[7] == 32 * f[0],
            "Invalid SHA finish/padding flow")
    require(total == u[8] + u[10] + f[0] + f[1], "Invalid cross-function SHA transform count")
    for ident, note in notes["functions"].items():
        if note["mangled_name"] == rows[0]:
            continue
        require(all(value < HIT_BOUND for value in data["functions"][ident]["arcs"]), "Non-transform raw arc exceeds ordinary bound")
        require(flows[ident]["entry_count"] < HIT_BOUND, "SHA nonloop entry exceeds ordinary bound")
    factors = {90: 17, **{line: 16 for line in range(91, 96)},
               97: 49, **{line: 48 for line in range(98, 105)},
               114: 65, **{line: 64 for line in range(115, 132)}}
    return {"transform_calls": total, "expected_lines": {str(line): factor * total for line, factor in factors.items()},
            "functions_flow_checked": len(flows)}


def safe_path(path: Path, *, directory: bool = False) -> Path:
    require(".." not in path.parts, f"Unsafe parent path: {path}")
    path = path.absolute()
    for ancestor in (path, *path.parents):
        require(not ancestor.is_symlink() and not (hasattr(ancestor, "is_junction") and ancestor.is_junction()),
                f"Symlink/junction path is not admitted: {path}")
    require(path.is_dir() if directory else path.is_file(), f"Missing {'directory' if directory else 'file'}: {path}")
    return path.resolve(strict=True)


def read_file(path: Path) -> tuple[bytes, dict]:
    safe_path(path)
    before = path.stat()
    require(stat.S_ISREG(before.st_mode), f"Not a regular file: {path}")
    with path.open("rb") as stream:
        raw = stream.read()
        opened = os.fstat(stream.fileno())
    after = path.stat()
    fields = ("st_size", "st_mtime_ns", "st_dev", "st_ino", "st_mode")
    # On Windows some Python versions expose creation time via stat.ctime but
    # metadata change time via fstat.ctime. Compare each API to itself there.
    same_ctime = before.st_ctime_ns == after.st_ctime_ns
    if os.name != "nt":
        same_ctime = same_ctime and before.st_ctime_ns == opened.st_ctime_ns
    require(same_ctime and all(getattr(before, name) == getattr(opened, name) == getattr(after, name) for name in fields)
            and len(raw) == before.st_size, f"File changed while reading: {path}")
    return raw, {"bytes": len(raw), "sha256": digest(raw), "mtime_ns": before.st_mtime_ns,
                 "ctime_ns": before.st_ctime_ns, "device": before.st_dev, "ino": before.st_ino,
                 "mode": before.st_mode}


def inventory(root: Path, suffixes: tuple[str, ...] | None = None) -> list[Path]:
    files = []
    for directory, directories, names in os.walk(root, followlinks=False):
        for name in directories:
            require(not (Path(directory) / name).is_symlink(), f"Symlink directory in inventory: {directory}/{name}")
        for name in names:
            if suffixes is None or name.endswith(suffixes):
                files.append(safe_path(Path(directory) / name))
    return sorted(files)


def prepare(source_root: Path, build_dir: Path) -> dict:
    source_root, build_dir = safe_path(source_root, directory=True), safe_path(build_dir, directory=True)
    sources = {}
    for subtree in ("src/jyppx", "include/jyppx"):
        directory = safe_path(source_root / subtree, directory=True)
        for path in inventory(directory):
            raw, identity = read_file(path)
            sources[path.relative_to(source_root).as_posix()] = {**identity, "lines": len(raw.splitlines())}
    require(bool(sources), "No selected project sources")
    files = inventory(build_dir, (".gcda", ".gcno"))
    notes_paths = [path for path in files if path.suffix == ".gcno"]
    require(bool(notes_paths), "No original coverage notes")
    file_set, pairs, sha = set(files), [], None
    require(all(path.with_suffix(".gcno") in file_set for path in files if path.suffix == ".gcda"),
            "Original data file is missing its notes pair")
    for path in notes_paths:
        try:
            note_raw, note_identity = read_file(path)
            notes = decode_notes(note_raw)
            data_path = path.with_suffix(".gcda")
            if data_path in file_set:
                data_raw, data_identity = read_file(data_path)
                data = decode_data(data_raw, notes)
            else:
                data_identity, data = None, zero_data(notes)
            contains_sha = any(is_sha_source(note["source"]) and note["mangled_name"] in SHA_GRAPHS
                               for note in notes["functions"].values())
            if contains_sha:
                require(sha is None and SHA_SOURCE in sources, "Duplicate SHA object or missing SHA source")
                raw, _ = read_file(source_root / SHA_SOURCE)
                require(digest(raw.replace(b"\r\n", b"\n")) == SHA_SOURCE_LF_SHA256, "Unreviewed SHA source")
                sha = {"notes": path.relative_to(build_dir).as_posix(), **validate_sha(notes, data)}
            else:
                require(all(value < HIT_BOUND for row in data["functions"].values() for value in row["arcs"]),
                        "Raw arc exceeds ordinary bound outside reviewed SHA")
            flow_bounds = ordinary_flow_bound(notes, data, admitted_sha=contains_sha)
            pairs.append({"notes": path.relative_to(build_dir).as_posix(), "notes_identity": note_identity,
                          "data_identity": data_identity, "stamp": notes["stamp"],
                          "functions": len(notes["functions"]), "unexecuted": data_identity is None,
                          "metadata_only": not notes["functions"],
                          "flow_bounds": flow_bounds})
        except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration, struct.error) as error:
            raise ValueError(f"Coverage pair {path}: {error}") from error
    require(inventory(build_dir, (".gcda", ".gcno")) == files, "Original pair inventory changed during preparation")
    maximum = 65 * sha["transform_calls"] if sha else 0
    return {"schema": SCHEMA, "source_root": str(source_root), "build_dir": str(build_dir),
            "ordinary_bound_exclusive": HIT_BOUND, "parser_threshold": max(HIT_BOUND, maximum + 1),
            "sources": sources, "pairs": pairs, "sha": sha}


def read_receipt(path: Path) -> dict:
    raw, _ = read_file(path)
    def unique(pairs: list) -> dict:
        result = {}
        for key, value in pairs:
            require(key not in result, f"Repeated receipt key: {key}")
            result[key] = value
        return result
    receipt = json.loads(raw, object_pairs_hook=unique)
    require(isinstance(receipt, dict) and receipt.get("schema") == SCHEMA, "Unknown guard receipt")
    require(isinstance(receipt.get("source_root"), str) and isinstance(receipt.get("build_dir"), str),
            "Missing guard roots")
    require(receipt == prepare(Path(receipt["source_root"]), Path(receipt["build_dir"])),
            "Original counter, source, inventory, or guard receipt changed")
    return receipt


def normalized_filename(value: str) -> str:
    path = PurePosixPath(value)
    require(bool(value) and "\\" not in value and not path.is_absolute() and ".." not in path.parts
            and path.as_posix() == value and ":" not in value, f"Unsafe XML filename: {value!r}")
    require(value.startswith(("src/jyppx/", "include/jyppx/")), f"Unselected XML source: {value}")
    return value


def decimal(value: str | None, label: str, *, positive: bool = False) -> int:
    require(isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value) is not None,
            f"Invalid nonnegative integer {label}: {value!r}")
    number = int(value)
    require(number < 1 << 63 and (not positive or number > 0), f"Out-of-range {label}")
    return number


def validate_xml(receipt_path: Path, xml_path: Path) -> dict:
    receipt = read_receipt(receipt_path)
    raw, xml_identity = read_file(xml_path)
    # The standard gcovr SYSTEM Cobertura doctype is harmless and unexpanded.
    # Internal subsets/entities introduce a second document language: refuse it.
    require(b"<!ENTITY" not in raw.upper(), "Unsupported XML entity")
    if b"<!DOCTYPE" in raw:
        require(b"[" not in raw.split(b"<!DOCTYPE", 1)[1].split(b">", 1)[0], "Unsupported XML doctype subset")
    root = ET.fromstring(raw)
    require(root.tag == "coverage", "Expected Cobertura coverage root")
    require(all(element.tag in {"coverage", "sources", "source", "packages", "package", "classes", "class",
                                "methods", "method", "lines", "line", "conditions", "condition"}
                for element in root.iter()), "Unknown XML record")
    packages = root.find("packages")
    require(packages is not None and len(root.findall("packages")) == 1, "Missing/repeated XML packages")
    files, records, high, seen_sha, seen_sha_method = set(), 0, 0, set(), set()
    class_records = class_covered = class_only_records = 0
    all_lines = set(root.iter("line"))
    visited_lines = set()
    for package in packages:
        require(package.tag == "package", "Unknown XML package record")
        classes = package.find("classes")
        require(classes is not None and len(package.findall("classes")) == 1, "Missing/repeated XML classes")
        for item in classes:
            require(item.tag == "class", "Unknown XML class record")
            filename = normalized_filename(item.get("filename", ""))
            require(filename not in files and filename in receipt["sources"], "Duplicate/unknown XML source")
            files.add(filename)
            lines = item.find("lines")
            require(lines is not None and len(item.findall("lines")) == 1, "Missing/repeated XML class lines")
            expected = receipt["sha"]["expected_lines"] if receipt["sha"] and filename == SHA_SOURCE else {}

            def record_key(element: ET.Element) -> tuple:
                return (element.tag, tuple(sorted(element.attrib.items())), tuple(record_key(child) for child in element))

            def check_lines(container: ET.Element, *, transform: bool = False, method: bool = False) -> Counter:
                nonlocal records, high, class_records, class_covered
                seen, inventory = set(), Counter()
                for line in container:
                    require(line.tag == "line", "Unknown XML line record")
                    number = decimal(line.get("number"), "line number", positive=True)
                    hits = decimal(line.get("hits"), "line hits")
                    require((not method or number not in seen) and number <= receipt["sources"][filename]["lines"],
                            f"Duplicate/out-of-range XML source line: {filename}:{number}")
                    seen.add(number)
                    visited_lines.add(line)
                    records += 1
                    inventory[record_key(line)] += 1
                    if not method:
                        class_records += 1
                        class_covered += hits > 0
                    if str(number) in expected:
                        require(not method or transform, "SHA loop attributed to an unknown XML method")
                        require(hits == expected[str(number)], "SHA XML count differs from exact raw loop count")
                        if method:
                            seen_sha_method.add(str(number))
                        else:
                            require(str(number) not in seen_sha, "Repeated SHA aggregate loop record")
                            seen_sha.add(str(number))
                    elif hits >= HIT_BOUND:
                        raise ValueError(f"Unreviewed high XML hit: {filename}:{number}={hits}")
                    if hits >= HIT_BOUND:
                        high += 1
                require(bool(seen), "Empty XML line container")
                return inventory

            class_inventory = check_lines(lines)
            methods = item.find("methods")
            require(methods is not None and len(item.findall("methods")) == 1, "Missing/repeated XML methods")
            method_keys, method_inventory = set(), Counter()
            for method in methods:
                require(method.tag == "method", "Unknown XML method record")
                key = (method.get("name"), method.get("signature"))
                require(isinstance(key[0], str) and bool(key[0]) and isinstance(key[1], str) and key not in method_keys,
                        "Missing/repeated XML method identity")
                method_keys.add(key)
                method_lines = method.find("lines")
                require(method_lines is not None and len(method.findall("lines")) == 1, "Missing/repeated XML method lines")
                method_inventory.update(check_lines(method_lines, method=True,
                    transform=key == ("jyppx::liuding::Sha256::Impl::transform", "(std::byte const*)")))
            # Template instantiations may legitimately have different hits at
            # the same source line. Check every record rather than requiring
            # a false equality between method and aggregate counts.
            require(not method_inventory - class_inventory, "XML method records missing from class inventory")
            class_only_records += sum((class_inventory - method_inventory).values())
    require(records > 0 and all_lines == visited_lines, "Empty or unaccounted XML line inventory")
    require(class_records == decimal(root.get("lines-valid"), "root lines-valid") and
            class_covered == decimal(root.get("lines-covered"), "root lines-covered"),
            "Incomplete/duplicate XML aggregate line inventory")
    if receipt["sha"]:
        expected_sha = set(receipt["sha"]["expected_lines"])
        require(SHA_SOURCE in files and seen_sha == expected_sha and seen_sha_method == expected_sha,
                "Missing SHA XML aggregate or transform-method loop records")
    # Recheck after parsing too, so a concurrent mutation cannot validate a
    # report against the identity observed only before reading it.
    require(read_receipt(receipt_path) == receipt, "Guard inputs changed during XML validation")
    return {"schema": SCHEMA + "/xml", "passed": True, "xml": xml_identity,
            "receipt_sha256": digest(read_file(receipt_path)[0]), "files": len(files),
            "line_records": records, "reviewed_high_records": high,
            "class_line_records": class_records, "class_only_records": class_only_records,
            "ordinary_bound_exclusive": HIT_BOUND, "parser_threshold": receipt["parser_threshold"]}


def write_json(path: Path, value: dict) -> None:
    require(path.suffix == ".json", "Guard output must be a JSON file")
    parent = safe_path(path.parent, directory=True)
    require(not path.is_symlink(), "Guard output must not be a symlink")
    # Never overwrite any original input, including an existing source JSON.
    with (parent / path.name).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    first = commands.add_parser("prepare")
    first.add_argument("--source-root", type=Path, required=True)
    first.add_argument("--build-dir", type=Path, required=True)
    first.add_argument("--output", type=Path, required=True)
    second = commands.add_parser("validate-xml")
    second.add_argument("--receipt", type=Path, required=True)
    second.add_argument("--xml", type=Path, required=True)
    second.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.source_root, args.build_dir)
            write_json(args.output, result)
            print(result["parser_threshold"])
        else:
            result = validate_xml(args.receipt, args.xml)
            write_json(args.output, result)
            print(json.dumps(result, sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError, UnicodeError, struct.error, ET.ParseError) as error:
        print(f"coverage hit guard: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
