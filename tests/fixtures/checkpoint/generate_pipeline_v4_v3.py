# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
"""Generate independent no-transform v4/v3 binary-format fixtures.

The independent encoding has also been compared byte-for-byte with the actual
v4/v3 C++ writer at BASELINE (LibTorch 2.14.1+cu126, Ubuntu). Preserve historical
v3/v2 fixtures and their provenance verbatim.
"""
from pathlib import Path
import base64
import hashlib
import json
import struct

ROOT = Path(__file__).resolve().parent
BASELINE = "22e88a8a095a87e0d19d4964c58d64042f6bcf98"

def u64(n):
    return struct.pack("<Q", n)

def string(s):
    b = s.encode() if isinstance(s, str) else s
    return u64(len(b)) + b

def integer_schema(identity):
    value = string("JYPPX_VALUE_SCHEMA_V1")
    value += bytes([5, 1, 0, 0]) + string("batch") + string("sequence_or_sum")
    value += bytes([0]) + u64(0) + u64(0)
    return string("JYPPX_BATCH_SCHEMA_V1") + string(identity) + struct.pack("<I", 1) + string(value)

def skip_string(b, pos):
    return pos + 8 + struct.unpack_from("<Q", b, pos)[0]

def make(old_dir, new_dir, old_version, prefix, schema_id, is_map):
    source = ROOT / old_dir
    target = ROOT / new_dir
    target.mkdir(exist_ok=True)
    original = json.loads((source / "metadata.json").read_text())
    fields = []
    for name in ("topology", "cursor"):
        data = base64.b64decode((source / (name + ".bin.b64")).read_text().strip(), validate=True)
        data = data.replace((prefix + "-" + name.upper() + "/v" + str(old_version)).encode(),
                            (prefix + "-" + name.upper() + "/v" + str(old_version + 1)).encode(), 1)
        if name == "topology":
            pos = 0
            for _ in range(3):
                pos = skip_string(data, pos)
            if is_map:
                pos += 8
            end = skip_string(data, pos)
            data = data[:pos] + string(integer_schema(schema_id)) + data[end:]
        (target / (name + ".bin.b64")).write_text(base64.b64encode(data).decode() + "\n", encoding="ascii")
        fields.append({"name": name, "media_type": "application/octet-stream", "required": True,
                       "file": name + ".bin.b64", "encoding": "base64",
                       "max_bytes": (32 * 1024 * 1024 if is_map else 2 * 1024 * 1024) if name == "topology" else (128 if is_map else 8 * 1024 * 1024),
                       "decoded_size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    metadata = {
        "fixture_schema": 1,
        "participant_schema": old_version + 1,
        "participant_id": original["participant_id"],
        "restore_level": original["restore_level"],
        "scenario": original["scenario"],
        "producer": {"jyppx_version": "0.3.0", "git_commit": BASELINE,
                     "libtorch_version": "2.14.1+cu126",
                     "writer": ("DataPipeline" if is_map else "IterableDataPipeline") + " schema-v" + str(old_version + 1) + " writer; independently reconstructed byte-for-byte"},
        "provenance": "Independent binary-format fixture; byte-identical historical C++ writer cross-check",
        "historical_writer_crosscheck": {"git_commit": BASELINE, "libtorch_version": "2.14.1+cu126", "platform": "Ubuntu", "all_blobs_byte_identical": True},
        "format_reference_commit": BASELINE,
        "derived_source_fixture": old_dir,
        "schema": {"id": schema_id, "version": 1, "kind": "integer", "name": "batch", "semantic_tag": "sequence_or_sum"},
        "fields": fields,
    }
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

make("data_pipeline_schema_v3_libtorch_2.5.1", "data_pipeline_schema_v4_spec", 3,
     "JYPPX-DATA-PIPELINE", "worker.batch", True)
for mode in ("exact", "bounded", "best_effort"):
    make("iterable_pipeline_schema_v2_" + mode + "_libtorch_2.5.1",
         "iterable_pipeline_schema_v3_" + mode + "_spec", 2,
         "JYPPX-ITERABLE-PIPELINE", "batch.iterable", False)
