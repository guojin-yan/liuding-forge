#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Independent official-schema/oracle checks of the native C++ INT8 exporter."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import onnx
import onnxruntime as ort

REPO = Path(__file__).resolve().parents[2]
PREPARE_TOOL: Path | None = None
EXPORT_TOOL: Path | None = None
EVIDENCE_ROOT: Path | None = None


class NativeStaticInt8ExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if PREPARE_TOOL is None or EXPORT_TOOL is None:
            raise RuntimeError("Native preparation/export binaries are required")
        cls.prepare_tool = PREPARE_TOOL.resolve(strict=True)
        cls.export_tool = EXPORT_TOOL.resolve(strict=True)
        if onnx.__version__ != "1.19.1" or ort.__version__ != "1.23.2":
            raise RuntimeError("Independent oracle requires ONNX1.19.1/ORT1.23.2")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="liuding-native-int8-")
        self.root = Path(self.temp.name)
        self.source = self.root / "source 中文.qint8"
        self.source.write_bytes((REPO / "tests/fixtures/fuzz/quantization/corpus/artifact-biased").read_bytes())
        self.output = self.root / "output 中文.onnx"

    def tearDown(self):
        try:
            if EVIDENCE_ROOT is not None:
                shutil.copytree(self.root, EVIDENCE_ROOT / self._testMethodName)
        finally:
            self.temp.cleanup()

    def export(self, output: Path | None = None):
        return subprocess.run([str(self.export_tool), str(self.source), str(output or self.output)],
                              capture_output=True, timeout=60)

    def prepare(self):
        result = subprocess.run([str(self.prepare_tool), str(self.source)], check=True,
                                capture_output=True, timeout=30)
        return json.loads(result.stdout.decode("utf-8"))

    def test_native_bytes_official_checker_and_identity(self):
        result = self.export()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout.decode("utf-8"))
        self.assertEqual(report["source_artifact_sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(report["onnx_sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assertFalse(report["compute_quantized"])
        self.assertFalse(report["qat"])
        model = onnx.load_model(self.output)
        onnx.checker.check_model(model, full_check=True)
        onnx.checker.check_model(onnx.shape_inference.infer_shapes(model, strict_mode=True), full_check=True)
        operations = {node.op_type for node in model.graph.node}
        self.assertIn("MatMulInteger", operations)
        self.assertIn("QuantizeLinear", operations)
        self.assertFalse(operations & {"Gemm", "MatMul", "DequantizeLinear"})
        self.assertEqual(model.opset_import[0].version, 17)
        self.assertEqual(model.graph.input[0].type.tensor_type.elem_type, onnx.TensorProto.FLOAT)
        self.assertEqual(model.graph.output[0].type.tensor_type.elem_type, onnx.TensorProto.FLOAT)
        initializers = {value.name: value for value in model.graph.initializer}
        self.assertEqual(initializers["weight_u8"].data_type, onnx.TensorProto.UINT8)
        self.assertEqual(initializers["weight_zero_point"].data_type, onnx.TensorProto.UINT8)
        self.assertEqual(onnx.numpy_helper.to_array(initializers["weight_zero_point"]).item(), 128)
        self.assertEqual(onnx.numpy_helper.to_array(initializers["activation_zero_point"]).item(), 128)
        (self.root / "checker-report.json").write_text(json.dumps({
            "onnx_sha256": report["onnx_sha256"],
            "source_artifact_sha256": report["source_artifact_sha256"],
            "ir_version": model.ir_version,
            "opset_import": [{"domain": item.domain, "version": item.version}
                             for item in model.opset_import],
            "operators": sorted(operations),
            "official_checker": "passed; full_check and strict shape inference",
            "execution_claim": "checker only; actual CPU trace is a separate test"
        }, indent=2), encoding="utf-8")

    def test_actual_cpu_integer_trace_and_independent_int64_oracle(self):
        metadata = self.prepare()
        result = self.export()
        self.assertEqual(result.returncode, 0, result.stderr)
        k, n = metadata["in_features"], metadata["out_features"]
        def scale(name):
            return np.float32(struct.unpack("<f", struct.pack("<I", metadata[name + "_bits"]))[0])
        activation = scale("activation_scale")
        product = scale("product_scale")
        weights = np.frombuffer(base64.b64decode(metadata["weights_u8_k_n_base64"]), dtype=np.uint8).reshape(k, n).astype(np.int64) - 128
        bias = np.frombuffer(base64.b64decode(metadata["bias_f32le_base64"]), dtype="<f4")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.enable_profiling = True
        options.profile_file_prefix = str(self.root / "actual-cpu")
        session = ort.InferenceSession(str(self.output), options, providers=["CPUExecutionProvider"])
        for batch in (1, 3, 7):
            inputs = np.resize(np.array([-1000, -128, -127, -3.5, -2.5, -1.5, -.5, 0, .5, 1.5, 2.5, 3.5, 127, 128, 1000],dtype=np.float32),(batch,k)) * activation
            quantized = np.clip(np.rint(inputs / activation) + 128, 0, 255).astype(np.uint8)
            accumulated = (quantized.astype(np.int64) - 128) @ weights
            expected = accumulated.astype(np.float32) * product
            if bias.size:
                expected = expected + bias
            actual = session.run(None, {session.get_inputs()[0].name: inputs})[0]
            np.testing.assert_array_equal(actual, expected)
        trace = json.loads(Path(session.end_profiling()).read_text(encoding="utf-8"))
        events = [event for event in trace if event.get("cat") == "Node" and event.get("args",{}).get("op_name") == "MatMulInteger"]
        self.assertTrue(events)
        self.assertTrue(all(event["args"].get("provider") == "CPUExecutionProvider" for event in events))
        self.assertFalse([event for event in trace if event.get("cat") == "Node" and event.get("args",{}).get("op_name") in {"MatMul","Gemm","FusedMatMul"}])

    def test_repeated_export_bytes_are_deterministic(self):
        result = self.export()
        self.assertEqual(result.returncode, 0, result.stderr)
        other = self.root / "other.onnx"
        repeated = self.export(other)
        self.assertEqual(repeated.returncode, 0, repeated.stderr)
        self.assertEqual(self.output.read_bytes(), other.read_bytes())

    def test_existing_output_is_never_overwritten(self):
        self.output.write_bytes(b"preserve-user-output")
        result = self.export()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.output.read_bytes(), b"preserve-user-output")
        self.assertFalse(Path(str(self.output) + ".liuding-int8-tmp").exists())

    def test_existing_temporary_file_is_never_overwritten_or_removed(self):
        temporary = Path(str(self.output) + ".liuding-int8-tmp")
        temporary.write_bytes(b"preserve-other-process")
        result = self.export()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(temporary.read_bytes(), b"preserve-other-process")
        self.assertFalse(self.output.exists())

    def test_concurrent_same_output_has_one_winner(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: self.export(), range(2)))
        self.assertEqual(sorted(result.returncode for result in results), [0,1])
        winner = json.loads(next(result.stdout for result in results if result.returncode == 0))
        self.assertEqual(hashlib.sha256(self.output.read_bytes()).hexdigest(),winner["onnx_sha256"])

    def test_truncated_source_fails_without_partial_output(self):
        self.source.write_bytes(self.source.read_bytes()[:-1])
        result = self.export()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout,b"")
        self.assertFalse(self.output.exists())
        self.assertFalse(Path(str(self.output)+".liuding-int8-tmp").exists())

    def test_wrong_version_and_trailing_bytes_fail(self):
        original = self.source.read_bytes()
        for content in (original[:8]+struct.pack("<I",2)+original[12:],original+b"trailing"):
            self.source.write_bytes(content)
            result = self.export()
            self.assertEqual(result.returncode,1)
            self.assertFalse(self.output.exists())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--prepare-tool", required=True,type=Path)
    parser.add_argument("--export-tool",required=True,type=Path)
    parser.add_argument("--evidence-directory",type=Path)
    options, remaining = parser.parse_known_args()
    PREPARE_TOOL, EXPORT_TOOL = options.prepare_tool,options.export_tool
    if options.evidence_directory is not None:
        EVIDENCE_ROOT = options.evidence_directory.resolve()
        EVIDENCE_ROOT.mkdir(parents=True, exist_ok=False)
        (EVIDENCE_ROOT / "environment.json").write_text(json.dumps({
            "python": sys.version,
            "onnx": onnx.__version__, "onnxruntime": ort.__version__,
            "numpy": np.__version__,
            "native_tools": [{"path": str(path.resolve(strict=True)),
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                             for path in (PREPARE_TOOL, EXPORT_TOOL)]}, indent=2), encoding="utf-8")
    unittest.main(argv=[sys.argv[0],*remaining])
