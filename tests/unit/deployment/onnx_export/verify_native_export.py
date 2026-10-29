#!/usr/bin/env python3
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
"""Independently check ONNX artifacts produced by the native C++ test binary."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import subprocess
import tempfile
import traceback
import xml.etree.ElementTree as ET


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False), encoding="utf-8")


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def read_tensor(root, specification):
    import numpy as np
    if specification["dtype"] != "float32" or specification["byte_order"] != "little":
        raise ValueError("unsupported native evidence tensor")
    name = pathlib.Path(specification["path"])
    if name.name != str(name):
        raise ValueError("evidence tensor name must be a local basename")
    payload = (root / name).read_bytes()
    if sha256(payload) != specification["sha256"]:
        raise ValueError("native evidence tensor identity mismatch")
    shape = specification["shape"]
    if not shape or any(type(extent) is not int or extent <= 0 for extent in shape):
        raise ValueError("invalid native evidence tensor shape")
    return np.frombuffer(payload, dtype="<f4").reshape(shape).copy()


def verify_model(root, name):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import torchvision
    from onnx import numpy_helper

    folder = root / name
    evidence = json.loads((folder / "native-evidence.json").read_text(encoding="utf-8"))
    payload = (folder / "model.onnx").read_bytes()
    if (evidence["production_exporter"] != "native-cpp" or
            evidence["model"] != name or evidence["external_data"] or
            sha256(payload) != evidence["onnx_sha256"]):
        raise ValueError("native graph provenance/identity mismatch")
    graph = onnx.load_model_from_string(payload)
    if [(item.domain, item.version) for item in graph.opset_import] != [("", 17)]:
        raise ValueError("native exporter did not use the fixed opset 17")
    if any(item.data_location == onnx.TensorProto.EXTERNAL or item.external_data
           for item in graph.graph.initializer):
        raise ValueError("native graph has external tensor data")
    if (len(graph.graph.input) != 1 or len(graph.graph.output) != 1 or
            graph.graph.input[0].name != "input" or graph.graph.output[0].name != "output"):
        raise ValueError("native graph has an unexpected public I/O signature")
    operators = sorted({"onnx::" + item.op_type for item in graph.graph.node})
    if any(item.domain for item in graph.graph.node) or operators != evidence["operators"]:
        raise ValueError("native graph operator inventory mismatch")
    onnx.checker.check_model(graph, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(graph, strict_mode=True, data_prop=True)
    onnx.checker.check_model(inferred, full_check=True)
    initializers = {item.name: torch.from_numpy(numpy_helper.to_array(item).copy())
                    for item in graph.graph.initializer}
    if name == "linear":
        model = torch.nn.Linear(8, 3)
    elif name == "novel":
        class Novel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first = torch.nn.Linear(8, 5)
                self.second = torch.nn.Linear(5, 3)

            def forward(self, values):
                return torch.tanh(self.second(torch.relu(self.first(values))))
        model = Novel()
    else:
        model = torchvision.models.resnet18(weights=None, num_classes=7)
    initializers = {key.removeprefix("state."): value
                    for key, value in initializers.items()}
    state = model.state_dict()
    expected_keys = {key for key in state if not key.endswith("num_batches_tracked")}
    if set(initializers) != expected_keys:
        raise ValueError("native graph initializer inventory differs from official model")
    state.update(initializers)
    model.load_state_dict(state, strict=True)
    model.eval()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    session = ort.InferenceSession(payload, sess_options=options,
                                   providers=["CPUExecutionProvider"])
    checked = []
    for case in evidence["cases"]:
        values = read_tensor(folder, case["input"])
        expected = read_tensor(folder, case["native_output"])
        with torch.no_grad():
            official = model(torch.from_numpy(values)).contiguous().numpy()
        actual = session.run(["output"], {"input": values})[0]
        if actual.shape != expected.shape or official.shape != expected.shape:
            raise ValueError("native/official/runtime output shape differs")
        if not np.isfinite(actual).all() or not np.isfinite(official).all():
            raise ValueError("independent output is nonfinite")
        np.testing.assert_allclose(official, expected, atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
        checked.append({"batch": values.shape[0],
                        "native_vs_official_max_abs": float(np.abs(official - expected).max()),
                        "native_vs_ort_max_abs": float(np.abs(actual - expected).max())})
    if [item["batch"] for item in checked] != [2, 5]:
        raise ValueError("both fixed-example and dynamic-batch cases are required")
    return {"onnx_sha256": sha256(payload), "checker": "passed",
            "strict_shape_inference": "passed", "operators": operators,
            "state_source": "exact-native-onnx-initializers",
            "cases": checked, "source_task_accuracy_claimed": False}


def verify_pair(root):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from onnx import numpy_helper

    folder = root / "pair"
    evidence = json.loads((folder / "native-evidence.json").read_text(encoding="utf-8"))
    payload = (folder / "model.onnx").read_bytes()
    if sha256(payload) != evidence["onnx_sha256"]:
        raise ValueError("multi-I/O native graph identity mismatch")
    graph = onnx.load_model_from_string(payload)
    onnx.checker.check_model(graph, full_check=True)
    onnx.shape_inference.infer_shapes(graph, strict_mode=True, data_prop=True)
    if ([value.name for value in graph.graph.input] != ["left", "right"] or
            [value.name for value in graph.graph.output] != ["sum_logits", "difference_logits"]):
        raise ValueError("multi-I/O public names changed")
    state = {value.name.removeprefix("state.layer."):
             torch.from_numpy(numpy_helper.to_array(value).copy())
             for value in graph.graph.initializer}
    if set(state) != {"weight", "bias"}:
        raise ValueError("multi-I/O state inventory changed")
    layer = torch.nn.Linear(8, 3)
    layer.load_state_dict(state, strict=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(payload, sess_options=options, providers=["CPUExecutionProvider"])
    cases = []
    for case in evidence["cases"]:
        left, right = [read_tensor(folder, item) for item in case["inputs"]]
        native = [read_tensor(folder, item) for item in case["native_outputs"]]
        actual = session.run(None, {"left": left, "right": right})
        with torch.no_grad():
            official = [layer(torch.from_numpy(left) + torch.from_numpy(right)).numpy(),
                        layer(torch.from_numpy(left) - torch.from_numpy(right)).numpy()]
        for current, oracle, expected in zip(actual, official, native, strict=True):
            np.testing.assert_allclose(current, expected, atol=1e-5, rtol=1e-4)
            np.testing.assert_allclose(oracle, expected, atol=1e-5, rtol=1e-4)
        cases.append({"batch": left.shape[0], "all_outputs_match": True})
    if [item["batch"] for item in cases] != [2, 5]:
        raise ValueError("multi-I/O dynamic cases missing")
    return {"onnx_sha256": sha256(payload), "checker": "passed",
            "strict_shape_inference": "passed", "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp-executable", type=pathlib.Path)
    parser.add_argument("--native-artifacts", type=pathlib.Path,
                        help="Verify an already preserved native test artifact directory")
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--jyppx-commit", required=True)
    parser.add_argument("--require-cpp-ort", action="store_true")
    args = parser.parse_args()
    if (args.cpp_executable is None) == (args.native_artifacts is None):
        parser.error("specify exactly one of --cpp-executable or --native-artifacts")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    root = pathlib.Path(tempfile.mkdtemp(prefix="native-onnx-", dir=args.work_dir.resolve()))
    artifacts = args.native_artifacts.resolve() if args.native_artifacts else root / "artifacts"
    report = {"schema_version": 1, "source_commit": args.jyppx_commit,
              "platform": platform.platform(), "python": platform.python_version(),
              "production_exporter": "native-cpp", "python_role": "independent-checker-oracle",
              "versions": {key: importlib.metadata.version(key)
                           for key in ("torch", "torchvision", "onnx", "onnxruntime", "numpy")},
              "cpp_gate": "not-run-here", "full_v1_onnx_claimed": False}
    try:
        if args.cpp_executable:
            env = dict(os.environ)
            env["JYPPX_LIUDING_ONNX_EXPORT_EVIDENCE_DIR"] = str(artifacts)
            command = [str(args.cpp_executable.resolve()),
                       "--gtest_output=xml:" + str(root / "cpp-results.xml")]
            report["cpp_command"] = command
            result = subprocess.run(command, env=env, capture_output=True, timeout=600, check=False)
            (root / "cpp.stdout").write_bytes(result.stdout)
            (root / "cpp.stderr").write_bytes(result.stderr)
            report["cpp_exit_code"] = result.returncode
            if result.returncode:
                raise RuntimeError("native C++ ONNX tests failed; diagnostics preserved")
            xml = ET.parse(root / "cpp-results.xml").getroot()
            report["cpp_tests"] = {key: int(xml.attrib.get(key, "0"))
                                   for key in ("tests", "failures", "errors", "disabled")}
            minimum = 14 if args.require_cpp_ort else 7
            if (report["cpp_tests"]["tests"] < minimum or
                    any(report["cpp_tests"][key] for key in ("failures", "errors", "disabled")) or
                    xml.findall(".//skipped")):
                raise RuntimeError("native tests were omitted, disabled, skipped or failed")
            if args.require_cpp_ort and len(xml.findall(".//testsuite[@name='NativeOnnxRuntime']/testcase")) < 7:
                raise RuntimeError("required native ORT tests were not executed")
            report["cpp_gate"] = "passed"
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        report["models"] = {name: verify_model(artifacts, name)
                            for name in ("linear", "resnet18", "novel")}
        report["models"]["pair"] = verify_pair(artifacts)
        report["native_artifacts"] = str(artifacts)
        report["status"] = "passed"
        write_json(root / "validation.json", report)
        print(json.dumps({"status": "passed", "work_dir": str(root)}), flush=True)
        return 0
    except Exception:
        report["status"] = "failed"
        write_json(root / "validation.json", report)
        (root / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(json.dumps({"status": "failed", "work_dir": str(root)}), flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
