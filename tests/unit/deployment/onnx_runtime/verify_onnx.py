#!/usr/bin/env python3
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
"""Build independent official ONNX fixtures for the optional C++ runtime gate.

This file is test-only. Production ONNX export is implemented in the native
onnx_export C++ component. These official reference graphs exercise runtime
acceptance/rejection independently of that exporter.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import subprocess
import sys
import tempfile
import traceback
import xml.etree.ElementTree as ET

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def artifact(root, payload, name, media="application/octet-stream"):
    identity = hashlib.sha256(payload).hexdigest()
    path = root / "artifacts" / identity
    if path.exists() and path.read_bytes() != payload:
        raise ValueError("test artifact collision")
    path.write_bytes(payload)
    return {"digest": {"algorithm": "sha256", "hex": identity},
            "size_bytes": len(payload), "logical_name": name,
            "media_type": media, "producer": "independent-runtime-test"}


def official_negative_graphs(root):
    import numpy as np
    import onnx
    from onnx import TensorProto as T, helper as h, numpy_helper as nh

    def model(name, nodes, inputs, outputs, initializers=(), custom=False):
        graph = h.make_graph(nodes, name, inputs, outputs, list(initializers))
        ops = [h.make_opsetid("", 18)]
        if custom:
            ops.append(h.make_opsetid("ai.liuding.test", 1))
        result = h.make_model(graph, opset_imports=ops, ir_version=10,
                              producer_name="liuding-independent-test")
        if name != "external_data":
            onnx.checker.check_model(result, full_check=True)
        return artifact(root, result.SerializeToString(), name, "application/onnx")

    def info(name, dtype=T.FLOAT, shape=("batch", 3)):
        return h.make_tensor_value_info(name, dtype, shape)

    negative = {"malformed": artifact(root, b"not ONNX", "malformed")}
    negative["missing_operator"] = model("missing_operator", [h.make_node("MissingOperator", ["input"], ["output"], domain="ai.liuding.test")], [info("input")], [info("output")], custom=True)
    negative["float64"] = model("float64", [h.make_node("Identity", ["input"], ["output"])], [info("input", T.DOUBLE)], [info("output", T.DOUBLE)])
    negative["mixed_output_dtypes"] = model("mixed_output_dtypes", [h.make_node("Identity", ["input"], ["output"]), h.make_node("Cast", ["input"], ["double_output"], to=T.DOUBLE)], [info("input")], [info("output"), info("double_output", T.DOUBLE)])
    negative["unbound_output_batch"] = model("unbound_output_batch", [h.make_node("Unique", ["input"], ["output"])], [info("input", shape=("batch",))], [info("output", shape=("unique_count",))])
    negative["zero_inputs"] = model("zero_inputs", [h.make_node("Constant", [], ["output"], value=nh.from_array(np.ones((1, 3), dtype=np.float32)))], [], [info("output", shape=(1, 3))])
    negative["dynamic_width"] = model("dynamic_width", [h.make_node("Identity", ["input"], ["output"])], [info("input", shape=("batch", "width"))], [info("output", shape=("batch", "width"))])
    negative["nonfinite_output"] = model("nonfinite_output", [h.make_node("Div", ["input", "zero"], ["output"])], [info("input")], [info("output")], [nh.from_array(np.array(0, dtype=np.float32), name="zero")])
    negative["multi_nonfinite_output"] = model("multi_nonfinite_output", [h.make_node("Identity", ["input"], ["finite"]), h.make_node("Div", ["input", "zero"], ["nonfinite"])], [info("input")], [info("finite"), info("nonfinite")], [nh.from_array(np.array(0, dtype=np.float32), name="zero")])
    external = h.make_tensor("external", T.FLOAT, (3,), [], raw=False)
    external.data_location = T.EXTERNAL
    item = external.external_data.add()
    item.key, item.value = "location", "missing-external-data.bin"
    negative["external_data"] = model("external_data", [h.make_node("Add", ["input", "external"], ["output"])], [info("input")], [info("output")], [external])
    nodes = [h.make_node("MatMul", ["input" if index == 0 else f"hidden{index - 1}", "weight"], ["output" if index == 31 else f"hidden{index}"]) for index in range(32)]
    slow = model("slow-cancellable-cpu", nodes, [info("input", shape=("batch", 512))], [info("output", shape=("batch", 512))], [nh.from_array(np.eye(512, dtype=np.float32), name="weight")])
    return negative, slow


def official_multi_io_models(root):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import safetensors.numpy
    from onnx import TensorProto as T, helper as h

    def info(name, symbol, width):
        return h.make_tensor_value_info(name, T.FLOAT, [symbol, width])

    def reference(name, nodes, inputs, outputs, cases):
        graph = h.make_graph(nodes, name, inputs, outputs)
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)],
                             ir_version=10, producer_name="liuding-independent-test")
        onnx.checker.check_model(model, full_check=True)
        onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        payload = model.SerializeToString()
        session = ort.InferenceSession(payload, providers=["CPUExecutionProvider"])
        refs = []
        for index, case in enumerate(cases):
            actual = session.run(None, {value.name: case[value.name]
                                        for value in inputs})
            for output, result in zip(outputs, actual, strict=True):
                np.testing.assert_array_equal(result, case[output.name])
            refs.append(artifact(root, safetensors.numpy.save(case),
                                 name + f"-case-{index}", "application/safetensors"))
        return {"artifact": artifact(root, payload, name, "application/onnx"),
                "cases": refs}

    cases = []
    for batch in (1, 2, 5):
        left = (np.arange(batch * 3, dtype=np.float32).reshape(batch, 3) - 3) / 4
        right = np.arange(batch, dtype=np.float32).reshape(batch, 1) + 2
        cases.append({"left": left, "right": right,
                      "sum": left + right, "product": left * right})
    shared = reference("multi_io", [h.make_node("Add", ["left", "right"], ["sum"]),
                                     h.make_node("Mul", ["left", "right"], ["product"])],
                       [info("left", "batch", 3), info("right", "batch", 1)],
                       [info("sum", "batch", 3), info("product", "batch", 3)], cases)
    left = np.arange(6, dtype=np.float32).reshape(2, 3)
    right = np.arange(5, dtype=np.float32).reshape(5, 1) - 7
    independent = reference("independent_batches", [h.make_node("Identity", ["left"], ["left_output"]),
                                                    h.make_node("Identity", ["right"], ["right_output"])],
                            [info("left", "left_batch", 3), info("right", "right_batch", 1)],
                            [info("left_output", "left_batch", 3), info("right_output", "right_batch", 1)],
                            [{"left": left, "right": right,
                              "left_output": left, "right_output": right}])
    return {"multi_io": shared, "independent_batches": independent}


def official_models(root):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import safetensors.torch
    import torch
    import torchvision

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    models = {}
    for index, name in enumerate(("linear", "resnet18")):
        torch.manual_seed(1729 + index)
        model = torch.nn.Linear(8, 3) if name == "linear" else torchvision.models.resnet18(weights=None, num_classes=7)
        model.eval()
        dimensions = [8] if name == "linear" else [3, 224, 224]
        state = safetensors.torch.save({key: value.detach().contiguous()
                                       for key, value in model.state_dict().items()})
        state_ref = artifact(root, state, name + "-reference-state", "application/safetensors")
        cases, tensors = [], []
        for batch in (2, 5):
            values = torch.randn([batch, *dimensions])
            with torch.no_grad():
                output = model(values).contiguous()
            tensors.append((values, output))
            cases.append(artifact(root, safetensors.torch.save({"input": values, "output": output}), name + f"-batch-{batch}", "application/safetensors"))
        program = torch.export.export(model, (tensors[0][0],),
                                      dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=8)},), strict=True)
        path = root / (name + "-official-reference.onnx")
        torch.onnx.export(program, f=path, opset_version=18, dynamo=True, external_data=False,
                          input_names=["input"], output_names=["output"])
        payload = path.read_bytes()
        graph = onnx.load_model_from_string(payload)
        onnx.checker.check_model(graph, full_check=True)
        onnx.shape_inference.infer_shapes(graph, strict_mode=True, data_prop=True)
        session = ort.InferenceSession(payload, providers=["CPUExecutionProvider"])
        for values, expected in tensors:
            actual = session.run(None, {session.get_inputs()[0].name: values.numpy()})[0]
            np.testing.assert_allclose(actual, expected.numpy(), atol=1e-5, rtol=1e-4)
        models[name] = {"artifact": artifact(root, payload, name + "-official-reference", "application/onnx"),
                        "state": state_ref, "cases": cases}
        print(name + " independent reference/checker/ORT passed", flush=True)
    return models


def official_cpu_only_tree(root):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import safetensors.numpy
    from onnx import TensorProto as T, helper as h

    node = h.make_node(
        "TreeEnsembleRegressor", ["input"], ["output"], domain="ai.onnx.ml",
        n_targets=1, aggregate_function="SUM", post_transform="NONE",
        nodes_treeids=[0, 0, 0], nodes_nodeids=[0, 1, 2],
        nodes_featureids=[0, 0, 0],
        nodes_modes=["BRANCH_LEQ", "LEAF", "LEAF"],
        nodes_values=[0.0, 0.0, 0.0], nodes_truenodeids=[1, 0, 0],
        nodes_falsenodeids=[2, 0, 0], target_treeids=[0, 0],
        target_nodeids=[1, 2], target_ids=[0, 0], target_weights=[1.0, 2.0])
    graph = h.make_graph(
        [node], "cpu-only-tree", [h.make_tensor_value_info("input", T.FLOAT, ["batch", 1])],
        [h.make_tensor_value_info("output", T.FLOAT, ["batch", 1])])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18), h.make_opsetid("ai.onnx.ml", 3)],
                         ir_version=10, producer_name="liuding-independent-test")
    onnx.checker.check_model(model, full_check=True)
    payload = model.SerializeToString()
    values = np.array([[-1.0], [0.0], [2.0]], dtype=np.float32)
    expected = np.array([[1.0], [1.0], [2.0]], dtype=np.float32)
    session = ort.InferenceSession(payload, providers=["CPUExecutionProvider"])
    np.testing.assert_array_equal(session.run(None, {"input": values})[0], expected)
    return {"artifact": artifact(root, payload, "cpu-only-tree", "application/onnx"),
            "cases": [artifact(root, safetensors.numpy.save({"input": values, "output": expected}),
                               "cpu-only-tree-case", "application/safetensors")]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp-executable", type=pathlib.Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--require-native-models", action="store_true",
                        help="Require both same-state native Linear/ResNet18 comparisons")
    parser.add_argument("--require-cuda", action="store_true",
                        help="Fail closed unless the C++ CUDA kernel, device and fallback gates pass")
    parser.add_argument("--cuda-device", type=int, default=0,
                        help="Explicit C++ CUDA device ordinal for --require-cuda")
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--environment-lock", type=pathlib.Path,
                        help="Optional independent environment lock to identify in evidence")
    parser.add_argument("--jyppx-commit", required=True)
    args = parser.parse_args()
    if not args.prepare_only and args.cpp_executable is None:
        parser.error("C++ executable is required unless --prepare-only")
    if not 0 <= args.cuda_device <= 2147483647:
        parser.error("--cuda-device must be a nonnegative int32 ordinal")
    if args.cuda_device != 0 and not args.require_cuda:
        parser.error("--cuda-device requires --require-cuda")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    root = pathlib.Path(tempfile.mkdtemp(prefix="onnx-runtime-reference-", dir=args.work_dir.resolve()))
    (root / "artifacts").mkdir()
    report = {"schema_version": 1, "source_commit": args.jyppx_commit,
              "platform": platform.platform(), "python": platform.python_version(),
              "environment_lock_sha256": (hashlib.sha256(args.environment_lock.read_bytes()).hexdigest()
                                          if args.environment_lock else None),
              "versions": {name: importlib.metadata.version(name)
                           for name in ("torch", "torchvision", "onnx", "onnxruntime", "safetensors")},
              "production_exporter_claimed": False, "cpp_gate": "not-run",
              "cuda_gate": "not-run" if args.require_cuda else "not-requested",
              "cuda_device": args.cuda_device if args.require_cuda else None,
              "cuda_execution_verified": False, "cuda_execution_scope": [],
              "cuda_performance_qualified": False,
              "other_cuda_models_qualified": False}
    try:
        models = official_models(root)
        models.update(official_multi_io_models(root))
        if args.require_cuda:
            models["cpu_only_tree"] = official_cpu_only_tree(root)
        negative, slow = official_negative_graphs(root)
        (root / "manifest.json").write_bytes(encoded({"schema_version": 1, "models": models,
                                                      "negative": negative, "slow": slow}))
        if not args.prepare_only:
            env = dict(os.environ)
            env["JYPPX_LIUDING_ONNX_FIXTURE_ROOT"] = str(root)
            test_filter = "*" if args.require_cuda else "-OnnxRuntimeCudaExecution.*"
            if args.require_cuda:
                env["JYPPX_LIUDING_ONNX_CUDA_DEVICE"] = str(args.cuda_device)
            else:
                env.pop("JYPPX_LIUDING_ONNX_CUDA_DEVICE", None)
            result = subprocess.run([str(args.cpp_executable.resolve()),
                                     "--gtest_output=xml:" + str(root / "cpp-results.xml"),
                                     "--gtest_filter=" + test_filter],
                                    env=env, capture_output=True, timeout=600, check=False)
            (root / "cpp.stdout").write_bytes(result.stdout)
            (root / "cpp.stderr").write_bytes(result.stderr)
            report["cpp_exit_code"] = result.returncode
            if result.returncode:
                raise RuntimeError("independent C++ runtime gate failed")
            xml = ET.parse(root / "cpp-results.xml").getroot()
            report["cpp_tests"] = {key: int(xml.attrib.get(key, "0")) for key in ("tests", "failures", "disabled", "errors")}
            required_suites = {"OnnxRuntimeSession": 16, "OnnxRuntimeMultiIo": 8,
                               "OnnxRuntimeCudaOptions": 3}
            if args.require_native_models:
                required_suites["OnnxRuntimeInterchange"] = 2
            if args.require_cuda:
                required_suites["OnnxRuntimeCudaExecution"] = 4
            suites = {suite.attrib["name"]: int(suite.attrib["tests"])
                      for suite in xml.findall("testsuite")}
            report["required_suites"] = required_suites
            if any(suites.get(name, 0) < count for name, count in required_suites.items()) or any(report["cpp_tests"][key] for key in ("failures", "disabled", "errors")) or xml.findall(".//skipped"):
                raise RuntimeError("independent runtime tests failed, were skipped or omitted")
            report["cpp_gate"] = "passed"
            if args.require_cuda:
                report["cuda_gate"] = "passed"
                report["cuda_execution_verified"] = True
                report["cuda_execution_scope"] = [
                    "fp32-linear-dynamic-batch-2-and-5",
                    "fp32-named-multi-io-concurrent-owned-results",
                    "invalid-device-no-primary-provider-fallback",
                    "explicit-cpu-only-tree-partition-policy"]
                report["cuda_profiles"] = [
                    {"path": str(path.relative_to(root)),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in sorted(root.glob("native-cuda-*.json"))]
        report["status"] = "prepared-cpp-not-run" if args.prepare_only else "passed"
        (root / "validation.json").write_bytes(encoded(report))
        print(json.dumps({"status": report["status"], "work_dir": str(root)}), flush=True)
        return 0
    except Exception:
        report["status"] = "failed"
        (root / "validation.json").write_bytes(encoded(report))
        (root / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
