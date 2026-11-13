# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Fault diagnostics in native dual-input classification/regression training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from test_image_failures import ImageFailureDiagnostics, sha


class PairedFailureDiagnostics(ImageFailureDiagnostics):
    fault_environment = "JYPPX_LIUDING_TEST_PAIRED_DIAGNOSTIC_FAULT"
    run_id = "run.paired-affine"

    @classmethod
    def expected_model_identity(cls, lineage):
        contract = {"recipe": "jyppx.liuding.paired-affine/v1", "left_features": 2,
                    "right_features": 2, "classification_outputs": 2, "regression_outputs": 1}
        return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @classmethod
    def setUpClass(cls):
        cls.commands = []
        cls.cases = {}
        cls.evidence = {"source": "native paired-affine preparation; bounded diagnostic subset",
                        "not_full_training_or_quality_qualification": True}
        cls.base = Path(tempfile.mkdtemp(prefix="paired-failure-", dir=cls.work_root))
        if cls.preparation_root:
            preparation_root = cls.preparation_root
        else:
            temporary = tempfile.TemporaryDirectory(prefix="paired-native-data-")
            cls.addClassCleanup(temporary.cleanup)
            preparation_root = Path(temporary.name)
        response, _ = cls.invoke("data", "prepare", "--name", "paired-affine",
                                  "--data-root", preparation_root)
        data = Path(response["data"]["destination"])
        selections = []
        for role, count in (("train", 64), ("validation", 32)):
            source = data / (role + ".jsonl")
            selected = b"".join(source.read_bytes().splitlines(keepends=True)[:count])
            target = cls.base / (role + ".jsonl")
            target.write_bytes(selected)
            if len(selected.splitlines()) != count:
                raise AssertionError("native prepared dataset has too few records")
            selections.append({"source": str(source), "source_sha256": sha(source),
                               "selected": target.name, "selected_sha256": sha(target), "rows": count})
        cls.evidence["input_selections"] = selections
        config = {"schema_version": 1, "recipe": "jyppx.liuding.paired-affine/v1",
                  "train_file": "train.jsonl", "validation_file": "validation.jsonl",
                  "data_license": "Apache-2.0; self-authored synthetic example data",
                  "device": "cpu", "seed": 20261028, "epochs": 2, "batch_size": 32,
                  "workers": 0, "prefetch_batches": 1, "learning_rate": .1,
                  "momentum": .9, "weight_decay": 0., "lr_gamma": .95, "regression_weight": 1.,
                  "checkpoint_every": 1, "minimum_accuracy": .95, "maximum_mse": .005,
                  "max_samples": 4096, "max_record_bytes": 512,
                  "max_data_bytes": 1048576, "max_batch_bytes": 65536}
        cls.config = cls.base / "config.json"
        cls.config.write_text(json.dumps(config), encoding="utf-8")
        cls.prefix = cls.base / "prefix"
        value, _ = cls.invoke("recipe", "run", "--file", cls.config, "--root", cls.prefix, "--pause-after", 1)
        if value["data"]["status"] != "paused" or value["data"]["global_step"] != 1:
            raise AssertionError("finite dual-input prefix did not persist one successful update")
        cls.evidence["prefix_checkpoint_sha256"] = sha(cls.prefix / "latest.jyck")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--preparation-root", type=Path)
    args = parser.parse_args()
    temporary = None
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        root = args.work_dir.resolve()
    else:
        temporary = tempfile.TemporaryDirectory(prefix="paired-failure-diagnostics-")
        root = Path(temporary.name)
    PairedFailureDiagnostics.cli = args.cli.resolve()
    PairedFailureDiagnostics.work_root = root
    preparation = args.preparation_root or os.environ.get("JYPPX_LIUDING_TEST_PAIRED_PREPARATION_ROOT")
    PairedFailureDiagnostics.preparation_root = Path(preparation).resolve() if preparation else None
    try:
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(PairedFailureDiagnostics))
        passed = result.wasSuccessful() and result.testsRun == 4 and not result.skipped
        print(json.dumps({"passed": passed, "tests_run": result.testsRun,
                          "skipped": len(result.skipped), "work_root": str(root)}))
        return 0 if passed else 1
    finally:
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
