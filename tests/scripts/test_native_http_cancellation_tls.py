# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Established HTTPS cancellation with a private Linux default-trust view."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trust_state(path):
    state = path.stat()
    return (digest(path), state.st_dev, state.st_ino, state.st_mtime_ns, state.st_ctime_ns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("harness", "work-dir", "system-ca-bundle"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--inside", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(sys.platform == "linux", "Private default-trust cancellation gate requires Linux")
    args.harness = args.harness.resolve(strict=True)
    args.system_ca_bundle = args.system_ca_bundle.resolve(strict=True)
    checker = Path(__file__).with_name("test_native_http_cancellation.py").resolve(strict=True)
    certificate_fixture = Path(__file__).with_name("test_native_otlp_trusted_tls.py").resolve(strict=True)
    inputs = {str(path): digest(path) for path in (args.harness, checker, certificate_fixture, Path(__file__).resolve())}
    environment = dict(os.environ)
    for key in ("SSL_CERT_FILE", "SSL_CERT_DIR", "CURL_CA_BUNDLE", "OPENSSL_CONF"):
        environment.pop(key, None)
    if args.inside:
        directory = args.inside.resolve(strict=True).parent
        manifest = json.loads(args.inside.read_bytes())
        require(inputs == manifest["inputs"], "Native cancellation inputs changed")
        namespace = os.readlink("/proc/self/ns/mnt")
        require(namespace != manifest["host_namespace"], "No private trust namespace")
        require(trust_state(args.system_ca_bundle) == tuple(manifest["host_trust"]), "Unexpected original CA view")
        for command in (["mount", "--bind", str(directory / "isolated-ca.pem"), str(args.system_ca_bundle)],
                        ["mount", "-o", "remount,bind,ro", str(args.system_ca_bundle)]):
            subprocess.run(command, check=True, capture_output=True, timeout=20, env=environment)
        require(digest(args.system_ca_bundle) == manifest["private_ca_sha256"], "Private default CA not mounted")
        control = subprocess.run(["openssl", "verify", "-CAfile", str(args.system_ca_bundle),
                                  "-purpose", "sslserver", "-verify_hostname", "localhost", str(directory / "valid.pem")],
                                 capture_output=True, timeout=10, env=environment)
        (directory / "certificate-verify.stdout").write_bytes(control.stdout)
        (directory / "certificate-verify.stderr").write_bytes(control.stderr)
        require(control.returncode == 0, "Independent fixture certificate verification failed")
        environment["JYPPX_LIUDING_HTTP_CANCELLATION_EVIDENCE_DIR"] = str(directory)
        result = subprocess.run([sys.executable, "-I", "-B", str(checker), "--harness", str(args.harness),
                                 "--tls-directory", str(directory)], capture_output=True, timeout=150, env=environment)
        (directory / "checker.stdout").write_bytes(result.stdout)
        (directory / "checker.stderr").write_bytes(result.stderr)
        require(result.returncode == 0, result.stderr.decode("utf-8", errors="replace"))
        report = json.loads((directory / "cancellation.json").read_bytes())
        require(report["passed"] and not report["failures"] and
                [case["mode"] for case in report["cases"]] ==
                ["stop-headers", "stop-body", "deadline-headers", "deadline-body"], "Original response modes changed")
        for case in report["cases"]:
            require(case["tls_trust_qualified"] and case["native_reaped"] and case["native_returncode"] == 0,
                    "Trusted native cancellation failed or process not reaped")
            require(case["tls_connections_closed"] == len(case["tls_connections"]) == len(case["requests"]),
                    "Trusted TLS connection cleanup failed")
        report.update(private_mount_namespace=namespace, host_mount_namespace=manifest["host_namespace"],
                      private_ca_sha256=digest(args.system_ca_bundle), certificate_sha256=digest(directory / "valid.pem"))
        (directory / "inside-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return 0
    for executable in ("openssl", "unshare", "mount"):
        require(shutil.which(executable), "Missing trusted cancellation prerequisite: " + executable)
    entries = [line.partition("#")[0].split() for line in Path("/etc/hosts").read_text("utf-8").splitlines()]
    addresses = [entry[0] for entry in entries if len(entry) >= 2 and "localhost" in entry[1:]]
    require(addresses and all(address in ("127.0.0.1", "::1") for address in addresses),
            "localhost must have only loopback hosts entries")
    os.umask(0o077)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="cancellation-tls-", dir=args.work_dir)).resolve()
    original = trust_state(args.system_ca_bundle)
    spec = importlib.util.spec_from_file_location("native_cancellation_certificate_fixture", certificate_fixture)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    fixture.prepare(directory, environment, "localhost")
    isolated = directory / "isolated-ca.pem"
    isolated.write_bytes(args.system_ca_bundle.read_bytes() + b"\n" + (directory / "ca.pem").read_bytes())
    manifest = dict(inputs=inputs, host_namespace=os.readlink("/proc/self/ns/mnt"),
                    host_trust=original, private_ca_sha256=digest(isolated))
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    command = ["unshare", "--user", "--map-root-user", "--mount", "--propagation", "private", "--",
               sys.executable, "-I", "-B", str(Path(__file__).resolve()), *sys.argv[1:], "--inside", str(manifest_path)]
    try:
        result = subprocess.run(command, capture_output=True, timeout=170, env=environment)
        (directory / "namespace.stdout").write_bytes(result.stdout)
        (directory / "namespace.stderr").write_bytes(result.stderr)
        require(result.returncode == 0, result.stderr.decode("utf-8", errors="replace"))
    finally:
        require(trust_state(args.system_ca_bundle) == original, "Host default CA changed")
        require(inputs == {path: digest(path) for path in inputs}, "Native cancellation inputs changed")
    report = json.loads((directory / "inside-report.json").read_bytes())
    report.update(work_directory=str(directory), host_trust_preserved=True,
                  scope="Original four response-phase cancellation/deadline contracts over localhost TLS1.2; "
                        "no kernel preemption, certificate revocation, process-tree termination or durable ACK qualification")
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(passed=report["passed"], cases=len(report["cases"]), report=str(directory / "report.json"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
