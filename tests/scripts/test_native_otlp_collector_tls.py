# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Real pinned Collector HTTPS interchange in an isolated Linux trust view.

The original three native product paths and official signal readback remain
unchanged. Python/OpenSSL prepare and supervise fixtures, never send OTLP.
"""
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
    for key in ("cli", "example", "collector", "artifacts", "config", "work-dir", "system-ca-bundle"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--inside", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(sys.platform == "linux", "Collector private trust gate requires Linux")
    for key in ("cli", "example", "collector", "artifacts", "config", "system_ca_bundle"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    environment = dict(os.environ)
    for key in ("SSL_CERT_FILE", "SSL_CERT_DIR", "CURL_CA_BUNDLE", "OPENSSL_CONF"):
        environment.pop(key, None)
    checker = Path(__file__).with_name("test_native_otlp_collector.py").resolve(strict=True)
    identity = {str(path): digest(path) for path in
                (args.cli, args.example, args.collector, args.artifacts, args.config, checker,
                 Path(__file__).resolve(), Path(__file__).with_name("test_native_otlp_trusted_tls.py"))}
    if args.inside:
        manifest = json.loads(args.inside.read_text("utf-8"))
        directory = args.inside.parent
        require(identity == manifest["inputs"], "Collector TLS input identity changed")
        namespace = os.readlink("/proc/self/ns/mnt")
        require(namespace != manifest["host_namespace"], "No private trust namespace")
        require(trust_state(args.system_ca_bundle) == tuple(manifest["host_trust"]),
                "Unexpected host trust before private mount")
        for command in (["mount", "--bind", str(directory / "isolated-ca.pem"), str(args.system_ca_bundle)],
                        ["mount", "-o", "remount,bind,ro", str(args.system_ca_bundle)]):
            subprocess.run(command, check=True, capture_output=True, timeout=20, env=environment)
        require(digest(args.system_ca_bundle) == manifest["private_ca_sha256"],
                "Private CA view not mounted")
        # Verify the independently generated certificate before native sends.
        control = subprocess.run(["openssl", "verify", "-CAfile", str(args.system_ca_bundle),
                                  "-purpose", "sslserver", "-verify_hostname", "localhost",
                                  str(directory / "valid.pem")], capture_output=True, timeout=10, env=environment)
        (directory / "certificate-verify.stdout").write_bytes(control.stdout)
        (directory / "certificate-verify.stderr").write_bytes(control.stderr)
        require(control.returncode == 0, "Independent Collector certificate validation failed")
        command = [sys.executable, "-I", "-B", str(checker)]
        for key in ("cli", "example", "collector", "artifacts", "config"):
            command.extend(("--" + key, str(getattr(args, key))))
        command.extend(("--tls-directory", str(directory), "--endpoint-host", "localhost",
                        "--work-dir", str(directory / "native")))
        result = subprocess.run(command, capture_output=True, env=environment, timeout=135)
        (directory / "native.stdout").write_bytes(result.stdout)
        (directory / "native.stderr").write_bytes(result.stderr)
        require(result.returncode == 0, result.stderr.decode("utf-8", errors="replace"))
        report = json.loads((directory / "native/report.json").read_text("utf-8"))
        require(report["tls_qualified"] and not report["durable_ack_qualified"] and
                not report["failures"] and report["skipped"] == 0, "Wrong native interchange result")
        require([case["name"] for case in report["cases"]] ==
                ["public-api", "cli-replay", "worker-six-payloads"], "Original product paths changed")
        require([case["events"] for case in report["cases"]] == [3, 3, 6], "Original events changed")
        for case in report["cases"]:
            require(case["tls"]["version"] == "TLSv1.2" and
                    case["tls"]["hostname_verified"] == "localhost" and
                    isinstance(case["collector_exit"], int), "Collector TLS or cleanup missing")
        report.update(private_mount_namespace=namespace, host_mount_namespace=manifest["host_namespace"],
                      private_ca_sha256=digest(args.system_ca_bundle))
        (directory / "inside-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return 0
    for executable in ("openssl", "unshare", "mount"):
        require(shutil.which(executable), "Missing TLS test prerequisite: " + executable)
    hosts = [line.partition("#")[0].split()
             for line in Path("/etc/hosts").read_text("utf-8").splitlines()]
    addresses = [entry[0] for entry in hosts if len(entry) >= 2 and "localhost" in entry[1:]]
    require(addresses and all(address in ("127.0.0.1", "::1") for address in addresses),
            "localhost must have only loopback hosts entries")
    os.umask(0o077)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="collector-tls-", dir=args.work_dir)).resolve()
    original = trust_state(args.system_ca_bundle)
    spec = importlib.util.spec_from_file_location("native_trusted_tls_fixture",
                                                Path(__file__).with_name("test_native_otlp_trusted_tls.py"))
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    fixture.prepare(directory, environment, "localhost")
    isolated = directory / "isolated-ca.pem"
    isolated.write_bytes(args.system_ca_bundle.read_bytes() + b"\n" + (directory / "ca.pem").read_bytes())
    manifest = {"inputs": identity, "host_namespace": os.readlink("/proc/self/ns/mnt"),
                "host_trust": original, "private_ca_sha256": digest(isolated)}
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    command = ["unshare", "--user", "--map-root-user", "--mount", "--propagation", "private", "--",
               sys.executable, "-I", "-B", str(Path(__file__).resolve()), *sys.argv[1:], "--inside", str(manifest_path)]
    try:
        result = subprocess.run(command, capture_output=True, env=environment, timeout=160)
        (directory / "namespace.stdout").write_bytes(result.stdout)
        (directory / "namespace.stderr").write_bytes(result.stderr)
        require(result.returncode == 0, result.stderr.decode("utf-8", errors="replace"))
    finally:
        require(trust_state(args.system_ca_bundle) == original, "Host CA changed")
        require(identity == {path: digest(path) for path in identity}, "Collector TLS input changed")
    report = json.loads((directory / "inside-report.json").read_text("utf-8"))
    report.update(host_trust_preserved=True, work_directory=str(directory),
                  scope="Exact pinned Collector TLS1.2 localhost receiver and original native interchange; "
                        "no revocation, durable ACK, HTTPS crash recovery or exactly-once qualification")
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
