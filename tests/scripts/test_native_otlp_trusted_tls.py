# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0.
"""Native system-trust HTTPS checks in a private Linux mount namespace.

OpenSSL and Python only create and supervise independent loopback fixtures.
Only the C++ harness sends OTLP, accepts responses and tracks identities.
The host trust store is never changed. Missing namespace support fails this
selected gate; it is not reported as a passing or skipped TLS qualification.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import sys
import tempfile
import threading


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, directory, environment):
    result = subprocess.run(command, cwd=directory, env=environment,
                            capture_output=True, timeout=20)
    require(result.returncode == 0,
            str(command[0]) + ": " + result.stderr.decode("utf-8", errors="replace"))
    return result


def prepare(directory, environment):
    (directory / "issued").mkdir()
    (directory / "index.txt").touch()
    (directory / "serial").write_text("01\n", encoding="ascii")
    config = """[ca]
default_ca = local
[local]
database = index.txt
new_certs_dir = issued
certificate = ca.pem
private_key = ca.key
serial = serial
default_md = sha256
default_days = 2
policy = name
unique_subject = no
[name]
commonName = supplied
[valid]
basicConstraints = critical,CA:FALSE
keyUsage = critical,digitalSignature,keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = IP:127.0.0.1
[mismatch]
basicConstraints = critical,CA:FALSE
keyUsage = critical,digitalSignature,keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = DNS:wrong.invalid
"""
    (directory / "ca.cnf").write_text(config, encoding="ascii")
    run(["openssl", "req", "-new", "-x509", "-newkey", "rsa:2048", "-nodes",
         "-keyout", "ca.key", "-out", "ca.pem", "-days", "30",
         "-subj", "/CN=Liuding Forge Ephemeral Loopback Test CA",
         "-addext", "basicConstraints=critical,CA:TRUE",
         "-addext", "keyUsage=critical,keyCertSign,cRLSign"], directory, environment)
    run(["openssl", "req", "-new", "-newkey", "rsa:2048", "-nodes",
         "-keyout", "server.key", "-out", "server.csr",
         "-subj", "/CN=Liuding Forge Loopback Fixture"], directory, environment)
    for name, extension, dates in (
            ("valid", "valid", []), ("mismatch", "mismatch", []),
            ("expired", "valid", ["-startdate", "20000101000000Z",
                                    "-enddate", "20010101000000Z"])):
        run(["openssl", "ca", "-batch", "-notext", "-config", "ca.cnf",
             "-extensions", extension, "-in", "server.csr", "-out", name + ".pem",
             *dates], directory, environment)


@contextmanager
def collector(certificate, key):
    observed = {"connections": 0, "handshakes": 0, "closed": 0,
                "rejected_handshakes": 0, "requests": [], "errors": []}
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = context.maximum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certificate, key)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "-1"))
            require(0 < length <= 16 * 1024 * 1024, "Invalid OTLP length")
            require(self.path == "/v1/logs", "Unexpected signal path")
            require(self.headers.get("Content-Type") == "application/x-protobuf",
                    "Unexpected OTLP media type")
            require(not any(self.headers.get(name) for name in
                            ("Authorization", "Proxy-Authorization", "Cookie")),
                    "Ambient credentials leaked")
            body = self.rfile.read(length)
            require(len(body) == length, "Incomplete OTLP body")
            observed["requests"].append({"path": self.path, "bytes": length,
                                         "sha256": hashlib.sha256(body).hexdigest()})
            self.send_response(200)
            self.send_header("Content-Type", "application/x-protobuf")
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

    class Server(HTTPServer):
        def get_request(self):
            connection, address = super().get_request()
            observed["connections"] += 1
            connection.settimeout(5)
            try:
                secured = context.wrap_socket(connection, server_side=True)
                observed["handshakes"] += 1
                return secured, address
            except (ssl.SSLError, ConnectionResetError):
                observed["rejected_handshakes"] += 1
                observed["closed"] += 1
                connection.close()
                raise OSError("Expected certificate rejection") from None

        def close_request(self, request):
            super().close_request(request)
            observed["closed"] += 1

        def handle_error(self, request, client_address):
            observed["errors"].append("Fixture request failed")

    server = Server(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever,
                              kwargs={"poll_interval": 0.01})
    worker.start()
    try:
        yield server.server_port, observed
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=6)
        require(not worker.is_alive(), "TLS fixture did not stop")


def inside(manifest_path):
    manifest = json.loads(manifest_path.read_text("utf-8"))
    directory = manifest_path.parent
    environment = dict(os.environ)
    namespace = os.readlink("/proc/self/ns/mnt")
    require(namespace != manifest["host_mount_namespace"], "No private mount namespace")
    bundle = Path(manifest["system_ca_bundle"])
    require(digest(bundle) == manifest["host_ca_sha256"], "Unexpected host trust input")
    run(["mount", "--bind", str(directory / "isolated-ca-bundle.pem"), str(bundle)],
        directory, environment)
    run(["mount", "-o", "remount,bind,ro", str(bundle)], directory, environment)
    require(digest(bundle) == manifest["isolated_ca_sha256"], "Private trust mount failed")
    cases = []
    for name in ("valid", "expired", "mismatch"):
        certificate = directory / (name + ".pem")
        control = subprocess.run(["openssl", "verify", "-CAfile", str(directory / "ca.pem"),
                                  "-purpose", "sslserver", "-verify_ip", "127.0.0.1",
                                  str(certificate)], capture_output=True, timeout=10)
        control_text = (control.stdout + control.stderr).decode("utf-8", errors="replace")
        (directory / (name + "-independent-verify.log")).write_text(control_text, encoding="utf-8")
        require((control.returncode == 0) == (name == "valid"), "Wrong independent certificate result")
        if name == "expired":
            require("certificate has expired" in control_text, "Expiry control failed for another reason")
        if name == "mismatch":
            require("IP address mismatch" in control_text, "Hostname control failed for another reason")
        with collector(certificate, directory / "server.key") as (port, observed):
            mode = "tls-logs" if name == "valid" else "tls-certificate-reject"
            native = subprocess.run([manifest["harness"], str(port), mode], cwd=directory,
                                    env=environment, capture_output=True, timeout=20)
        (directory / (name + "-native.stdout")).write_bytes(native.stdout)
        (directory / (name + "-native.stderr")).write_bytes(native.stderr)
        (directory / (name + "-observed.json")).write_text(json.dumps(observed, indent=2) + "\n", encoding="utf-8")
        stdout = native.stdout.decode("utf-8", errors="replace")
        require(native.returncode == 0, native.stderr.decode("utf-8", errors="replace"))
        require("native-http-contract-ok" in stdout, "Native contract did not complete")
        require(not observed["errors"], "Fixture error")
        expected_connections = 1 if name == "valid" else 2
        require(observed["connections"] == observed["closed"] == expected_connections,
                "Unexpected retry or incomplete connection cleanup")
        require(observed["handshakes"] + observed["rejected_handshakes"] == expected_connections,
                "Incomplete TLS observations")
        require(len(observed["requests"]) == (1 if name == "valid" else 0),
                "HTTP accepted an invalid certificate or duplicate request")
        if name != "valid":
            require(stdout.count("native-http-certificate-rejected: ") == 2,
                    "Missing native certificate diagnostics on the same-identity retries")
        cases.append({"case": name, "independent_verify_exit": control.returncode,
                      "independent_verify_output": control_text, "native_stdout": stdout,
                      "native_exit": native.returncode, "observed": observed})
    report = {"passed": True, "cases": cases, "private_mount_namespace": namespace,
              "host_mount_namespace": manifest["host_mount_namespace"],
              "system_ca_sha256_inside": digest(bundle), "native_processes": 3,
              "native_network_connections": 5, "native_http_requests": 1,
              "qualification": "Linux system trust: trusted success, expiry and IP-name rejection; revocation and durable collector ACK remain unqualified"}
    (directory / "inside-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--system-ca-bundle", type=Path,
                        default=Path("/etc/ssl/certs/ca-certificates.crt"))
    parser.add_argument("--inside", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(sys.platform == "linux", "Private system-trust TLS gate requires Linux")
    if args.inside:
        inside(args.inside.resolve(strict=True))
        return 0
    require(args.harness and args.work_dir, "--harness and --work-dir are required")
    for executable in ("openssl", "unshare", "mount"):
        require(shutil.which(executable), "Missing TLS test prerequisite: " + executable)
    harness = args.harness.resolve(strict=True)
    bundle = args.system_ca_bundle.resolve(strict=True)
    original = bundle.read_bytes()
    original_stat = bundle.stat()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="trusted-tls-", dir=args.work_dir)).resolve()
    environment = dict(os.environ)
    for key in ("SSL_CERT_FILE", "SSL_CERT_DIR", "CURL_CA_BUNDLE", "OPENSSL_CONF"):
        environment.pop(key, None)
    prepare(directory, environment)
    isolated = directory / "isolated-ca-bundle.pem"
    isolated.write_bytes(original + b"\n" + (directory / "ca.pem").read_bytes())
    manifest = {"harness": str(harness), "harness_sha256": digest(harness),
                "system_ca_bundle": str(bundle), "host_ca_sha256": hashlib.sha256(original).hexdigest(),
                "isolated_ca_sha256": digest(isolated), "host_mount_namespace": os.readlink("/proc/self/ns/mnt")}
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    command = ["unshare", "--user", "--map-root-user", "--mount", "--propagation", "private", "--",
               sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--inside", str(manifest_path)]
    try:
        result = subprocess.run(command, env=environment, capture_output=True, timeout=80)
        (directory / "namespace.stdout").write_bytes(result.stdout)
        (directory / "namespace.stderr").write_bytes(result.stderr)
        require(result.returncode == 0, result.stderr.decode("utf-8", errors="replace"))
    finally:
        after = bundle.stat()
        require(bundle.read_bytes() == original and after.st_ino == original_stat.st_ino and
                after.st_mtime_ns == original_stat.st_mtime_ns and after.st_ctime_ns == original_stat.st_ctime_ns,
                "Host CA file changed")
        require(digest(harness) == manifest["harness_sha256"], "Native harness changed")
    report = json.loads((directory / "inside-report.json").read_text("utf-8"))
    report.update({"host_trust_bytes_inode_mtime_ctime_preserved": True,
                   "work_directory": str(directory)})
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
