#!/usr/bin/env python3
"""Real subprocess faults for the product sidecar byte transport.

This fixture is a test artifact and is never part of the user-facing tools.
It intentionally blocks, crashes, emits bad frames, or starts descendant
processes. The supervising test must always terminate its process tree.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


SECRET = b"credential-do-not-echo-in-errors"


def write_all(descriptor: int, value: bytes) -> None:
    while value:
        value = value[os.write(descriptor, value):]


def reply(value) -> None:
    write_all(1, json.dumps(value, ensure_ascii=False, allow_nan=False,
                           separators=(",", ":")).encode("utf-8") + b"\n")


def marker(prefix: Path, suffix: str, value: str) -> None:
    Path(str(prefix) + suffix).write_text(value, encoding="utf-8")


def spawn_leaf(prefix: Path, level: int) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-I", "-u", str(Path(__file__).resolve()),
                             "--mode", "leaf", "--marker-prefix", str(prefix),
                             "--levels", str(level)], shell=False,
                            stdin=subprocess.DEVNULL, stdout=sys.stdout.buffer,
                            stderr=sys.stderr.buffer)


def inherited_handle(number: int, object_name: str, file_identity: str) -> bool:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetHandleInformation.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.GetHandleInformation.restype = wintypes.BOOL
        flags = wintypes.DWORD()
        if not kernel.GetHandleInformation(wintypes.HANDLE(number), ctypes.byref(flags)):
            return False
        # Handle numbers can be reused independently in the child. Compare
        # actual object identity so an unrelated Python handle cannot fail the
        # inheritance test merely by having the same numeric value.
        kernel.OpenEventW.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR)
        kernel.OpenEventW.restype = wintypes.HANDLE
        kernel.CompareObjectHandles.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        kernel.CompareObjectHandles.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        reference = kernel.OpenEventW(0x00100000, False, object_name)
        if not reference:
            raise RuntimeError("fixture object identity unavailable")
        try:
            return bool(kernel.CompareObjectHandles(wintypes.HANDLE(number), reference))
        finally:
            kernel.CloseHandle(reference)
    try:
        observed = os.fstat(number)
        return f"{observed.st_dev}:{observed.st_ino}" == file_identity
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--marker-prefix", type=Path)
    parser.add_argument("--levels", type=int, default=1)
    parser.add_argument("--handle", type=int, default=-1)
    parser.add_argument("--object-name", default="")
    parser.add_argument("--file-identity", default="")
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    if args.marker_prefix:
        marker(args.marker_prefix, ".parent.pid" if args.mode != "leaf" else f".{args.levels}.pid",
               str(os.getpid()))
    if args.mode == "leaf":
        if args.levels:
            spawn_leaf(args.marker_prefix, args.levels - 1)
        time.sleep(60)
        return 0
    if args.mode == "no_read":
        time.sleep(60)
        return 0
    for line in sys.stdin.buffer:
        if args.mode == "echo":
            write_all(1, line)
        elif args.mode == "inspect":
            reply({"arguments": extra, "cwd": os.getcwd(),
                   "environment": os.environ.get("JYPPX_CHILD_TOKEN")})
        elif args.mode == "handle":
            reply({"inherited": inherited_handle(args.handle, args.object_name, args.file_identity)})
        elif args.mode == "delay":
            time.sleep(60)
        elif args.mode == "eof":
            os.close(1)
            time.sleep(60)
        elif args.mode == "partial":
            write_all(1, b'{"unfinished":')
            time.sleep(60)
        elif args.mode == "flood":
            while True:
                write_all(1, b"x" * 4096)
        elif args.mode == "stderr_flood":
            while True:
                write_all(2, SECRET * 128)
        elif args.mode == "stderr":
            write_all(2, SECRET + b"\n")
            write_all(1, line)
        elif args.mode == "extra":
            write_all(1, line + line)
        elif args.mode == "late_extra":
            write_all(1, line)
            # The native test releases this only after it received the first
            # reply. A scheduling pause cannot turn this into an immediate
            # extra-line fault and make the delayed-output gate flaky.
            while not Path(str(args.marker_prefix) + ".late.release").exists():
                time.sleep(0.005)
            write_all(1, b'{"unsolicited":true}\n')
            marker(args.marker_prefix, ".late.ready", "ready")
        elif args.mode == "crash":
            os._exit(37)
        elif args.mode == "invalid_utf8":
            write_all(1, b'{"invalid":"\xff"}\n')
        elif args.mode == "nul":
            write_all(1, b'{"invalid":"\0"}\n')
        elif args.mode == "cr":
            write_all(1, b'{"invalid":true}\r\n')
        elif args.mode in ("child", "child_reply"):
            spawn_leaf(args.marker_prefix, args.levels)
            if args.mode == "child_reply":
                reply({"spawned": True})
            else:
                time.sleep(60)
        elif args.mode == "breakaway":
            if os.name != "nt":
                reply({"blocked": True})
            else:
                try:
                    child = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(60)"],
                        shell=False, creationflags=subprocess.CREATE_BREAKAWAY_FROM_JOB,
                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except OSError:
                    reply({"blocked": True})
                else:
                    # Nested jobs can allow the launch while retaining the
                    # descendant in the restrictive inner job. The supervisor
                    # must verify actual termination when the transport closes.
                    reply({"blocked": False, "child_pid": child.pid})
        else:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
