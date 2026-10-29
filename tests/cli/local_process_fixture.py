# Copyright 2026 JYPPX Training Framework Contributors
# Product: 六丁炼算 (Liuding Forge)
# Project alias: JYPPX Training Framework
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Real subprocess workloads for the tool-layer lifecycle contract."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write_all(descriptor, data):
    view = memoryview(data)
    while view:
        view = view[os.write(descriptor, view):]


def await_file(path):
    deadline = time.monotonic() + 10
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise RuntimeError("Descendant startup did not complete")
        time.sleep(0.005)


def main():
    mode, root_text, *arguments = sys.argv[1:]
    root = Path(root_text)
    if mode == "inspect":
        payload = {"arguments": arguments, "cwd": str(Path.cwd()),
                   "token": os.environ.get("JYPPX_LOCAL_PROCESS_TEST_TOKEN"),
                   "parent_path": os.environ.get("PATH"),
                   "stdin": sys.stdin.buffer.read().hex()}
        write_all(1, json.dumps(payload, ensure_ascii=True).encode("ascii"))
    elif mode == "exit":
        write_all(1, b"ordinary-output\x00\xff")
        write_all(2, b"ordinary-error\x00\xfe")
        os._exit(int(arguments[0]))
    elif mode == "volume":
        write_all(2, b"e" * 90000)
        write_all(1, b"o" * 110000)
        write_all(2, b"last-stderr")
        write_all(1, b"last-stdout")
    elif mode == "stdout-eof":
        os.close(1)
        time.sleep(0.05)
        write_all(2, b"stderr-after-stdout-eof")
    elif mode == "flood":
        descriptor = int(arguments[0])
        while True:
            write_all(descriptor, b"x" * 4096)
    elif mode == "grandchild":
        (root / "grandchild.pid").write_text(str(os.getpid()), encoding="ascii")
        while True:
            time.sleep(0.01)
    elif mode == "descendant":
        if os.name != "nt":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        subprocess.Popen([sys.executable, "-I", "-u", __file__, "grandchild", str(root)])
        await_file(root / "grandchild.pid")
        (root / "descendant.pid").write_text(str(os.getpid()), encoding="ascii")
        while True:
            time.sleep(0.01)
    elif mode in ("tree", "leader-exits", "tree-flood"):
        subprocess.Popen([sys.executable, "-I", "-u", __file__, "descendant", str(root)])
        await_file(root / "descendant.pid")
        (root / "ready").write_text("ready", encoding="ascii")
        if mode == "leader-exits":
            write_all(1, b"leader-done")
            os._exit(23)
        if mode == "tree-flood":
            while True:
                write_all(1, b"x" * 4096)
        while True:
            time.sleep(0.01)
    elif mode == "signal":
        os.kill(os.getpid(), signal.SIGTERM)
    elif mode == "sleep":
        (root / "ready").write_text("ready", encoding="ascii")
        while True:
            time.sleep(0.01)
    else:
        raise ValueError("Unknown fixture mode")


if __name__ == "__main__":
    main()
