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


def cpu_ids():
    if os.name != "nt":
        return sorted(os.sched_getaffinity(0))
    import ctypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    process, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    return [i for i in range(ctypes.sizeof(process) * 8) if process.value & (1 << i)]


def main():
    mode, root_text, *arguments = sys.argv[1:]
    root = Path(root_text)
    if mode in ("cpu-affinity", "cpu-affinity-child"):
        observed = cpu_ids()
        if mode == "cpu-affinity-child":
            write_all(1, json.dumps(observed).encode("ascii"))
        else:
            child = subprocess.check_output([sys.executable, "-I", "-u", __file__, "cpu-affinity-child", str(root)])
            write_all(1, json.dumps({"leader": observed, "descendant": json.loads(child)}).encode("ascii"))
    elif mode == "memory-boundary":
        payload = {"touched_bytes": 8 * 1024 * 1024}
        memory = bytearray(payload["touched_bytes"])
        for offset in range(0, len(memory), 4096):
            memory[offset] = 19
        if sys.platform == "linux":
            group = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
            controls = Path("/sys/fs/cgroup") / group.lstrip("/")
            payload.update(cgroup=group, memory_max=int((controls / "memory.max").read_text()),
                           swap_max=int((controls / "memory.swap.max").read_text()),
                           oom_group=int((controls / "memory.oom.group").read_text()))
        write_all(1, json.dumps(payload).encode("ascii"))
    elif mode == "memory-aggregate":
        child = subprocess.Popen([sys.executable, "-I", "-u", __file__,
                                  "memory-aggregate-child", str(root)])
        await_file(root / "memory-child.pid")
        memory = bytearray(24 * 1024 * 1024)
        for offset in range(0, len(memory), 4096):
            memory[offset] = 31
        (root / "memory-aggregate.go").write_text("go", encoding="ascii")
        os._exit(child.wait())
    elif mode == "memory-aggregate-child":
        (root / "memory-child.pid").write_text(str(os.getpid()), encoding="ascii")
        await_file(root / "memory-aggregate.go")
        try:
            memory = bytearray(32 * 1024 * 1024)
            for offset in range(0, len(memory), 4096):
                memory[offset] = 37
        except MemoryError:
            os._exit(79)
    elif mode in ("memory-exceed", "memory-tree-exceed", "memory-tree-hold"):
        if mode.startswith("memory-tree"):
            child = subprocess.Popen([sys.executable, "-I", "-u", __file__,
                                      "memory-escaped-child", str(root)],
                                     start_new_session=os.name != "nt")
            await_file(root / "memory-child.pid")
        (root / "ready").write_text(str(os.getpid()), encoding="ascii")
        if mode == "memory-tree-hold":
            while True:
                time.sleep(0.01)
        blocks = []
        try:
            # Finite even if enforcement regresses: at most 256 MiB.
            for _ in range(32):
                block = bytearray(8 * 1024 * 1024)
                for offset in range(0, len(block), 4096):
                    block[offset] = 29
                blocks.append(block)
        except MemoryError:
            write_all(1, json.dumps({"allocation_rejected": True,
                                     "touched_bytes": sum(map(len, blocks))}).encode("ascii"))
            # Windows rejects commitment instead of invoking a Linux OOM kill.
            os._exit(79)
        write_all(1, json.dumps({"allocation_rejected": False,
                                 "touched_bytes": sum(map(len, blocks))}).encode("ascii"))
    elif mode == "memory-escaped-child":
        (root / "memory-child.pid").write_text(str(os.getpid()), encoding="ascii")
        while True:
            time.sleep(0.01)
    elif mode == "inspect":
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
    elif mode in ("resources", "resources-hold"):
        memory = bytearray(24 * 1024 * 1024)
        for offset in range(0, len(memory), 4096):
            memory[offset] = 23
        target = time.process_time() + 0.08
        while time.process_time() < target:
            sum(range(10000))
        payload = {"cpu_time_us": time.process_time_ns() // 1000,
                   "process_id": os.getpid(), "touched_bytes": len(memory)}
        if sys.platform == "linux":
            # Independent /proc interface, not the runner's wait4 result.
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmHWM:"):
                    payload["self_peak_resident_memory_bytes"] = int(line.split()[1]) * 1024
        write_all(1, json.dumps(payload).encode("ascii"))
        (root / "ready").write_text("ready", encoding="ascii")
        if mode == "resources-hold":
            while True:
                time.sleep(0.01)
    elif mode == "flood":
        descriptor = int(arguments[0])
        while True:
            write_all(descriptor, b"x" * 4096)
    elif mode == "grandchild":
        if arguments == ["close-streams"]:
            os.close(1)
            os.close(2)
        (root / "grandchild.pid").write_text(str(os.getpid()), encoding="ascii")
        while True:
            time.sleep(0.01)
    elif mode == "descendant":
        if os.name != "nt":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        subprocess.Popen([sys.executable, "-I", "-u", __file__, "grandchild", str(root),
                          *arguments])
        await_file(root / "grandchild.pid")
        if arguments == ["close-streams"]:
            os.close(1)
            os.close(2)
        (root / "descendant.pid").write_text(str(os.getpid()), encoding="ascii")
        while True:
            time.sleep(0.01)
    elif mode in ("tree", "leader-exits", "tree-flood", "volume-leader-exits",
                  "tree-closed-streams", "leader-exits-closed-streams"):
        descendant_arguments = ["close-streams"] if mode.endswith("-closed-streams") else []
        subprocess.Popen([sys.executable, "-I", "-u", __file__, "descendant", str(root),
                          *descendant_arguments])
        await_file(root / "descendant.pid")
        if descendant_arguments:
            os.close(1)
            os.close(2)
        (root / "ready").write_text("ready", encoding="ascii")
        if mode == "leader-exits-closed-streams":
            os._exit(23)
        if mode == "leader-exits":
            write_all(1, b"leader-done")
            os._exit(23)
        if mode == "volume-leader-exits":
            # More than pipe capacity on both streams, with descendants still
            # retaining their inherited write ends after the leader exits.
            write_all(2, b"e" * (2 * 1024 * 1024 + 43))
            write_all(1, b"o" * (2 * 1024 * 1024 + 29))
            write_all(2, b"final-stderr")
            write_all(1, b"final-stdout")
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
