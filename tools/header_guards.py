#!/usr/bin/env python3
# Copyright 2026 JYPPX Training Framework Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Validate include-guard identity for public JYPPX headers."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify #ifndef/#define guards for public JYPPX headers."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "include" / "jyppx",
        help="public header root",
    )
    return parser.parse_args()


def _expected_guard(root: Path, header: Path) -> str:
    relative = header.relative_to(root.parent)
    normalized = re.sub(r"[^A-Za-z0-9]", "_", relative.as_posix()).upper()
    return normalized


def main() -> int:
    args = _parse_arguments()
    root = args.root.resolve()
    if not root.is_dir():
        raise ValueError(f"public header root does not exist: {root}")

    problems: list[str] = []
    headers = sorted((*root.rglob("*.h"), *root.rglob("*.hpp")))
    if not headers:
        raise ValueError(f"no public headers found under {root}")

    for header in headers:
        lines = header.read_text(encoding="utf-8-sig").splitlines()
        directives = [line.strip() for line in lines if line.lstrip().startswith("#")]
        if len(directives) < 3:
            problems.append(f"{header}: missing include guard directives")
            continue
        ifndef = next((line for line in directives if line.startswith("#ifndef ")), "")
        define = next((line for line in directives if line.startswith("#define ")), "")
        if not ifndef or not define:
            problems.append(f"{header}: missing #ifndef or #define")
            continue
        ifndef_guard = ifndef.removeprefix("#ifndef ").strip()
        define_guard = define.removeprefix("#define ").strip()
        expected = _expected_guard(root, header)
        if ifndef_guard != define_guard:
            problems.append(
                f"{header}: #ifndef {ifndef_guard} differs from #define {define_guard}"
            )
        if ifndef_guard != expected:
            problems.append(
                f"{header}: guard {ifndef_guard} does not match path-derived {expected}"
            )

    if problems:
        print(f"Header guard validation failed with {len(problems)} issue(s):")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print(f"Header guard validation passed: {len(headers)} public headers")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"header guard validation error: {error}")
        raise SystemExit(2) from error
