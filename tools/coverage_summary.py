#!/usr/bin/env python3
"""Validate and print gcovr line, function, and branch coverage totals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a gcovr JSON summary and print stable metric totals."
    )
    parser.add_argument("report", type=Path, help="gcovr --json-summary report")
    return parser.parse_args()


def _metric(report: dict[str, Any], name: str) -> tuple[int, int, float]:
    keys = (f"{name}_covered", f"{name}_total", f"{name}_percent")
    missing = [key for key in keys if key not in report]
    if missing:
        raise ValueError(
            f"coverage report lacks {name} metrics ({', '.join(missing)}); "
            "use a gcovr release that reports function coverage"
        )

    covered, total, percent = (report[key] for key in keys)
    if not isinstance(covered, int) or not isinstance(total, int):
        raise ValueError(f"{name} coverage counts must be integers")
    if total <= 0 or covered < 0 or covered > total:
        raise ValueError(f"invalid {name} coverage counts: {covered}/{total}")
    if not isinstance(percent, (int, float)) or not math.isfinite(percent):
        raise ValueError(f"invalid {name} coverage percentage: {percent!r}")

    calculated = covered * 100.0 / total
    if abs(float(percent) - calculated) > 0.11:
        raise ValueError(
            f"inconsistent {name} coverage: report={percent}, calculated={calculated:.2f}"
        )
    return covered, total, calculated


def main() -> int:
    args = _parse_arguments()
    with args.report.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    if not isinstance(report, dict):
        raise ValueError("coverage report root must be an object")

    for name in ("line", "function", "branch"):
        covered, total, percent = _metric(report, name)
        print(f"{name.capitalize()}: {covered}/{total} ({percent:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
