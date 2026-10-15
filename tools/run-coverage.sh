#!/usr/bin/env bash
# Run the Linux coverage lane in an isolated, reproducible output directory.
set -euo pipefail

source_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build_root="$source_root/out/coverage/linux/build"
output_root="$source_root/out/coverage/linux/report"

case "$build_root" in
  "$source_root/out/coverage/linux/"*) ;;
  *) echo "coverage build path escaped the repository output root" >&2; exit 2 ;;
esac
case "$output_root" in
  "$source_root/out/coverage/linux/"*) ;;
  *) echo "coverage report path escaped the repository output root" >&2; exit 2 ;;
esac

command -v cmake >/dev/null
command -v ctest >/dev/null
command -v python3 >/dev/null
command -v gcovr >/dev/null || {
  echo "gcovr is required for the coverage lane" >&2
  exit 2
}

build_parallelism="${JYPPX_BUILD_PARALLEL_LEVEL:-2}"
case "$build_parallelism" in
  ''|*[!0-9]*)
    echo "JYPPX_BUILD_PARALLEL_LEVEL must be a positive integer" >&2
    exit 2
    ;;
esac
if [ "$build_parallelism" -lt 1 ]; then
  echo "JYPPX_BUILD_PARALLEL_LEVEL must be a positive integer" >&2
  exit 2
fi

rm -rf "$build_root" "$output_root"
mkdir -p "$output_root"

run_logged() {
  local log_name=$1
  shift
  "$@" 2>&1 | tee "$output_root/$log_name"
}

run_logged 01-configure.log cmake \
  -S "$source_root" \
  -B "$build_root" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DJYPPX_BUILD_TESTS=ON \
  -DJYPPX_BUILD_EXAMPLES=OFF \
  -DJYPPX_BUILD_BENCHMARKS=OFF \
  -DJYPPX_ENABLE_COVERAGE=ON \
  -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:?CMAKE_PREFIX_PATH must point to LibTorch}"

run_logged 02-build.log cmake --build "$build_root" --parallel "$build_parallelism"
run_logged 03-ctest.log ctest --test-dir "$build_root" --output-on-failure

coverage_args=(
  --root "$source_root"
  --filter "$source_root/(src|include)/jyppx/"
  --exclude "$source_root/third_party/"
  # GCC 11 can emit a negative hit for an unreachable generated branch in
  # inline template code. Keep the report usable while retaining a warning in
  # the lane log; the coverage summary still validates all three metrics.
  --gcov-ignore-parse-errors=negative_hits.warn_once_per_file
)
gcovr "${coverage_args[@]}" \
  --xml-pretty \
  --output "$output_root/coverage.xml"
gcovr "${coverage_args[@]}" \
  --json-summary-pretty \
  --json-summary "$output_root/coverage-summary.json"
python3 "$source_root/tools/coverage_summary.py" \
  "$output_root/coverage-summary.json" \
  > "$output_root/coverage-summary.txt"

cat > "$output_root/coverage-metadata.txt" <<EOF
commit=$(git -C "$source_root" rev-parse HEAD)
dirty=$(test -n "$(git -C "$source_root" status --porcelain)" && echo true || echo false)
build_type=Debug
build_parallelism=$build_parallelism
coverage_tool=$(gcovr --version | head -n 1)
EOF

echo "Coverage report: $output_root/coverage.xml"
