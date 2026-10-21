#!/usr/bin/env bash
# Run the Linux coverage lane in an isolated, reproducible output directory.
set -euo pipefail

source_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
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

# Resolve one explicit LibTorch prefix for both CMake lookup mechanisms. An old
# Torch_DIR in this persistent cache must not select a different dependency.
torch_prefix="${CMAKE_PREFIX_PATH:?CMAKE_PREFIX_PATH must point to one LibTorch installation}"
test -f "$torch_prefix/share/cmake/Torch/TorchConfig.cmake" || {
  echo "CMAKE_PREFIX_PATH must point to one LibTorch installation" >&2
  exit 2
}
torch_prefix=$(cd "$torch_prefix" && pwd)

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

mkdir -p "$build_root" "$output_root"

source_commit=$(git -C "$source_root" rev-parse HEAD)
source_tree=$(git -C "$source_root" rev-parse 'HEAD^{tree}')
source_dirty=$(test -n "$(git -C "$source_root" status --porcelain)" && echo true || echo false)
started_utc=$(date -u +%FT%TZ)
ctest_status=not_run
policy_status=not_run
stage=configure
packaging_excluded="${JYPPX_COVERAGE_EXCLUDE_PACKAGING:-0}"
case "$packaging_excluded" in
  0|1) ;;
  *) echo "JYPPX_COVERAGE_EXCLUDE_PACKAGING must be 0 or 1" >&2; exit 2 ;;
esac

write_metadata() {
  local lane_status=$?
  trap - EXIT
  cat > "$output_root/coverage-metadata.txt" <<EOF
commit=$source_commit
tree=$source_tree
dirty=$source_dirty
final_commit=$(git -C "$source_root" rev-parse HEAD)
final_dirty=$(test -n "$(git -C "$source_root" status --porcelain)" && echo true || echo false)
started_utc=$started_utc
finished_utc=$(date -u +%FT%TZ)
build_type=Debug
build_parallelism=$build_parallelism
libtorch_prefix=$torch_prefix
coverage_tool=$(gcovr --version | head -n 1)
coverage_policy=coverage-policy.json
coverage_policy_sha256=$(sha256sum "$source_root/coverage-policy.json" | awk '{print $1}')
gcov_search_root=$build_root
previews=models,quantization,serving
ctest_excluded_label=$(test "$packaging_excluded" = 1 && echo packaging || echo none)
ctest_exit=$ctest_status
policy_exit=$policy_status
last_stage=$stage
lane_exit=$lane_status
EOF
  exit "$lane_status"
}
trap write_metadata EXIT
if [ "$source_dirty" = true ]; then
  echo "Coverage evidence requires a clean committed source tree" >&2
  exit 2
fi
rm -f "$output_root/ctest.exit" "$output_root/policy.exit"

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
  -DJYPPX_ENABLE_ASAN=OFF -DJYPPX_ENABLE_UBSAN=OFF -DJYPPX_ENABLE_TSAN=OFF \
  -DJYPPX_BUILD_MODELS=ON -DJYPPX_BUILD_QUANTIZATION=ON -DJYPPX_BUILD_SERVING=ON \
  -DCMAKE_PREFIX_PATH="$torch_prefix" \
  -DTorch_DIR="$torch_prefix/share/cmake/Torch"

stage=build
run_logged 02-build.log cmake --build "$build_root" --parallel "$build_parallelism"
find "$build_root" -type f -name '*.gcda' -delete
ctest_args=(--test-dir "$build_root" --output-on-failure --no-tests=error
  --output-junit "$output_root/coverage-ctest.xml")
if [ "$packaging_excluded" = 1 ]; then
  ctest_args+=(-LE packaging)
fi
stage=ctest
set +e
run_logged 03-ctest.log ctest "${ctest_args[@]}"
ctest_status=$?
set -e
printf '%s\n' "$ctest_status" > "$output_root/ctest.exit"

coverage_args=(
  --root "$source_root"
  --filter "$source_root/(src|include)/jyppx/"
  --exclude "$source_root/third_party/"
  # GCC 11 can emit a negative hit for an unreachable generated branch in
  # inline template code. Keep the report usable while retaining a warning in
  # the lane log; the coverage summary still validates all three metrics.
  --gcov-ignore-parse-errors=negative_hits.warn_once_per_file
  -j "$build_parallelism"
)
stage=gcov_xml
run_logged 04-gcov-xml.log gcovr "${coverage_args[@]}" \
  --xml-pretty \
  --output "$output_root/coverage.xml" "$build_root"
stage=gcov_summary
run_logged 05-gcov-summary.log gcovr "${coverage_args[@]}" \
  --json-summary-pretty \
  --json-summary "$output_root/coverage-summary.json" "$build_root"
python3 "$source_root/scripts/validation/coverage_summary.py" \
  "$output_root/coverage-summary.json" \
  > "$output_root/coverage-summary.txt"
stage=policy
set +e
python3 "$source_root/scripts/validation/coverage_gate.py" \
  "$output_root/coverage-summary.json" \
  "$source_root/coverage-policy.json" \
  --output "$output_root/coverage-targets.json" \
  > "$output_root/coverage-targets.txt"
policy_status=$?
set -e
printf '%s\n' "$policy_status" > "$output_root/policy.exit"
stage=complete
if [ "$(git -C "$source_root" rev-parse HEAD)" != "$source_commit" ] ||
   [ "$(test -n "$(git -C "$source_root" status --porcelain)" && echo true || echo false)" != "$source_dirty" ]; then
  echo "Source identity changed during coverage collection" >&2
  exit 2
fi
if [ "$ctest_status" -ne 0 ]; then
  exit "$ctest_status"
fi
if [ "$policy_status" -ne 0 ]; then
  exit "$policy_status"
fi

echo "Coverage report: $output_root/coverage.xml"
