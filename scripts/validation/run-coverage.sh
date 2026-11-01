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

mkdir -p "$build_root" "$output_root"
source_commit=not_run
source_tree=not_run
source_dirty=not_run
started_utc=$(date -u +%FT%TZ)
ctest_status=not_run
policy_status=not_run
stage=preflight
torch_prefix="${CMAKE_PREFIX_PATH:-not_set}"
build_parallelism="${JYPPX_LIUDING_BUILD_PARALLEL_LEVEL:-2}"
packaging_excluded="${JYPPX_COVERAGE_EXCLUDE_PACKAGING:-0}"
models_reference_dir="${JYPPX_LIUDING_MODELS_REFERENCE_DIR:-}"
coverage_profile=v1-selected-cpu
dependency_cache="${JYPPX_COVERAGE_DEPENDENCY_CACHE:-}"
dependency_cache_expected_sha256="${JYPPX_COVERAGE_DEPENDENCY_CACHE_SHA256:-}"
offline_inputs="${JYPPX_COVERAGE_OFFLINE_INPUTS:-}"
offline_inputs_expected_sha256="${JYPPX_COVERAGE_OFFLINE_INPUTS_SHA256:-}"
dependency_cache_sha256=not_run
offline_inputs_sha256=not_run
configured_cache_sha256=not_run
coverage_tool=not_run
policy_sha256=not_run

write_metadata() {
  local lane_status=$?
  trap - EXIT
  set +e
  cat > "$output_root/coverage-metadata.txt" <<EOF
commit=$source_commit
tree=$source_tree
dirty=$source_dirty
final_commit=$(git -C "$source_root" rev-parse HEAD 2>/dev/null || echo unavailable)
final_dirty=$(test -n "$(git -C "$source_root" status --porcelain)" && echo true || echo false)
started_utc=$started_utc
finished_utc=$(date -u +%FT%TZ)
build_type=Debug
build_parallelism=$build_parallelism
libtorch_prefix=$torch_prefix
models_reference_dir=$models_reference_dir
coverage_profile=$coverage_profile
dependency_cache=$dependency_cache
dependency_cache_sha256=$dependency_cache_sha256
offline_inputs=$offline_inputs
offline_inputs_sha256=$offline_inputs_sha256
ctest_parallelism=1
configured_cache_sha256=$configured_cache_sha256
coverage_tool=$coverage_tool
coverage_policy=coverage-policy.json
coverage_policy_sha256=$policy_sha256
gcov_search_root=$build_root
components=core,interop,data,nn,optim,training,experiment,data_image,vision_render,models,serving,onnx_export,onnx_runtime,onnx_serving,bundle_signing,telemetry_native,telemetry_http,tools
ctest_excluded_label=$(test "$packaging_excluded" = 1 && echo packaging || echo none)
ctest_exit=$ctest_status
policy_exit=$policy_status
last_stage=$stage
lane_exit=$lane_status
EOF
  exit "$lane_status"
}
trap write_metadata EXIT
# Invalidate prior successful fixed-name metadata before any preflight check.
printf 'last_stage=preflight\nlane_exit=running\n' > "$output_root/coverage-metadata.txt"
rm -f "$output_root/ctest.exit" "$output_root/policy.exit"
rm -f "$output_root/coverage-offline-inputs.json" "$output_root/coverage-offline-inputs-final.json" \
  "$output_root/coverage-offline-cmake-args.txt"
command -v cmake >/dev/null
command -v ctest >/dev/null
command -v python3 >/dev/null
command -v git >/dev/null
command -v gcovr >/dev/null || {
  echo "gcovr is required for the coverage lane" >&2
  exit 2
}
coverage_tool=$(gcovr --version | head -n 1)
policy_sha256=$(sha256sum "$source_root/coverage-policy.json" | awk '{print $1}')
source_commit=$(git -C "$source_root" rev-parse HEAD)
source_tree=$(git -C "$source_root" rev-parse 'HEAD^{tree}')
source_dirty=$(test -n "$(git -C "$source_root" status --porcelain)" && echo true || echo false)

# An old Torch_DIR must not select a different dependency in this cache.
test -f "$torch_prefix/share/cmake/Torch/TorchConfig.cmake" || {
  echo "CMAKE_PREFIX_PATH must point to one LibTorch installation" >&2
  exit 2
}
torch_prefix=$(cd "$torch_prefix" && pwd)
case "$build_parallelism" in
  ''|*[!0-9]*)
    echo "JYPPX_LIUDING_BUILD_PARALLEL_LEVEL must be a positive integer" >&2
    exit 2
    ;;
esac
if [ "$build_parallelism" != 1 ] && [ "$build_parallelism" != 2 ]; then
  echo "JYPPX_LIUDING_BUILD_PARALLEL_LEVEL must be 1 or 2" >&2
  exit 2
fi
export CMAKE_BUILD_PARALLEL_LEVEL="$build_parallelism"
export CTEST_PARALLEL_LEVEL=1
test -f "$dependency_cache" || {
  echo "JYPPX_COVERAGE_DEPENDENCY_CACHE must name reviewed explicit dependency inputs" >&2
  exit 2
}
case "$dependency_cache_expected_sha256" in
  *[!0-9a-f]*|'') echo "Reviewed lowercase dependency cache SHA-256 is required" >&2; exit 2 ;;
esac
if [ "${#dependency_cache_expected_sha256}" -ne 64 ]; then
  echo "Reviewed dependency cache SHA-256 must have 64 characters" >&2
  exit 2
fi
dependency_cache=$(cd "$(dirname "$dependency_cache")" && pwd)/$(basename "$dependency_cache")
dependency_cache_sha256=$(sha256sum "$dependency_cache" | awk '{print $1}')
if [ "$dependency_cache_sha256" != "$dependency_cache_expected_sha256" ]; then
  echo "Reviewed dependency cache SHA-256 mismatch" >&2
  exit 2
fi
case "$packaging_excluded" in
  0|1) ;;
  *) echo "JYPPX_COVERAGE_EXCLUDE_PACKAGING must be 0 or 1" >&2; exit 2 ;;
esac
if [ "$source_dirty" = true ]; then
  echo "Coverage evidence requires a clean committed source tree" >&2
  exit 2
fi
python3 -I -B "$source_root/scripts/validation/coverage_dependencies.py" prepare \
  --inputs-file "$offline_inputs" --inputs-sha256 "$offline_inputs_expected_sha256" \
  --init-cache "$dependency_cache" --init-cache-sha256 "$dependency_cache_sha256" \
  --output "$output_root/coverage-offline-inputs.json" \
  --cmake-args-output "$output_root/coverage-offline-cmake-args.txt"
offline_inputs_sha256=$offline_inputs_expected_sha256
mapfile -t offline_source_args < "$output_root/coverage-offline-cmake-args.txt"
stage=configure

run_logged() {
  local log_name=$1
  shift
  "$@" 2>&1 | tee "$output_root/$log_name"
}

run_logged 01-configure.log cmake \
  -C "$dependency_cache" \
  -DFETCHCONTENT_FULLY_DISCONNECTED:BOOL=ON \
  -DFETCHCONTENT_UPDATES_DISCONNECTED:BOOL=ON \
  "${offline_source_args[@]}" \
  -S "$source_root" \
  -B "$build_root" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DJYPPX_LIUDING_BUILD_TESTS=ON \
  -DJYPPX_LIUDING_BUILD_EXAMPLES=OFF \
  -DJYPPX_LIUDING_BUILD_BENCHMARKS=OFF \
  -DJYPPX_LIUDING_ENABLE_COVERAGE=ON \
  -DJYPPX_LIUDING_ENABLE_ASAN=OFF -DJYPPX_LIUDING_ENABLE_UBSAN=OFF -DJYPPX_LIUDING_ENABLE_TSAN=OFF \
  -DJYPPX_LIUDING_BUILD_CORE=ON -DJYPPX_LIUDING_BUILD_INTEROP=ON \
  -DJYPPX_LIUDING_BUILD_DATA=ON -DJYPPX_LIUDING_BUILD_NN=ON \
  -DJYPPX_LIUDING_BUILD_OPTIM=ON -DJYPPX_LIUDING_BUILD_TRAINING=ON \
  -DJYPPX_LIUDING_BUILD_EXPERIMENT=ON -DJYPPX_LIUDING_BUILD_TOOLS=ON \
  -DJYPPX_LIUDING_BUILD_DATA_IMAGE=ON -DJYPPX_LIUDING_BUILD_MODELS=ON \
  -DJYPPX_LIUDING_BUILD_VISION_RENDER=ON \
  -DJYPPX_LIUDING_BUILD_SERVING=ON \
  -DJYPPX_LIUDING_BUILD_ONNX_EXPORT=ON -DJYPPX_LIUDING_BUILD_ONNX_RUNTIME=ON \
  -DJYPPX_LIUDING_BUILD_ONNX_SERVING=ON -DJYPPX_LIUDING_BUILD_BUNDLE_SIGNING=ON \
  -DJYPPX_LIUDING_BUILD_TELEMETRY_NATIVE=ON -DJYPPX_LIUDING_BUILD_TELEMETRY_HTTP=ON \
  -DJYPPX_LIUDING_MODELS_REFERENCE_DIR="$models_reference_dir" \
  -DCMAKE_PREFIX_PATH="$torch_prefix" \
  -DTorch_DIR="$torch_prefix/share/cmake/Torch"

stage=ownership_profile
if [ -f "$build_root/CMakeCache.txt" ]; then
  configured_cache_sha256=$(sha256sum "$build_root/CMakeCache.txt" | awk '{print $1}')
fi
run_logged 01-offline-cache.log python3 -I -B "$source_root/scripts/validation/coverage_dependencies.py" verify-cache \
  --receipt "$output_root/coverage-offline-inputs.json" --cmake-cache "$build_root/CMakeCache.txt"
run_logged 01-owner-profile.log python3 "$source_root/scripts/validation/coverage_gate.py" \
  "$source_root/coverage-policy.json" --audit-owners --source-root "$source_root" \
  --cmake-cache "$build_root/CMakeCache.txt" --output "$output_root/coverage-owners.json"

stage=build
run_logged 02-build.log cmake --build "$build_root" --parallel "$build_parallelism"
find "$build_root" -type f -name '*.gcda' -delete
ctest_args=(--parallel 1 --test-dir "$build_root" --output-on-failure --no-tests=error
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
  # The fixed artifact/reference tests hash over a GiB: 22,615,392 SHA-256
  # blocks * 576 rotations = 13,026,465,792 legitimate positive hits. Retain
  # these counters and a finite sanity bound rather than suppressing errors.
  --gcov-suspicious-hits-threshold=20000000000
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
if [ -f "$build_root/CMakeCache.txt" ]; then
  configured_cache_sha256=$(sha256sum "$build_root/CMakeCache.txt" | awk '{print $1}')
fi
set +e
python3 "$source_root/scripts/validation/coverage_gate.py" \
  "$output_root/coverage-summary.json" \
  "$source_root/coverage-policy.json" \
  --source-root "$source_root" --cmake-cache "$build_root/CMakeCache.txt" \
  --output "$output_root/coverage-targets.json" \
  > "$output_root/coverage-targets.txt"
policy_status=$?
set -e
printf '%s\n' "$policy_status" > "$output_root/policy.exit"
stage=verify_dependencies
run_logged 06-offline-inputs-final.log python3 -I -B "$source_root/scripts/validation/coverage_dependencies.py" recheck \
  --receipt "$output_root/coverage-offline-inputs.json" --cmake-cache "$build_root/CMakeCache.txt" \
  --output "$output_root/coverage-offline-inputs-final.json"
stage=complete
if [ "$(sha256sum "$dependency_cache" | awk '{print $1}')" != "$dependency_cache_sha256" ]; then
  echo "Dependency input identity changed during coverage collection" >&2
  exit 2
fi
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
