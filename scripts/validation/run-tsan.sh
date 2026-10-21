#!/usr/bin/env bash
set -euo pipefail

source_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
build_root="$source_root/out/build/tsan-linux"
output_root="$source_root/out/sanitizer/linux/tsan"

case "$build_root" in
  "$source_root/out/build/"*) ;;
  *) echo "TSan build path escaped the repository output root" >&2; exit 2 ;;
esac
case "$output_root" in
  "$source_root/out/sanitizer/linux/"*) ;;
  *) echo "TSan report path escaped the repository output root" >&2; exit 2 ;;
esac

command -v cmake >/dev/null
command -v ctest >/dev/null
command -v clang++ >/dev/null || {
  echo "clang++ with ThreadSanitizer support is required" >&2
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

mkdir -p "$build_root" "$output_root"

run_logged() {
  local log_name=$1
  shift
  "$@" 2>&1 | tee "$output_root/$log_name"
}

run_logged 01-configure.log cmake \
  -S "$source_root" \
  -B "$build_root" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DJYPPX_BUILD_TESTS=ON \
  -DJYPPX_BUILD_EXAMPLES=OFF \
  -DJYPPX_BUILD_BENCHMARKS=OFF \
  -DJYPPX_ENABLE_TSAN=ON \
  -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:?CMAKE_PREFIX_PATH must point to LibTorch}"

run_logged 02-build.log cmake --build "$build_root" \
  --target jyppx_test_data jyppx_test_concurrent_stress \
  --parallel "$build_parallelism"

export TSAN_OPTIONS="${TSAN_OPTIONS:-halt_on_error=1:second_deadlock_stack=1}"
tsan_test_regex='^(DataPipelineTest|DataPipelineCheckpointTest|DataPipelineWorkerTest|IterableDataPipelineTest|IterableDataPipelineCheckpointTest|BatchMemoryPipelineTest|ValueSchemaPipelineTest|ConcurrentStressTest)\.'
address_layout_runner=()
address_layout_mode=native
if command -v setarch >/dev/null && setarch "$(uname -m)" -R true 2>/dev/null; then
  # GCC's TSan runtime can collide with randomized PIE mappings on newer
  # kernels. Disabling ASLR for the test process gives TSan its documented
  # shadow-memory layout without changing the host configuration.
  address_layout_runner=(setarch "$(uname -m)" -R)
  address_layout_mode=setarch-no-randomize
fi
run_logged 00-test-discovery.log "${address_layout_runner[@]}" ctest \
  --test-dir "$build_root" \
  -N \
  --tests-regex "$tsan_test_regex"
run_logged 03-ctest.log "${address_layout_runner[@]}" ctest --test-dir "$build_root" \
  --output-on-failure \
  --tests-regex "$tsan_test_regex"

discovered_test_count=$(grep -cE '^ *Test +#[0-9]+:' \
  "$output_root/00-test-discovery.log" || true)
if [ "$discovered_test_count" -lt 1 ]; then
  echo "TSan test discovery selected no tests" >&2
  exit 2
fi

cat > "$output_root/metadata.txt" <<EOF
commit=$(git -C "$source_root" rev-parse HEAD)
dirty=$(test -n "$(git -C "$source_root" status --porcelain)" && echo true || echo false)
build_type=Debug
build_parallelism=$build_parallelism
compiler=$(clang++ --version | head -n 1)
tsan_options=$TSAN_OPTIONS
address_layout=$address_layout_mode
test_regex=$tsan_test_regex
discovered_test_count=$discovered_test_count
discovered_test_list=00-test-discovery.log
EOF

echo "TSan report: $output_root"
