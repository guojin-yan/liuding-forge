#!/usr/bin/env bash
set -euo pipefail

source_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build_root="$source_root/out/build/fuzz-linux"
output_root="$source_root/out/fuzz/linux/device-parse"
fixture_root="$source_root/tests/fixtures/fuzz/device_parse"
seed_root="$fixture_root/corpus"
corpus_root="$output_root/corpus"
artifact_root="$output_root/artifacts"

case "$build_root" in
  "$source_root/out/build/"*) ;;
  *) echo "fuzz build path escaped the repository output root" >&2; exit 2 ;;
esac
case "$output_root" in
  "$source_root/out/fuzz/linux/"*) ;;
  *) echo "fuzz output path escaped the repository output root" >&2; exit 2 ;;
esac

command -v cmake >/dev/null
command -v clang++ >/dev/null || {
  echo "clang++ with libFuzzer support is required" >&2
  exit 2
}

build_parallelism="${JYPPX_BUILD_PARALLEL_LEVEL:-2}"
fuzz_runs="${JYPPX_FUZZ_RUNS:-10000}"
for value_name in build_parallelism fuzz_runs; do
  value=${!value_name}
  case "$value" in
    ''|*[!0-9]*) echo "$value_name must be a positive integer" >&2; exit 2 ;;
  esac
  if [ "$value" -lt 1 ]; then
    echo "$value_name must be a positive integer" >&2
    exit 2
  fi
done

rm -rf "$build_root" "$output_root"
mkdir -p "$corpus_root" "$artifact_root"
cp -R "$seed_root/." "$corpus_root/"

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
  -DJYPPX_BUILD_FUZZERS=ON \
  -DJYPPX_ENABLE_ASAN=ON \
  -DJYPPX_ENABLE_UBSAN=ON \
  -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:?CMAKE_PREFIX_PATH must point to LibTorch}"

run_logged 02-build.log cmake --build "$build_root" \
  --target jyppx_fuzz_device_parse \
  --parallel "$build_parallelism"

export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0:halt_on_error=1:strict_string_checks=1}"
export UBSAN_OPTIONS="${UBSAN_OPTIONS:-halt_on_error=1:print_stacktrace=1}"
address_layout_runner=()
address_layout_mode=native
if command -v setarch >/dev/null && setarch "$(uname -m)" -R true 2>/dev/null; then
  # Older sanitizer runtimes can collide with randomized PIE mappings on
  # newer kernels. Scope the workaround to the fuzz process.
  address_layout_runner=(setarch "$(uname -m)" -R)
  address_layout_mode=setarch-no-randomize
fi
run_logged 03-fuzz.log "${address_layout_runner[@]}" \
  "$build_root/tests/fuzz/jyppx_fuzz_device_parse" \
  "$corpus_root" \
  -dict="$fixture_root/device_parse.dict" \
  -artifact_prefix="$artifact_root/" \
  -max_len=256 \
  -timeout=5 \
  -runs="$fuzz_runs"

cat > "$output_root/metadata.txt" <<EOF
commit=$(git -C "$source_root" rev-parse HEAD)
dirty=$(test -n "$(git -C "$source_root" status --porcelain)" && echo true || echo false)
build_type=Debug
build_parallelism=$build_parallelism
fuzz_runs=$fuzz_runs
compiler=$(clang++ --version | head -n 1)
asan_options=$ASAN_OPTIONS
ubsan_options=$UBSAN_OPTIONS
address_layout=$address_layout_mode
EOF

echo "Fuzz report: $output_root"
