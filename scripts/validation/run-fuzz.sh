#!/usr/bin/env bash
set -euo pipefail

source_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# Keep this one persistent cache/archive directory for every fuzz chapter run.
build_root="$source_root/out/build/fuzz-linux"
output_root="$source_root/out/fuzz/linux"
fixture_root="$source_root/tests/fixtures/fuzz"
targets=(
  device_parse
  data_pipeline_state
  iterable_pipeline_state
  telemetry_codec
  checkpoint_manifest
  model_bundle_manifest
  safetensors
  evaluation_json
  backend_protocol
)
fixtures=(
  device_parse
  data_pipeline
  iterable_pipeline
  telemetry_codec
  checkpoint_manifest
  model_bundle_manifest
  safetensors
  evaluation_json
  backend_protocol
)
maximum_lengths=(256 4096 4096 65536 65536 65536 65536 65536 65536)
output_names=(device-parse data-pipeline iterable-pipeline telemetry-codec
  checkpoint-manifest model-bundle-manifest safetensors evaluation-json backend-protocol)
fuzz_targets=()
for target in "${targets[@]}"; do
  fuzz_targets+=("jyppx_fuzz_$target")
done

case "$build_root" in
  "$source_root/out/build/"*) ;;
  *) echo "fuzz build path escaped the repository output root" >&2; exit 2 ;;
esac
case "$output_root" in
  "$source_root/out/fuzz/linux") ;;
  *) echo "fuzz output path escaped the repository output root" >&2; exit 2 ;;
esac

command -v cmake >/dev/null
command -v clang++ >/dev/null || {
  echo "clang++ with libFuzzer support is required" >&2
  exit 2
}

build_parallelism="${JYPPX_LIUDING_BUILD_PARALLEL_LEVEL:-2}"
fuzz_runs="${JYPPX_FUZZ_RUNS:-10000}"
fuzz_seed="${JYPPX_FUZZ_SEED:-20261021}"
fuzz_rss_mb="${JYPPX_FUZZ_RSS_LIMIT_MB:-4096}"
for value_name in build_parallelism fuzz_runs fuzz_seed fuzz_rss_mb; do
  value=${!value_name}
  case "$value" in
    ''|*[!0-9]*) echo "$value_name must be a positive integer" >&2; exit 2 ;;
  esac
  if [ "$value" -lt 1 ]; then
    echo "$value_name must be a positive integer" >&2
    exit 2
  fi
done

mkdir -p "$build_root" "$output_root"
for index in "${!targets[@]}"; do
  target_output="$output_root/${output_names[$index]}"
  mkdir -p "$target_output/corpus" "$target_output/artifacts"
  # Seed copies do not discard corpus discoveries or the incremental cache.
  cp -R "$fixture_root/${fixtures[$index]}/corpus/." "$target_output/corpus/"
done

run_logged() {
  local log_path=$1
  shift
  "$@" 2>&1 | tee "$log_path"
}

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

started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
active_target=configure
finish_metadata() {
  local exit_status=$?
  trap - EXIT
  cat > "$output_root/metadata.txt" <<EOF
commit=$(git -C "$source_root" rev-parse HEAD)
dirty=$(test -n "$(git -C "$source_root" status --porcelain)" && echo true || echo false)
started_utc=$started_utc
finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
exit_status=$exit_status
last_stage=$active_target
build_type=Debug
build_directory=$build_root
build_parallelism=$build_parallelism
fuzz_runs=$fuzz_runs
fuzz_seed=$fuzz_seed
fuzz_rss_limit_mb=$fuzz_rss_mb
compiler=$(clang++ --version | head -n 1)
asan_options=$ASAN_OPTIONS
ubsan_options=$UBSAN_OPTIONS
address_layout=$address_layout_mode
implementation_coverage=fuzzer-no-link
fuzz_targets=$(IFS=,; echo "${targets[*]}")
EOF
  exit "$exit_status"
}
trap finish_metadata EXIT

run_logged "$output_root/01-configure.log" cmake \
  -S "$source_root" \
  -B "$build_root" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DJYPPX_LIUDING_BUILD_TESTS=ON \
  -DJYPPX_LIUDING_BUILD_EXAMPLES=OFF \
  -DJYPPX_LIUDING_BUILD_BENCHMARKS=OFF \
  -DJYPPX_LIUDING_BUILD_FUZZERS=ON \
  -DJYPPX_LIUDING_ENABLE_ASAN=ON \
  -DJYPPX_LIUDING_ENABLE_UBSAN=ON \
  -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:?CMAKE_PREFIX_PATH must point to LibTorch}"

active_target=build
run_logged "$output_root/02-build.log" cmake --build "$build_root" \
  --target "${fuzz_targets[@]}" --parallel "$build_parallelism"

for index in "${!targets[@]}"; do
  target=${targets[$index]}
  target_name="liuding-forge-fuzz-${target//_/-}"
  fixture=${fixtures[$index]}
  active_target=$target
  target_output="$output_root/${output_names[$index]}"
  cat > "$target_output/metadata.txt" <<EOF
target=jyppx_fuzz_$target
executable=$target_name
fixture=$fixture
max_input_bytes=${maximum_lengths[$index]}
runs=$fuzz_runs
seed=$fuzz_seed
rss_limit_mb=$fuzz_rss_mb
initial_corpus_files=$(find "$target_output/corpus" -type f | wc -l)
dictionary_sha256=$(sha256sum "$fixture_root/$fixture/$fixture.dict" | cut -d ' ' -f 1)
EOF
  run_logged "$target_output/03-fuzz.log" "${address_layout_runner[@]}" \
    "$build_root/tests/fuzz/$target_name" \
    "$target_output/corpus" \
    -dict="$fixture_root/$fixture/$fixture.dict" \
    -artifact_prefix="$target_output/artifacts/" \
    -max_len="${maximum_lengths[$index]}" \
    -rss_limit_mb="$fuzz_rss_mb" \
    -timeout=10 \
    -seed="$fuzz_seed" \
    -runs="$fuzz_runs"
  printf 'completed_utc=%s\nfinal_corpus_files=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$(find "$target_output/corpus" -type f | wc -l)" \
    >> "$target_output/metadata.txt"
done
active_target=completed
echo "Fuzz report: $output_root"
