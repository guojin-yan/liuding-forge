// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
// Fixed-task validation tool; not part of the installed SDK.

#include "exclusive_file.hpp"

#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>
#include <jyppx/liuding/deployment/onnx_export/report.hpp>
#include <jyppx/liuding/deployment/onnx_runtime/session.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/foundation/version.hpp>

#include <torch/torch.h>

#include <ATen/Parallel.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
// clang-format off
#include <Windows.h>
#include <Psapi.h>
// clang-format on
#endif

namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;
namespace ex = jyppx::liuding::onnx_export;
namespace rt = jyppx::liuding::onnx_runtime;
constexpr std::size_t kSamples = 10000U;
constexpr std::size_t kInputs = 784U;
constexpr std::size_t kOutputs = 10U;
constexpr std::size_t kBatch = 64U;
constexpr std::size_t kHotPasses = 5U;

void Require(bool value, std::string_view message) {
  if (!value) throw std::runtime_error(std::string(message));
}

double Elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

// Exact-size and EOF checks reject truncated or concurrently grown files.
template <class T>
std::vector<T> Read(const fs::path &path, std::size_t count) {
  Require(count > 0U && count <= (64U << 20U) / sizeof(T),
          "Read exceeds the fixed fixture bound");
  const auto size = count * sizeof(T);
  Require(fs::file_size(path) == size, "Wrong exact fixture size");
  std::vector<T> values(count);
  std::ifstream stream(path, std::ios::binary);
  Require(static_cast<bool>(stream.read(reinterpret_cast<char *>(values.data()),
                                        static_cast<std::streamsize>(size))),
          "Cannot read complete fixture");
  Require(stream.peek() == std::char_traits<char>::eof() && !stream.bad() &&
              fs::file_size(path) == size,
          "Fixture changed during read");
  if constexpr (std::is_same_v<T, float>)
    Require(std::all_of(values.begin(), values.end(),
                        [](float value) { return std::isfinite(value); }),
            "Nonfinite frozen value");
  return values;
}

std::vector<std::byte> ReadBounded(const fs::path &path, std::size_t bound) {
  const auto size = fs::file_size(path);
  Require(size > 0U && size <= bound, "File exceeds its byte bound");
  return Read<std::byte>(path, static_cast<std::size_t>(size));
}

template <class T>
std::string Digest(const std::vector<T> &values) {
  return jyppx::liuding::sha256_hex(std::as_bytes(std::span(values)));
}

Json ReadJson(const fs::path &path) {
  const auto bytes = ReadBounded(path, 1U << 20U);
  return Json::parse(
      reinterpret_cast<const char *>(bytes.data()),
      reinterpret_cast<const char *>(bytes.data() + bytes.size()));
}

Json LoadedRuntime() {
#ifdef _WIN32
  const auto module = GetModuleHandleW(L"onnxruntime.dll");
  Require(module != nullptr, "Cannot locate actually loaded ONNX Runtime DLL");
  std::vector<wchar_t> name(32768U);
  const auto path_length =
      GetModuleFileNameW(module, name.data(), static_cast<DWORD>(name.size()));
  Require(path_length > 0U && path_length < name.size(),
          "Cannot identify actual runtime DLL");
  const fs::path path(std::wstring(name.data(), path_length));
#elif defined(__linux__)
  std::ifstream maps("/proc/self/maps");
  fs::path path;
  std::string line;
  while (std::getline(maps, line)) {
    const auto slash = line.find('/');
    if (slash == std::string::npos) continue;
    const fs::path found(line.substr(slash));
    if (found.filename().string().starts_with("libonnxruntime.so")) {
      Require(!line.ends_with(" (deleted)"),
              "Loaded runtime library was deleted");
      path = found;
      break;
    }
  }
  Require(!path.empty(), "Cannot identify actual runtime in process maps");
#else
  throw std::runtime_error(
      "Actual runtime identity unsupported on this platform");
#endif
#if defined(_WIN32) || defined(__linux__)
  const auto size = fs::file_size(path);
  Require(size > 0U && size <= (1ULL << 30U),
          "Runtime library size is invalid");
  std::ifstream stream(path, std::ios::binary);
  Require(stream.is_open(), "Cannot open actual runtime library");
  jyppx::liuding::Sha256 hash;
  std::array<std::byte, 64U << 10U> buffer{};
  std::uintmax_t count = 0U;
  while (stream.read(reinterpret_cast<char *>(buffer.data()), buffer.size()) ||
         stream.gcount() > 0) {
    const auto read = static_cast<std::size_t>(stream.gcount());
    Require(count <= size && read <= size - count,
            "Runtime library grew during hashing");
    hash.update(std::span(buffer).first(read));
    count += read;
  }
  Require(stream.eof() && !stream.bad() && count == size &&
              fs::file_size(path) == size,
          "Runtime library changed during hashing");
  const auto utf8 = path.u8string();
  return {{"path", std::string(utf8.begin(), utf8.end())},
          {"sha256", jyppx::liuding::sha256_digest_hex(hash.finish())},
          {"bytes", size}};
#endif
}

void Write(const fs::path &path, std::span<const std::byte> bytes) {
  jyppx::liuding::quantization::tools_detail::PublishExclusive(path, bytes);
}

void WriteJson(const fs::path &path, const Json &value) {
  const auto text = value.dump(2) + "\n";
  Write(path, std::as_bytes(std::span(text.data(), text.size())));
}

Json PeakRss() {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS memory{};
  Require(
      GetProcessMemoryInfo(GetCurrentProcess(), &memory, sizeof(memory)) != 0,
      "GetProcessMemoryInfo failed");
  return {{"bytes", memory.PeakWorkingSetSize},
          {"scope", "process lifetime PeakWorkingSetSize including runtime"}};
#elif defined(__linux__)
  std::ifstream stream("/proc/self/status");
  std::string field;
  while (stream >> field) {
    if (field == "VmHWM:") {
      std::uint64_t kib = 0U;
      std::string unit;
      Require(
          static_cast<bool>(stream >> kib >> unit) && unit == "kB" && kib > 0U,
          "Invalid Linux VmHWM");
      return {{"bytes", kib * 1024U},
              {"scope",
               "process lifetime /proc/self/status VmHWM including runtime"}};
    }
    std::string rest;
    std::getline(stream, rest);
  }
  throw std::runtime_error("Linux VmHWM unavailable");
#else
  throw std::runtime_error("Peak RSS unsupported on this platform");
#endif
}

void Prepare(const fs::path &fixture, const fs::path &qualified_int8,
             const std::string &expected_int8_digest, const fs::path &output) {
  // The independent quality checker owns integer-kernel/oracle qualification.
  // Retain its exact original INT8 graph; do not profile a replacement graph.
  const auto integer_model = ReadBounded(qualified_int8, 16U << 20U);
  Require(Digest(integer_model) == expected_int8_digest,
          "Qualified original INT8 graph digest mismatch");
  auto weights = Read<float>(fixture / "weights.f32", kOutputs * kInputs);
  auto bias = Read<float>(fixture / "bias.f32", kOutputs);
  const auto input = Read<float>(fixture / "input.f32", kSamples * kInputs);
  const auto labels = Read<std::uint8_t>(fixture / "labels.u8", kSamples);
  const auto reference = ReadJson(fixture / "reference-result.json");
  const auto thresholds = reference.at("thresholds");
  Require(thresholds.at("minimum_float_accuracy") == 0.75 &&
              thresholds.at("minimum_int8_accuracy") == 0.75 &&
              thresholds.at("maximum_accuracy_drop") == 0.005 &&
              thresholds.at("threads") == 1 &&
              thresholds.at("memory_budget_bytes") == (1ULL << 30U),
          "Original frozen task policy changed");
  Require(reference.at("float_accuracy") == 0.8115,
          "Original frozen FP32 task changed");

  at::set_num_threads(1);
  at::set_num_interop_threads(1);
  torch::NoGradGuard no_grad;
  torch::nn::Linear layer(static_cast<std::int64_t>(kInputs),
                          static_cast<std::int64_t>(kOutputs));
  layer->weight.copy_(torch::from_blob(
      weights.data(),
      {static_cast<std::int64_t>(kOutputs), static_cast<std::int64_t>(kInputs)},
      torch::kFloat32));
  layer->bias.copy_(torch::from_blob(
      bias.data(), {static_cast<std::int64_t>(kOutputs)}, torch::kFloat32));
  layer->eval();
  ex::ExportOptions options;
  options.minimum_batch = 1;
  options.maximum_batch = static_cast<std::int64_t>(kBatch);
  options.max_output_bytes = 16U << 20U;
  const std::array<std::int64_t, 2U> shape{static_cast<std::int64_t>(kBatch),
                                           static_cast<std::int64_t>(kInputs)};
  const auto baseline = ex::export_linear(layer, shape, options);
  Require(baseline.report.has_value(), "Missing generic C++ export report");
  bool weight_bound = false;
  bool bias_bound = false;
  for (const auto &state : baseline.report->state) {
    if (state.name == "state.weight")
      weight_bound = state.digest.hex == Digest(weights);
    if (state.name == "state.bias")
      bias_bound = state.digest.hex == Digest(bias);
  }
  Require(weight_bound && bias_bound,
          "Generic C++ exporter did not bind the frozen original parameters");
  ex::verify_export_report_model(*baseline.report, baseline.onnx);

  Require(fs::create_directory(output),
          "Prepare output must be a new directory");
  Write(output / "baseline.onnx", baseline.onnx);
  Write(output / "candidate.onnx", integer_model);
  const auto export_report = ex::serialize_export_report(*baseline.report);
  Write(output / "baseline-export.json",
        std::as_bytes(std::span(export_report.data(), export_report.size())));
  Json manifest{
      {"schema", "jyppx.frozen-linear-comparison/v1"},
      {"build", Json::parse(jyppx::liuding::build_fingerprint_json())},
      {"fixture",
       {{"weights.f32", Digest(weights)},
        {"bias.f32", Digest(bias)},
        {"input.f32", Digest(input)},
        {"labels.u8", Digest(labels)},
        {"reference-result.json",
         Digest(ReadBounded(fixture / "reference-result.json", 1U << 20U))}}},
      {"models",
       {{"baseline",
         {{"sha256", baseline.digest.hex}, {"bytes", baseline.onnx.size()}}},
        {"candidate",
         {{"sha256", expected_int8_digest}, {"bytes", integer_model.size()}}}}},
      {"thresholds", thresholds},
      {"frozen_float_correct", 8115},
      {"kernel_evidence",
       "external checker must bind candidate digest and original trace"}};
  WriteJson(output / "manifest.json", manifest);
  std::cout << manifest.dump() << '\n';
}

struct TaskPass {
  std::vector<float> output;
  double elapsed_ms = 0.0;
  double first_batch_ms = 0.0;
  double first_batch_ready_ms = 0.0;
};

TaskPass RunTask(rt::Session &session, const std::vector<float> &input,
                 Clock::time_point process_entry) {
  TaskPass pass;
  // Capacity allocation is outside the inference timer, identically for both.
  pass.output.resize(kSamples * kOutputs);
  const auto task_start = Clock::now();
  for (std::size_t offset = 0U; offset < kSamples; offset += kBatch) {
    const auto count = std::min(kBatch, kSamples - offset);
    const std::array<std::int64_t, 2U> shape{
        static_cast<std::int64_t>(count), static_cast<std::int64_t>(kInputs)};
    const auto batch_start = Clock::now();
    const auto result = session.run(
        std::span(input).subspan(offset * kInputs, count * kInputs), shape);
    if (offset == 0U) {
      pass.first_batch_ms = Elapsed(batch_start);
      pass.first_batch_ready_ms = Elapsed(process_entry);
    }
    Require(result.shape ==
                    std::vector<std::int64_t>{
                        static_cast<std::int64_t>(count),
                        static_cast<std::int64_t>(kOutputs)} &&
                result.values.size() == count * kOutputs,
            "Wrong task output signature");
    std::copy(
        result.values.begin(), result.values.end(),
        pass.output.begin() + static_cast<std::ptrdiff_t>(offset * kOutputs));
  }
  pass.elapsed_ms = Elapsed(task_start);
  return pass;
}

std::size_t Correct(const std::vector<float> &output,
                    const std::vector<std::uint8_t> &labels) {
  std::size_t correct = 0U;
  for (std::size_t row = 0U; row < kSamples; ++row) {
    Require(labels[row] < kOutputs, "Invalid frozen task label");
    const auto values = std::span(output).subspan(row * kOutputs, kOutputs);
    Require(std::all_of(values.begin(), values.end(),
                        [](float value) { return std::isfinite(value); }),
            "Nonfinite inference output");
    if (static_cast<std::size_t>(
            std::max_element(values.begin(), values.end()) - values.begin()) ==
        labels[row])
      ++correct;
  }
  return correct;
}

void Sample(const fs::path &fixture, const fs::path &prepared,
            const std::string &variant, const fs::path &output,
            Clock::time_point process_entry) {
  Require(variant == "baseline" || variant == "candidate", "Unknown variant");
  const auto read_start = Clock::now();
  const auto manifest_bytes =
      ReadBounded(prepared / "manifest.json", 1U << 20U);
  const auto manifest =
      Json::parse(reinterpret_cast<const char *>(manifest_bytes.data()),
                  reinterpret_cast<const char *>(manifest_bytes.data() +
                                                 manifest_bytes.size()));
  Require(manifest.at("schema") == "jyppx.frozen-linear-comparison/v1",
          "Wrong prepared manifest schema");
  const auto model = ReadBounded(prepared / (variant + ".onnx"), 16U << 20U);
  const auto input = Read<float>(fixture / "input.f32", kSamples * kInputs);
  const auto labels = Read<std::uint8_t>(fixture / "labels.u8", kSamples);
  Require(Digest(model) == manifest.at("models")
                               .at(variant)
                               .at("sha256")
                               .get<std::string>() &&
              Digest(input) ==
                  manifest.at("fixture").at("input.f32").get<std::string>() &&
              Digest(labels) ==
                  manifest.at("fixture").at("labels.u8").get<std::string>(),
          "Model or immutable task content changed");
  const auto read_and_verify_ms = Elapsed(read_start);
  rt::SessionOptions options;
  options.provider.name = "CPUExecutionProvider";
  options.allow_cpu_fallback = false;
  options.cpu_threads = 1U;
  options.disable_graph_optimizations = true;
  options.max_model_bytes = 16U << 20U;
  options.max_batch_size = kBatch;
  // The empty prefix is deliberate: formal timing NEVER enables profiling.
  Require(options.profile_prefix.empty(), "Profiling must be disabled");
  const auto load_start = Clock::now();
  rt::Session session(
      model,
      {.hex =
           manifest.at("models").at(variant).at("sha256").get<std::string>()},
      options);
  const auto load_ms = Elapsed(load_start);
  Require(session.runtime_version() == "1.23.2" &&
              session.execution_provider().name == "CPUExecutionProvider",
          "Actual runtime/backend changed");
  const auto cold = RunTask(session, input, process_entry);
  const auto cold_task_ready_ms = Elapsed(process_entry);
  const auto digest = Digest(cold.output);
  const auto correct = Correct(cold.output, labels);
  // An additional untimed full-data warmup separates cold and hot passes.
  const auto warm = RunTask(session, input, process_entry);
  Require(
      Digest(warm.output) == digest && Correct(warm.output, labels) == correct,
      "Warmup changed task output");
  Json hot = Json::array();
  for (std::size_t i = 0U; i < kHotPasses; ++i) {
    const auto pass = RunTask(session, input, process_entry);
    Require(Digest(pass.output) == digest &&
                Correct(pass.output, labels) == correct,
            "Hot pass changed task output");
    hot.push_back({{"task_ms", pass.elapsed_ms},
                   {"samples_per_second", 1000.0 * kSamples / pass.elapsed_ms},
                   {"correct", correct},
                   {"output_sha256", digest}});
  }
  const auto build = Json::parse(jyppx::liuding::build_fingerprint_json());
  const auto rss = PeakRss();
  // Use fixed 64KiB streaming metadata buffers, outside formal task timers.
  const auto runtime_identity_start = Clock::now();
  const auto runtime_library = LoadedRuntime();
  const auto runtime_identity_ms = Elapsed(runtime_identity_start);
  const auto accuracy = static_cast<double>(correct) / kSamples;
  const auto &thresholds = manifest.at("thresholds");
  const auto float_accuracy =
      static_cast<double>(
          manifest.at("frozen_float_correct").get<std::size_t>()) /
      kSamples;
  const bool quality =
      variant == "baseline"
          ? correct == manifest.at("frozen_float_correct").get<std::size_t>() &&
                accuracy >=
                    thresholds.at("minimum_float_accuracy").get<double>()
          : accuracy >= thresholds.at("minimum_int8_accuracy").get<double>() &&
                float_accuracy - accuracy <=
                    thresholds.at("maximum_accuracy_drop").get<double>();
  const bool memory = rss.at("bytes").get<std::uint64_t>() <=
                      thresholds.at("memory_budget_bytes").get<std::uint64_t>();
  Json report{
      {"schema", "jyppx.frozen-linear-sample/v1"},
      {"variant", variant},
      {"prepared_manifest_sha256", Digest(manifest_bytes)},
      {"model_sha256", Digest(model)},
      {"model_bytes", model.size()},
      {"input_sha256", Digest(input)},
      {"labels_sha256", Digest(labels)},
      {"build", build},
      {"runtime_version", session.runtime_version()},
      {"provider", session.execution_provider().name},
      {"runtime_library", runtime_library},
      {"runtime_identity_ms", runtime_identity_ms},
      {"cpu_threads", 1},
      {"interop_threads", 1},
      {"batch_size", kBatch},
      {"samples", kSamples},
      {"graph_optimizations", false},
      {"profiling", false},
      {"read_and_verify_ms", read_and_verify_ms},
      {"session_load_ms", load_ms},
      {"first_batch_ms", cold.first_batch_ms},
      {"first_batch_ready_ms", cold.first_batch_ready_ms},
      {"cold_task_ms", cold.elapsed_ms},
      {"cold_task_ready_ms", cold_task_ready_ms},
      {"warmup_full_passes", 1},
      {"hot", hot},
      {"peak_rss", rss},
      {"correct", correct},
      {"accuracy", accuracy},
      {"output_sha256", digest},
      {"quality_accepted", quality},
      {"frozen_memory_budget_accepted", memory},
      {"kernel_claim",
       "none from timing; bind separately qualified original graph digest"},
      {"timing_scope",
       "steady_clock; Session::run and result collection; "
       "validation/hash/output file excluded"},
      {"cold_scope",
       "new process/session; ordinary warm OS file cache; no "
       "storage cold claim"}};
  Require(fs::create_directory(output),
          "Sample output must be a new directory");
  Write(output / "output.f32", std::as_bytes(std::span(cold.output)));
  WriteJson(output / "sample.json", report);
  std::cout << report.dump() << '\n';
  Require(quality && memory,
          "Frozen quality or task memory budget failed; report retained");
}
}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t **argv) {
#else
int main(int argc, char **argv) {
#endif
  const auto process_entry = Clock::now();
  try {
    static_assert(std::endian::native == std::endian::little);
    Require(
        argc == 6,
        "Usage: compare prepare FIXTURE QUALIFIED_INT8 INT8_SHA NEW_OUTPUT; "
        "compare run FIXTURE PREPARED baseline|candidate NEW_OUTPUT");
    if (fs::path(argv[1]) == "prepare")
      Prepare(fs::path(argv[2]), fs::path(argv[3]), fs::path(argv[4]).string(),
              fs::path(argv[5]));
    else if (fs::path(argv[1]) == "run")
      Sample(fs::path(argv[2]), fs::path(argv[3]), fs::path(argv[4]).string(),
             fs::path(argv[5]), process_entry);
    else
      throw std::runtime_error("Unknown comparison command");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Frozen linear comparison failed: " << error.what() << '\n';
    return 1;
  }
}
