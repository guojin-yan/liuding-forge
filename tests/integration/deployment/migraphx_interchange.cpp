// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <migraphx/migraphx.h>
#include <migraphx/version.h>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
// psapi.h requires the Windows types declared first.
// clang-format off
#include <windows.h>
#include <psapi.h>
// clang-format on
#else
#include <link.h>
#endif

namespace {
static_assert(std::endian::native == std::endian::little);
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(MIGRAPHX_VERSION_MAJOR == 2 && MIGRAPHX_VERSION_MINOR == 18 &&
              MIGRAPHX_VERSION_PATCH == 0);

void Check(migraphx_status status, std::string_view operation) {
  if (status != migraphx_status_success)
    throw std::runtime_error(std::string(operation) + ": MIGraphX status " +
                             std::to_string(status));
}

template <class Handle, auto Destroy>
struct Owner {
  Handle value{};
  Owner() = default;
  Owner(const Owner&) = delete;
  Owner& operator=(const Owner&) = delete;
  ~Owner() {
    if (value != nullptr) Destroy(value);
  }
};

struct Fixture {
  std::string_view graph_sha;
  std::string_view input_sha;
  std::string_view expected_sha;
  std::vector<std::size_t> input_shape;
  std::vector<std::size_t> output_shape;
};

// Fixed clean native-exporter artifacts; no regeneration or tolerance
// overrides.
Fixture Select(std::string_view model, std::string_view index) {
  const bool second = index == "1";
  if (index != "0" && !second) throw std::runtime_error("case must be 0 or 1");
  const std::size_t batch = second ? 5U : 2U;
  if (model == "linear")
    return {
        "7b4f9bf96cd72143e22c906b0063958ef05b10534785eac3e9fc1b07afcb990a",
        second
            ? "bafca57808f22a3d4d4d70225cabecb085132ff4c9de22fd4692abd7abee94b7"
            : "d55f7abaa5bb2c75cc098aa3dc97579967a294f802b308803833fef7964a94e"
              "9",
        second
            ? "509b6a760633711b6655feca75f3d873d5184fc1076cc55213773a8b91be5529"
            : "3b106db38a6dca7aafe95d1af546237ca38ac4d74a084a8f1b0cea5a361c709"
              "a",
        {batch, 8},
        {batch, 3}};
  if (model == "resnet18")
    return {
        "95fbd4284bb53e1559c3448d49200836f50731bc89ac9694ed1f478c1581a399",
        second
            ? "697c148d7f14443e742ca11f62dd028a582fb3a216647a6cd0fd99b56f6f538e"
            : "e72a6421bd642adf1c13e70636c800abdbbda4fecfbce5413b21ad42a186351"
              "e",
        second
            ? "a80ff115fbbddc54314cdebace0f302765ba4fc311e2f52328952ec23c653165"
            : "4c8b23c87357417e3c647d9bc7d8a3db1cc8f1a8897bda5088865910b56b489"
              "f",
        {batch, 3, 224, 224},
        {batch, 7}};
  throw std::runtime_error("model must be linear or resnet18");
}

std::size_t Elements(std::span<const std::size_t> shape) {
  std::size_t result = 1;
  for (const auto dimension : shape) {
    if (dimension == 0 || dimension > (64U << 20U) / sizeof(float) / result)
      throw std::runtime_error("tensor shape exceeds test bound");
    result *= dimension;
  }
  return result;
}

std::vector<std::byte> Read(const std::filesystem::path& path,
                            std::size_t maximum, std::string_view digest) {
  const auto size = std::filesystem::file_size(path);
  if (size == 0 || size > maximum)
    throw std::runtime_error("fixture size bound");
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  std::ifstream stream(path, std::ios::binary);
  stream.read(reinterpret_cast<char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
  if (!stream || jyppx::liuding::sha256_hex(bytes) != digest)
    throw std::runtime_error("fixed fixture SHA-256 mismatch: " +
                             path.string());
  return bytes;
}

std::vector<float> Tensor(const std::filesystem::path& path,
                          std::span<const std::size_t> shape,
                          std::string_view digest) {
  const auto count = Elements(shape);
  const auto bytes = Read(path, count * sizeof(float), digest);
  if (bytes.size() != count * sizeof(float))
    throw std::runtime_error("fixed tensor byte count mismatch");
  std::vector<float> result(count);
  std::memcpy(result.data(), bytes.data(), bytes.size());
  if (!std::all_of(result.begin(), result.end(),
                   [](float x) { return std::isfinite(x); }))
    throw std::runtime_error("nonfinite fixed tensor");
  return result;
}

void Shape(const_migraphx_shape_t shape,
           std::span<const std::size_t> expected) {
  const std::size_t* lengths = nullptr;
  std::size_t rank = 0, count = 0, bytes = 0;
  bool standard = false;
  migraphx_shape_datatype_t type{};
  Check(migraphx_shape_lengths(&lengths, &rank, shape), "shape_lengths");
  Check(migraphx_shape_elements(&count, shape), "shape_elements");
  Check(migraphx_shape_bytes(&bytes, shape), "shape_bytes");
  Check(migraphx_shape_type(&type, shape), "shape_type");
  Check(migraphx_shape_standard(&standard, shape), "shape_standard");
  if (rank != expected.size() ||
      !std::equal(expected.begin(), expected.end(), lengths) ||
      count != Elements(expected) || bytes != count * sizeof(float) ||
      !standard || type != migraphx_shape_float_type)
    throw std::runtime_error("unexpected MIGraphX tensor signature");
}

struct Trace {
  std::vector<std::string> names;
  bool failed = false;
};

void Modules(const std::filesystem::path& path) {
  std::ofstream output(path, std::ios::binary);
#ifdef _WIN32
  std::vector<HMODULE> handles(4096);
  DWORD needed = 0;
  if (!EnumProcessModulesEx(
          GetCurrentProcess(), handles.data(),
          static_cast<DWORD>(handles.size() * sizeof(HMODULE)), &needed,
          LIST_MODULES_ALL) ||
      needed > handles.size() * sizeof(HMODULE))
    throw std::runtime_error("module enumeration failed");
  handles.resize(needed / sizeof(HMODULE));
  for (auto handle : handles) {
    std::wstring name(32768, L'\0');
    const auto count = GetModuleFileNameW(handle, name.data(),
                                          static_cast<DWORD>(name.size()));
    if (count == 0 || count >= name.size())
      throw std::runtime_error("module path failed");
    const auto bytes = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                           name.data(), static_cast<int>(count),
                                           nullptr, 0, nullptr, nullptr);
    if (bytes <= 0) throw std::runtime_error("module path encoding failed");
    std::string utf8(static_cast<std::size_t>(bytes), '\0');
    if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, name.data(),
                            static_cast<int>(count), utf8.data(), bytes,
                            nullptr, nullptr) != bytes)
      throw std::runtime_error("module path conversion failed");
    output << utf8 << '\n';
  }
#else
  dl_iterate_phdr(
      [](dl_phdr_info* info, std::size_t, void* data) {
        if (info->dlpi_name != nullptr && *info->dlpi_name != '\0')
          *static_cast<std::ofstream*>(data) << info->dlpi_name << '\n';
        return 0;
      },
      &output);
#endif
  output.close();
  if (!output) throw std::runtime_error("module receipt write failure");
}

migraphx_status Record(migraphx_trace_info_t info, void* opaque) noexcept {
  auto& trace = *static_cast<Trace*>(opaque);
  try {
    const char* name = nullptr;
    Check(migraphx_trace_info_get_name(&name, info), "trace_info_get_name");
    if (name == nullptr || trace.names.size() >= 10000U)
      throw std::runtime_error("invalid or excessive instruction trace");
    trace.names.emplace_back(name);
    return migraphx_status_success;
  } catch (...) {
    trace.failed = true;
    return migraphx_status_unknown_error;
  }
}

std::filesystem::path Evidence(const std::filesystem::path& parent,
                               std::string_view model, std::string_view index) {
  std::filesystem::create_directories(parent);
  const auto tick = std::chrono::system_clock::now().time_since_epoch().count();
  for (unsigned counter = 0; counter < 100U; ++counter) {
    const auto path =
        parent / (std::string(model) + "-" + std::string(index) + "-" +
                  std::to_string(tick) + "-" + std::to_string(counter));
    if (std::filesystem::create_directory(path)) return path;
  }
  throw std::runtime_error("cannot allocate fresh evidence directory");
}

std::filesystem::path Utf8Path(const char* text) {
  std::u8string value;
  for (const unsigned char character : std::string_view(text))
    value.push_back(static_cast<char8_t>(character));
  return std::filesystem::path(value);
}

bool Digest(std::string_view value) {
  return value.size() == 64U &&
         std::all_of(value.begin(), value.end(), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

void Write(const std::filesystem::path& path,
           std::span<const std::byte> bytes) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  stream.close();
  if (!stream) throw std::runtime_error("artifact write failure");
}

int Run(int argc, char** argv) {
  const bool reload = argc == 10 && std::string_view(argv[5]) == "--reload";
  if (argc != 5 && !reload)
    throw std::runtime_error(
        "expected fixture_root model case evidence_parent "
        "[--reload compiled_program program_sha baseline_output output_sha]");
  const auto fixture = Select(argv[2], argv[3]);
  const auto folder = Utf8Path(argv[1]) / argv[2];
  const auto graph =
      Read(folder / "model.onnx", 256U << 20U, fixture.graph_sha);
  const auto prefix = "case-" + std::string(argv[3]);
  auto input = Tensor(folder / (prefix + "-input.f32"), fixture.input_shape,
                      fixture.input_sha);
  const auto expected = Tensor(folder / (prefix + "-output.f32"),
                               fixture.output_shape, fixture.expected_sha);
  std::vector<std::byte> serialized;
  std::vector<float> baseline;
  if (reload) {
    if (!Digest(argv[7]) || !Digest(argv[9]))
      throw std::runtime_error("reload requires lowercase SHA-256 digests");
    serialized = Read(Utf8Path(argv[6]), 512U << 20U, argv[7]);
    baseline = Tensor(Utf8Path(argv[8]), fixture.output_shape, argv[9]);
  }
  const auto output = Evidence(Utf8Path(argv[4]), argv[2], argv[3]);
  std::cout << "evidence=" << output.string() << '\n';
  std::cout << "MIGraphX 2.18.0 gpu FP32; separately specialized shape; "
               "offload_copy=true fast_math=false exhaustive_tune=false\n";
  Owner<migraphx_onnx_options_t, migraphx_onnx_options_destroy> onnx;
  auto dimensions = fixture.input_shape;
  Owner<migraphx_program_t, migraphx_program_destroy> program;
  Owner<migraphx_target_t, migraphx_target_destroy> target;
  Owner<migraphx_compile_options_t, migraphx_compile_options_destroy> options;
  Owner<migraphx_file_options_t, migraphx_file_options_destroy> file_options;
  Check(migraphx_file_options_create(&file_options.value),
        "file_options_create");
  Check(migraphx_file_options_set_file_format(file_options.value, "msgpack"),
        "file_format");
  const auto compiled_path = (output / "compiled.mxr").u8string();
  if (reload) {
    // Load a private copy of the verified bytes, never the caller's mutable
    // path.
    Write(output / "compiled.mxr", serialized);
    Write(output / "baseline.f32", std::as_bytes(std::span(baseline)));
    Check(migraphx_load(&program.value,
                        reinterpret_cast<const char*>(compiled_path.c_str()),
                        file_options.value),
          "load_compiled_program");
    Read(output / "compiled.mxr", 512U << 20U, argv[7]);
    std::cout << "reload_program_sha256=" << argv[7]
              << " reload_baseline_sha256=" << argv[9]
              << "; no ONNX parse or program_compile API call\n";
  } else {
    Check(migraphx_onnx_options_create(&onnx.value), "onnx_options_create");
    Check(migraphx_onnx_options_set_input_parameter_shape(
              onnx.value, "input", dimensions.data(), dimensions.size()),
          "set_input_parameter_shape");
    Check(migraphx_parse_onnx_buffer(&program.value, graph.data(), graph.size(),
                                     onnx.value),
          "parse_onnx_buffer");
    Check(migraphx_target_create(&target.value, "gpu"), "target_create_gpu");
    Check(migraphx_compile_options_create(&options.value),
          "compile_options_create");
    Check(migraphx_compile_options_set_offload_copy(options.value, true),
          "offload_copy");
    Check(migraphx_compile_options_set_fast_math(options.value, false),
          "fast_math");
    Check(
        migraphx_compile_options_set_exhaustive_tune_flag(options.value, false),
        "exhaustive_tune");
    Check(migraphx_program_compile(program.value, target.value, options.value),
          "program_compile_gpu");
    Check(migraphx_save(program.value,
                        reinterpret_cast<const char*>(compiled_path.c_str()),
                        file_options.value),
          "save_compiled_program");
  }
  Check(migraphx_program_print(program.value), "program_print");
  Owner<migraphx_program_parameter_shapes_t,
        migraphx_program_parameter_shapes_destroy>
      shapes;
  Check(migraphx_program_get_parameter_shapes(&shapes.value, program.value),
        "parameter_shapes");
  std::size_t inputs = 0;
  Check(migraphx_program_parameter_shapes_size(&inputs, shapes.value),
        "parameter_count");
  if (inputs != 1U) throw std::runtime_error("unexpected parameter count");
  const char* name = nullptr;
  Check(migraphx_program_parameter_shapes_names(&name, shapes.value),
        "parameter_name");
  if (name == nullptr || std::string_view(name) != "input")
    throw std::runtime_error("unexpected input name");
  const_migraphx_shape_t input_shape = nullptr;
  Check(migraphx_program_parameter_shapes_get(&input_shape, shapes.value, name),
        "parameter_shape");
  Shape(input_shape, fixture.input_shape);
  Owner<migraphx_argument_t, migraphx_argument_destroy> argument;
  Check(migraphx_argument_create(&argument.value, input_shape, input.data()),
        "input_argument");
  Owner<migraphx_program_parameters_t, migraphx_program_parameters_destroy>
      parameters;
  Check(migraphx_program_parameters_create(&parameters.value),
        "parameters_create");
  Check(migraphx_program_parameters_add(parameters.value, name, argument.value),
        "parameters_add");
  Trace trace;
  Owner<migraphx_arguments_t, migraphx_arguments_destroy> results;
  const auto run_status = migraphx_program_run_trace(
      &results.value, program.value, parameters.value, Record, &trace);
  std::ofstream dispatch(output / "dispatch.txt", std::ios::binary);
  for (const auto& operation : trace.names) dispatch << operation << '\n';
  dispatch.close();
  if (!dispatch) throw std::runtime_error("trace write failure");
  Modules(output / "modules.txt");
  Check(run_status, "program_run_trace");
  const auto any = [&](std::string_view prefix_value) {
    return std::any_of(
        trace.names.begin(), trace.names.end(), [&](const auto& operation) {
          return std::string_view(operation).starts_with(prefix_value);
        });
  };
  if (trace.failed ||
      !(any("gpu::hip_gemm") || any("gpu::convolution") ||
        any("gpu::code_object")) ||
      !any("hip::copy_from_gpu") || !any("hip::sync_stream") || any("ref::") ||
      any("cpu::"))
    throw std::runtime_error(
        "missing GPU arithmetic/synchronized host output trace");
  std::size_t count = 0;
  Check(migraphx_arguments_size(&count, results.value), "output_count");
  if (count != 1U) throw std::runtime_error("unexpected output count");
  const_migraphx_argument_t result = nullptr;
  const_migraphx_shape_t output_shape = nullptr;
  Check(migraphx_arguments_get(&result, results.value, 0), "output_argument");
  Check(migraphx_argument_shape(&output_shape, result), "output_shape");
  Shape(output_shape, fixture.output_shape);
  char* buffer = nullptr;
  Check(migraphx_argument_buffer(&buffer, result), "host_output_buffer");
  if (buffer == nullptr) throw std::runtime_error("missing output buffer");
  std::vector<float> actual(expected.size());
  std::memcpy(actual.data(), buffer, actual.size() * sizeof(float));
  const auto actual_bytes = std::as_bytes(std::span(actual));
  std::ofstream payload(output / "actual.f32", std::ios::binary);
  payload.write(reinterpret_cast<const char*>(actual_bytes.data()),
                static_cast<std::streamsize>(actual_bytes.size()));
  payload.close();
  if (!payload) throw std::runtime_error("output write failure");
  std::size_t violations = 0;
  double max_abs = 0, max_ratio = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double difference =
        std::abs(static_cast<double>(actual[i]) - expected[i]);
    const double tolerance =
        1e-5 + 1e-4 * std::abs(static_cast<double>(expected[i]));
    if (!std::isfinite(actual[i]) || difference > tolerance) ++violations;
    max_abs = std::max(max_abs, difference);
    max_ratio = std::max(max_ratio, difference / tolerance);
  }
  std::cout.precision(17);
  std::cout << "output_sha256=" << jyppx::liuding::sha256_hex(actual_bytes)
            << " elements=" << actual.size() << " violations=" << violations
            << " max_abs=" << max_abs << " max_tolerance_ratio=" << max_ratio
            << " trace_instructions=" << trace.names.size() << '\n';
  if (reload) {
    const bool identical =
        std::memcmp(actual.data(), baseline.data(), actual_bytes.size()) == 0;
    std::cout << "byte_identical_to_compiled_output=" << std::boolalpha
              << identical << '\n';
    if (!identical) return 3;
  }
  return violations == 0U ? 0 : 2;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::cout.setf(std::ios::unitbuf);
    return Run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
