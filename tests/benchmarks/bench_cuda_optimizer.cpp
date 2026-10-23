// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/liuding.hpp>

#include <torch/cuda.h>

#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kParameterCount = 8;
constexpr std::size_t kElementsPerParameter = 262144;
constexpr std::size_t kSteps = 200;
constexpr std::uint64_t kSeed = 20261020;
constexpr std::string_view kWorkloadId = "jyppx.cuda.optimizer.dense/v1";
constexpr std::string_view kGradientPolicy = "fixed-dense-per-parameter/v1";
constexpr std::string_view kOptionProfile = "stateful-representative/v1";
constexpr std::string_view kWorkloadDescriptor =
    "id=jyppx.cuda.optimizer.dense/v1;parameters=8;elements_each=262144;"
    "steps=200;dtype=float32;seed=20261020;gradient=fixed-dense-per-"
    "parameter/v1;options=stateful-representative/v1";

/** @brief Selects one optimizer under the frozen CUDA workload. */
enum class BenchmarkMode {
  sgd,       ///< Delegated SGD with Nesterov momentum.
  adam,      ///< Delegated Adam with AMSGrad state.
  adamw,     ///< Delegated AdamW with AMSGrad state.
  rmsprop,   ///< Delegated centered RMSprop with momentum.
  adagrad,   ///< Delegated Adagrad with decay and initialized accumulators.
  nadam,     ///< Native decoupled-weight-decay NAdam.
  adadelta,  ///< Native stateful Adadelta.
  lamb,      ///< Native LAMB with trust-ratio updates.
};

/** @brief Owns measured values and semantic postconditions. */
struct BenchmarkResult {
  double parameter_elements_per_second = 0.0;  ///< Updated element rate.
  double mean_step_milliseconds = 0.0;         ///< Mean synchronized step time.
  std::int64_t peak_allocated_bytes = 0;       ///< CUDA allocated-byte peak.
  std::int64_t peak_reserved_bytes = 0;        ///< CUDA reserved-byte peak.
  std::uint64_t completed_steps = 0;           ///< Committed optimizer steps.
  bool parameters_finite = false;   ///< Whether all outputs are finite.
  bool parameters_changed = false;  ///< Whether any output changed.
};

/** @brief Returns the stable command-line spelling of a benchmark mode. */
std::string_view mode_name(BenchmarkMode mode) {
  switch (mode) {
    case BenchmarkMode::sgd:
      return "cuda-fp32-optimizer-sgd";
    case BenchmarkMode::adam:
      return "cuda-fp32-optimizer-adam";
    case BenchmarkMode::adamw:
      return "cuda-fp32-optimizer-adamw";
    case BenchmarkMode::rmsprop:
      return "cuda-fp32-optimizer-rmsprop";
    case BenchmarkMode::adagrad:
      return "cuda-fp32-optimizer-adagrad";
    case BenchmarkMode::nadam:
      return "cuda-fp32-optimizer-nadam";
    case BenchmarkMode::adadelta:
      return "cuda-fp32-optimizer-adadelta";
    case BenchmarkMode::lamb:
      return "cuda-fp32-optimizer-lamb";
  }
  throw std::logic_error("unreachable optimizer benchmark mode");
}

/** @brief Parses exactly one required `--mode` argument. */
BenchmarkMode parse_arguments(int argc, char** argv) {
  if (argc != 3 || std::string_view(argv[1]) != "--mode") {
    throw std::invalid_argument(
        "usage: liuding-forge-bench-cuda-optimizer --mode "
        "<cuda-fp32-optimizer-{sgd|adam|adamw|rmsprop|adagrad|nadam|"
        "adadelta|lamb}>");
  }
  const std::string_view value(argv[2]);
  for (const auto mode :
       {BenchmarkMode::sgd, BenchmarkMode::adam, BenchmarkMode::adamw,
        BenchmarkMode::rmsprop, BenchmarkMode::adagrad, BenchmarkMode::nadam,
        BenchmarkMode::adadelta, BenchmarkMode::lamb}) {
    if (value == mode_name(mode)) {
      return mode;
    }
  }
  throw std::invalid_argument("unsupported CUDA optimizer benchmark mode: " +
                              std::string(value));
}

/** @brief Computes the stable FNV-1a digest used for workload identity. */
std::uint64_t fnv1a64(std::string_view value) noexcept {
  std::uint64_t digest = 14695981039346656037ULL;
  for (const unsigned char byte : value) {
    digest ^= byte;
    digest *= 1099511628211ULL;
  }
  return digest;
}

/** @brief Formats the frozen workload digest as lowercase hexadecimal. */
std::string workload_digest() {
  std::ostringstream output;
  output << std::hex << std::setfill('0') << std::setw(16)
         << fnv1a64(kWorkloadDescriptor);
  return output.str();
}

/** @brief Escapes one string for compact JSON output. */
std::string json_escape(std::string_view value) {
  std::ostringstream output;
  for (const unsigned char byte : value) {
    switch (byte) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (byte < 0x20U) {
          output << "\\u" << std::hex << std::setfill('0') << std::setw(4)
                 << static_cast<unsigned int>(byte) << std::dec;
        } else {
          output << static_cast<char>(byte);
        }
        break;
    }
  }
  return output.str();
}

/** @brief Builds the common stable parameter bindings for one sample. */
std::vector<jyppx::liuding::optim::OptimizerParameterBinding> make_bindings(
    const std::vector<jyppx::liuding::Tensor>& parameters) {
  std::vector<jyppx::liuding::optim::OptimizerParameterBinding> bindings;
  bindings.reserve(parameters.size());
  for (std::size_t index = 0; index < parameters.size(); ++index) {
    bindings.push_back({.parameter_id = "parameter." + std::to_string(index),
                        .parameter = parameters[index]});
  }
  return bindings;
}

/** @brief Creates the selected adapter with a stateful option profile. */
std::unique_ptr<jyppx::liuding::optim::OptimizerAdapter> make_optimizer(
    BenchmarkMode mode,
    const std::vector<jyppx::liuding::optim::OptimizerParameterBinding>& bindings) {
  switch (mode) {
    case BenchmarkMode::sgd:
      return std::make_unique<jyppx::liuding::optim::SgdOptimizerAdapter>(
          "optimizer.benchmark.cuda.sgd",
          std::vector<jyppx::liuding::optim::SgdParameterGroup>{
              {.group_id = "group.dense",
               .parameters = bindings,
               .options = {
                   .learning_rate = 1e-3, .momentum = 0.9, .nesterov = true}}});
    case BenchmarkMode::adam:
      return std::make_unique<jyppx::liuding::optim::AdamOptimizerAdapter>(
          "optimizer.benchmark.cuda.adam",
          std::vector<jyppx::liuding::optim::AdamParameterGroup>{
              {.group_id = "group.dense",
               .parameters = bindings,
               .options = {.learning_rate = 1e-3,
                           .weight_decay = 0.01,
                           .amsgrad = true}}});
    case BenchmarkMode::adamw:
      return std::make_unique<jyppx::liuding::optim::AdamWOptimizerAdapter>(
          "optimizer.benchmark.cuda.adamw",
          std::vector<jyppx::liuding::optim::AdamWParameterGroup>{
              {.group_id = "group.dense",
               .parameters = bindings,
               .options = {.learning_rate = 1e-3,
                           .weight_decay = 0.01,
                           .amsgrad = true}}});
    case BenchmarkMode::rmsprop:
      return std::make_unique<jyppx::liuding::optim::RmspropOptimizerAdapter>(
          "optimizer.benchmark.cuda.rmsprop",
          std::vector<jyppx::liuding::optim::RmspropParameterGroup>{
              {.group_id = "group.dense",
               .parameters = bindings,
               .options = {.learning_rate = 1e-3,
                           .weight_decay = 0.01,
                           .momentum = 0.9,
                           .centered = true}}});
    case BenchmarkMode::adagrad:
      return std::make_unique<jyppx::liuding::optim::AdagradOptimizerAdapter>(
          "optimizer.benchmark.cuda.adagrad",
          std::vector<jyppx::liuding::optim::AdagradParameterGroup>{
              {.group_id = "group.dense",
               .parameters = bindings,
               .options = {.learning_rate = 1e-2,
                           .learning_rate_decay = 1e-4,
                           .weight_decay = 0.01,
                           .initial_accumulator_value = 0.1}}});
    case BenchmarkMode::nadam:
      return std::make_unique<jyppx::liuding::optim::NAdamOptimizerAdapter>(
          "optimizer.benchmark.cuda.nadam",
          std::vector<jyppx::liuding::optim::NAdamParameterGroup>{
              {.group_id = "group.dense",
               .parameters = bindings,
               .options = {.learning_rate = 2e-3,
                           .weight_decay = 0.01,
                           .decoupled_weight_decay = true}}});
    case BenchmarkMode::adadelta:
      return std::make_unique<jyppx::liuding::optim::AdadeltaOptimizerAdapter>(
          "optimizer.benchmark.cuda.adadelta",
          std::vector<jyppx::liuding::optim::AdadeltaParameterGroup>{
              {.group_id = "group.dense",
               .parameters = bindings,
               .options = {.learning_rate = 1.0, .weight_decay = 0.01}}});
    case BenchmarkMode::lamb:
      return std::make_unique<jyppx::liuding::optim::LambOptimizerAdapter>(
          "optimizer.benchmark.cuda.lamb",
          std::vector<jyppx::liuding::optim::LambParameterGroup>{
              {.group_id = "group.dense",
               .parameters = bindings,
               .options = {.learning_rate = 1e-3, .weight_decay = 0.01}}});
  }
  throw std::logic_error("unreachable optimizer construction mode");
}

/** @brief Runs and validates one isolated CUDA optimizer sample. */
BenchmarkResult run_benchmark(BenchmarkMode mode) {
  torch::manual_seed(kSeed);
  torch::cuda::manual_seed_all(kSeed);
  const auto tensor_options = torch::TensorOptions()
                                  .dtype(torch::kFloat32)
                                  .device(torch::Device(torch::kCUDA, 0));
  std::vector<jyppx::liuding::Tensor> parameters;
  std::vector<jyppx::liuding::Tensor> originals;
  parameters.reserve(kParameterCount);
  originals.reserve(kParameterCount);
  for (std::size_t index = 0; index < kParameterCount; ++index) {
    auto parameter =
        torch::randn({static_cast<std::int64_t>(kElementsPerParameter)},
                     tensor_options)
            .set_requires_grad(true);
    parameter.mutable_grad() =
        torch::full_like(parameter, 1e-4 * static_cast<double>(index + 1));
    originals.push_back(parameter.detach().clone());
    parameters.push_back(std::move(parameter));
  }
  auto optimizer = make_optimizer(mode, make_bindings(parameters));

  constexpr c10::DeviceIndex kDeviceIndex = 0;
  torch::cuda::synchronize(kDeviceIndex);
  c10::cuda::CUDACachingAllocator::resetPeakStats(kDeviceIndex);
  const auto started = std::chrono::steady_clock::now();
  for (std::size_t step = 0; step < kSteps; ++step) {
    optimizer->step();
  }
  torch::cuda::synchronize(kDeviceIndex);
  const auto elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  const auto allocator_stats =
      c10::cuda::CUDACachingAllocator::getDeviceStats(kDeviceIndex);
  // LibTorch stores the aggregate statistic in slot zero across supported
  // 2.5.x CUDA distributions even though the owning enum moved namespaces.
  constexpr std::size_t kAggregate = 0;

  bool parameters_finite = true;
  bool parameters_changed = false;
  for (std::size_t index = 0; index < parameters.size(); ++index) {
    parameters_finite = parameters_finite &&
                        torch::isfinite(parameters[index]).all().item<bool>();
    parameters_changed = parameters_changed ||
                         !torch::equal(parameters[index], originals[index]);
  }
  const auto total_elements = kParameterCount * kElementsPerParameter;
  BenchmarkResult result{
      .parameter_elements_per_second =
          static_cast<double>(total_elements * kSteps) / elapsed,
      .mean_step_milliseconds = elapsed * 1000.0 / kSteps,
      .peak_allocated_bytes = allocator_stats.allocated_bytes[kAggregate].peak,
      .peak_reserved_bytes = allocator_stats.reserved_bytes[kAggregate].peak,
      .completed_steps = optimizer->completed_steps(),
      .parameters_finite = parameters_finite,
      .parameters_changed = parameters_changed};
  if (elapsed <= 0.0 || result.peak_allocated_bytes <= 0 ||
      result.peak_reserved_bytes <= 0) {
    throw std::runtime_error("CUDA timing or allocator statistics are invalid");
  }
  if (result.completed_steps != kSteps) {
    throw std::runtime_error("optimizer did not commit the frozen step count");
  }
  if (!result.parameters_finite || !result.parameters_changed) {
    throw std::runtime_error("optimizer output postconditions failed");
  }
  return result;
}

/** @brief Writes the validated sample as one compact JSON document. */
void write_sample(BenchmarkMode mode, const BenchmarkResult& result) {
  const auto* properties = at::cuda::getDeviceProperties(0);
  std::cout << std::setprecision(17) << '{' << "\"schema_version\":1,"
            << "\"kind\":\"jyppx-cuda-optimizer-benchmark-sample\","
            << "\"mode\":\"" << mode_name(mode) << "\","
            << "\"workload\":{"
            << "\"id\":\"" << kWorkloadId << "\","
            << "\"digest\":\"fnv1a64:" << workload_digest() << "\","
            << "\"seed\":" << kSeed << ',' << "\"steps\":" << kSteps << ','
            << "\"parameter_count\":" << kParameterCount << ','
            << "\"parameter_elements\":"
            << kParameterCount * kElementsPerParameter << ','
            << "\"dtype\":\"float32\","
            << "\"gradient_policy\":\"" << kGradientPolicy << "\","
            << "\"option_profile\":\"" << kOptionProfile << "\"},"
            << "\"build_fingerprint\":" << jyppx::liuding::build_fingerprint_json()
            << ',' << "\"cuda\":{"
            << "\"device_index\":0,"
            << "\"device_name\":\"" << json_escape(properties->name) << "\","
            << "\"compute_capability\":\"" << properties->major << '.'
            << properties->minor << "\","
            << "\"compile_toolkit_version\":" << CUDART_VERSION << "},"
            << "\"runtime\":{"
            << "\"intraop_threads\":" << torch::get_num_threads() << ','
            << "\"interop_threads\":" << torch::get_num_interop_threads()
            << "},"
            << "\"result\":{"
            << "\"parameter_elements_per_second\":"
            << result.parameter_elements_per_second << ','
            << "\"mean_step_milliseconds\":" << result.mean_step_milliseconds
            << ',' << "\"peak_allocated_bytes\":" << result.peak_allocated_bytes
            << ',' << "\"peak_reserved_bytes\":" << result.peak_reserved_bytes
            << ',' << "\"completed_steps\":" << result.completed_steps << ','
            << "\"parameters_finite\":"
            << (result.parameters_finite ? "true" : "false") << ','
            << "\"parameters_changed\":"
            << (result.parameters_changed ? "true" : "false") << "}}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (!jyppx::liuding::cuda_is_available()) {
      throw std::runtime_error(
          "CUDA optimizer benchmark requires CUDA LibTorch and a visible GPU");
    }
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    const auto mode = parse_arguments(argc, argv);
    write_sample(mode, run_benchmark(mode));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CUDA optimizer benchmark error: " << error.what() << '\n';
    return 2;
  }
}
