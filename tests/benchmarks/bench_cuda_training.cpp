// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/jyppx.hpp>

#include <torch/cuda.h>

#include <ATen/cuda/CUDAContextLight.h>
#include <c10/core/Allocator.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kBatchSize = 128;
constexpr std::size_t kSteps = 200;
constexpr std::size_t kInputFeatures = 1024;
constexpr std::size_t kHiddenFeatures = 2048;
constexpr std::size_t kOutputFeatures = 256;
constexpr std::size_t kWorkerCount = 2;
constexpr std::size_t kPrefetchBatches = 4;
constexpr std::uint64_t kSeed = 20261020;
constexpr double kInitialScale = 128.0;
constexpr std::string_view kWorkloadId = "jyppx.cuda.training.mlp/v1";
constexpr std::string_view kWorkloadDescriptor =
    "id=jyppx.cuda.training.mlp/v1;batch=128;steps=200;input=1024;"
    "hidden=2048;output=256;workers=2;prefetch=4;seed=20261020;"
    "pin_memory=true;non_blocking=true;stream=caller_current";

/** @brief Selects one frozen CUDA training path. */
enum class BenchmarkMode {
  fp32_single_sgd,     ///< Full precision with one SGD optimizer.
  fp16_multi_sgd_amp,  ///< Float16 autocast with two SGD optimizers and AMP.
};

/** @brief Dense regression network used by the CUDA Trainer benchmark. */
class CudaBenchmarkModel final : public torch::nn::Module {
 public:
  /** @brief Registers the frozen three-layer benchmark network. */
  CudaBenchmarkModel()
      : network_(register_module(
            "network",
            torch::nn::Sequential(
                torch::nn::Linear(kInputFeatures, kHiddenFeatures),
                torch::nn::ReLU(),
                torch::nn::Linear(kHiddenFeatures, kHiddenFeatures),
                torch::nn::ReLU(),
                torch::nn::Linear(kHiddenFeatures, kOutputFeatures)))) {}

  /** @brief Produces one regression output tensor. */
  jyppx::Tensor forward(const jyppx::Tensor& input) {
    return network_->forward(input);
  }

 private:
  torch::nn::Sequential network_{nullptr};  ///< Registered dense network.
};

/** @brief Owns the measured values and validated training counters. */
struct BenchmarkResult {
  double samples_per_second = 0.0;        ///< End-to-end committed sample rate.
  double mean_step_milliseconds = 0.0;    ///< Mean wall time per Trainer step.
  std::int64_t peak_allocated_bytes = 0;  ///< CUDA allocator allocated peak.
  std::int64_t peak_reserved_bytes = 0;   ///< CUDA allocator reserved peak.
  std::uint64_t global_step = 0;          ///< Trainer committed update count.
  std::vector<std::uint64_t>
      optimizer_steps;  ///< Completed count for every configured optimizer.
  std::optional<double> scaler_scale;  ///< Final scale for the AMP mode.
  bool scaler_checkpoint_ready =
      true;  ///< Whether scaler state is at a checkpoint boundary.
};

/** @brief Returns the stable command-line spelling of a benchmark mode. */
std::string_view mode_name(BenchmarkMode mode) {
  switch (mode) {
    case BenchmarkMode::fp32_single_sgd:
      return "cuda-fp32-single-sgd";
    case BenchmarkMode::fp16_multi_sgd_amp:
      return "cuda-fp16-multi-sgd-amp";
  }
  throw std::logic_error("unreachable benchmark mode");
}

/** @brief Parses exactly one required `--mode` argument. */
BenchmarkMode parse_arguments(int argc, char** argv) {
  if (argc != 3 || std::string_view(argv[1]) != "--mode") {
    throw std::invalid_argument(
        "usage: liuding-forge-bench-cuda-training --mode "
        "<cuda-fp32-single-sgd|cuda-fp16-multi-sgd-amp>");
  }
  const std::string_view value(argv[2]);
  if (value == mode_name(BenchmarkMode::fp32_single_sgd)) {
    return BenchmarkMode::fp32_single_sgd;
  }
  if (value == mode_name(BenchmarkMode::fp16_multi_sgd_amp)) {
    return BenchmarkMode::fp16_multi_sgd_amp;
  }
  throw std::invalid_argument("unsupported CUDA benchmark mode: " +
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

/** @brief Formats a 64-bit workload digest as fixed-width lowercase hex. */
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
      case '\"':
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

/** @brief Creates deterministic CPU tensors for the frozen workload. */
std::pair<jyppx::Tensor, jyppx::Tensor> make_dataset() {
  const auto sample_count = static_cast<std::int64_t>(kBatchSize * kSteps);
  const auto options = torch::TensorOptions().dtype(torch::kFloat32);
  auto inputs = torch::randn(
      {sample_count, static_cast<std::int64_t>(kInputFeatures)}, options);
  auto projection = torch::randn({static_cast<std::int64_t>(kInputFeatures),
                                  static_cast<std::int64_t>(kOutputFeatures)},
                                 options);
  auto targets =
      inputs.matmul(projection) / static_cast<double>(kInputFeatures);
  return {std::move(inputs), std::move(targets)};
}

/** @brief Creates the CPU-to-CUDA map pipeline used by both modes. */
std::shared_ptr<jyppx::data::DataPipeline> make_pipeline(
    const jyppx::Tensor& inputs, const jyppx::Tensor& targets) {
  constexpr std::size_t kSampleCount = kBatchSize * kSteps;
  return std::make_shared<jyppx::data::DataPipeline>(
      std::make_shared<jyppx::data::samplers::SequentialSampler>(kSampleCount),
      jyppx::data::DataPipelineOptions{
          .pipeline_id = "data.benchmark.cuda",
          .dataset_version = "benchmark.cuda.synthetic/v1",
          .dataset_size = kSampleCount,
          .schema = jyppx::data::BatchSchema::any(
              "benchmark.cuda.regression.batch", 1),
          .batch_factory_id = "benchmark.cuda.tensor-slice/v1",
          .batch_size = kBatchSize,
          .drop_last = true,
          .num_workers = kWorkerCount,
          .prefetch_batches = kPrefetchBatches,
          .persistent_workers = false,
          .deterministic_batch_factory = true,
          .worker_seed = kSeed,
          .memory = {.pin_memory = true,
                     .target_device = jyppx::Device(torch::kCUDA, 0),
                     .non_blocking = true,
                     .stage_capacity = kPrefetchBatches,
                     .max_tensors = 2,
                     .max_value_nodes = 3,
                     .max_bytes = 8ULL * 1024ULL * 1024ULL}},
      [inputs, targets](std::span<const std::size_t> indices,
                        const jyppx::data::DataWorkerContext&,
                        std::stop_token) {
        std::vector<std::int64_t> rows;
        rows.reserve(indices.size());
        for (const auto index : indices) {
          rows.push_back(static_cast<std::int64_t>(index));
        }
        const auto row_tensor = torch::tensor(rows, torch::kLong);
        return c10::ivalue::Tuple::create(
            {inputs.index_select(0, row_tensor),
             targets.index_select(0, row_tensor)});
      });
}

/** @brief Creates one SGD adapter over the supplied stable bindings. */
std::shared_ptr<jyppx::optim::SgdOptimizerAdapter> make_optimizer(
    std::string optimizer_id, std::string group_id,
    std::vector<jyppx::optim::OptimizerParameterBinding> bindings) {
  return std::make_shared<jyppx::optim::SgdOptimizerAdapter>(
      std::move(optimizer_id), std::vector<jyppx::optim::SgdParameterGroup>{
                                   {.group_id = std::move(group_id),
                                    .parameters = std::move(bindings),
                                    .options = {.learning_rate = 1e-4}}});
}

/** @brief Runs and validates one end-to-end CUDA Trainer sample. */
BenchmarkResult run_benchmark(BenchmarkMode mode) {
  torch::manual_seed(kSeed);
  torch::cuda::manual_seed_all(kSeed);
  auto [inputs, targets] = make_dataset();
  auto pipeline = make_pipeline(inputs, targets);
  auto holder = std::make_shared<CudaBenchmarkModel>();
  auto model = std::make_shared<jyppx::nn::ModelAdapter>(
      jyppx::nn::adapt_tensor_module(holder));

  std::vector<jyppx::optim::OptimizerParameterBinding> all_bindings;
  std::vector<jyppx::optim::OptimizerParameterBinding> weight_bindings;
  std::vector<jyppx::optim::OptimizerParameterBinding> bias_bindings;
  for (const auto& item : model->state().named_parameters()) {
    jyppx::optim::OptimizerParameterBinding binding{.parameter_id = item.key(),
                                                    .parameter = item.value()};
    all_bindings.push_back(binding);
    if (item.key().ends_with(".bias")) {
      bias_bindings.push_back(std::move(binding));
    } else {
      weight_bindings.push_back(std::move(binding));
    }
  }

  std::vector<std::shared_ptr<jyppx::optim::SgdOptimizerAdapter>> optimizers;
  if (mode == BenchmarkMode::fp32_single_sgd) {
    optimizers.push_back(make_optimizer("optimizer.benchmark.cuda.all", "all",
                                        std::move(all_bindings)));
  } else {
    optimizers.push_back(make_optimizer("optimizer.benchmark.cuda.weights",
                                        "weights", std::move(weight_bindings)));
    optimizers.push_back(make_optimizer("optimizer.benchmark.cuda.biases",
                                        "biases", std::move(bias_bindings)));
  }

  std::vector<jyppx::training::OptimizerExecutionBinding> executions;
  executions.reserve(optimizers.size());
  for (const auto& optimizer : optimizers) {
    executions.push_back(
        {.optimizer = optimizer,
         .checkpoint =
             std::make_shared<jyppx::optim::SgdOptimizerCheckpointParticipant>(
                 optimizer)});
  }

  auto scaler = mode == BenchmarkMode::fp16_multi_sgd_amp
                    ? std::make_shared<jyppx::training::amp::GradScaler>(
                          jyppx::training::amp::GradScalerOptions{
                              .init_scale = kInitialScale,
                              .growth_factor = 2.0,
                              .backoff_factor = 0.5,
                              .growth_interval = 100000,
                              .enabled = true,
                              .scaler_id = "grad_scaler.benchmark.cuda"})
                    : nullptr;
  auto scaler_checkpoint =
      scaler
          ? std::make_shared<
                jyppx::training::amp::GradScalerCheckpointParticipant>(scaler)
          : nullptr;

  const auto mapper = [](const jyppx::data::BatchEnvelope& batch) {
    return jyppx::nn::Invocation{
        .positional = {batch.value.toTuple()->elements().at(0)}};
  };
  const auto target = [](const jyppx::data::BatchEnvelope& batch)
      -> std::optional<c10::IValue> {
    return batch.value.toTuple()->elements().at(1);
  };
  const auto step = [](const jyppx::training::StepInput& input) {
    const auto prediction =
        input.model.invoke(input.invocation).value.toTensor();
    return jyppx::training::StepOutput{
        .loss = (prediction - input.target->toTensor()).pow(2).mean(),
        .sample_count = input.batch.sample_ids.size()};
  };

  jyppx::training::Trainer trainer(
      {.trainer_id = "trainer.benchmark.cuda",
       .run_id = "run.benchmark.cuda",
       .trace_id = "trace.benchmark.cuda",
       .max_epochs = 1,
       .precision = mode == BenchmarkMode::fp16_multi_sgd_amp
                        ? jyppx::training::PrecisionMode::cuda_float16
                        : jyppx::training::PrecisionMode::full});
  trainer.prepare(
      {.model = model,
       .optimizers = std::move(executions),
       .training_data = pipeline,
       .training_batch_mapper = mapper,
       .training_target_mapper = target,
       .training_step = step,
       .strategy = std::make_shared<jyppx::training::SingleDeviceStrategy>(
           jyppx::training::SingleDeviceStrategyOptions{
               .strategy_id = "strategy.benchmark.cuda",
               .device = jyppx::Device(torch::kCUDA, 0),
               .non_blocking_transfer = true}),
       .gradient_scaler = scaler,
       .gradient_scaler_checkpoint = scaler_checkpoint,
       .model_checkpoint =
           std::make_shared<jyppx::nn::ModelCheckpointParticipant>(*model),
       .rng_checkpoint =
           std::make_shared<jyppx::core::RngCheckpointParticipant>(
               jyppx::core::RngCheckpointParticipantOptions{.include_cuda =
                                                                true})});

  constexpr c10::DeviceIndex kDeviceIndex = 0;
  torch::cuda::synchronize(kDeviceIndex);
  c10::cuda::CUDACachingAllocator::resetPeakStats(kDeviceIndex);
  const auto started = std::chrono::steady_clock::now();
  const auto fit_result = trainer.fit();
  torch::cuda::synchronize(kDeviceIndex);
  const auto elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  const auto allocator_stats =
      c10::cuda::CUDACachingAllocator::getDeviceStats(kDeviceIndex);
  // LibTorch keeps the aggregate allocator statistic in slot zero across the
  // candidate 2.5.x and current development headers, while the owning enum's
  // namespace changed between those releases.
  constexpr std::size_t kAggregate = 0;

  BenchmarkResult result{
      .samples_per_second = static_cast<double>(kBatchSize * kSteps) / elapsed,
      .mean_step_milliseconds = elapsed * 1000.0 / kSteps,
      .peak_allocated_bytes = allocator_stats.allocated_bytes[kAggregate].peak,
      .peak_reserved_bytes = allocator_stats.reserved_bytes[kAggregate].peak,
      .global_step = fit_result.global_step,
      .optimizer_steps = {},
      .scaler_scale =
          scaler ? std::optional<double>(scaler->get_scale()) : std::nullopt,
      .scaler_checkpoint_ready =
          scaler == nullptr || scaler->checkpoint_ready()};
  for (const auto& optimizer : optimizers) {
    result.optimizer_steps.push_back(optimizer->completed_steps());
  }
  trainer.close();

  if (fit_result.status != jyppx::training::FitStatus::completed ||
      result.global_step != kSteps) {
    throw std::runtime_error("Trainer did not commit the frozen step count");
  }
  for (const auto completed_steps : result.optimizer_steps) {
    if (completed_steps != kSteps) {
      throw std::runtime_error(
          "an optimizer did not commit the frozen step count");
    }
  }
  if (scaler && (!result.scaler_checkpoint_ready ||
                 result.scaler_scale.value_or(0.0) != kInitialScale)) {
    throw std::runtime_error(
        "GradScaler did not finish at the expected checkpoint boundary");
  }
  if (elapsed <= 0.0 || result.peak_allocated_bytes <= 0 ||
      result.peak_reserved_bytes <= 0) {
    throw std::runtime_error("CUDA timing or allocator statistics are invalid");
  }
  return result;
}

/** @brief Writes the validated sample as one compact JSON document. */
void write_sample(BenchmarkMode mode, const BenchmarkResult& result) {
  const auto* properties = at::cuda::getDeviceProperties(0);
  std::cout << std::setprecision(17) << '{' << "\"schema_version\":1,"
            << "\"kind\":\"jyppx-cuda-training-benchmark-sample\","
            << "\"mode\":\"" << mode_name(mode) << "\","
            << "\"workload\":{"
            << "\"id\":\"" << kWorkloadId << "\","
            << "\"digest\":\"fnv1a64:" << workload_digest() << "\","
            << "\"seed\":" << kSeed << ',' << "\"batch_size\":" << kBatchSize
            << ',' << "\"steps\":" << kSteps << ','
            << "\"sample_count\":" << kBatchSize * kSteps << ','
            << "\"input_features\":" << kInputFeatures << ','
            << "\"hidden_features\":" << kHiddenFeatures << ','
            << "\"output_features\":" << kOutputFeatures << ','
            << "\"worker_count\":" << kWorkerCount << ','
            << "\"prefetch_batches\":" << kPrefetchBatches << "},"
            << "\"build_fingerprint\":" << jyppx::build_fingerprint_json()
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
            << "\"samples_per_second\":" << result.samples_per_second << ','
            << "\"mean_step_milliseconds\":" << result.mean_step_milliseconds
            << ',' << "\"peak_allocated_bytes\":" << result.peak_allocated_bytes
            << ',' << "\"peak_reserved_bytes\":" << result.peak_reserved_bytes
            << ',' << "\"global_step\":" << result.global_step << ','
            << "\"optimizer_steps\":[";
  for (std::size_t index = 0; index < result.optimizer_steps.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << result.optimizer_steps[index];
  }
  std::cout << "],\"grad_scaler\":";
  if (result.scaler_scale) {
    std::cout << "{\"enabled\":true,\"scale\":" << *result.scaler_scale
              << ",\"checkpoint_ready\":"
              << (result.scaler_checkpoint_ready ? "true" : "false") << '}';
  } else {
    std::cout << "null";
  }
  std::cout << "}}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (!jyppx::cuda_is_available()) {
      throw std::runtime_error(
          "CUDA benchmark requires CUDA LibTorch and a visible NVIDIA GPU");
    }
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    const auto mode = parse_arguments(argc, argv);
    write_sample(mode, run_benchmark(mode));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CUDA training benchmark error: " << error.what() << '\n';
    return 2;
  }
}
