// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "resource_support.hpp"

#include <jyppx/liuding/data_image/image_batch.hpp>
#include <jyppx/liuding/data_image/image_dataset.hpp>
#include <jyppx/liuding/data_image/joint_transform.hpp>
#include <jyppx/liuding/data_image/manifests.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/liuding.hpp>
#if defined(JYPPX_BENCH_MODELS)
#include <jyppx/liuding/models/vision.hpp>
#endif
#if defined(JYPPX_BENCH_CUDA)
#include <torch/cuda.h>

#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace bench = jyppx::liuding::benchmarks;
namespace data = jyppx::liuding::data;
namespace image = jyppx::liuding::data::image;
#if defined(JYPPX_BENCH_MODELS)
namespace models = jyppx::liuding::models;
#endif
namespace nn = jyppx::liuding::nn;
namespace optim = jyppx::liuding::optim;
namespace training = jyppx::liuding::training;
using Clock = std::chrono::steady_clock;

struct Options {
  std::filesystem::path root;
  std::size_t batch_size = 8;
  std::uint32_t image_size = 32;
  std::uint64_t epochs = 1;
  int threads = 1;
  std::size_t workers = 0;
  std::size_t prefetch_batches = 2;
  std::string model = "linear64";
  std::string device = "cpu";
  std::string source_mode = "image-folder";
  std::uint64_t max_cached_bytes = 512ULL << 20U;
};

std::uint64_t positive_integer(std::string_view value, std::string_view name,
                               std::uint64_t maximum) {
  std::uint64_t result = 0;
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc{} || end != value.data() + value.size() ||
      result == 0 || result > maximum) {
    throw std::invalid_argument(std::string(name) + " must be in [1, " +
                                std::to_string(maximum) + "]");
  }
  return result;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 == argc) {
      throw std::invalid_argument("each option requires a value");
    }
    const std::string_view name(argv[index]);
    const std::string_view value(argv[index + 1]);
    if (name == "--image-root") {
      options.root = argv[index + 1];
    } else if (name == "--batch-size") {
      options.batch_size = positive_integer(value, name, 256);
    } else if (name == "--image-size") {
      options.image_size =
          static_cast<std::uint32_t>(positive_integer(value, name, 512));
    } else if (name == "--epochs") {
      options.epochs = positive_integer(value, name, 100);
    } else if (name == "--threads") {
      options.threads = static_cast<int>(positive_integer(value, name, 64));
    } else if (name == "--workers") {
      if (value == "0") {
        options.workers = 0;
      } else {
        options.workers = positive_integer(value, name, 16);
      }
    } else if (name == "--prefetch-batches") {
      options.prefetch_batches = positive_integer(value, name, 4);
    } else if (name == "--model") {
      if (value != "linear64" && value != "resnet18") {
        throw std::invalid_argument("--model must be linear64 or resnet18");
      }
      options.model = value;
    } else if (name == "--device") {
      if (value != "cpu" && value != "cuda:0") {
        throw std::invalid_argument("--device must be cpu or cuda:0");
      }
      options.device = value;
    } else if (name == "--source-mode") {
      if (value != "image-folder" && value != "tensor-cache" &&
          value != "tensor-cache-resized") {
        throw std::invalid_argument(
            "--source-mode must be image-folder, tensor-cache or "
            "tensor-cache-resized");
      }
      options.source_mode = value;
    } else if (name == "--max-cached-bytes") {
      options.max_cached_bytes = positive_integer(value, name, 64ULL << 30U);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(name));
    }
  }
  if (options.root.empty()) {
    throw std::invalid_argument("--image-root is required");
  }
  if (options.model == "resnet18" && options.image_size < 64) {
    throw std::invalid_argument("--model resnet18 requires --image-size >= 64");
  }
  return options;
}

std::uint64_t nanoseconds_since(Clock::time_point start) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start)
          .count());
}

class TimedPipeline final : public data::BatchPipeline {
 public:
  explicit TimedPipeline(std::shared_ptr<data::DataPipeline> inner)
      : inner_(std::move(inner)) {}

  void set_epoch(std::uint64_t epoch) override { inner_->set_epoch(epoch); }
  std::optional<data::BatchEnvelope> next(std::stop_token stop = {}) override {
    const auto start = Clock::now();
    auto result = inner_->next(stop);
    next_ns_ += nanoseconds_since(start);
    return result;
  }
  void close() noexcept override { inner_->close(); }
  bool closed() const noexcept override { return inner_->closed(); }
  data::DataCapability capability() const override {
    return inner_->capability();
  }
  std::string participant_id() const override {
    return inner_->participant_id();
  }
  jyppx::liuding::StateManifest describe_state() const override {
    return inner_->describe_state();
  }
  void save_state(jyppx::liuding::StateWriter& writer) const override {
    inner_->save_state(writer);
  }
  void load_state(const jyppx::liuding::StateReader& reader,
                  jyppx::liuding::RestorePolicy policy) override {
    inner_->load_state(reader, policy);
  }
  std::uint64_t next_ns() const noexcept { return next_ns_; }

 private:
  std::shared_ptr<data::DataPipeline> inner_;
  std::uint64_t next_ns_ = 0;
};

struct StageTimes {
  std::atomic<std::uint64_t> read_decode_ns{0};
  std::atomic<std::uint64_t> transform_ns{0};
  std::atomic<std::uint64_t> collate_ns{0};
  std::atomic<std::uint64_t> encode_ns{0};
};

class Classifier final : public torch::nn::Module {
 public:
  Classifier(std::uint32_t image_size, std::size_t classes)
      : network_(register_module(
            "network",
            torch::nn::Sequential(
                torch::nn::Linear(3 * image_size * image_size, 64),
                torch::nn::ReLU(),
                torch::nn::Linear(64, static_cast<std::int64_t>(classes))))) {}

  jyppx::liuding::Tensor forward(const jyppx::liuding::Tensor& input) {
    return network_->forward(input.reshape({input.size(0), -1}));
  }

 private:
  torch::nn::Sequential network_{nullptr};
};

std::string parameter_digest(const nn::ModelAdapter& model) {
  jyppx::liuding::Sha256 digest;
  for (const auto& item : model.state().named_parameters()) {
    const auto tensor = item.value().detach().to(torch::kCPU).contiguous();
    const auto name = item.key();
    const auto name_bytes = static_cast<std::uint64_t>(name.size());
    const auto tensor_bytes = static_cast<std::uint64_t>(tensor.nbytes());
    digest.update(std::as_bytes(std::span(&name_bytes, 1)));
    digest.update(std::as_bytes(std::span(name.data(), name.size())));
    digest.update(std::as_bytes(std::span(&tensor_bytes, 1)));
    digest.update(std::span(static_cast<const std::byte*>(tensor.data_ptr()),
                            tensor.nbytes()));
  }
  return jyppx::liuding::sha256_digest_hex(digest.finish());
}

bench::Json measure(const Options& options) {
  const auto workload_start = Clock::now();
  torch::manual_seed(1729);
  torch::set_num_threads(options.threads);
  const bool cuda = options.device == "cuda:0";
#if defined(JYPPX_BENCH_CUDA)
  if (cuda)
    bench::require(torch::cuda::is_available(),
                   "CUDA requested but unavailable at runtime");
#else
  bench::require(!cuda, "CUDA requested from a CPU-only benchmark binary");
#endif
  const auto device = cuda ? jyppx::liuding::Device(torch::kCUDA, 0)
                           : jyppx::liuding::Device(torch::kCPU);
  const auto source_start = Clock::now();
  image::ImageFolderOptions folder_options;
  folder_options.extensions = {".jpg", ".jpeg", ".png"};
  auto index = image::index_image_folder(options.root, folder_options);
  bench::require(index.records().size() >= 2,
                 "ImageFolder needs at least two image files");
  bench::require(index.categories().size() >= 2,
                 "ImageFolder needs at least two nonempty classes");
  const auto inventory_sha256 = index.source_sha256();
  const auto samples_per_epoch = index.records().size();
  const auto classes = index.categories().size();
  const bool cache_source = options.source_mode != "image-folder";
  const bool resized_cache = options.source_mode == "tensor-cache-resized";
  auto budget = std::make_shared<image::ImageResourceBudget>();
  std::shared_ptr<image::ImageResourceBudget> cache_budget;
  if (cache_source) {
    image::ImageResourceBudgetLimits cache_limits;
    cache_limits.max_decoded_bytes = options.max_cached_bytes;
    cache_budget = std::make_shared<image::ImageResourceBudget>(cache_limits);
  }
  const image::ImageGeometryStep resize{
      .kind = image::ImageGeometryKind::resize,
      .extent = {options.image_size, options.image_size}};
  image::JointTransformOptions transform_options;
  transform_options.interpolation = image::ImageInterpolation::bilinear;
  transform_options.resource_budget = budget;
  image::ImageDatasetOptions dataset_options;
  dataset_options.resource_budget = cache_budget ? cache_budget : budget;
  dataset_options.image.channels = image::ImageChannels::rgb;
  dataset_options.image.dtype = image::ImageDtype::float32_normalized;
  auto categories = index.categories();
  auto file_source = std::make_shared<image::ImageDataset>(
      options.root, std::move(index), dataset_options);
  std::shared_ptr<image::TensorImageDataset> tensor_source;
  std::vector<std::shared_ptr<image::ImageResourceLeaseSet>> cache_leases;
  std::uint64_t cached_bytes = 0;
  if (cache_budget) {
    std::vector<image::TensorImageRecord> records;
    records.reserve(samples_per_epoch);
    cache_leases.reserve(samples_per_epoch);
    auto cache_transform_options = transform_options;
    cache_transform_options.resource_budget = cache_budget;
    for (std::size_t i = 0; i < samples_per_epoch; ++i) {
      auto sample = file_source->get(i);
      if (resized_cache)
        sample = image::transform_sample(sample, std::span(&resize, 1),
                                         cache_transform_options);
      const auto bytes = static_cast<std::uint64_t>(sample.image.numel()) *
                         sample.image.element_size();
      bench::require(bytes <= options.max_cached_bytes - cached_bytes,
                     "decoded Tensor cache exceeds --max-cached-bytes");
      cached_bytes += bytes;
      image::TensorImageRecord record{std::move(sample.image), sample.pixels,
                                      std::move(sample.image_path),
                                      std::move(sample.labels)};
      record.geometry = std::move(sample.geometry);
      records.push_back(std::move(record));
      cache_leases.push_back(std::move(sample.image_resource_leases));
    }
    tensor_source = std::make_shared<image::TensorImageDataset>(
        std::move(records), std::move(categories));
    file_source.reset();
    bench::require(
        cache_budget->usage(image::ImageResourceKind::decoded_bytes).current ==
            cached_bytes,
        "Tensor cache accounting differs from retained storage");
  }
  const auto source_preparation_seconds =
      static_cast<double>(nanoseconds_since(source_start)) / 1e9;
  const auto sample_at = [file_source, tensor_source](std::size_t index,
                                                      std::stop_token stop) {
    return tensor_source ? tensor_source->get(index, stop)
                         : file_source->get(index, stop);
  };
  image::ImageCollateOptions collate_options;
  collate_options.resource_budget = budget;
  StageTimes stages;
  data::DataPipelineOptions pipeline_options;
  pipeline_options.pipeline_id = "data.benchmark.image-training";
  pipeline_options.dataset_version = "image-folder:" + inventory_sha256;
  pipeline_options.dataset_size = samples_per_epoch;
  pipeline_options.batch_factory_id = "image-training.resize-bilinear/v1";
  pipeline_options.batch_size = options.batch_size;
  pipeline_options.num_workers = options.workers;
  pipeline_options.prefetch_batches = options.prefetch_batches;
  if (cuda) {
    pipeline_options.memory.pin_memory = true;
    pipeline_options.memory.target_device = device;
    pipeline_options.memory.non_blocking = true;
  }
  {
    const auto exemplar =
        resized_cache
            ? sample_at(0, {})
            : image::transform_sample(sample_at(0, {}), std::span(&resize, 1),
                                      transform_options);
    const std::array exemplar_keys{image::ImageBatchKey{"exemplar", 0, 0}};
    const auto exemplar_batch = image::collate_images(
        std::span(&exemplar, 1), exemplar_keys, collate_options);
    pipeline_options.schema = image::image_batch_schema(exemplar_batch, device);
  }
  pipeline_options.admission_factory =
      image::make_image_batch_byte_admission_factory(budget,
                                                     collate_options.limits);
  pipeline_options.resource_lease_factory =
      image::make_image_batch_resource_lease_factory(budget,
                                                     collate_options.limits);
  auto raw_pipeline = std::make_shared<data::DataPipeline>(
      std::make_shared<data::samplers::SequentialSampler>(samples_per_epoch),
      pipeline_options,
      [sample_at, resize, resized_cache, transform_options, collate_options,
       &stages](std::span<const std::size_t> indices,
                const data::DataWorkerContext& context, std::stop_token stop) {
        std::vector<image::VisionSample> samples;
        std::vector<image::ImageBatchKey> keys;
        samples.reserve(indices.size());
        keys.reserve(indices.size());
        for (const auto index : indices) {
          auto start = Clock::now();
          auto sample = sample_at(index, stop);
          stages.read_decode_ns.fetch_add(nanoseconds_since(start),
                                          std::memory_order_relaxed);
          keys.push_back({sample.image_path, context.epoch, 0});
          if (resized_cache) {
            samples.push_back(std::move(sample));
          } else {
            start = Clock::now();
            samples.push_back(image::transform_sample(
                sample, std::span(&resize, 1), transform_options, stop));
            stages.transform_ns.fetch_add(nanoseconds_since(start),
                                          std::memory_order_relaxed);
          }
        }
        auto start = Clock::now();
        auto batch =
            image::collate_images(samples, keys, collate_options, stop);
        stages.collate_ns.fetch_add(nanoseconds_since(start),
                                    std::memory_order_relaxed);
        start = Clock::now();
        auto value = image::image_batch_value(batch);
        stages.encode_ns.fetch_add(nanoseconds_since(start),
                                   std::memory_order_relaxed);
        return value;
      });
  auto pipeline = std::make_shared<TimedPipeline>(raw_pipeline);

  std::shared_ptr<nn::ModelAdapter> model;
  if (options.model == "resnet18") {
#if defined(JYPPX_BENCH_MODELS)
    models::ResNetOptions model_options;
    model_options.num_classes = static_cast<std::int64_t>(classes);
    model = std::make_shared<nn::ModelAdapter>(
        models::adapt(models::resnet18(model_options)));
#else
    throw std::invalid_argument(
        "--model resnet18 requires the models component");
#endif
  } else {
    auto holder = std::make_shared<Classifier>(options.image_size, classes);
    model = std::make_shared<nn::ModelAdapter>(nn::adapt_tensor_module(holder));
  }
  std::vector<optim::OptimizerParameterBinding> bindings;
  for (const auto& item : model->state().named_parameters()) {
    bindings.push_back({.parameter_id = item.key(), .parameter = item.value()});
  }
  auto optimizer = std::make_shared<optim::SgdOptimizerAdapter>(
      "optimizer.benchmark.image-training",
      std::vector<optim::SgdParameterGroup>{
          {.group_id = "main",
           .parameters = std::move(bindings),
           .options = {.learning_rate = 1e-2}}});
  const auto mapper = [device](const data::BatchEnvelope& batch) {
    const auto images = image::image_batch_from_value(batch.value).images;
    bench::require(images.device() == device,
                   "image batch was not transferred to the selected device");
    return nn::Invocation{.positional = {images}};
  };
  const auto target =
      [device](const data::BatchEnvelope& batch) -> std::optional<c10::IValue> {
    const auto labels = image::image_batch_from_value(batch.value).labels;
    bench::require(labels.device() == device,
                   "labels were not transferred to the selected device");
    return labels;
  };
  std::uint64_t observed_samples = 0;
  const auto step = [&observed_samples](const training::StepInput& input) {
    const auto prediction =
        input.model.invoke(input.invocation).value.toTensor();
    const auto labels = input.target->toTensor().to(torch::kLong);
    bench::require(labels.numel() ==
                       static_cast<std::int64_t>(input.batch.sample_ids.size()),
                   "classification labels do not match batch size");
    observed_samples += input.batch.sample_ids.size();
    return training::StepOutput{
        .loss = torch::nn::functional::cross_entropy(prediction, labels),
        .sample_count = input.batch.sample_ids.size()};
  };
  training::Trainer trainer({.trainer_id = "trainer.benchmark.image-training",
                             .run_id = "run.benchmark.image-training",
                             .trace_id = "trace.benchmark.image-training",
                             .max_epochs = options.epochs});
  trainer.prepare(
      {.model = model,
       .optimizers =
           {{.optimizer = optimizer,
             .checkpoint =
                 std::make_shared<optim::SgdOptimizerCheckpointParticipant>(
                     optimizer)}},
       .training_data = pipeline,
       .training_batch_mapper = mapper,
       .training_target_mapper = target,
       .training_step = step,
       .strategy = std::make_shared<training::SingleDeviceStrategy>(
           training::SingleDeviceStrategyOptions{.device = device}),
       .model_checkpoint =
           std::make_shared<nn::ModelCheckpointParticipant>(*model),
       .rng_checkpoint =
           std::make_shared<jyppx::liuding::core::RngCheckpointParticipant>(
               jyppx::liuding::core::RngCheckpointParticipantOptions{
                   .include_cuda = cuda})});
  const auto initial_parameter =
      model->state().module().parameters().front().detach().clone();
#if defined(JYPPX_BENCH_CUDA)
  if (cuda) {
    torch::cuda::synchronize(0);
    c10::cuda::CUDACachingAllocator::resetPeakStats(0);
  }
#endif
  const auto start = Clock::now();
  const auto result = trainer.fit();
#if defined(JYPPX_BENCH_CUDA)
  if (cuda) torch::cuda::synchronize(0);
#endif
  const auto finish = Clock::now();
  const auto fit_seconds =
      std::chrono::duration<double>(finish - start).count();
  const auto workload_seconds =
      std::chrono::duration<double>(finish - workload_start).count();
  if (cache_budget)
    bench::require(
        cache_budget->usage(image::ImageResourceKind::decoded_bytes).current ==
            cached_bytes,
        "Tensor cache was released before training completed");
  auto cuda_allocated = bench::unavailable("CPU-only workload", "bytes");
  auto cuda_reserved = bench::unavailable("CPU-only workload", "bytes");
  bench::Json execution = {
      {"device", options.device},
      {"timing", cuda ? "steady_clock with CUDA sync" : "steady_clock"}};
#if defined(JYPPX_BENCH_CUDA)
  if (cuda) {
    const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
    constexpr std::size_t aggregate = 0;
    cuda_allocated = bench::metric(
        static_cast<double>(stats.allocated_bytes[aggregate].peak), "bytes",
        "lower", true, "LibTorch allocator peak since before Trainer::fit");
    cuda_reserved = bench::metric(
        static_cast<double>(stats.reserved_bytes[aggregate].peak), "bytes",
        "lower", true, "LibTorch allocator peak since before Trainer::fit");
    const auto* properties = at::cuda::getDeviceProperties(0);
    execution["cuda_device_name"] = properties->name;
    execution["compute_capability"] = std::to_string(properties->major) + "." +
                                      std::to_string(properties->minor);
  }
#endif
  trainer.close();
  const auto expected_samples = samples_per_epoch * options.epochs;
  const auto expected_steps =
      ((samples_per_epoch + options.batch_size - 1) / options.batch_size) *
      options.epochs;
  bench::require(result.status == training::FitStatus::completed &&
                     result.completed_epochs == options.epochs &&
                     result.global_step == expected_steps &&
                     observed_samples == expected_samples,
                 "Trainer did not complete the expected image workload");
  bench::require(
      !torch::allclose(initial_parameter,
                       model->state().module().parameters().front().detach()),
      "SGD did not change model parameters");
  bench::require(fit_seconds > 0, "fit duration must be positive");
  const auto peak_rss = bench::peak_rss();
  const auto final_parameter_sha256 = parameter_digest(*model);

  const bench::Json descriptor = {
      {"task", "imagefolder-classification"},
      {"dataset_inventory_sha256", inventory_sha256},
      {"samples_per_epoch", samples_per_epoch},
      {"classes", classes},
      {"image_size", options.image_size},
      {"batch_size", options.batch_size},
      {"epochs", options.epochs},
      {"threads", options.threads},
      {"workers", options.workers},
      {"prefetch_batches", options.prefetch_batches},
      {"source_mode", options.source_mode},
      {"max_cached_bytes", cache_source ? options.max_cached_bytes : 0},
      {"device", options.device},
      {"batch_transfer", cuda ? "pin-and-nonblocking-transfer" : "none"},
      {"image_decode", "rgb-float32-normalized"},
      {"resize", "bilinear-antialias"},
      {"model", options.model == "resnet18" ? "resnet18-random-init"
                                            : "flatten-linear64-relu-linear"},
      {"optimizer", "sgd-lr0.01"}};
  const auto seconds = [](std::uint64_t ns) {
    return static_cast<double>(ns) / 1e9;
  };
  return {{"schema_version", 1},
          {"kind", "liuding-image-training-benchmark-sample"},
          {"workload",
           {{"descriptor", descriptor},
            {"digest", bench::digest(descriptor.dump())}}},
          {"build_fingerprint",
           bench::Json::parse(jyppx::liuding::build_fingerprint_json())},
          {"execution", execution},
          {"validation",
           {{"completed_epochs", result.completed_epochs},
            {"global_step", result.global_step},
            {"observed_samples", observed_samples},
            {"parameter_updated", true},
            {"final_parameter_sha256", final_parameter_sha256}}},
          {"metrics",
           {{"fit_seconds", fit_seconds},
            {"workload_seconds", workload_seconds},
            {"source_preparation_seconds", source_preparation_seconds},
            {"cached_source_bytes", cached_bytes},
            {"images_per_second", expected_samples / fit_seconds},
            {"workload_images_per_second", expected_samples / workload_seconds},
            {"epoch_seconds", fit_seconds / options.epochs},
            {"data_wait_seconds", seconds(pipeline->next_ns())},
            {"read_decode_seconds", seconds(stages.read_decode_ns.load())},
            {"transform_seconds", seconds(stages.transform_ns.load())},
            {"collate_seconds", seconds(stages.collate_ns.load())},
            {"encode_seconds", seconds(stages.encode_ns.load())},
            {"peak_rss_bytes", peak_rss},
            {"peak_cuda_allocated_bytes", cuda_allocated},
            {"peak_cuda_reserved_bytes", cuda_reserved}}},
          {"measurement_scope",
           {{"timer",
             "source preparation includes indexing and optional Tensor "
             "caching; fit excludes it; workload includes runtime setup, "
             "source preparation, model setup and fit"},
            {"cached_source_bytes",
             "retained decoded or resized Tensor payload, excluding "
             "allocator overhead"},
            {"data_wait",
             "DataPipeline::next consumer wall time, including result/schema "
             "processing; not isolated queue wait"},
            {"stage_times",
             "sum of producer factory CPU wall intervals across workers; "
             "read_decode_seconds measures shared-handle access for "
             "Tensor cache; resized-cache transform runs during source "
             "preparation; intervals may overlap and exceed fit time"},
            {"rss", "process lifetime high-water mark, including setup"},
            {"cuda",
             "LibTorch allocator peak during Trainer::fit; excludes "
             "driver/context and setup peak"}}}};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::cout << measure(parse_options(argc, argv)).dump() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "image training benchmark: " << error.what() << '\n';
    return 1;
  }
}
