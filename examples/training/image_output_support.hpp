// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
/** @file image_output_support.hpp
 * @brief Runs bounded ImageFolder training and post-update preview evidence. */
#ifndef JYPPX_EXAMPLE_IMAGE_OUTPUT_SUPPORT_HPP
#define JYPPX_EXAMPLE_IMAGE_OUTPUT_SUPPORT_HPP

#include <jyppx/liuding/data_image/image_batch_copy_stream.hpp>
#include <jyppx/liuding/data_image/image_output_sink.hpp>
#include <jyppx/liuding/data_image/joint_transform.hpp>
#include <jyppx/liuding/foundation/random.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/liuding.hpp>
#include <jyppx/liuding/training/callback_dispatcher.hpp>

#include <torch/cuda.h>
#include <torch/nn.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <nlohmann/json.hpp>

namespace image_output_example {
namespace forge = jyppx::liuding;
namespace data = forge::data;
namespace image = data::image;
namespace fs = std::filesystem;
using Json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;
inline constexpr std::size_t kBatchBytes = 1U << 20U;
inline constexpr std::size_t kEncodedBytes = 256U << 10U;

inline void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct Options {
  fs::path images, output;
  forge::Device device{torch::kCPU};
  bool explicit_copy = false;
  std::size_t batch_size = 8;
  std::stop_token stop;
  std::shared_ptr<image::ImageResourceBudget> budget;
  // An optional observer runs on the sink worker with source admission held.
  // It must not reenter the sink and its captures must outlive Run().
  std::function<void(std::uint64_t)> evidence_observer;
};

struct Activity {
  std::mutex mutex;
  std::condition_variable changed;
  std::size_t next_calls = 0;
  std::atomic<std::size_t> decoded{0}, attempts{0}, acquired{0};
  std::uint64_t next_ns = 0;
};

// This observer delegates every cursor/checkpoint operation unchanged. It
// neither fetches ahead nor commits an additional source batch.
class ObservedPipeline final : public data::BatchPipeline {
 public:
  ObservedPipeline(std::shared_ptr<data::DataPipeline> inner,
                   Activity& activity)
      : inner_(std::move(inner)), activity_(activity) {}
  void set_epoch(std::uint64_t epoch) override { inner_->set_epoch(epoch); }
  std::optional<data::BatchEnvelope> next(std::stop_token stop = {}) override {
    {
      std::lock_guard lock(activity_.mutex);
      ++activity_.next_calls;
      activity_.changed.notify_all();
    }
    const auto start = Clock::now();
    auto batch = inner_->next(stop);
    activity_.next_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                             start)
            .count());
    return batch;
  }
  void close() noexcept override { inner_->close(); }
  bool closed() const noexcept override { return inner_->closed(); }
  data::DataCapability capability() const override {
    return inner_->capability();
  }
  std::string participant_id() const override {
    return inner_->participant_id();
  }
  forge::StateManifest describe_state() const override {
    return inner_->describe_state();
  }
  void save_state(forge::StateWriter& writer) const override {
    inner_->save_state(writer);
  }
  void load_state(const forge::StateReader& reader,
                  forge::RestorePolicy policy) override {
    inner_->load_state(reader, policy);
  }

 private:
  std::shared_ptr<data::DataPipeline> inner_;
  Activity& activity_;
};

class Classifier final : public torch::nn::Module {
 public:
  explicit Classifier(std::size_t classes)
      : network_(register_module(
            "network",
            torch::nn::Sequential(
                torch::nn::Linear(3 * 32 * 32, 64), torch::nn::ReLU(),
                torch::nn::Linear(64, static_cast<int64_t>(classes))))) {}
  torch::Tensor forward(const torch::Tensor& images) {
    return network_->forward(images.flatten(1));
  }

 private:
  torch::nn::Sequential network_{nullptr};
};

inline std::string TensorDigest(const torch::Tensor& value) {
  const auto cpu = value.detach().cpu().contiguous();
  return forge::sha256_hex(
      {static_cast<const std::byte*>(cpu.const_data_ptr()), cpu.nbytes()});
}

inline std::string StateDigest(const forge::nn::ModelAdapter& model) {
  forge::Sha256 digest;
  for (const auto& item : model.state().named_parameters()) {
    const auto cpu = item.value().detach().cpu().contiguous();
    digest.update(
        std::as_bytes(std::span(item.key().data(), item.key().size())));
    digest.update(
        {static_cast<const std::byte*>(cpu.const_data_ptr()), cpu.nbytes()});
  }
  return forge::sha256_digest_hex(digest.finish());
}

class AfterBatch final : public forge::training::CallbackHandler {
 public:
  explicit AfterBatch(
      std::function<void(const forge::training::CallbackEvent&)> fn)
      : fn_(std::move(fn)) {}
  const forge::training::CallbackDescriptor& descriptor()
      const noexcept override {
    return descriptor_;
  }
  forge::training::CallbackCommandRequest handle(
      const forge::training::CallbackEvent& event) override {
    fn_(event);
    return {};
  }

 private:
  forge::training::CallbackDescriptor descriptor_{
      .type_id = "example.image-output.after-update.v1",
      .callback_id = "preview",
      .subscriptions = {forge::training::CallbackEventKind::batch_finished}};
  std::function<void(const forge::training::CallbackEvent&)> fn_;
};

inline image::EncodedImage Preview(
    const torch::Tensor& input,
    const std::shared_ptr<image::ImageResourceBudget>& budget,
    std::stop_token stop) {
  // D2H, float conversion and RGB8 are charged before allocating. D2H and
  // CPU conversion deliberately synchronize this example's consumer stream.
  auto reservation = budget->reserve(image::ImageResourceKind::decoded_bytes,
                                     32U * 32U * 3U * 9U);
  const auto cpu = input.detach().cpu();
  auto scaled = cpu.mul(255);
  scaled.round_().clamp_(0, 255);
  const auto pixels = scaled.to(torch::kUInt8);
  image::PixelSpec spec;
  spec.layout = image::ImageLayout::chw;
  image::ImageEncodeOptions encode;
  encode.resource_budget = budget;
  encode.limits.max_output_bytes = kEncodedBytes;
  return image::encode_image(pixels, spec, encode, stop);
}

inline Json Run(const Options& options) {
  Require(options.batch_size > 0 && options.batch_size <= 32,
          "Batch size must be in [1,32]");
  Require(options.device.is_cpu() ||
              options.device == forge::Device(torch::kCUDA, 0),
          "Choose cpu or cuda:0");
  Require(!options.explicit_copy || options.device.is_cuda(),
          "Explicit copy requires cuda:0");
  Require(!options.device.is_cuda() || torch::cuda::is_available(),
          "CUDA is unavailable");
  Require(!fs::exists(options.output), "Use a new output directory");
  auto index = image::index_image_folder(options.images);
  Require(index.categories().size() >= 2 &&
              index.records().size() > options.batch_size,
          "Use at least two classes and more than one batch");
  const auto sample_count = index.records().size();
  const auto classes = index.categories().size();
  const auto inventory = index.source_sha256();
  forge::core::RngGuard seeded(1729, options.device.is_cuda());
  auto limits = image::ImageResourceBudgetLimits{};
  limits.max_pipeline_inflight_batches = 1;
  limits.max_pipeline_inflight_bytes = kBatchBytes;
  const auto budget =
      options.budget ? options.budget
                     : std::make_shared<image::ImageResourceBudget>(limits);
  Require(budget->usage(image::ImageResourceKind::pipeline_inflight_batches)
                      .limit == 1 &&
              budget->usage(image::ImageResourceKind::pipeline_inflight_bytes)
                      .limit == kBatchBytes,
          "Source admission must be one batch and 1MiB");
  fs::create_directories(options.output);
  forge::experiment::SQLiteRunStore store(options.output / "run-store");
  constexpr std::int64_t timestamp = 1;
  store.start({.run_id = "image-training",
               .project_id = "example",
               .experiment_id = "training-output",
               .display_name = "ImageFolder training previews",
               .status = forge::experiment::RunStatus::running,
               .created_at_unix_ms = timestamp,
               .updated_at_unix_ms = timestamp});
  Activity activity;
  image::ImageDatasetOptions read;
  read.resource_budget = budget;
  read.image.channels = image::ImageChannels::rgb;
  read.image.dtype = image::ImageDtype::float32_normalized;
  auto dataset = std::make_shared<image::ImageDataset>(options.images,
                                                       std::move(index), read);
  image::ImageCollateOptions collate;
  collate.resource_budget = budget;
  collate.limits.max_samples = options.batch_size;
  collate.limits.max_output_bytes = kBatchBytes;
  data::DataPipelineOptions pipeline_options;
  pipeline_options.pipeline_id = "example.image-training-output";
  pipeline_options.dataset_version = inventory;
  pipeline_options.dataset_size = sample_count;
  pipeline_options.batch_factory_id = "rgb-float32.resize32-bilinear/v1";
  pipeline_options.batch_size = options.batch_size;
  pipeline_options.num_workers = pipeline_options.prefetch_batches = 2;
  const auto admission =
      image::make_image_batch_byte_admission_factory(budget, collate.limits);
  pipeline_options.admission_factory = [&](std::stop_token stop) {
    {
      std::lock_guard lock(activity.mutex);
      ++activity.attempts;
      activity.changed.notify_all();
    }
    auto owner = admission(stop);
    ++activity.acquired;
    return owner;
  };
  pipeline_options.stage_resource_factory =
      image::make_image_batch_stage_resource_factory(budget, collate.limits);
  auto raw = std::make_shared<data::DataPipeline>(
      std::make_shared<data::samplers::SequentialSampler>(sample_count),
      pipeline_options,
      [&, dataset, collate](std::span<const std::size_t> ids,
                            const data::DataWorkerContext& context,
                            std::stop_token stop) {
        std::vector<image::VisionSample> samples;
        std::vector<image::ImageBatchKey> keys;
        image::JointTransformOptions transform;
        transform.resource_budget = budget;
        const image::ImageGeometryStep resize{
            .kind = image::ImageGeometryKind::resize, .extent = {32, 32}};
        for (const auto id : ids) {
          auto sample = dataset->get(id, stop);
          ++activity.decoded;
          keys.push_back({sample.image_path, context.epoch, 0});
          samples.push_back(image::transform_sample(
              sample, std::span(&resize, 1), transform, stop));
        }
        return image::image_batch_value(
            image::collate_images(samples, keys, collate, stop));
      });
  auto pipeline = std::make_shared<ObservedPipeline>(raw, activity);
  data::BatchMemoryOptions memory;
  memory.target_device = options.device;
  memory.pin_memory = memory.non_blocking = options.device.is_cuda();
  std::unique_ptr<image::ImageBatchCopyStream> copy;
  if (options.explicit_copy)
    copy = std::make_unique<image::ImageBatchCopyStream>(
        memory, 1, true, collate.limits, budget);
  struct Pending {
    data::BatchEnvelope source;  // Opaque pipeline leases are not in IValue.
    image::ImageBatch batch;
    std::optional<image::ImageBatchTransfer> transfer;
    std::string input_digest, logits_digest;
  };
  std::unique_ptr<Pending> pending;
  auto holder = std::make_shared<Classifier>(classes);
  auto model = std::make_shared<forge::nn::ModelAdapter>(
      forge::nn::adapt_tensor_module(holder));
  const auto initial_digest = StateDigest(*model);
  std::vector<forge::optim::OptimizerParameterBinding> bindings;
  for (const auto& item : model->state().named_parameters())
    bindings.push_back({.parameter_id = item.key(), .parameter = item.value()});
  auto optimizer = std::make_shared<forge::optim::SgdOptimizerAdapter>(
      "optimizer.example", std::vector<forge::optim::SgdParameterGroup>{
                               {.group_id = "main",
                                .parameters = std::move(bindings),
                                .options = {.learning_rate = .01}}});
  image::ImageOutputSinkOptions sink_options;
  sink_options.max_pending_items = 1;
  sink_options.max_pending_bytes = sink_options.max_item_bytes = kEncodedBytes;
  sink_options.resource_budget = budget;
  sink_options.wait_for_aggregate_queue = true;
  image::ImageOutputSink sink(sink_options);
  Json rows = Json::array(), pressure, timelines = Json::array();
  std::vector<image::ImageSinkSubmission> tickets;
  auto after = std::make_shared<
      AfterBatch>([&](const forge::training::CallbackEvent& event) {
    Require(pending && event.global_step == pending->source.sequence + 1,
            "Preview must follow a committed SGD update");
    const auto sequence = pending->source.sequence;
    rows.push_back({{"sequence", sequence},
                    {"global_step", event.global_step},
                    {"sample_ids", pending->source.sample_ids},
                    {"input_sha256", pending->input_digest},
                    {"logits_sha256", pending->logits_digest},
                    {"loss", event.metrics.at("training.loss")},
                    {"state_sha256", StateDigest(*model)}});
    tickets.push_back(sink.produce_and_submit(
        options.output / ("step-" + std::to_string(event.global_step) + ".png"),
        kEncodedBytes,
        [&](std::stop_token stop) {
          return Preview(pending->batch.images[0], budget, stop);
        },
        {}, options.stop,
        [&, sequence, step = event.global_step,
         lease = pending->source.admission_lease](
            const image::ImageSinkWriteResult& file) {
          (void)lease;
          if (sequence == 0) {
            std::unique_lock lock(activity.mutex);
            Require(activity.changed.wait_for(lock, std::chrono::seconds(10),
                                              [&] {
                                                return activity.next_calls >=
                                                           2 &&
                                                       activity.attempts >= 2;
                                              }),
                    "Timed out observing training-to-source backpressure");
            Require(
                activity.decoded == options.batch_size &&
                    activity.acquired == 1,
                "Decoded a second batch before evidence released admission");
            Require(
                budget->usage(
                          image::ImageResourceKind::pipeline_inflight_batches)
                            .current == 1 &&
                    budget->usage(
                              image::ImageResourceKind::pipeline_inflight_bytes)
                            .current == kBatchBytes,
                "Evidence lost source count/byte admission");
            for (const auto kind : {image::ImageResourceKind::decoded_bytes,
                                    image::ImageResourceKind::pinned_bytes,
                                    image::ImageResourceKind::gpu_bytes})
              Require(budget->usage(kind).current == 0,
                      "Input storage lease survived the next training read");
            pressure = {{"decoded_samples", activity.decoded.load()},
                        {"acquired_batches", activity.acquired.load()},
                        {"next_read_waiting", true},
                        {"input_storage_released", true}};
          }
          if (options.evidence_observer) options.evidence_observer(step);
          const auto id = "step-" + std::to_string(step);
          forge::experiment::PublishedFileEvidenceRequest request;
          request.run_id = "image-training";
          request.reference_id = id;
          request.path = file.path;
          request.descriptor = {.logical_name = id + ".png",
                                .media_type = "image/png",
                                .producer = "example.image-training",
                                .expected_size_bytes = file.bytes};
          request.expected_digest = {.hex = file.sha256};
          request.attached_at_unix_ms = timestamp;
          request.event_batch_key = id;
          request.event.identity = {.stream_id = "previews",
                                    .event_id = id,
                                    .idempotency_key = id,
                                    .sequence = step};
          request.event.correlation.project_id = "example";
          request.event.correlation.run_id = "image-training";
          request.event.payload =
              forge::TelemetryArtifactReference{.role = "image.preview"};
          const auto receipt =
              store.record_published_file_evidence_atomic(request);
          Require(!receipt.reference_replay && !receipt.event.batch_replay,
                  "Duplicate preview evidence");
        }));
    Require(tickets.back().disposition == image::ImageSinkDisposition::queued,
            "Preview was not admitted");
    if (pending->transfer) {
      pending->transfer->wait();
      const auto timeline = pending->transfer->timeline();
      timelines.push_back({{"copy_stream_id", timeline.copy_stream_id},
                           {"consumer_stream_id", timeline.consumer_stream_id},
                           {"copy_ms", *timeline.copy_elapsed_ms}});
    }
    pending.reset();
  });
  forge::training::Trainer trainer({.trainer_id = "trainer.example",
                                    .run_id = "image-training",
                                    .trace_id = "trace.image-training",
                                    .max_epochs = 1});
  trainer.prepare(
      {.model = model,
       .optimizers = {{.optimizer = optimizer,
                       .checkpoint = std::make_shared<
                           forge::optim::SgdOptimizerCheckpointParticipant>(
                           optimizer)}},
       .training_data = pipeline,
       .training_batch_mapper =
           [&](const data::BatchEnvelope& envelope) {
             Require(!pending, "Previous training owners remain live");
             pending = std::make_unique<Pending>();
             pending->source = envelope;
             auto batch = image::image_batch_from_value(envelope.value);
             pending->input_digest = TensorDigest(batch.images);
             if (options.stop.stop_requested())
               throw forge::Cancelled("Image training mapper cancelled");
             auto prepared = image::prepare_image_batch(batch, memory,
                                                        collate.limits, budget);
             if (copy) {
               pending->transfer.emplace(
                   copy->submit(std::move(prepared), options.stop));
               pending->batch = pending->transfer->receive();
             } else {
               pending->batch = image::transfer_image_batch(
                   prepared, memory, collate.limits, budget);
             }
             return forge::nn::Invocation{
                 .positional = {pending->batch.images}};
           },
       .training_target_mapper = [&](const data::BatchEnvelope&)
           -> std::optional<c10::IValue> { return pending->batch.labels; },
       .training_step =
           [&](const forge::training::StepInput& input) {
             const auto prediction =
                 input.model.invoke(input.invocation).value.toTensor();
             pending->logits_digest = TensorDigest(prediction);
             return forge::training::StepOutput{
                 .loss = torch::nn::functional::cross_entropy(
                     prediction, input.target->toTensor()),
                 .sample_count = input.batch.sample_ids.size()};
           },
       .strategy = std::make_shared<forge::training::SingleDeviceStrategy>(
           forge::training::SingleDeviceStrategyOptions{.device =
                                                            options.device}),
       .callbacks = std::make_shared<forge::training::CallbackDispatcher>(
           std::vector<std::shared_ptr<forge::training::CallbackHandler>>{
               after}),
       .model_checkpoint =
           std::make_shared<forge::nn::ModelCheckpointParticipant>(*model),
       .rng_checkpoint =
           std::make_shared<forge::core::RngCheckpointParticipant>(
               forge::core::RngCheckpointParticipantOptions{
                   .include_cuda = options.device.is_cuda()})});
  const auto started = Clock::now();
  forge::training::FitResult result;
  try {
    result = trainer.fit(options.stop);
    trainer.close();
    pending.reset();
    if (copy) copy->close();
    sink.close();
  } catch (...) {
    raw->close();
    pending.reset();
    if (copy) copy->close();
    try {
      sink.close();
    } catch (...) {
    }
    throw;
  }
  Json files = Json::array();
  for (auto& ticket : tickets) {
    const auto file = ticket.completion.get();
    ticket.evidence_completion.get();
    files.push_back({{"name", file.path.filename().string()},
                     {"bytes", file.bytes},
                     {"sha256", file.sha256}});
  }
  Require(store.references("image-training").size() == 2 * rows.size() &&
              store.events("image-training").events.size() == rows.size() &&
              store.integrity_check().ok(),
          "Incomplete preview CAS/reference/event evidence");
  if (result.status == forge::training::FitStatus::completed)
    Require(activity.decoded == sample_count &&
                result.global_step == (sample_count + options.batch_size - 1) /
                                          options.batch_size,
            "Incomplete training epoch");
  const auto final_digest = StateDigest(*model);
  Require(initial_digest != final_digest &&
              optimizer->completed_steps() == result.global_step,
          "SGD did not complete its advertised updates");
  Json resources = Json::array();
  for (unsigned i = 0; i < 14; ++i) {
    const auto usage = budget->usage(static_cast<image::ImageResourceKind>(i));
    Require(usage.current == 0 && usage.peak <= usage.limit,
            "Image resources did not fully release");
    resources.push_back({{"kind", i},
                         {"current", usage.current},
                         {"peak", usage.peak},
                         {"limit", usage.limit}});
  }
  return {{"device", options.device.str()},
          {"explicit_copy", options.explicit_copy},
          {"dataset_inventory_sha256", inventory},
          {"samples", sample_count},
          {"decoded", activity.decoded.load()},
          {"updates", result.global_step},
          {"cancelled", result.status == forge::training::FitStatus::cancelled},
          {"initial_state_sha256", initial_digest},
          {"final_state_sha256", final_digest},
          {"steps", rows},
          {"files", files},
          {"pressure", pressure},
          {"resources", resources},
          {"timelines", timelines},
          {"fit_output_seconds",
           std::chrono::duration<double>(Clock::now() - started).count()},
          {"data_wait_seconds", activity.next_ns / 1e9},
          {"overlap_qualified", false}};
}
}  // namespace image_output_example
#endif  // JYPPX_EXAMPLE_IMAGE_OUTPUT_SUPPORT_HPP
