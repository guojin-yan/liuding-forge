// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "model/detector.hpp"

#include <jyppx/liuding/data/data_pipeline.hpp>
#include <jyppx/liuding/data/samplers/random_sampler.hpp>
#include <jyppx/liuding/data_image/joint_transform.hpp>
#include <jyppx/liuding/experiment/evaluation.hpp>
#include <jyppx/liuding/foundation/random_checkpoint_participant.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/foundation/version.hpp>
#include <jyppx/liuding/interop/safetensors.hpp>
#include <jyppx/liuding/nn/model_checkpoint_participant.hpp>
#include <jyppx/liuding/optim/adam_adapter.hpp>
#include <jyppx/liuding/training/checkpoint_coordinator.hpp>
#include <jyppx/liuding/training/detection_evaluation.hpp>
#include <jyppx/liuding/training/file_checkpoint_store.hpp>
#include <jyppx/liuding/training/single_device_strategy.hpp>
#include <jyppx/liuding/training/trainer.hpp>
#if defined(JYPPX_LIUDING_DETECTOR_CUDA_METRICS)
#include <c10/cuda/CUDACachingAllocator.h>
#endif

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
// clang-format off
#include <windows.h>
#include <psapi.h>
// clang-format on
#elif defined(__linux__)
#include <sys/resource.h>
#endif

namespace {
using namespace jyppx::liuding;
using namespace jyppx::liuding::training;
namespace img = jyppx::liuding::data::image;
using Json = nlohmann::json;
using jyppx::liuding::examples::detection::Detector;
namespace fs = std::filesystem;
fs::path data_root;

// External bytes are pinned independently of the optional local paths.
constexpr const char* kManifestSha =
    "38386432386c0fc3b9560c98575677e150708b3dbfb4fc11825035309f4dbd95";
constexpr const char* kConfig =
    "penn-fudan.direct-anchor-detector-v1.rgb128.bilinear-aa.mask-nearest-"
    "exact."
    "seed20261108.batch8.epochs80.adam001.channels16-32-64-64-64-64."
    "anchors16x48-32x80-64x112.grid8.focal2-smoothl1x5."
    "score001.nms05.max100.per-image-fp32";
constexpr std::array<std::array<double, 2>, 3> kAnchors{
    {{16., 48.}, {32., 80.}, {64., 112.}}};
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::uint64_t step_argument(const std::string& text, const char* name) {
  std::uint64_t value = 0;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  require(error == std::errc{} && end == text.data() + text.size() &&
              value > 0 && value < 1200,
          std::string(name) + " must be a step between 1 and 1199");
  return value;
}

std::string config_sha() {
  const std::string config =
      std::string(kConfig) + ";test25;batch4;sources:penn,fudan";
  return sha256_hex(std::as_bytes(std::span(config)));
}

std::vector<std::byte> read(const fs::path& path,
                            std::uint64_t limit = 16ULL << 20U) {
  const auto size = fs::file_size(path);
  if (size > limit) throw std::runtime_error("external fixture size limit");
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  std::ifstream in(path, std::ios::binary);
  if (!in.read(reinterpret_cast<char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size())))
    throw std::runtime_error("external fixture read failed");
  return bytes;
}
void write(const fs::path& path, std::span<const std::byte> bytes) {
  if (fs::exists(path))
    throw std::runtime_error("refuse to overwrite evidence");
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  if (!out) throw std::runtime_error("external fixture write failed");
}
void write_json(const fs::path& path, const Json& value) {
  const auto text = value.dump(2) + "\n";
  write(path, std::as_bytes(std::span(text)));
}
void write_tensor(const fs::path& path, const Tensor& tensor) {
  const auto cpu = tensor.detach().cpu().contiguous();
  write(path,
        {static_cast<const std::byte*>(cpu.const_data_ptr()), cpu.nbytes()});
}
fs::path root() { return data_root; }
Json manifest() {
  const auto bytes = read(root() / "manifest.json", 256U << 10U);
  if (sha256_hex(bytes) != kManifestSha)
    throw std::runtime_error("dataset manifest SHA-256 mismatch");
  const auto* begin = reinterpret_cast<const char*>(bytes.data());
  auto result = Json::parse(begin, begin + bytes.size());
  if (result.at("rows").size() != 170)
    throw std::runtime_error("complete 170-image manifest required");
  return result;
}
struct Data {
  Tensor images, instances;
  std::vector<std::string> ids;
  std::vector<std::size_t> sources;
};
Data load(const std::string& split) {
  std::vector<Tensor> photos, masks;
  Data result;
  const auto definition = manifest();
  for (const auto& row : definition.at("rows")) {
    if (row.at("split").get<std::string>() != split) continue;
    const auto photo_bytes = read(root() / row.at("image").get<std::string>());
    const auto mask_bytes = read(root() / row.at("mask").get<std::string>());
    if (sha256_hex(photo_bytes) != row.at("image_sha256").get<std::string>() ||
        sha256_hex(mask_bytes) != row.at("mask_sha256").get<std::string>())
      throw std::runtime_error("real image/mask SHA-256 mismatch");
    img::ImageDecodeOptions photo_options;
    photo_options.channels = img::ImageChannels::rgb;
    photo_options.dtype = img::ImageDtype::float32_normalized;
    const auto decoded = img::decode_image(photo_bytes, photo_options);
    img::MaskDecodeOptions mask_options;
    mask_options.encoding = std::to_integer<unsigned>(mask_bytes.at(25)) == 3
                                ? img::MaskEncoding::palette_index
                                : img::MaskEncoding::class_index;
    auto mask = img::decode_mask(mask_bytes, mask_options);
    img::VisionSample sample;
    sample.image = decoded.tensor;
    sample.pixels = decoded.pixels;
    sample.width = decoded.width;
    sample.height = decoded.height;
    sample.image_path = row.at("image").get<std::string>();
    sample.mask_target = std::move(mask);
    sample.geometry.emplace(img::ImageExtent{sample.width, sample.height});
    const std::vector<img::ImageGeometryStep> steps{
        {.kind = img::ImageGeometryKind::resize, .extent = {128, 128}}};
    auto resized = img::transform_sample(
        sample, steps, {.interpolation = img::ImageInterpolation::bilinear});
    photos.push_back(std::move(resized.image));
    masks.push_back(std::move(resized.mask_target->ids));
    result.ids.push_back(row.at("id").get<std::string>());
    result.sources.push_back(
        row.at("source").get<std::string>() == "penn" ? 0U : 1U);
  }
  const auto count = split == "train" ? 120U : 25U;
  if (photos.size() != count) throw std::runtime_error("fixed split mismatch");
  result.images = torch::stack(photos);
  result.instances = torch::stack(masks);
  return result;
}

std::vector<metrics::DetectionTarget> targets(const Tensor& mask) {
  std::vector<metrics::DetectionTarget> result;
  const auto max = mask.max().item<std::int64_t>();
  for (std::int64_t id = 1; id <= max; ++id) {
    const auto pixels = torch::nonzero(mask == id);
    if (pixels.size(0) == 0) continue;
    result.push_back({.box = {pixels.select(1, 1).min().item<double>(),
                              pixels.select(1, 0).min().item<double>(),
                              pixels.select(1, 1).max().item<double>() + 1.,
                              pixels.select(1, 0).max().item<double>() + 1.},
                      .class_index = 0});
  }
  return result;
}
Tensor training_targets(const Data& data) {
  auto result = torch::zeros({data.instances.size(0), 3, 5, 8, 8});
  auto a = result.accessor<float, 5>();
  for (std::int64_t image = 0; image < data.instances.size(0); ++image) {
    for (const auto& target : targets(data.instances[image])) {
      const auto& b = target.box;
      const auto cx = (b[0] + b[2]) / 32., cy = (b[1] + b[3]) / 32.;
      const auto x = static_cast<std::int64_t>(cx),
                 y = static_cast<std::int64_t>(cy);
      const auto w = b[2] - b[0], h = b[3] - b[1];
      std::int64_t best = -1;
      double score = -1.;
      for (std::int64_t anchor = 0; anchor < 3; ++anchor) {
        if (a[image][anchor][4][y][x] != 0.f) continue;
        const auto& shape = kAnchors[static_cast<std::size_t>(anchor)];
        const auto intersection = std::min(w, shape[0]) * std::min(h, shape[1]);
        const auto iou =
            intersection / (w * h + shape[0] * shape[1] - intersection);
        if (iou > score) {
          best = anchor;
          score = iou;
        }
      }
      require(best >= 0,
              "More than three target centers in one cell; no target is "
              "silently dropped");
      const auto& shape = kAnchors[static_cast<std::size_t>(best)];
      a[image][best][0][y][x] = static_cast<float>(cx - static_cast<double>(x));
      a[image][best][1][y][x] = static_cast<float>(cy - static_cast<double>(y));
      a[image][best][2][y][x] = static_cast<float>(std::log(w / shape[0]));
      a[image][best][3][y][x] = static_cast<float>(std::log(h / shape[1]));
      a[image][best][4][y][x] = 1.f;
    }
  }
  return result;
}
Tensor loss(const Tensor& output, const Tensor& target) {
  const auto raw = output.reshape({output.size(0), 3, 5, 8, 8});
  const auto positive = target.select(2, 4), logit = raw.select(2, 4);
  const auto p = logit.sigmoid(), count = positive.sum().clamp_min(1.);
  const auto bce =
      torch::logaddexp(torch::zeros_like(logit), logit) - positive * logit;
  const auto focal =
      positive * (1. - p).square() + (1. - positive) * p.square();
  const auto predicted =
      torch::cat({raw.slice(2, 0, 2).sigmoid(), raw.slice(2, 2, 4)}, 2);
  const auto error = (predicted - target.slice(2, 0, 4)).abs();
  const auto smooth = torch::where(error < 1., .5 * error.square(), error - .5);
  return (bce * focal).sum() / count +
         5. * (smooth * positive.unsqueeze(2)).sum() / count;
}
double sigmoid(double value) {
  return 1. / (1. + std::exp(-std::clamp(value, -80., 80.)));
}
double iou(const metrics::DetectionBox& a, const metrics::DetectionBox& b) {
  const auto intersection =
      std::max(0., std::min(a[2], b[2]) - std::max(a[0], b[0])) *
      std::max(0., std::min(a[3], b[3]) - std::max(a[1], b[1]));
  return intersection / ((a[2] - a[0]) * (a[3] - a[1]) +
                         (b[2] - b[0]) * (b[3] - b[1]) - intersection);
}
std::vector<metrics::DetectionPrediction> decode(const Tensor& output) {
  const auto raw = output.detach().cpu().contiguous().reshape({3, 5, 8, 8});
  require(torch::isfinite(raw).all().item<bool>(), "Nonfinite detector output");
  const auto a = raw.accessor<float, 4>();
  std::vector<metrics::DetectionPrediction> candidates, selected;
  for (std::size_t anchor = 0; anchor < kAnchors.size(); ++anchor) {
    for (std::int64_t y = 0; y < 8; ++y) {
      for (std::int64_t x = 0; x < 8; ++x) {
        const auto score = sigmoid(a[anchor][4][y][x]);
        if (score < .01) continue;
        const auto cx =
            (static_cast<double>(x) + sigmoid(a[anchor][0][y][x])) * 16.;
        const auto cy =
            (static_cast<double>(y) + sigmoid(a[anchor][1][y][x])) * 16.;
        const auto w = kAnchors[anchor][0] *
                       std::exp(std::clamp(
                           static_cast<double>(a[anchor][2][y][x]), -4., 4.));
        const auto h = kAnchors[anchor][1] *
                       std::exp(std::clamp(
                           static_cast<double>(a[anchor][3][y][x]), -4., 4.));
        metrics::DetectionBox box{
            std::clamp(cx - w / 2, 0., 128.), std::clamp(cy - h / 2, 0., 128.),
            std::clamp(cx + w / 2, 0., 128.), std::clamp(cy + h / 2, 0., 128.)};
        if (box[2] > box[0] && box[3] > box[1])
          candidates.push_back({.box = box, .score = score, .class_index = 0});
      }
    }
  }
  std::stable_sort(
      candidates.begin(), candidates.end(),
      [](const auto& a, const auto& b) { return a.score > b.score; });
  for (const auto& candidate : candidates) {
    if (std::none_of(selected.begin(), selected.end(), [&](const auto& kept) {
          return iou(candidate.box, kept.box) > .5;
        }))
      selected.push_back(candidate);
    if (selected.size() == 100) break;
  }
  return selected;
}
c10::IValue boxes(const Tensor& raw) {
  c10::impl::GenericList result(c10::AnyType::get());
  for (std::int64_t image = 0; image < raw.size(0); ++image) {
    const auto predictions = decode(raw[image]);
    auto b = torch::empty({static_cast<std::int64_t>(predictions.size()), 4},
                          torch::kFloat64);
    auto s = torch::empty({b.size(0)}, torch::kFloat64);
    for (std::size_t i = 0; i < predictions.size(); ++i) {
      std::copy(predictions[i].box.begin(), predictions[i].box.end(),
                b.data_ptr<double>() + i * 4);
      s.data_ptr<double>()[i] = predictions[i].score;
    }
    result.push_back(c10::ivalue::Tuple::create(
        {b, s, torch::zeros({b.size(0)}, torch::kInt64)}));
  }
  return result;
}
Tensor predict(Detector& module, const Tensor& input) {
  std::vector<Tensor> rows;
  for (std::int64_t i = 0; i < input.size(0); ++i)
    rows.push_back(module.forward(input.narrow(0, i, 1)));
  return torch::cat(rows);
}
void save_weights(const Detector& module, const fs::path& path) {
  std::vector<Tensor> values;
  for (const auto& p : module.named_parameters())
    values.push_back(p.value().detach().cpu().contiguous());
  std::vector<interop::SafeTensorSource> sources;
  std::size_t i = 0;
  for (const auto& p : module.named_parameters()) {
    const auto& value = values.at(i++);
    sources.push_back(
        {.name = p.key(),
         .dtype = interop::SafeTensorDtype::f32,
         .shape = {value.sizes().begin(), value.sizes().end()},
         .data = {static_cast<const std::byte*>(value.const_data_ptr()),
                  value.nbytes()}});
  }
  write(path, interop::serialize_safetensors(
                  sources, {{"dataset_manifest_sha256", kManifestSha},
                            {"recipe", kConfig}}));
}
std::shared_ptr<Detector> trained(const fs::path& path,
                                  const std::string& expected,
                                  const torch::Device& device) {
  const auto bytes = read(path, 4U << 20U);
  require(sha256_hex(bytes) == expected,
          "Expected immutable detector weight SHA-256 differs");
  const interop::SafeTensorsView view(bytes);
  auto module = std::make_shared<Detector>();
  require(
      view.directory().metadata.at("dataset_manifest_sha256") == kManifestSha &&
          view.directory().metadata.at("recipe") == kConfig,
      "Detector recipe identity differs");
  require(view.directory().tensors.size() == module->named_parameters().size(),
          "Detector topology differs");
  torch::NoGradGuard guard;
  for (const auto& p : module->named_parameters()) {
    const auto info = std::find_if(
        view.directory().tensors.begin(), view.directory().tensors.end(),
        [&](const auto& item) { return item.name == p.key(); });
    const std::vector<std::uint64_t> shape(p.value().sizes().begin(),
                                           p.value().sizes().end());
    require(info != view.directory().tensors.end() &&
                info->dtype == interop::SafeTensorDtype::f32 &&
                info->shape == shape,
            "Detector parameter dtype or shape differs");
    const auto value = view.tensor_bytes(p.key());
    require(value.size() == p.value().nbytes(),
            "Detector parameter bytes differ");
    std::memcpy(p.value().data_ptr(), value.data(), value.size());
    require(torch::isfinite(p.value()).all().item<bool>(),
            "Nonfinite detector weight");
  }
  module->to(device);
  return module;
}
std::uint64_t peak_rss() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS counters{};
  if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)))
    return static_cast<std::uint64_t>(counters.PeakWorkingSetSize);
#elif defined(__linux__)
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) == 0)
    return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024U;
#endif
  throw std::runtime_error("Process RSS counter unavailable");
}
void resources(Json& result, std::chrono::steady_clock::time_point start,
               const torch::Device& device) {
  result["elapsed_seconds"] =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  result["peak_rss_bytes"] = peak_rss();
  result["device"] = device.str();
  result["manifest_sha256"] = kManifestSha;
  result["config_sha256"] = config_sha();
  result["build_fingerprint"] = build_fingerprint_json();
  require(peak_rss() < (6ULL << 30U), "Task exceeds 6 GiB RSS");
#if defined(JYPPX_LIUDING_DETECTOR_CUDA_METRICS)
  if (device.is_cuda()) {
    const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
    result["cuda_peak_allocated_bytes"] = stats.allocated_bytes[0].peak;
    result["cuda_peak_reserved_bytes"] = stats.reserved_bytes[0].peak;
    require(stats.reserved_bytes[0].peak < (3LL << 30U),
            "Task exceeds 3 GiB reserved CUDA memory");
  }
#endif
}
void train(const fs::path& out, const torch::Device& device,
           const fs::path& resume, std::uint64_t until,
           std::uint64_t checkpoint_every) {
  const auto started = std::chrono::steady_clock::now();
  auto data = load("train");
  const auto labels = training_targets(data);
  auto holder = std::make_shared<Detector>();
  auto model =
      std::make_shared<nn::ModelAdapter>(nn::adapt_tensor_module(holder));
  std::vector<optim::OptimizerParameterBinding> bindings;
  for (const auto& p : model->state().named_parameters())
    bindings.push_back({.parameter_id = p.key(), .parameter = p.value()});
  auto optimizer = std::make_shared<optim::AdamOptimizerAdapter>(
      "optimizer.detector", std::vector<optim::AdamParameterGroup>{
                                {.group_id = "main",
                                 .parameters = std::move(bindings),
                                 .options = {.learning_rate = .001}}});
  auto pipeline = std::make_shared<data::DataPipeline>(
      std::make_shared<data::samplers::RandomSampler>(120, 20261108),
      data::DataPipelineOptions{
          .pipeline_id = "data.detector.train",
          .dataset_version = kManifestSha,
          .dataset_size = 120,
          .schema = data::BatchSchema::any("detector.batch", 1),
          .batch_factory_id = kConfig,
          .batch_size = 8,
          .worker_seed = 20261108},
      [images = data.images, labels](std::span<const std::size_t> indices,
                                     const data::DataWorkerContext&,
                                     std::stop_token) {
        std::vector<std::int64_t> rows(indices.begin(), indices.end());
        const auto selected = torch::tensor(rows, torch::kInt64);
        return c10::ivalue::Tuple::create({images.index_select(0, selected),
                                           labels.index_select(0, selected)});
      });
  Trainer trainer({.trainer_id = "trainer.detector",
                   .run_id = "penn-fudan.direct-detector.final80",
                   .trace_id = "trace.detector",
                   .max_epochs = 80});
  trainer.prepare(
      {.model = model,
       .optimizers =
           {{.optimizer = optimizer,
             .checkpoint =
                 std::make_shared<optim::AdamOptimizerCheckpointParticipant>(
                     optimizer)}},
       .training_data = pipeline,
       .training_batch_mapper =
           [device](const data::BatchEnvelope& batch) {
             return nn::Invocation{
                 .positional = {
                     batch.value.toTuple()->elements()[0].toTensor().to(
                         device)}};
           },
       .training_target_mapper = [device](const data::BatchEnvelope& batch)
           -> std::optional<c10::IValue> {
         return batch.value.toTuple()->elements()[1].toTensor().to(device);
       },
       .training_step =
           [](const StepInput& input) {
             const auto output =
                 input.model.invoke(input.invocation).value.toTensor();
             return StepOutput{.loss = loss(output, input.target->toTensor()),
                               .sample_count = input.batch.sample_ids.size()};
           },
       .strategy =
           std::make_shared<SingleDeviceStrategy>(SingleDeviceStrategyOptions{
               .strategy_id = "strategy.detector", .device = device}),
       .model_checkpoint =
           std::make_shared<nn::ModelCheckpointParticipant>(*model),
       .rng_checkpoint = std::make_shared<core::RngCheckpointParticipant>()});
  if (!resume.empty()) {
    FileCheckpointStore source(resume);
    trainer.restore_checkpoint(source, RestorePolicy::exact_only);
  }
  const auto before = trainer.global_step();
  require(until == 0 || until > before,
          "Pause must be after the restored step");
  const auto target = until == 0 ? 1200 : until;
  FileCheckpointStore sink(out / "checkpoint.jyck");
  FitResult result;
  std::size_t participant_count = 0;
  std::uint64_t checkpoints_written = 0;
  do {
    const auto boundary =
        checkpoint_every == 0
            ? target
            : std::min(target, (trainer.global_step() / checkpoint_every + 1) *
                                   checkpoint_every);
    result = trainer.fit_until(boundary == 1200 ? 0 : boundary);
    require(result.global_step == boundary,
            "Training did not reach its checkpoint boundary");
    const auto descriptor = trainer.save_checkpoint(sink);
    participant_count = descriptor.participants.size();
    ++checkpoints_written;
    if (checkpoint_every != 0) {
      std::cout << Json{{"event", "checkpoint"},
                        {"global_step", result.global_step},
                        {"checkpoint_sha256",
                         sha256_hex(read(out / "checkpoint.jyck", 16U << 20U))}}
                       .dump()
                << std::endl;
    }
  } while (result.global_step < target);
  require(result.global_step == (until == 0 ? 1200 : until),
          "Training did not reach its frozen boundary");
  require(
      result.status == (until == 0 ? FitStatus::completed : FitStatus::paused),
      "Unexpected training outcome");
  save_weights(*holder, out / "model.safetensors");
  trainer.close();
  Json summary{{"global_step", result.global_step},
               {"initial_global_step", before},
               {"applied_updates", result.global_step - before},
               {"completed_epochs", result.completed_epochs},
               {"participant_count", participant_count},
               {"checkpoint_every", checkpoint_every},
               {"checkpoints_written", checkpoints_written},
               {"paused", result.status == FitStatus::paused},
               {"weights_sha256", sha256_hex(read(out / "model.safetensors"))},
               {"checkpoint_sha256",
                sha256_hex(read(out / "checkpoint.jyck", 16U << 20U))}};
  resources(summary, started, device);
  write_json(out / "training.json", summary);
  std::cout << summary.dump() << '\n';
}

std::string report(const DetectionEvaluationResult& result,
                   const std::string& weights, const std::string& device) {
  auto points = result.observations;
  auto ap50 = points.front();
  ap50.definition.metric_id = "detection.box_ap50";
  ap50.definition.parameters["iou_thresholds"] = "0.5";
  ap50.observation.value = result.overall.per_class.front().ap_by_iou.front();
  points.push_back(std::move(ap50));
  experiment::EvaluationSuite suite{
      .suite_id = "penn-fudan.direct-detector.test25",
      .task = "axis_aligned_box_detection",
      .dataset_digest = {.hex = kManifestSha},
      .environment_id = device + ".fp32",
      .seed = 20261108};
  for (std::size_t i = 0; i < points.size(); ++i) {
    const auto& p = points[i];
    suite.rules.push_back(
        {.rule_id = "r" + std::to_string(i),
         .definition = p.definition,
         .context = p.context,
         .minimum_samples = 1,
         .minimum = p.definition.metric_id == "detection.box_ap50" ? .10 : 0.});
  }
  experiment::EvaluationIdentity identity{
      .evaluation_id = "penn-fudan.direct-detector." + device,
      .run_id = "penn-fudan.direct-detector.final80",
      .model_digest = {.hex = weights},
      .build_fingerprint = build_fingerprint_json(),
      .adapter_id = kConfig,
      .config_digest = {.hex = config_sha()}};
  const auto value = experiment::run_evaluation(
      suite, identity, [&](const auto&, auto, auto) { return points; });
  return experiment::quality_report_json(value);
}
void evaluate(const fs::path& out, const torch::Device& device,
              const fs::path& weights, const std::string& weight_sha) {
  const auto started = std::chrono::steady_clock::now();
  auto data = load("test");
  Tensor raw;
  {
    torch::NoGradGuard guard;
    auto model = trained(weights, weight_sha, device);
    model->eval();
    raw = predict(*model, data.images.to(device)).cpu();
  }
  write_tensor(out / "raw.f32", raw);
  write_tensor(out / "instances.i64", data.instances);
  Json predictions{{"ids", data.ids},
                   {"sources", data.sources},
                   {"predictions", Json::array()},
                   {"targets", Json::array()}};
  for (std::int64_t i = 0; i < raw.size(0); ++i) {
    Json p = Json::array(), t = Json::array();
    for (const auto& box : decode(raw[i]))
      p.push_back({{"box", box.box}, {"score", box.score}});
    for (const auto& box : targets(data.instances[i])) t.push_back(box.box);
    predictions["predictions"].push_back(std::move(p));
    predictions["targets"].push_back(std::move(t));
  }
  write_json(out / "predictions.json", predictions);
  std::uint64_t calls = 0;
  const auto make_model = [&] {
    return nn::adapt_module(
        trained(weights, weight_sha, device),
        [&](Detector& module, const nn::Invocation& in) -> c10::IValue {
          ++calls;
          require(!module.is_training() && !torch::GradMode::is_enabled(),
                  "Evaluation guard missing");
          return boxes(predict(module, in.positional.front().toTensor()));
        },
        {.positional = {ValueSchema::tensor()}},
        ValueSchema::list(
            ValueSchema::tuple({ValueSchema::tensor(), ValueSchema::tensor(),
                                ValueSchema::tensor()})));
  };
  DetectionEvaluationOptions options;
  options.metric = {
      .dataset_id = std::string(kManifestSha) + ".test25",
      .class_vocabulary_id = "person/v1",
      .num_classes = 1,
      .iou_thresholds = {.5, .55, .6, .65, .7, .75, .8, .85, .9, .95},
      .sources = {"penn", "fudan"}};
  options.expected_images = 25;
  options.limits.max_images = 25;
  options.max_batches = 25;
  const auto source = [&](std::uint64_t batch, std::stop_token,
                          std::uint64_t size =
                              4) -> std::optional<DetectionEvaluationBatch> {
    const auto begin = static_cast<std::int64_t>(batch * size);
    if (begin >= 25) return std::nullopt;
    const auto n = std::min(static_cast<std::int64_t>(size), 25 - begin);
    DetectionEvaluationBatch value{
        .invocation =
            {.positional = {data.images.narrow(0, begin, n).to(device)}},
        .images = {}};
    for (std::int64_t i = begin; i < begin + n; ++i)
      value.images.push_back(
          {.image_id = static_cast<std::uint64_t>(i + 1),
           .source_index = data.sources.at(static_cast<std::size_t>(i)),
           .targets = targets(data.instances[i]),
           .predictions = {}});
    return value;
  };
  auto model = make_model();
  std::uint64_t cursor = 0;
  const auto full = evaluate_detection(
      model, [&](auto stop) { return source(cursor++, stop); }, options);
  cursor = 0;
  const auto split = evaluate_detection(
      model, [&](auto stop) { return source(cursor++, stop, 1); }, options);
  const auto encoded = report(full, weight_sha, device.str());
  require(encoded == report(split, weight_sha, device.str()),
          "Partitioned detection report differs");
  const auto make_session = [&] {
    return std::make_shared<DetectionEvaluationSession>(
        DetectionEvaluationIdentity{
            .evaluation_id = "penn-fudan.direct-detector",
            .dataset_digest = {.hex = kManifestSha},
            .model_digest = {.hex = weight_sha},
            .config_digest = {.hex = config_sha()},
            .build_fingerprint = build_fingerprint_json()},
        options);
  };
  struct Interrupted {};
  {
    auto state = make_session();
    CheckpointParticipantRegistry registry;
    registry.register_participant(state);
    FileCheckpointStore checkpoint(out / "evaluation.jyck");
    CheckpointCoordinator coordinator(registry);
    bool interrupted = false;
    try {
      state->run(
          model, [&](auto batch, auto stop) { return source(batch, stop); },
          [&](const auto& current) {
            if (current.progress().batches == 2) {
              coordinator.save(checkpoint);
              throw Interrupted{};
            }
          });
    } catch (const Interrupted&) {
      interrupted = true;
    }
    require(interrupted, "Evaluation did not stop at its durable boundary");
  }
  auto state = make_session();
  CheckpointParticipantRegistry registry;
  registry.register_participant(state);
  FileCheckpointStore checkpoint(out / "evaluation.jyck");
  CheckpointCoordinator coordinator(registry);
  coordinator.restore(checkpoint);
  require(state->progress().batches == 2, "Evaluation cursor was not restored");
  // A new model object is rebuilt from the immutable weights before
  // continuation.
  auto resumed_model = make_model();
  const auto resumed = state->run(
      resumed_model, [&](auto batch, auto stop) { return source(batch, stop); },
      [&](const auto&) { coordinator.save(checkpoint); });
  require(encoded == report(resumed, weight_sha, device.str()),
          "Resumed detection report differs");
  auto complete = make_session();
  CheckpointParticipantRegistry complete_registry;
  complete_registry.register_participant(complete);
  CheckpointCoordinator(complete_registry).restore(checkpoint);
  const auto complete_calls = calls;
  complete->run(resumed_model,
                [](auto, auto) -> std::optional<DetectionEvaluationBatch> {
                  throw std::runtime_error("Completed evaluator reread data");
                });
  require(calls == complete_calls, "Completed evaluator invoked model");
  write(out / "quality.json", std::as_bytes(std::span(encoded)));
  Json result{{"ap50", *full.overall.per_class.front().ap_by_iou.front()},
              {"map", *full.overall.mean_average_precision},
              {"ground_truth", full.overall.per_class.front().ground_truth},
              {"predictions", full.overall.per_class.front().predictions},
              {"by_source", Json::array()},
              {"weights_sha256", weight_sha},
              {"inference_calls", calls},
              {"partition_report_equal", true},
              {"resumed_report_equal", true},
              {"complete_no_reread", true}};
  for (const auto& s : full.by_source)
    result["by_source"].push_back(
        {{"ap50", *s.per_class.front().ap_by_iou.front()},
         {"map", *s.mean_average_precision},
         {"ground_truth", s.per_class.front().ground_truth},
         {"predictions", s.per_class.front().predictions}});
  resources(result, started, device);
  write_json(out / "results.json", result);
  std::cout << result.dump() << '\n';
  require(*full.overall.per_class.front().ap_by_iou.front() >= .10,
          "Held-out AP50 failed frozen 0.10 threshold");
}
}  // namespace
int main(int argc, char** argv) {
  try {
    require(argc >= 5,
            "Usage: pedestrian-detector train DATA NEW_OUTPUT cpu|cuda "
            "[CHECKPOINT|-] [PAUSE_STEP] [--checkpoint-every STEP]; "
            "or evaluate DATA NEW_OUTPUT cpu|cuda "
            "WEIGHTS SHA256");
    const std::string action = argv[1];
    int positional_end = argc;
    std::uint64_t checkpoint_every = 0;
    if (action == "train" && argc >= 7 &&
        std::string(argv[argc - 2]) == "--checkpoint-every") {
      checkpoint_every = step_argument(argv[argc - 1], "Checkpoint interval");
      positional_end -= 2;
    }
    require((action == "train" && positional_end >= 5 && positional_end <= 7) ||
                (action == "evaluate" && argc == 7),
            "Invalid detector arguments");
    fs::path resume;
    std::uint64_t until = 0;
    if (action == "train") {
      if (positional_end >= 6) {
        const std::string text = argv[5];
        require(!text.starts_with("--"), "Unexpected detector option");
        if (text != "-") resume = fs::path(text);
      }
      if (positional_end == 7) until = step_argument(argv[6], "Pause");
    }
    data_root = fs::absolute(argv[2]);
    const fs::path out = fs::absolute(argv[3]);
    const std::string device_name = argv[4];
    require(device_name == "cpu" || device_name == "cuda",
            "Device must be cpu or cuda");
    const torch::Device device(device_name == "cuda" ? "cuda:0" : "cpu");
    require(!device.is_cuda() || torch::cuda::is_available(),
            "Requested CUDA is unavailable");
    torch::manual_seed(20261108);
    torch::set_num_threads(1);
    at::globalContext().setAllowTF32CuDNN(false);
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setDeterministicCuDNN(true);
    at::globalContext().setBenchmarkCuDNN(false);
    at::globalContext().setDeterministicAlgorithms(true, false);
    require(fs::create_directories(out), "Output must be a fresh directory");
    if (action == "train") {
      train(out, device, resume, until, checkpoint_every);
    } else
      evaluate(out, device, fs::path(argv[5]), argv[6]);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
