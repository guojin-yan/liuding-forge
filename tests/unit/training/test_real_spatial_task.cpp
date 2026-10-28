// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
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
#include <jyppx/liuding/training/segmentation_evaluation.hpp>
#include <jyppx/liuding/training/single_device_strategy.hpp>
#include <jyppx/liuding/training/trainer.hpp>
#if defined(JYPPX_LIUDING_SPATIAL_CUDA_METRICS)
#include <c10/cuda/CUDACachingAllocator.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
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
namespace fs = std::filesystem;

// External bytes are pinned independently of the optional local paths.
constexpr const char* kManifestSha =
    "38386432386c0fc3b9560c98575677e150708b3dbfb4fc11825035309f4dbd95";
constexpr const char* kWeightsSha =
    "983cddb5b67bd9bf6bc03819e128f914c41ef72de1ed7ca7c0bd31303bfac4d8";
constexpr const char* kConfig =
    "penn-fudan.tiny-unet-8.final80.rgb128.bilinear-aa.mask-nearest-exact."
    "per-image-fp32.argmax.cc4-area16.mean-score.xyxy-exclusive/v1";

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
fs::path root() {
  const auto* path = std::getenv("JYPPX_LIUDING_SPATIAL_TASK_DIR");
  if (!path || !*path) throw std::runtime_error("explicit task root required");
  return fs::path(path);
}
fs::path output() {
  const auto* path = std::getenv("JYPPX_LIUDING_SPATIAL_OUTPUT_DIR");
  if (!path || !*path)
    throw std::runtime_error("explicit new output directory required");
  fs::path result(path);
  if (!fs::create_directories(result))
    throw std::runtime_error("task output must be a fresh directory");
  return result;
}
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

torch::nn::Sequential block(std::int64_t in, std::int64_t out) {
  return torch::nn::Sequential(
      torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 3).padding(1)),
      torch::nn::ReLU(),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(out, out, 3).padding(1)),
      torch::nn::ReLU());
}
class TinyUNet final : public torch::nn::Module {
 public:
  TinyUNet()
      : e1(register_module("e1", block(3, 8))),
        e2(register_module("e2", block(8, 16))),
        middle(register_module("middle", block(16, 32))),
        d2(register_module("d2", block(48, 16))),
        d1(register_module("d1", block(24, 8))),
        head(register_module("head", torch::nn::Conv2d(8, 2, 1))) {}
  Tensor forward(const Tensor& x) {
    const auto a = e1->forward(x);
    const auto b = e2->forward(torch::max_pool2d(a, 2));
    const auto c = middle->forward(torch::max_pool2d(b, 2));
    const auto up = [](const Tensor& v) {
      // Exact nearest 2x, with a deterministic reduction in backward.
      return v.unsqueeze(3)
          .unsqueeze(5)
          .expand({v.size(0), v.size(1), v.size(2), 2, v.size(3), 2})
          .reshape({v.size(0), v.size(1), v.size(2) * 2, v.size(3) * 2});
    };
    const auto d = d2->forward(torch::cat({up(c), b}, 1));
    return head->forward(d1->forward(torch::cat({up(d), a}, 1)));
  }

 private:
  torch::nn::Sequential e1, e2, middle, d2, d1;
  torch::nn::Conv2d head;
};
void save_weights(TinyUNet& module, const fs::path& path) {
  std::vector<Tensor> values;
  for (const auto& p : module.named_parameters())
    values.push_back(p.value().detach().cpu().contiguous());
  std::vector<interop::SafeTensorSource> sources;
  std::size_t i = 0;
  for (const auto& p : module.named_parameters()) {
    const auto& v = values.at(i++);
    sources.push_back(
        {.name = p.key(),
         .dtype = interop::SafeTensorDtype::f32,
         .shape = {v.sizes().begin(), v.sizes().end()},
         .data = {static_cast<const std::byte*>(v.const_data_ptr()),
                  v.nbytes()}});
  }
  write(path, interop::serialize_safetensors(
                  sources, {{"dataset_manifest_sha256", kManifestSha},
                            {"recipe", kConfig}}));
}
std::shared_ptr<TinyUNet> trained(const torch::Device& device) {
  auto module = std::make_shared<TinyUNet>();
  const auto bytes = read(root() / "trained.safetensors", 1U << 20U);
  if (sha256_hex(bytes) != kWeightsSha)
    throw std::runtime_error("trained weight SHA-256 mismatch");
  const interop::SafeTensorsView view(bytes);
  if (view.directory().tensors.size() != module->named_parameters().size())
    throw std::runtime_error("trained weight topology mismatch");
  if (view.directory().metadata.at("dataset_manifest_sha256") != kManifestSha ||
      view.directory().metadata.at("recipe") != kConfig)
    throw std::runtime_error("trained weight recipe mismatch");
  torch::NoGradGuard guard;
  for (const auto& p : module->named_parameters()) {
    const auto value = view.tensor_bytes(p.key());
    if (value.size() != p.value().nbytes())
      throw std::runtime_error("trained parameter size mismatch");
    std::memcpy(p.value().data_ptr(), value.data(), value.size());
  }
  module->to(device);
  return module;
}
void setup() {
  torch::manual_seed(20261028);
  torch::set_num_threads(2);
  at::globalContext().setAllowTF32CuDNN(false);
  at::globalContext().setAllowTF32CuBLAS(false);
  at::globalContext().setDeterministicCuDNN(true);
  at::globalContext().setBenchmarkCuDNN(false);
  at::globalContext().setDeterministicAlgorithms(true, false);
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
  throw std::runtime_error("peak process RSS unavailable");
}
Tensor predict(TinyUNet& module, const Tensor& x) {
  std::vector<Tensor> rows;
  for (std::int64_t i = 0; i < x.size(0); ++i)
    rows.push_back(module.forward(x.narrow(0, i, 1)));
  return torch::cat(rows);
}
std::vector<metrics::DetectionPrediction> proposals(const Tensor& logits) {
  const auto probabilities = logits.softmax(0)[1].cpu().contiguous();
  const auto mask = logits.argmax(0).cpu().contiguous();
  const auto* p = probabilities.const_data_ptr<float>();
  const auto* m = mask.const_data_ptr<std::int64_t>();
  std::vector<bool> seen(128 * 128, false);
  std::vector<metrics::DetectionPrediction> result;
  std::vector<std::size_t> queue;
  queue.reserve(128 * 128);
  for (std::size_t index = 0; index < seen.size(); ++index) {
    if (seen[index] || m[index] != 1) continue;
    queue.clear();
    queue.push_back(index);
    seen[index] = true;
    std::size_t left = 128, top = 128, right = 0, bottom = 0;
    double score = 0;
    for (std::size_t at = 0; at < queue.size(); ++at) {
      const auto q = queue[at], y = q / 128, x = q % 128;
      left = std::min(left, x);
      right = std::max(right, x + 1);
      top = std::min(top, y);
      bottom = std::max(bottom, y + 1);
      score += p[q];
      const auto visit = [&](std::size_t next) {
        if (!seen[next] && m[next] == 1) {
          seen[next] = true;
          queue.push_back(next);
        }
      };
      if (x > 0) visit(q - 1);
      if (x + 1 < 128) visit(q + 1);
      if (y > 0) visit(q - 128);
      if (y + 1 < 128) visit(q + 128);
    }
    if (queue.size() >= 16)
      result.push_back(
          {.box = {static_cast<double>(left), static_cast<double>(top),
                   static_cast<double>(right), static_cast<double>(bottom)},
           .score = score / static_cast<double>(queue.size()),
           .class_index = 0});
  }
  return result;
}
c10::IValue boxes(const Tensor& logits) {
  c10::impl::GenericList output(c10::AnyType::get());
  for (std::int64_t i = 0; i < logits.size(0); ++i) {
    const auto predictions = proposals(logits[i]);
    auto b = torch::empty({static_cast<std::int64_t>(predictions.size()), 4},
                          torch::kFloat64);
    auto s = torch::empty({static_cast<std::int64_t>(predictions.size())},
                          torch::kFloat64);
    for (std::size_t j = 0; j < predictions.size(); ++j) {
      std::copy(predictions[j].box.begin(), predictions[j].box.end(),
                b.data_ptr<double>() + j * 4);
      s.data_ptr<double>()[j] = predictions[j].score;
    }
    output.push_back(c10::ivalue::Tuple::create(
        {b, s, torch::zeros({b.size(0)}, torch::kInt64)}));
  }
  return output;
}
nn::ModelAdapter adapter(const torch::Device& device, bool detection,
                         int& calls) {
  return nn::adapt_module(
      trained(device),
      [detection, &calls](TinyUNet& module,
                          const nn::Invocation& in) -> c10::IValue {
        ++calls;
        if (module.is_training() || torch::GradMode::is_enabled())
          throw std::runtime_error("actual inference guard missing");
        const auto logits = predict(module, in.positional.front().toTensor());
        return detection ? boxes(logits) : c10::IValue(logits);
      },
      {.positional = {ValueSchema::tensor()}},
      detection ? ValueSchema::list(ValueSchema::tuple({ValueSchema::tensor(),
                                                        ValueSchema::tensor(),
                                                        ValueSchema::tensor()}))
                : ValueSchema::tensor());
}
template <class Result>
std::string report(const Result& result, const std::string& task,
                   const std::string& device) {
  experiment::EvaluationSuite suite{.suite_id = "penn-fudan.fixed25." + task,
                                    .task = task,
                                    .dataset_digest = {.hex = kManifestSha},
                                    .environment_id = device + ".fp32"};
  for (std::size_t i = 0; i < result.observations.size(); ++i) {
    const auto& point = result.observations[i];
    double minimum = 0.;
    if (point.context.at("slice") == "overall") {
      const auto& cls = point.context.at("class");
      if (task == "iou") minimum = cls == "all" ? .60 : cls == "1" ? .35 : 0.;
      if (task == "box") minimum = .05;  // AP50 >= .10 implies mean AP >= .05.
    }
    suite.rules.push_back({.rule_id = "r" + std::to_string(i),
                           .definition = point.definition,
                           .context = point.context,
                           .minimum_samples = 1,
                           .minimum = minimum});
  }
  experiment::EvaluationIdentity identity{
      .evaluation_id = "penn-fudan." + task + "." + device,
      .run_id = "penn-fudan.tiny-unet.final80",
      .model_digest = {.hex = kWeightsSha},
      .build_fingerprint = build_fingerprint_json(),
      .adapter_id = kConfig,
      .config_digest = {.hex = config_sha()}};
  const auto value = experiment::run_evaluation(
      suite, identity,
      [&](const auto&, auto, auto) { return result.observations; });
  EXPECT_EQ(value.verdict, experiment::QualityVerdict::pass);
  return experiment::quality_report_json(value);
}

template <class Session, class ModelFactory, class SessionFactory, class Source>
auto reopen(const fs::path& checkpoint, const ModelFactory& make_model,
            const SessionFactory& make_session, const Source& source) {
  struct Interrupted {};
  {
    auto s = make_session();
    auto model = make_model();
    CheckpointParticipantRegistry registry;
    registry.register_participant(s);
    CheckpointCoordinator coordinator(registry);
    FileCheckpointStore store(checkpoint);
    EXPECT_THROW(s->run(model, source,
                        [&](const auto& state) {
                          if (state.progress().batches == 2) {
                            coordinator.save(store);
                            throw Interrupted{};
                          }
                        }),
                 Interrupted);
  }
  auto s = make_session();
  auto model = make_model();
  CheckpointParticipantRegistry registry;
  registry.register_participant(s);
  CheckpointCoordinator coordinator(registry);
  FileCheckpointStore store(checkpoint);
  coordinator.restore(store);
  EXPECT_EQ(s->progress().batches, 2);
  auto result =
      s->run(model, source, [&](const auto&) { coordinator.save(store); });
  EXPECT_TRUE(s->progress().complete);
  auto complete = make_session();
  CheckpointParticipantRegistry final_registry;
  final_registry.register_participant(complete);
  FileCheckpointStore final_store(checkpoint);
  CheckpointCoordinator(final_registry).restore(final_store);
  complete->run(model,
                [](auto, auto) -> decltype(source(0, std::stop_token{})) {
                  throw std::runtime_error(
                      "complete evaluation must not reread real data");
                });
  return result;
}

void task(bool cuda) {
  setup();
  const auto started = std::chrono::steady_clock::now();
  const torch::Device device(cuda ? "cuda:0" : "cpu");
  const auto out = output();
  auto data = load("test");
  int calls = 0;
  const auto make_model = [&](bool detection) {
    return adapter(device, detection, calls);
  };
  // Export actual model predictions, not labels, for an independent evaluator.
  Tensor logits;
  {
    torch::NoGradGuard guard;
    auto module = trained(device);
    module->eval();
    logits = predict(*module, data.images.to(device)).cpu();
  }
  write_tensor(out / "images.f32", data.images);
  write_tensor(out / "instances.i64", data.instances);
  write_tensor(out / "logits.f32", logits);
  Json raw = {{"ids", data.ids},
              {"sources", data.sources},
              {"proposals", Json::array()}};
  for (std::int64_t i = 0; i < 25; ++i) {
    Json predictions = Json::array();
    for (const auto& p : proposals(logits[i]))
      predictions.push_back({{"box", p.box}, {"score", p.score}});
    raw["proposals"].push_back(std::move(predictions));
  }
  Json results;
  for (const auto score :
       {metrics::SegmentationScore::iou, metrics::SegmentationScore::dice}) {
    const std::string name =
        score == metrics::SegmentationScore::iou ? "iou" : "dice";
    SegmentationEvaluationOptions options;
    options.metric.num_classes = 2;
    options.metric.score = score;
    options.sources = {"penn", "fudan"};
    options.max_images = 25;
    options.max_batches = 25;
    const auto source =
        [&](std::uint64_t batch, std::stop_token,
            std::uint64_t batch_size =
                4) -> std::optional<SegmentationEvaluationBatch> {
      const auto begin = static_cast<std::int64_t>(batch * batch_size);
      if (begin >= 25) return std::nullopt;
      const auto n = std::min<std::int64_t>(
          static_cast<std::int64_t>(batch_size), 25 - begin);
      return SegmentationEvaluationBatch{
          .invocation =
              {.positional = {data.images.narrow(0, begin, n).to(device)}},
          .targets = data.instances.narrow(0, begin, n)
                         .gt(0)
                         .to(torch::kInt64)
                         .to(device),
          .source_indices = {data.sources.begin() + begin,
                             data.sources.begin() + begin + n}};
    };
    auto model = make_model(false);
    std::uint64_t offset = 0;
    const auto result = evaluate_semantic_segmentation(
        model, [&](auto stop) { return source(offset++, stop); }, options);
    offset = 0;
    const auto partitioned = evaluate_semantic_segmentation(
        model, [&](auto stop) { return source(offset++, stop, 1); }, options);
    EXPECT_EQ(result.overall.score, partitioned.overall.score);
    EXPECT_EQ(result.overall.per_class, partitioned.overall.per_class);
    for (std::size_t i = 0; i < result.by_source.size(); ++i)
      EXPECT_EQ(result.by_source[i].per_class,
                partitioned.by_source[i].per_class);
    for (std::size_t i = 0; i < result.overall.counts.size(); ++i) {
      EXPECT_EQ(result.overall.counts[i].intersection,
                partitioned.overall.counts[i].intersection);
      EXPECT_EQ(result.overall.counts[i].target,
                partitioned.overall.counts[i].target);
      EXPECT_EQ(result.overall.counts[i].predicted,
                partitioned.overall.counts[i].predicted);
    }
    const auto make_session = [&] {
      return std::make_shared<SegmentationEvaluationSession>(
          SegmentationEvaluationIdentity{
              .evaluation_id = "penn-fudan." + name,
              .dataset_digest = {.hex = kManifestSha},
              .model_digest = {.hex = kWeightsSha},
              .config_digest = {.hex = config_sha()},
              .build_fingerprint = build_fingerprint_json()},
          options, 25, 25 * 128 * 128);
    };
    const auto resumed = reopen<SegmentationEvaluationSession>(
        out / (name + ".state"), [&] { return make_model(false); },
        make_session, source);
    const auto encoded = report(result, name, cuda ? "cuda" : "cpu");
    EXPECT_EQ(encoded, report(resumed, name, cuda ? "cuda" : "cpu"));
    write(out / (name + "-report.json"), std::as_bytes(std::span(encoded)));
    results[name] = {
        {"overall", *result.overall.score},
        {"per_class",
         {*result.overall.per_class[0], *result.overall.per_class[1]}},
        {"counts", Json::array()}};
    for (const auto& c : result.overall.counts)
      results[name]["counts"].push_back(
          {c.intersection, c.target, c.predicted});
    results[name]["by_source"] = Json::array();
    for (const auto& summary : result.by_source) {
      Json slice{{"overall", *summary.score},
                 {"per_class", {*summary.per_class[0], *summary.per_class[1]}},
                 {"counts", Json::array()}};
      for (const auto& c : summary.counts)
        slice["counts"].push_back({c.intersection, c.target, c.predicted});
      results[name]["by_source"].push_back(std::move(slice));
    }
    if (score == metrics::SegmentationScore::iou) {
      EXPECT_GE(*result.overall.score, .60);
      EXPECT_GE(*result.overall.per_class[1], .35);
    }
  }
  DetectionEvaluationOptions options;
  options.metric.dataset_id = kManifestSha;
  options.metric.class_vocabulary_id = "person/v1";
  options.metric.num_classes = 1;
  options.metric.iou_thresholds = {.5, .75};
  options.metric.sources = {"penn", "fudan"};
  options.expected_images = 25;
  options.max_batches = 25;
  const auto source = [&](std::uint64_t batch, std::stop_token,
                          std::uint64_t batch_size =
                              4) -> std::optional<DetectionEvaluationBatch> {
    const auto begin = static_cast<std::int64_t>(batch * batch_size);
    if (begin >= 25) return std::nullopt;
    const auto n = std::min<std::int64_t>(static_cast<std::int64_t>(batch_size),
                                          25 - begin);
    DetectionEvaluationBatch result{
        .invocation = {
            .positional = {data.images.narrow(0, begin, n).to(device)}}};
    for (std::int64_t i = begin; i < begin + n; ++i) {
      metrics::DetectionImage image{
          .image_id = static_cast<std::uint64_t>(i),
          .source_index = data.sources.at(static_cast<std::size_t>(i))};
      const auto mask = data.instances[i];
      const auto max = mask.max().item<std::int64_t>();
      for (std::int64_t id = 1; id <= max; ++id) {
        const auto pixels = torch::nonzero(mask == id);
        if (pixels.size(0) == 0) continue;
        image.targets.push_back(
            {.box = {pixels.select(1, 1).min().item<double>(),
                     pixels.select(1, 0).min().item<double>(),
                     pixels.select(1, 1).max().item<double>() + 1,
                     pixels.select(1, 0).max().item<double>() + 1},
             .class_index = 0});
      }
      result.images.push_back(std::move(image));
    }
    return result;
  };
  auto model = make_model(true);
  std::uint64_t offset = 0;
  const auto detection = evaluate_detection(
      model, [&](auto stop) { return source(offset++, stop); }, options);
  offset = 0;
  const auto partitioned = evaluate_detection(
      model, [&](auto stop) { return source(offset++, stop, 1); }, options);
  EXPECT_EQ(detection.overall.per_class.front().ap_by_iou,
            partitioned.overall.per_class.front().ap_by_iou);
  EXPECT_EQ(detection.overall.per_class.front().recall_by_iou,
            partitioned.overall.per_class.front().recall_by_iou);
  for (std::size_t i = 0; i < detection.by_source.size(); ++i)
    EXPECT_EQ(detection.by_source[i].per_class.front().ap_by_iou,
              partitioned.by_source[i].per_class.front().ap_by_iou);
  const auto make_session = [&] {
    return std::make_shared<DetectionEvaluationSession>(
        DetectionEvaluationIdentity{
            .evaluation_id = "penn-fudan.box",
            .dataset_digest = {.hex = kManifestSha},
            .model_digest = {.hex = kWeightsSha},
            .config_digest = {.hex = config_sha()},
            .build_fingerprint = build_fingerprint_json()},
        options);
  };
  const auto resumed = reopen<DetectionEvaluationSession>(
      out / "box.state", [&] { return make_model(true); }, make_session,
      source);
  const auto encoded = report(detection, "box", cuda ? "cuda" : "cpu");
  EXPECT_EQ(encoded, report(resumed, "box", cuda ? "cuda" : "cpu"));
  write(out / "box-report.json", std::as_bytes(std::span(encoded)));
  const auto& c = detection.overall.per_class.front();
  results["detection"] = {{"ap50", *c.ap_by_iou[0]},
                          {"ap75", *c.ap_by_iou[1]},
                          {"map", *c.average_precision},
                          {"gt", c.ground_truth},
                          {"predictions", c.predictions}};
  results["detection"]["by_source"] = Json::array();
  for (const auto& summary : detection.by_source) {
    const auto& cls = summary.per_class.front();
    results["detection"]["by_source"].push_back(
        {{"ap50", *cls.ap_by_iou[0]},
         {"ap75", *cls.ap_by_iou[1]},
         {"map", *cls.average_precision},
         {"gt", cls.ground_truth},
         {"predictions", cls.predictions}});
  }
  EXPECT_GE(*c.ap_by_iou[0], .10);
  write_json(out / "predictions.json", raw);
  write_json(out / "results.json", results);
  Json resources{
      {"device", device.str()},
      {"data_manifest_sha256", kManifestSha},
      {"weights_sha256", kWeightsSha},
      {"inference_calls", calls},
      {"peak_rss_bytes", peak_rss()},
      {"elapsed_seconds",
       std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
           .count()}};
  EXPECT_LT(peak_rss(), 6ULL << 30U);
#if defined(JYPPX_LIUDING_SPATIAL_CUDA_METRICS)
  if (cuda) {
    const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
    resources["cuda_peak_allocated_bytes"] = stats.allocated_bytes[0].peak;
    resources["cuda_peak_reserved_bytes"] = stats.reserved_bytes[0].peak;
    EXPECT_LT(stats.allocated_bytes[0].peak, 3LL << 30U);
  }
#endif
  write_json(out / "resources.json", resources);
  std::cout << "REAL_SPATIAL_RESULT " << results.dump() << '\n';
}

TEST(RealSpatialTask, TrainCUDA) {
  if (!std::getenv("JYPPX_LIUDING_TRAIN_SPATIAL_TASK"))
    GTEST_SKIP() << "Explicit external training run required";
  ASSERT_TRUE(torch::cuda::is_available());
  setup();
  const auto start = std::chrono::steady_clock::now();
  const auto out = output();
  auto data = load("train");
  auto holder = std::make_shared<TinyUNet>();
  auto model =
      std::make_shared<nn::ModelAdapter>(nn::adapt_tensor_module(holder));
  std::vector<optim::OptimizerParameterBinding> bindings;
  for (const auto& p : model->state().named_parameters())
    bindings.push_back({.parameter_id = p.key(), .parameter = p.value()});
  auto optimizer = std::make_shared<optim::AdamOptimizerAdapter>(
      "optimizer.spatial", std::vector<optim::AdamParameterGroup>{
                               {.group_id = "main",
                                .parameters = std::move(bindings),
                                .options = {.learning_rate = .001}}});
  auto pipeline = std::make_shared<data::DataPipeline>(
      std::make_shared<data::samplers::RandomSampler>(120, 20261028),
      data::DataPipelineOptions{
          .pipeline_id = "data.penn-fudan.train",
          .dataset_version = kManifestSha,
          .dataset_size = 120,
          .schema = data::BatchSchema::any("penn-fudan.batch", 1),
          .batch_factory_id = "tensor-slice/v1",
          .batch_size = 8,
          .worker_seed = 20261028},
      [images = data.images, labels = data.instances.gt(0).to(torch::kInt64)](
          std::span<const std::size_t> indices, const data::DataWorkerContext&,
          std::stop_token) {
        std::vector<std::int64_t> rows(indices.begin(), indices.end());
        auto selected = torch::tensor(rows, torch::kInt64);
        return c10::ivalue::Tuple::create({images.index_select(0, selected),
                                           labels.index_select(0, selected)});
      });
  Trainer trainer({.trainer_id = "trainer.spatial",
                   .run_id = "penn-fudan.tiny-unet.final80",
                   .trace_id = "trace.spatial",
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
           [](const data::BatchEnvelope& batch) {
             return nn::Invocation{
                 .positional = {
                     batch.value.toTuple()->elements()[0].toTensor().to(
                         torch::kCUDA)}};
           },
       .training_target_mapper =
           [](const data::BatchEnvelope& batch) -> std::optional<c10::IValue> {
         return batch.value.toTuple()->elements()[1].toTensor().to(
             torch::kCUDA);
       },
       .training_step =
           [](const StepInput& input) {
             try {
               const auto logits =
                   input.model.invoke(input.invocation).value.toTensor();
               const auto target = input.target->toTensor();
               const auto p = logits.softmax(1).select(1, 1);
               const auto y = target.to(torch::kFloat32);
               // The equivalent gather/mean avoids nondeterministic CUDA NLL2d.
               auto loss = -logits.log_softmax(1)
                                .gather(1, target.unsqueeze(1))
                                .mean() +
                           1 -
                           (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1);
               return StepOutput{.loss = loss,
                                 .sample_count = input.batch.sample_ids.size()};
             } catch (const std::exception& error) {
               std::cerr << "Real spatial training step: " << error.what()
                         << '\n';
               throw;
             }
           },
       .strategy = std::make_shared<SingleDeviceStrategy>(
           SingleDeviceStrategyOptions{.strategy_id = "strategy.spatial",
                                       .device = torch::Device("cuda:0")}),
       .model_checkpoint =
           std::make_shared<nn::ModelCheckpointParticipant>(*model),
       .rng_checkpoint = std::make_shared<core::RngCheckpointParticipant>()});
  const auto result = trainer.fit();
  EXPECT_EQ(result.completed_epochs, 80);
  EXPECT_EQ(result.global_step, 1200);
  save_weights(*holder, out / "trained.safetensors");
  EXPECT_EQ(sha256_hex(read(out / "trained.safetensors")), kWeightsSha);
  trainer.close();
  const auto elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  EXPECT_LT(elapsed, 900.);
  Json summary{
      {"epochs", result.completed_epochs},
      {"updates", result.global_step},
      {"final_loss", result.training_metrics.at("training.loss")},
      {"elapsed_seconds", elapsed},
      {"peak_rss_bytes", peak_rss()},
      {"weights_sha256", sha256_hex(read(out / "trained.safetensors"))},
      {"manifest_sha256", kManifestSha}};
  EXPECT_LT(peak_rss(), 6ULL << 30U);
#if defined(JYPPX_LIUDING_SPATIAL_CUDA_METRICS)
  const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
  summary["cuda_peak_allocated_bytes"] = stats.allocated_bytes[0].peak;
  summary["cuda_peak_reserved_bytes"] = stats.reserved_bytes[0].peak;
  EXPECT_LT(stats.allocated_bytes[0].peak, 3LL << 30U);
#endif
  write_json(out / "training.json", summary);
  std::cout << "REAL_SPATIAL_TRAINING " << summary.dump() << '\n';
}
TEST(RealSpatialTask, HeldOutInferenceAndDurableEvaluationCPU) {
  if (!std::getenv("JYPPX_LIUDING_SPATIAL_TASK_DIR"))
    GTEST_SKIP() << "Pinned external real task data and weights required";
  task(false);
}
TEST(RealSpatialTask, HeldOutInferenceAndDurableEvaluationCUDA) {
  if (!std::getenv("JYPPX_LIUDING_SPATIAL_TASK_DIR"))
    GTEST_SKIP() << "Pinned external real task data and weights required";
  ASSERT_TRUE(torch::cuda::is_available());
  task(true);
}
}  // namespace
