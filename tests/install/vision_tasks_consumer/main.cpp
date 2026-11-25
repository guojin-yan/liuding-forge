// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/core/runtime.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/foundation/version.hpp>
#include <jyppx/liuding/vision/tasks/training/musgd.hpp>
#include <jyppx/liuding/vision/tasks/training/weight_conversion.hpp>
#include <jyppx/liuding/vision/tasks/workflow/export.hpp>
#include <jyppx/liuding/vision/tasks/workflow/output.hpp>
#include <jyppx/liuding/vision/tasks/workflow/preflight.hpp>
#include <jyppx/liuding/vision/tasks/workflow/search.hpp>
#include <jyppx/liuding/vision/tasks/workflow/suite.hpp>
#ifdef VISION_CONSUMER_MODELS_YOLO
#include <jyppx/liuding/vision/models/yolo/model.hpp>
#include <jyppx/liuding/vision/models/yolo/yolov5.hpp>
#endif
#ifdef VISION_CONSUMER_MODELS_RCNN
#include <jyppx/liuding/vision/models/rcnn/faster_rcnn.hpp>
#endif
#ifdef VISION_CONSUMER_ONNX_RUNTIME
#include <jyppx/liuding/deployment/onnx_runtime/typed_session.hpp>
#endif
#include <jyppx/liuding/vision/tasks/data/context.hpp>
#include <jyppx/liuding/vision/tasks/tracking/workflow.hpp>
#ifdef VISION_CONSUMER_MODELS_VISION
#include <jyppx/liuding/vision/models/extended/pyramid.hpp>
#include <jyppx/liuding/vision/models/extended/rtdetr.hpp>
#endif
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
int main() try {
  namespace tasks = jyppx::liuding::vision::tasks;
  jyppx::liuding::initialize_runtime();
  {
    auto parameter = torch::ones({2, 2}, torch::requires_grad());
    parameter.mutable_grad() = torch::ones_like(parameter);
    tasks::MuSgdOptimizer optimizer(
        "installed.musgd",
        {{.group_id = "matrix",
          .parameters = {{"weight", parameter}},
          .options = {.learning_rate = .01, .momentum = .9}}},
        {.math = tasks::MuSgdMath::bf16_batched});
    optimizer.step();
    if (optimizer.completed_steps() != 1 ||
        torch::equal(parameter, torch::ones_like(parameter)) ||
        !torch::isfinite(parameter).all().item<bool>())
      throw std::runtime_error("installed BF16 MuSGD update failed");
  }
  const auto catalog = tasks::vision_catalog();
#ifdef VISION_CONSUMER_MODELS_VISION
  {
    namespace ex = jyppx::liuding::models::extended;
    ex::RtDetr model(
        ex::RtDetrOptions{.backbone = ex::RtDetrBackbone::resnet18_vd,
                          .num_classes = 3,
                          .profile = ex::RtDetrProfile::r18_coco72,
                          .max_encoder_elements = 67108864});
    if (!model->named_parameters().contains(
            "model.decoder2.self_attention.packed.in_proj_weight") ||
        !model->named_parameters().contains(
            "model.backbone.stem.0.norm.weight"))
      throw std::runtime_error("installed R18 COCO profile topology mismatch");
  }
#endif
#if defined(VISION_CONSUMER_MODELS) || defined(VISION_CONSUMER_MODELS_YOLO) || \
    defined(VISION_CONSUMER_MODELS_RCNN) ||                                    \
    defined(VISION_CONSUMER_MODELS_VISION)
  if (catalog.empty())
    throw std::runtime_error("installed vision model catalog is empty");
#else
  if (!catalog.empty())
    throw std::runtime_error("disabled model packs leaked into the catalog");
#endif
  auto recipe = tasks::load_recipe(
      std::filesystem::path(VISION_INSTALLED_RECIPES) / "yolov8-detect.json");
  auto roundtrip = tasks::parse_recipe(tasks::recipe_json(recipe),
                                       std::filesystem::current_path());
  if (tasks::recipe_identity(recipe) != tasks::recipe_identity(roundtrip))
    throw std::runtime_error("installed recipe roundtrip mismatch");
  {
    auto jpeg = recipe;
    jpeg.jpeg_backend =
        jyppx::liuding::data::image::JpegDecodeBackend::libjpeg_turbo;
    if (jyppx::liuding::data::image::jpeg_decode_backend_available(
            jpeg.jpeg_backend)) {
      const auto restored = tasks::parse_recipe(
          tasks::recipe_json(jpeg), std::filesystem::current_path());
      if (restored.jpeg_backend != jpeg.jpeg_backend ||
          tasks::recipe_identity(jpeg) != tasks::recipe_identity(restored))
        throw std::runtime_error("installed JPEG backend recipe mismatch");
    }
  }
  {
    auto deferred = recipe;
    deferred.data.numeric_on_consumer = true;
    deferred.non_blocking_transfer = true;
    const auto restored = tasks::parse_recipe(tasks::recipe_json(deferred),
                                              std::filesystem::current_path());
    if (!restored.data.numeric_on_consumer || !restored.non_blocking_transfer ||
        tasks::recipe_identity(restored) != tasks::recipe_identity(deferred))
      throw std::runtime_error("installed deferred numeric recipe mismatch");
  }
  recipe.training.limits.max_input_bytes = 600ULL << 20;
  recipe.optimizer.weight_decay_overrides = {{"model.example.weight", .0001}};
  roundtrip = tasks::parse_recipe(tasks::recipe_json(recipe),
                                  std::filesystem::current_path());
  if (roundtrip.training.limits.max_input_bytes != (600ULL << 20) ||
      roundtrip.optimizer.weight_decay_overrides.size() != 1 ||
      tasks::recipe_identity(recipe) != tasks::recipe_identity(roundtrip))
    throw std::runtime_error("installed annotation limits roundtrip mismatch");
  recipe.optimizer.weight_decay_overrides.clear();
  {
    auto rtdetr = recipe;
    rtdetr.model.architecture = "rtdetr_resnet18_vd_coco72";
    rtdetr.data = {};
    rtdetr.data.rtdetr.enabled = true;
    rtdetr.drop_last_training_batch = true;
    rtdetr.evaluation.clip_boxes = rtdetr.prediction.clip_boxes = false;
    rtdetr.evaluation.confidence = 0;
    rtdetr.evaluation.multi_label = true;
    rtdetr.rtdetr_inference_validation = true;
    rtdetr.save_best_checkpoint = false;
    rtdetr.ema_update_rule = tasks::ModelEmaUpdateRule::multiply_add_floating_only;
    rtdetr.rtdetr_training_sizes = {480, 512, 544, 576, 608, 640, 640,
                                    640, 672, 704, 736, 768, 800};
    const auto restored = tasks::parse_recipe(tasks::recipe_json(rtdetr),
                                              std::filesystem::current_path());
    if (!restored.data.rtdetr.enabled || !restored.drop_last_training_batch ||
        !restored.rtdetr_inference_validation ||
        restored.save_best_checkpoint ||
        restored.ema_update_rule != rtdetr.ema_update_rule ||
        restored.evaluation.clip_boxes || restored.prediction.clip_boxes ||
        restored.rtdetr_training_sizes != rtdetr.rtdetr_training_sizes ||
        tasks::recipe_identity(restored) != tasks::recipe_identity(rtdetr))
      throw std::runtime_error(
          "installed RT-DETR data/multiscale profile mismatch");
  }
  {
    auto accumulation_recipe = recipe;
    accumulation_recipe.accumulation_tail =
        jyppx::liuding::training::AccumulationTailPolicy::carry;
    accumulation_recipe.accumulation_reduction =
        jyppx::liuding::training::AccumulationReduction::sum;
    const auto restored =
        tasks::parse_recipe(tasks::recipe_json(accumulation_recipe),
                            std::filesystem::current_path());
    if (restored.accumulation_tail != accumulation_recipe.accumulation_tail ||
        restored.accumulation_reduction !=
            accumulation_recipe.accumulation_reduction ||
        tasks::recipe_identity(restored) !=
            tasks::recipe_identity(accumulation_recipe))
      throw std::runtime_error(
          "installed gradient accumulation roundtrip mismatch");
  }
  auto criterion_recipe = recipe;
  criterion_recipe.model.architecture = "yolo26n";
  criterion_recipe.yolo26_loss =
      tasks::Yolo26LossRecipe{.box_gain = 5.62767,
                              .class_gain = .56099,
                              .dfl_gain = 9.03871,
                              .one_to_one_topk = 8,
                              .fixed_branch_gains = true};
  const auto criterion_roundtrip = tasks::parse_recipe(
      tasks::recipe_json(criterion_recipe), std::filesystem::current_path());
  if (!criterion_roundtrip.yolo26_loss ||
      criterion_roundtrip.yolo26_loss->one_to_one_topk != 8 ||
      tasks::recipe_identity(criterion_recipe) !=
          tasks::recipe_identity(criterion_roundtrip))
    throw std::runtime_error("installed YOLO26 criterion roundtrip mismatch");
  tasks::TrainRequest measured;
  measured.phase_observer = [](std::string_view) {};
  measured.event_observer = [](const jyppx::liuding::training::CallbackEvent&) {
  };
  if (std::string_view(jyppx::liuding::version()) !=
      jyppx::liuding::build_info().framework_version)
    throw std::runtime_error("installed library and header version mismatch");
#ifdef VISION_CONSUMER_ONNX_RUNTIME
  try {
    jyppx::liuding::onnx_runtime::TypedSession session(
        {}, {.hex = std::string(64, '0')});
    throw std::runtime_error("empty typed session model was accepted");
  } catch (const jyppx::liuding::ResourceExhausted&) {
  }
#endif
#if defined(VISION_CONSUMER_MODELS_YOLO) && \
    defined(VISION_CONSUMER_ONNX_EXPORT) && \
    defined(VISION_CONSUMER_ONNX_RUNTIME)
  // One independent path check for installed fusion/export/runtime contracts;
  // it does not qualify upstream architecture, task accuracy or other devices.
  recipe.model.num_classes = 2;
  recipe.data.num_classes = 2;
  recipe.data.letterbox.canvas = {64, 64};
  recipe.epochs = 1;
  recipe.data.augmentation.multi_source.total_epochs = 1;
  recipe.data.augmentation.multi_source.close_epochs = 0;
  torch::manual_seed(9);
  auto binding = tasks::bind_task(recipe);
  binding.model->state().eval();
  const auto directory =
      std::filesystem::current_path() /
      ("bundle-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  tasks::VisionExportOptions profile;
  profile.fuse = true;
  static_cast<void>(tasks::export_vision(binding, recipe, directory, profile));
  std::ifstream file(directory / "model.onnx",
                     std::ios::binary | std::ios::ate);
  auto count = file.tellg();
  if (count < 1 || count > (64 << 20))
    throw std::runtime_error("installed export model size mismatch");
  std::vector<std::byte> bytes(static_cast<std::size_t>(count));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  if (!file) throw std::runtime_error("installed export model read failed");
  jyppx::liuding::onnx_runtime::TypedSession session(
      bytes, {.hex = jyppx::liuding::sha256_hex(bytes)});
  torch::NoGradGuard guard;
  auto input = torch::rand({1, 3, 64, 64});
  const std::vector<std::int64_t> shape{1, 3, 64, 64};
  const std::array<jyppx::liuding::onnx_runtime::TypedTensorInput, 1> inputs{
      {{"images",
        std::span<const float>(input.const_data_ptr<float>(), input.numel()),
        shape}}};
  auto actual = session.run(inputs);
  auto expected = tasks::raw_tensors(
      binding.model->invoke(jyppx::liuding::nn::Invocation{{input}, {}}).value);
  if (actual.size() != expected.size())
    throw std::runtime_error("installed ONNX output count mismatch");
  double maximum = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    auto& data = std::get<std::vector<float>>(actual[i].values);
    auto tensor =
        torch::from_blob(data.data(), actual[i].shape, torch::kFloat32);
    if (tensor.sizes() != expected[i].sizes() ||
        !torch::allclose(tensor, expected[i], 1e-4, 1e-3))
      throw std::runtime_error("installed fused ONNX/native numeric mismatch");
    maximum =
        (std::max)(maximum, (tensor - expected[i]).abs().max().item<double>());
  }
  std::cout << "installed fused ONNX/native maximum absolute error: " << maximum
            << '\n';
#endif
#if defined(VISION_CONSUMER_MODELS_RCNN) && \
    defined(VISION_CONSUMER_ONNX_EXPORT) && \
    defined(VISION_CONSUMER_ONNX_RUNTIME)
  {
    auto rcnn_recipe = tasks::load_recipe(
        std::filesystem::path(VISION_INSTALLED_RECIPES) / "faster-rcnn.json");
    rcnn_recipe.model.num_classes = 2;
    rcnn_recipe.data.num_classes = 2;
    rcnn_recipe.data.letterbox.canvas = {64, 64};
    rcnn_recipe.epochs = 1;
    rcnn_recipe.data.augmentation.multi_source.total_epochs = 1;
    rcnn_recipe.data.augmentation.multi_source.close_epochs = 0;
    torch::manual_seed(19);
    auto rcnn_binding = tasks::bind_task(rcnn_recipe);
    rcnn_binding.model->state().eval();
    const auto directory =
        std::filesystem::current_path() /
        ("rcnn-bundle-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    tasks::VisionExportOptions rcnn_profile;
    rcnn_profile.rcnn_proposals = 16;
    static_cast<void>(tasks::export_vision(rcnn_binding, rcnn_recipe, directory,
                                           rcnn_profile));
    std::ifstream file(directory / "model.onnx",
                       std::ios::binary | std::ios::ate);
    const auto size = file.tellg();
    if (size < 1 || size > (256 << 20))
      throw std::runtime_error("installed RCNN export model size mismatch");
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    if (!file) throw std::runtime_error("installed RCNN export read failed");
    jyppx::liuding::onnx_runtime::TypedSession session(
        bytes, {.hex = jyppx::liuding::sha256_hex(bytes)});
    torch::NoGradGuard no_grad;
    auto input = torch::rand({1, 3, 64, 64});
    const std::vector<std::int64_t> shape{1, 3, 64, 64};
    const std::array<jyppx::liuding::onnx_runtime::TypedTensorInput, 1> inputs{
        {{"images",
          std::span<const float>(input.const_data_ptr<float>(), input.numel()),
          shape}}};
    auto actual = session.run(inputs);
    if (actual.size() != 4)
      throw std::runtime_error("installed RCNN ONNX output count mismatch");
    auto detector =
        std::dynamic_pointer_cast<jyppx::liuding::models::rcnn::FasterRcnnImpl>(
            rcnn_binding.model->state().shared_module());
    if (!detector)
      throw std::runtime_error("installed RCNN binding type mismatch");
    auto native = detector->forward(input);
    const auto count =
        (std::min)(rcnn_profile.rcnn_proposals, native.proposals.size(0));
    auto& markers = std::get<std::vector<std::int64_t>>(actual[3].values);
    if (markers.size() != 16)
      throw std::runtime_error("installed RCNN valid-row shape mismatch");
    for (std::size_t i = 0; i < markers.size(); ++i)
      if (markers[i] != (static_cast<std::int64_t>(i) < count ? 1 : 0))
        throw std::runtime_error("installed RCNN valid-row identity mismatch");
    const std::array<torch::Tensor, 3> expected{
        native.proposals, native.class_logits, native.box_deltas};
    double maximum = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
      auto& data = std::get<std::vector<float>>(actual[i].values);
      auto tensor =
          torch::from_blob(data.data(), actual[i].shape, torch::kFloat32)
              .slice(0, 0, count);
      auto reference = expected[i].slice(0, 0, count);
      const auto error =
          count ? (tensor - reference).abs().max().item<double>() : 0.;
      if (tensor.sizes() != reference.sizes() ||
          !torch::allclose(tensor, reference, 1e-4, 1e-3))
        throw std::runtime_error(
            "installed RCNN ONNX/native numeric mismatch: output " +
            std::to_string(i) + ", max error " + std::to_string(error));
      maximum = (std::max)(maximum, error);
    }
    std::cout << "installed RCNN ONNX/native maximum absolute error: "
              << maximum << '\n';
  }
#endif
  std::cout
      << "installed vision headers/catalog/recipe/preview metadata passed: "
      << catalog.size() << '\n';
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
