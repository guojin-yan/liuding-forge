// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "artifact_io.hpp"
#include "extended_io.hpp"

#include <jyppx/liuding/deployment/onnx_runtime/typed_session.hpp>
#include <jyppx/liuding/vision/tasks/faster_rcnn.hpp>
#include <jyppx/liuding/vision/tasks/original_v5.hpp>
#include <jyppx/liuding/vision/tasks/workflow/export.hpp>

#include <chrono>
#include <nlohmann/json.hpp>
namespace jyppx::liuding::vision::tasks {
namespace {
using J = nlohmann::json;
using I = std::int64_t;
namespace image = data::image;
}  // namespace
std::string predict_vision_bundle(const std::filesystem::path& bundle,
                                  std::span<const image::VisionSample> samples,
                                  std::stop_token stop) {
#if defined(JYPPX_VISION_WITH_ONNX_RUNTIME)
  const auto bytes = detail::read_artifact(bundle / "manifest.json", 4U << 20U);
  auto manifest = J::parse(bytes.begin(), bytes.end());
  if(manifest.at("schema")=="vision.onnx_bundle/v3")return detail::predict_extended_bundle(bundle,samples,stop);
  if ((manifest.at("schema") != "vision.onnx_bundle/v1" &&
       manifest.at("schema") != "vision.onnx_bundle/v2") ||
      manifest.at("model_file") != "model.onnx" ||
      samples.size() != manifest.at("fixed_batch").get<std::size_t>())
    throw CompatibilityError("Offline bundle schema or fixed batch mismatch");
  auto recipe = parse_recipe(manifest.at("recipe").dump(), bundle);
  recipe.device = Device(torch::kCPU);
  const bool v2 = manifest.at("schema") == "vision.onnx_bundle/v2";
  const auto batch = manifest.at("fixed_batch").get<I>(),
             width = manifest.at("width").get<I>(),
             height = manifest.at("height").get<I>();
  const auto decoder = manifest.at("decoder").get<std::string>();
  const bool rcnn = recipe.model.architecture == "faster_rcnn_resnet50_fpn";
  if (batch < 1 || batch > 64 || width != recipe.data.letterbox.canvas.width ||
      height != recipe.data.letterbox.canvas.height ||
      manifest.at("class_count").get<I>() != recipe.model.num_classes ||
      manifest.at("end_to_end").get<bool>() !=
          (recipe.model.architecture.starts_with("yolo26") &&
           recipe.model.task != TaskKind::classify) ||
      (v2 && manifest.at("io_profile") != "cpu.dense_typed/v1") ||
      (rcnn ? decoder !=
                  (v2 ? "rcnn.fixed_proposals/v2" : "rcnn.fixed_proposals/v1")
            : decoder != "vision.raw/v1") ||
      (rcnn && recipe.model.task != TaskKind::detect) ||
      manifest.at("geometry") != (recipe.model.task == TaskKind::classify
                                      ? "deterministic_center_crop"
                                      : "letterbox_with_inverse_trace") ||
      manifest.at("pixel_contract") !=
          J{{"color", "RGB"},
            {"layout", "NCHW"},
            {"dtype", "float32"},
            {"range", "unit_then_recipe_normalization"}})
    throw CompatibilityError(
        "Offline bundle recipe/preprocessing/decoder contract mismatch");
  // Decoder construction is independent of model availability and does not
  // allocate weights.
  YoloTaskOptions task;
  task.task = recipe.model.task;
  task.detection.num_classes = recipe.model.num_classes;
  task.generation26 = recipe.model.architecture.starts_with("yolo26");
  task.end_to_end = manifest.at("end_to_end").get<bool>();
  task.detection.reg_max = task.generation26 ? 1 : 16;
  task.mask_channels = recipe.model.mask_channels;
  task.keypoints = recipe.model.keypoints;
  task.keypoint_dimensions = recipe.model.keypoint_dimensions;
  task.detection.strides =
      recipe.model.variant.find("p2") != std::string::npos
          ? std::vector<I>{4, 8, 16, 32}
      : recipe.model.variant.find("p6") != std::string::npos
          ? std::vector<I>{8, 16, 32, 64}
          : std::vector<I>{8, 16, 32};
  const auto started = std::chrono::steady_clock::now();
  std::vector<image::ImageBatchKey> keys;
  for (const auto& s : samples) keys.push_back({.id = s.image_path});
  auto data = yolo_evaluation_options(recipe.data);
  data.num_classes = recipe.model.num_classes;
  data.allow_unannotated_negatives = true;
  data.augmentation.dataset_id="onnx-bundle:"+manifest.at("sha256").get<std::string>();
  auto prepared = prepare_yolo_batch(samples, keys, data, stop);
  auto input = prepared.images.contiguous();
  const auto preprocessing_done = std::chrono::steady_clock::now();
  auto model = detail::read_artifact(bundle / "model.onnx", 4ULL << 30U);
  onnx_runtime::TypedSessionOptions settings;
  settings.max_model_bytes = 4ULL << 30U;
  settings.max_input_bytes = 1ULL << 30U;
  settings.max_output_bytes = 1ULL << 30U;
  onnx_runtime::TypedSession session(
      model, {.hex = manifest.at("sha256").get<std::string>()}, settings);
  const auto input_names =
      manifest.at("input_names").get<std::vector<std::string>>();
  const auto output_names =
      manifest.at("output_names").get<std::vector<std::string>>();
  const auto input_shapes =
      manifest.at("input_shapes").get<std::vector<std::vector<I>>>();
  const auto output_shapes =
      manifest.at("output_shapes").get<std::vector<std::vector<I>>>();
  if (input_names != std::vector<std::string>{"images"} ||
      input_shapes != std::vector<std::vector<I>>{{batch, 3, height, width}} ||
      session.inputs().size() != 1 ||
      session.outputs().size() != output_names.size() ||
      output_shapes.size() != output_names.size() ||
      session.inputs()[0].name != "images" ||
      session.inputs()[0].shape != input_shapes[0] ||
      session.inputs()[0].type != onnx_runtime::TensorElementType::float32)
    throw CompatibilityError("Offline bundle ONNX input signature mismatch");
  if (rcnn &&
      output_names != std::vector<std::string>{"proposals", "class_logits",
                                               "box_deltas", "valid_proposals"})
    throw CompatibilityError("Offline RCNN bundle output order mismatch");
  for (std::size_t i = 0; i < output_names.size(); ++i) {
    const auto& signature = session.outputs()[i];
    const auto type = rcnn && v2 && i == 3
                          ? onnx_runtime::TensorElementType::int64
                          : onnx_runtime::TensorElementType::float32;
    if (signature.name != output_names[i] ||
        signature.shape != output_shapes[i] || signature.type != type ||
        !signature.leading_symbol.empty())
      throw CompatibilityError("Offline bundle ONNX output signature mismatch");
  }
  const std::vector<I> shape(input.sizes().begin(), input.sizes().end());
  const std::array<onnx_runtime::TypedTensorInput, 1> inputs{
      {{"images",
        std::span<const float>(input.const_data_ptr<float>(),
                               static_cast<std::size_t>(input.numel())),
        shape}}};
  const auto model_started = std::chrono::steady_clock::now();
  auto outputs = session.run(inputs, stop);
  const auto model_done = std::chrono::steady_clock::now();
  std::vector<Tensor> raw;
  for (auto& output : outputs)
    std::visit(
        [&](auto& data) {
          using T = typename std::decay_t<decltype(data)>::value_type;
          const auto type = std::is_same_v<T, float>          ? torch::kFloat32
                            : std::is_same_v<T, std::int32_t> ? torch::kInt32
                                                              : torch::kInt64;
          raw.push_back(
              data.empty()
                  ? torch::empty(output.shape, type)
                  : torch::from_blob(data.data(), output.shape, type).clone());
        },
        output.values);
  std::vector<ImageResult> results;
  if (manifest.at("decoder") == "rcnn.fixed_proposals/v1" ||
      manifest.at("decoder") == "rcnn.fixed_proposals/v2") {
    if (raw.size() != 4)
      throw CompatibilityError("RCNN bundle output schema mismatch");
    const auto valid = raw[3] > 0;
    auto rois = raw[0].index({valid}), logits = raw[1].index({valid}),
         deltas = raw[2].index({valid});
    auto sizes =
        torch::tensor(std::vector<float>{static_cast<float>(input.size(2)),
                                         static_cast<float>(input.size(3))})
            .view({1, 2})
            .expand({input.size(0), 2});
    std::vector<Tensor> reconstructed{torch::zeros({input.size(0), 1}),
                                      Tensor{},
                                      Tensor{},
                                      rois,
                                      logits,
                                      deltas,
                                      Tensor{},
                                      Tensor{},
                                      sizes};
    results = decode_faster_rcnn_results(reconstructed, recipe.prediction);
  } else if (recipe.model.task == TaskKind::classify) {
    results = decode_classification_results(
        raw.at(0), {input.size(2), input.size(3)}, recipe.prediction);
  } else if (recipe.model.architecture.starts_with("yolov5")) {
    OriginalV5TaskOptions o;
    o.num_classes = recipe.model.num_classes;
    const bool p6 = recipe.model.variant == "p6";
    if (p6)
      o.anchors = {{19, 27, 44, 40, 38, 94},
                   {96, 68, 86, 152, 180, 137},
                   {140, 301, 303, 264, 238, 542},
                   {436, 615, 739, 380, 925, 792}};
    if (p6) {
      o.strides = {8, 16, 32, 64};
      o.objectness_balance = {4, 1, .25, .06};
    }
    results = decode_detection_results(
        decode_original_v5(raw.at(0), {input.size(2), input.size(3)}, o),
        {input.size(2), input.size(3)}, recipe.prediction);
  } else
    results = decode_yolo_results(raw, {input.size(2), input.size(3)}, task,
                                  recipe.prediction);
  J json = J::array();
  for (std::size_t i = 0; i < results.size(); ++i) {
    results[i].source_id = samples[i].image_path;
    if (recipe.model.task != TaskKind::classify)
      results[i] = restore_result(results[i], *prepared.items[i].geometry,
                                  recipe.prediction.max_mask_elements);
    else
      results[i].extent = {samples[i].width, samples[i].height};
    json.push_back(J::parse(result_json(results[i])));
  }
  const auto finished = std::chrono::steady_clock::now();
  auto ms = [](auto d) {
    return std::chrono::duration<double, std::milli>(d).count();
  };
  return J{{"schema", "vision.bundle_prediction/v1"},
           {"sha256", manifest.at("sha256")},
           {"results", json},
           {"timing_ms",
            {{"preprocess", ms(preprocessing_done - started)},
             {"session_load", ms(model_started - preprocessing_done)},
             {"model", ms(model_done - model_started)},
             {"postprocess", ms(finished - model_done)}}}}
      .dump(2);
#else
  static_cast<void>(bundle);
  static_cast<void>(samples);
  static_cast<void>(stop);
  throw CompatibilityError("Offline vision execution requires onnx_runtime");
#endif
}
}  // namespace jyppx::liuding::vision::tasks
