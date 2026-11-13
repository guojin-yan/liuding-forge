// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/vision/tasks/evaluation.hpp>

#include <algorithm>
#include <limits>

namespace jyppx::liuding::vision::tasks {
training::DetectionEvaluationResult evaluate_yolo_detection(
    nn::ModelAdapter& model, const data::image::ImageDataset& dataset,
    const YoloEvaluationOptions& options, std::stop_token stop,
    TelemetryDeadline deadline) {
  if (options.batch_size == 0 || options.batch_size > 256 ||
      dataset.size() == 0 || options.data.num_classes <= 0 ||
      options.prediction.detection.num_classes != options.data.num_classes ||
      options.evaluation.metric.num_classes !=
          static_cast<std::size_t>(options.data.num_classes) ||
      !options.evaluation.metric.sources.empty() ||
      (options.evaluation.expected_images &&
       *options.evaluation.expected_images != dataset.size()))
    throw InvalidArgument("Invalid YOLO evaluation split, classes or batching");
  if (dataset.size() > options.evaluation.limits.max_images)
    throw ResourceExhausted("YOLO evaluation image count exceeds metric limit");
  const auto preparation = yolo_evaluation_options(options.data);
  auto scoring = options.evaluation;
  scoring.expected_images = dataset.size();
  const std::array<std::int64_t, 2> canvas{preparation.letterbox.canvas.height,
                                           preparation.letterbox.canvas.width};
  std::vector<data::image::GeometryTrace> geometry;
  std::size_t cursor = 0;
  const auto source = [&](std::stop_token token)
      -> std::optional<training::DetectionEvaluationBatch> {
    if (token.stop_requested()) throw Cancelled("YOLO evaluation cancelled");
    if (cursor == dataset.size()) return std::nullopt;
    const auto count = (std::min)(options.batch_size, dataset.size() - cursor);
    std::vector<data::image::VisionSample> samples;
    std::vector<data::image::ImageBatchKey> keys;
    training::DetectionEvaluationBatch evaluation_batch;
    samples.reserve(count);
    keys.reserve(count);
    evaluation_batch.images.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      auto sample = dataset.get(cursor + i, token);
      if (!sample.annotation || !sample.categories ||
          sample.categories->size() != scoring.metric.num_classes)
        throw InvalidArgument(
            "YOLO AP requires canonical annotations and classes");
      training::metrics::DetectionImage target;
      target.image_id = cursor + i + 1;
      if (sample.annotation->objects.size() >
          scoring.limits.max_targets_per_image)
        throw ResourceExhausted("YOLO AP target count exceeds per-image limit");
      target.targets.reserve(sample.annotation->objects.size());
      for (const auto& object : sample.annotation->objects) {
        if (!object.box || object.category >= scoring.metric.num_classes)
          throw InvalidArgument(
              "YOLO AP requires a valid box for every object");
        const auto& box = *object.box;
        target.targets.push_back(
            {.box = {box.x_min, box.y_min, box.x_max, box.y_max},
             .class_index = object.category,
             .crowd = object.crowd,
             .ignore = object.ignore ||
                       (options.ignore_difficult && object.difficult),
             .area = object.area});
      }
      keys.push_back({.id = sample.image_path});
      samples.push_back(std::move(sample));
      evaluation_batch.images.push_back(std::move(target));
    }
    auto batch = prepare_yolo_batch(samples, keys, preparation, token);
    geometry.clear();
    geometry.reserve(count);
    for (const auto& item : batch.items) {
      if (!item.geometry)
        throw InvalidArgument("YOLO AP needs an original-coordinate trace");
      geometry.push_back(*item.geometry);
    }
    evaluation_batch.invocation.positional = {batch.images.to(options.device)};
    cursor += count;
    return evaluation_batch;
  };
  // The private synchronous adapter sees only the traces of its current source
  // batch. The underlying module owner is shared so the existing evaluator
  // guards and restores every actual model submodule's mode.
  nn::ModelAdapter decoded(
      std::make_shared<nn::ModelStateView>(model.state().shared_module()),
      [&](const nn::Invocation& invocation) -> c10::IValue {
        const auto raw = model.invoke(invocation).value.toTensor();
        const auto canvas_predictions =
            postprocess_yolo_detection(raw, canvas, options.prediction);
        if (canvas_predictions.size() != geometry.size())
          throw InvalidArgument(
              "YOLO AP prediction count differs from source batch");
        c10::impl::GenericList results(c10::AnyType::get());
        std::uint64_t tensor_bytes = 0;
        for (std::size_t i = 0; i < geometry.size(); ++i) {
          const auto predictions =
              restore_detection_coordinates(canvas_predictions[i], geometry[i]);
          const auto count = static_cast<std::int64_t>(predictions.size());
          constexpr std::uint64_t bytes_per_prediction = 6 * sizeof(double);
          if (static_cast<std::uint64_t>(count) >
              (scoring.max_conversion_bytes - tensor_bytes) /
                  bytes_per_prediction)
            throw ResourceExhausted(
                "YOLO AP output tensor staging exceeds limit");
          tensor_bytes +=
              static_cast<std::uint64_t>(count) * bytes_per_prediction;
          auto boxes = torch::empty({count, 4}, torch::kFloat64);
          auto scores = torch::empty({count}, torch::kFloat64);
          auto labels = torch::empty({count}, torch::kInt64);
          auto* box_data = boxes.data_ptr<double>();
          auto* score_data = scores.data_ptr<double>();
          auto* label_data = labels.data_ptr<std::int64_t>();
          for (std::size_t row = 0; row < predictions.size(); ++row) {
            std::copy(predictions[row].box.begin(), predictions[row].box.end(),
                      box_data + row * 4);
            score_data[row] = predictions[row].score;
            label_data[row] =
                static_cast<std::int64_t>(predictions[row].class_index);
          }
          results.push_back(c10::ivalue::Tuple::create(
              {std::move(boxes), std::move(scores), std::move(labels)}));
        }
        return results;
      },
      model.input_schema(),
      ValueSchema::list(
          ValueSchema::tuple({ValueSchema::tensor(), ValueSchema::tensor(),
                              ValueSchema::tensor()})));
  return training::evaluate_detection(decoded, source, scoring, stop, deadline);
}

}  // namespace jyppx::liuding::vision::tasks
