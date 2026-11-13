// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "artifact_io.hpp"

#include <jyppx/liuding/data_image/image_encoder.hpp>
#include <jyppx/liuding/vision/tasks/workflow/output.hpp>

#include <cmath>
#include <nlohmann/json.hpp>
namespace jyppx::liuding::vision::tasks {
namespace detail {
Tensor render_vision_annotations(const Tensor&, const data::image::PixelSpec&,
                                 const data::image::VisionAnnotationRecord&,
                                 std::span<const data::image::VisionCategory>,
                                 std::stop_token);
}
namespace {
namespace im = data::image;
using J = nlohmann::json;
using I = std::int64_t;
void Png(const std::filesystem::path& path, const Tensor& pixels,
         const im::PixelSpec& spec, std::stop_token stop) {
  auto encoded = im::encode_image(pixels, spec, {}, stop);
  detail::publish_artifact(path, encoded.bytes);
}
im::PixelSpec Rgb() {
  im::PixelSpec spec;
  spec.dtype = torch::kUInt8;
  spec.layout = im::ImageLayout::chw;
  spec.color = im::PixelColorModel::rgb;
  spec.range = im::PixelValueRange::unsigned_full_scale;
  return spec;
}
im::VisionAnnotationRecord Annotation(const ImageResult& result) {
  im::VisionAnnotationRecord record;
  record.image_path = result.source_id;
  record.width = result.extent.width;
  record.height = result.extent.height;
  if (result.task == TaskKind::classify) return record;
  for (I n = 0; n < result.boxes.size(0); ++n) {
    im::VisionObject object;
    object.source_id = n + 1;
    object.category = static_cast<std::size_t>(result.classes[n].item<I>());
    object.box = im::ImageBox{
        result.boxes[n][0].item<double>(), result.boxes[n][1].item<double>(),
        result.boxes[n][2].item<double>(), result.boxes[n][3].item<double>()};
    if (result.oriented_corners.defined()) {
      std::array<im::ImagePoint, 4> points;
      for (I k = 0; k < 4; ++k)
        points[k] = {result.oriented_corners[n][k][0].item<double>(),
                     result.oriented_corners[n][k][1].item<double>()};
      object.oriented_box = points;
    }
    if (result.keypoints.defined())
      for (I k = 0; k < result.keypoints.size(1); ++k)
        object.keypoints.push_back(
            {{result.keypoints[n][k][0].item<double>(),
              result.keypoints[n][k][1].item<double>()},
             static_cast<std::uint8_t>(
                 result.keypoints[n][k][2].item<double>() >= .5 ? 2 : 0)});
    if (result.masks.defined()) {
      auto pixels = result.masks[n]
                        .to(torch::kCPU, torch::kUInt8)
                        .transpose(0, 1)
                        .contiguous();
      const auto* data = pixels.const_data_ptr<std::uint8_t>();
      im::ImageMaskRle rle;
      rle.width = record.width;
      rle.height = record.height;
      bool current = false;
      std::uint64_t count = 0;
      for (I k = 0; k < pixels.numel(); ++k) {
        const bool next = data[k] != 0;
        if (next != current) {
          rle.counts.push_back(count);
          count = 0;
          current = next;
        }
        ++count;
      }
      rle.counts.push_back(count);
      object.rle = std::move(rle);
    }
    record.objects.push_back(std::move(object));
  }
  return record;
}
}  // namespace
std::string coco_annotations(const im::AnnotationIndex& index,
                             std::size_t max_bytes) {
  if (!max_bytes || max_bytes > (1ULL << 30U))
    throw InvalidArgument("Invalid COCO output bound");
  J categories = J::array(), images = J::array(), annotations = J::array();
  std::uint64_t id = 0, image_id = 0;
  for (std::size_t c = 0; c < index.categories.size(); ++c) {
    const auto& category = index.categories[c];
    J value = {{"id", c}, {"name", category.name}};
    if (!category.keypoint_names.empty())
      value["keypoints"] = category.keypoint_names;
    if (!category.skeleton.empty()) {
      value["skeleton"] = J::array();
      for (const auto& edge : category.skeleton)
        value["skeleton"].push_back({edge[0] + 1, edge[1] + 1});
    }
    categories.push_back(std::move(value));
  }
  for (const auto& record : index.images) {
    ++image_id;
    images.push_back({{"id", image_id},
                      {"file_name", record.image_path},
                      {"width", record.width},
                      {"height", record.height}});
    for (const auto& object : record.objects) {
      if (object.category >= index.categories.size())
        throw InvalidArgument("COCO output category out of range");
      J value = {{"id", ++id},
                 {"image_id", image_id},
                 {"category_id", object.category},
                 {"iscrowd", object.crowd ? 1 : 0},
                 {"ignore", object.ignore || object.difficult}};
      if (object.box) {
        const auto b = *object.box;
        value["bbox"] = {b.x_min, b.y_min, b.x_max - b.x_min,
                         b.y_max - b.y_min};
        value["area"] =
            object.area.value_or((b.x_max - b.x_min) * (b.y_max - b.y_min));
      }
      if (object.rle)
        value["segmentation"] = {
            {"size", {object.rle->height, object.rle->width}},
            {"counts", object.rle->counts}};
      else if (!object.polygons.empty()) {
        value["segmentation"] = J::array();
        for (const auto& ring : object.polygons) {
          J points = J::array();
          for (auto p : ring) {
            points.push_back(p.x);
            points.push_back(p.y);
          }
          value["segmentation"].push_back(points);
        }
      }
      if (object.oriented_box) {
        J points = J::array();
        for (auto p : *object.oriented_box) {
          points.push_back(p.x);
          points.push_back(p.y);
        }
        value["oriented_bbox"] = points;
        if (!value.contains("segmentation"))
          value["segmentation"] = J::array({points});
      }
      if (!object.keypoints.empty()) {
        value["keypoints"] = J::array();
        std::size_t visible = 0;
        for (auto k : object.keypoints) {
          value["keypoints"].push_back(k.point.x);
          value["keypoints"].push_back(k.point.y);
          value["keypoints"].push_back(k.visibility);
          visible += k.visibility > 0;
        }
        value["num_keypoints"] = visible;
      }
      annotations.push_back(std::move(value));
    }
  }
  auto text = J{{"images", images},
                {"annotations", annotations},
                {"categories", categories}}
                  .dump(2);
  if (text.size() > max_bytes)
    throw ResourceExhausted("COCO output exceeds explicit byte bound");
  return text;
}
std::string preview_vision(RecipeSpec r, const std::filesystem::path& output,
                           const PreviewOptions& options,
                           std::stop_token stop) {
  if (!options.samples || options.samples > 256 || output.empty() ||
      (std::filesystem::exists(output) && !std::filesystem::is_empty(output)))
    throw InvalidArgument(
        "Preview needs 1..256 samples and a new output directory");
  auto splits = load_vision_data(r, stop);
  r.data.num_classes = r.model.num_classes;
  r.data.size_divisor =
      static_cast<std::uint32_t>(describe_binding(r.model).input_divisor);
  r.data.rectangular = r.rectangular_batches;
  r.data.classification.enabled = r.model.task == TaskKind::classify;
  r.data.allow_unannotated_negatives = r.model.task == TaskKind::classify;
  r.data.augmentation.dataset_id = splits.training_identity;
  r.data.augmentation.seed = r.seed;
  r.data.augmentation.multi_source.total_epochs = r.epochs;
  data::DataPipelineOptions pipeline;
  pipeline.pipeline_id = "vision.preview";
  pipeline.dataset_version = splits.training_identity;
  pipeline.batch_size = 1;
  auto source = make_yolo_data_pipeline(splits.training, pipeline, r.data);
  source->set_epoch(options.epoch);
  std::filesystem::create_directories(output);
  J rows = J::array();
  try {
    for (std::size_t i = 0;
         i < (std::min)(options.samples, splits.training->size()); ++i) {
      if (stop.stop_requested()) throw Cancelled("Preview cancelled");
      auto envelope = source->next(stop);
      if (!envelope) break;
      auto batch = im::image_batch_from_value(envelope->value);
      auto original = splits.training->get(i, stop);
      auto pixels = batch.images[0].to(torch::kCPU);
      if (!r.data.normalization_std.empty())
        pixels =
            pixels * torch::tensor(r.data.normalization_std, torch::kFloat32)
                         .view({3, 1, 1}) +
            torch::tensor(r.data.normalization_mean, torch::kFloat32)
                .view({3, 1, 1});
      pixels = (pixels * 255).round().clamp(0, 255).to(torch::kUInt8);
      auto spec = Rgb();
      auto original_pixels = original.image;
      auto original_spec = original.pixels;
      if (options.annotations) {
        if (original.annotation) {
          original_pixels = detail::render_vision_annotations(
              original.image, original.pixels, *original.annotation,
              splits.categories, stop);
          original_spec.layout = im::ImageLayout::hwc;
        }
        if (batch.items[0].annotation) {
          pixels = detail::render_vision_annotations(pixels, spec,
                                                     *batch.items[0].annotation,
                                                     splits.categories, stop);
          spec.layout = im::ImageLayout::hwc;
        }
      }
      const auto name = std::to_string(i);
      Png(output / (name + ".original.png"), original_pixels, original_spec,
          stop);
      Png(output / (name + ".augmented.png"), pixels, spec, stop);
      im::AnnotationIndex annotation;
      annotation.categories = splits.categories;
      if (batch.items[0].annotation)
        annotation.images.push_back(*batch.items[0].annotation);
      rows.push_back({{"index", i},
                      {"source", original.image_path},
                      {"original", name + ".original.png"},
                      {"augmented", name + ".augmented.png"},
                      {"labels", J::parse(coco_annotations(annotation))}});
    }
  } catch (...) {
    source->close();
    throw;
  }
  source->close();
  auto text = J{
      {"schema", "vision.preview/v1"},
      {"data", splits.training_identity},
      {"recipe", recipe_identity(r)},
      {"epoch", options.epoch},
      {"samples",
       rows}}.dump(2);
  detail::publish_text(output / "preview.json", text);
  return text;
}
void save_prediction(const im::VisionSample& sample, const ImageResult& result,
                     const std::filesystem::path& basename,
                     const PredictionOutputOptions& options,
                     std::stop_token stop) {
  if (basename.empty() || result.extent.width != sample.width ||
      result.extent.height != sample.height || options.max_crops > 10000)
    throw InvalidArgument(
        "Prediction output requires matching source coordinates and bounded "
        "crops");
  std::filesystem::create_directories(basename.parent_path());
  if (options.json)
    detail::publish_text(basename.string() + ".json", result_json(result));
  if (options.text) write_result(basename.string() + ".txt", result, false);
  if (options.render) {
    auto record = Annotation(result);
    auto rendered = detail::render_vision_annotations(
        sample.image, sample.pixels, record,
        sample.categories
            ? std::span<const im::VisionCategory>(*sample.categories)
            : std::span<const im::VisionCategory>{},
        stop);
    auto spec = Rgb();
    spec.layout = im::ImageLayout::hwc;
    Png(basename.string() + ".png", rendered, spec, stop);
  }
  if (options.crops && result.task != TaskKind::classify)
    for (I i = 0; i < (std::min)(result.boxes.size(0),
                                 static_cast<I>(options.max_crops));
         ++i) {
      auto b = result.boxes[i];
      const auto x1 = std::clamp<I>(
                     static_cast<I>(std::floor(b[0].item<double>())), 0,
                     sample.width),
                 y1 = std::clamp<I>(
                     static_cast<I>(std::floor(b[1].item<double>())), 0,
                     sample.height);
      const auto x2 = std::clamp<I>(
                     static_cast<I>(std::ceil(b[2].item<double>())), 0,
                     sample.width),
                 y2 = std::clamp<I>(
                     static_cast<I>(std::ceil(b[3].item<double>())), 0,
                     sample.height);
      if (x2 <= x1 || y2 <= y1) continue;
      auto crop = sample.pixels.layout == im::ImageLayout::chw
                      ? sample.image.slice(1, y1, y2).slice(2, x1, x2)
                      : sample.image.slice(0, y1, y2).slice(1, x1, x2);
      Png(basename.string() + ".crop-" + std::to_string(i) + ".png", crop,
          sample.pixels, stop);
    }
}
}  // namespace jyppx::liuding::vision::tasks
