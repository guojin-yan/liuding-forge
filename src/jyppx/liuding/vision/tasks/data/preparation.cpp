// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data/samplers/sequential_sampler.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/vision/tasks/data/composition.hpp>
#include <jyppx/liuding/vision/tasks/data/preparation.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <locale>
#include <sstream>
#include <vector>

namespace jyppx::liuding::vision::tasks {
namespace {
namespace image = data::image;

bool PhotoEnabled(const PhotometricOptions& options) {
  return options.hue_shift > 0 || options.saturation_gain > 0 ||
         options.value_gain > 0 || options.grayscale_probability > 0 ||
         options.blur_probability > 0 || options.median_blur_probability > 0 ||
         options.clahe_probability > 0;
}

void CheckTargets(const YoloDataOptions& options) {
  if (options.num_classes < 1 || options.num_classes > 10000 ||
      options.max_targets <= 0) {
    throw InvalidArgument(
        "YOLO data requires positive bounded classes and target rows");
  }
}

void CheckProjection(const YoloDataOptions& options) {
  CheckTargets(options);
  for (const auto probability :
       {options.augmentation.horizontal_flip_probability,
        options.augmentation.vertical_flip_probability})
    if (!std::isfinite(probability) || probability < 0 || probability > 1)
      throw InvalidArgument("YOLO flip probability must be in [0,1]");
  static_cast<void>(scale_translate_identity(options.augmentation.spatial));
  static_cast<void>(photometric_identity(options.augmentation.photo));
  if (options.normalization_mean.size() != options.normalization_std.size() ||
      (!options.normalization_mean.empty() &&
       options.normalization_mean.size() != 3))
    throw InvalidArgument("Normalization requires matching RGB mean/std");
  for (std::size_t i = 0; i < options.normalization_mean.size(); ++i)
    if (!std::isfinite(options.normalization_mean[i]) ||
        !std::isfinite(options.normalization_std[i]) ||
        options.normalization_std[i] <= 0)
      throw InvalidArgument("Invalid RGB normalization values");
  if ((options.augmentation.horizontal_flip_probability > 0 ||
       options.augmentation.vertical_flip_probability > 0 ||
       options.augmentation.spatial.scale_gain > 0 ||
       options.augmentation.spatial.translate_fraction > 0 ||
       PhotoEnabled(options.augmentation.photo)) &&
      options.augmentation.dataset_id.empty())
    throw InvalidArgument(
        "YOLO augmentation requires a stable dataset identity");
}
}  // namespace

image::VisionSample letterbox_sample(const image::VisionSample& sample,
                                     const LetterBoxOptions& options,
                                     std::stop_token stop) {
  if (sample.width == 0 || sample.height == 0 || options.canvas.width == 0 ||
      options.canvas.height == 0 || options.canvas.width > 65536 ||
      options.canvas.height > 65536 || !std::isfinite(options.padding_value)) {
    throw InvalidArgument(
        "LetterBox requires positive bounded source/canvas extents and finite "
        "fill");
  }
  double ratio =
      (std::min)(static_cast<double>(options.canvas.width) / sample.width,
                 static_cast<double>(options.canvas.height) / sample.height);
  if (!options.scale_up) {
    ratio = (std::min)(ratio, 1.0);
  }
  const auto width = (std::min)(
      options.canvas.width,
      (std::max)(std::uint32_t{1}, static_cast<std::uint32_t>(std::floor(
                                       sample.width * ratio + 0.5))));
  const auto height = (std::min)(
      options.canvas.height,
      (std::max)(std::uint32_t{1}, static_cast<std::uint32_t>(std::floor(
                                       sample.height * ratio + 0.5))));
  const auto horizontal = options.canvas.width - width;
  const auto vertical = options.canvas.height - height;
  const auto left = options.center ? horizontal / 2 : 0;
  const auto top = options.center ? vertical / 2 : 0;
  std::array<image::ImageGeometryStep, 2> steps{
      image::ImageGeometryStep{.kind = image::ImageGeometryKind::resize,
                               .extent = {width, height}},
      image::ImageGeometryStep{.kind = image::ImageGeometryKind::pad,
                               .left = left,
                               .top = top,
                               .right = horizontal - left,
                               .bottom = vertical - top}};
  auto geometry = options.geometry;
  geometry.image_padding = options.padding_value;
  return image::transform_sample(sample, steps, geometry, stop);
}

image::ImageBatch prepare_yolo_batch(
    std::span<const image::VisionSample> samples,
    std::span<const image::ImageBatchKey> keys, const YoloDataOptions& options,
    std::stop_token stop) {
  CheckProjection(options);
  if (samples.empty() || samples.size() != keys.size() ||
      samples.size() > options.limits.max_samples ||
      !options.size_divisor || options.size_divisor > 256 ||
      options.letterbox.canvas.width % options.size_divisor != 0 ||
      options.letterbox.canvas.height % options.size_divisor != 0) {
    throw InvalidArgument(
        "Vision batch requires bounded samples/keys and a canvas aligned to size_divisor");
  }
  std::vector<image::VisionSample> prepared;
  prepared.reserve(samples.size());
  for (std::size_t index = 0; index < samples.size(); ++index) {
    const auto& sample = samples[index];
    if (stop.stop_requested()) {
      throw Cancelled("YOLO data preparation cancelled");
    }
    if (!sample.image.defined() || !sample.image.is_cpu() ||
        sample.pixels.dtype != torch::kUInt8 ||
        sample.pixels.color != image::PixelColorModel::rgb ||
        sample.pixels.range != image::PixelValueRange::unsigned_full_scale) {
      throw InvalidArgument(
          "YOLO preparation currently requires explicit CPU uint8 RGB images");
    }
    if (!sample.annotation && !options.allow_unannotated_negatives) {
      throw InvalidArgument(
          "Unannotated YOLO samples need explicit negative-sample opt-in");
    }
    auto letterbox = options.letterbox;
    letterbox.geometry.interpolation = image::ImageInterpolation::bilinear;
    letterbox.geometry.image_padding = letterbox.padding_value;
    if (sample.categories && sample.categories->size() !=
                                 static_cast<std::size_t>(options.num_classes))
      throw InvalidArgument(
          "YOLO source category count differs from the model class count");
    if (index > 0 && sample.categories && samples.front().categories) {
      for (std::size_t category = 0; category < sample.categories->size();
           ++category)
        if ((*sample.categories)[category].source_id !=
                (*samples.front().categories)[category].source_id ||
            (*sample.categories)[category].name !=
                (*samples.front().categories)[category].name)
          throw InvalidArgument(
              "YOLO batch samples have inconsistent class mappings");
    }
    auto output = sample;
    const auto stream = [&](const std::string& transform_id) {
      return image::SampleRandom({.dataset_id = options.augmentation.dataset_id,
                                  .sample_id = keys[index].id,
                                  .transform_id = transform_id,
                                  .seed = options.augmentation.seed,
                                  .epoch = options.augmentation.epoch,
                                  .occurrence = keys[index].occurrence,
                                  .view = keys[index].view});
    };
    output = options.classification.enabled
                 ? prepare_classification_sample(
                       sample, letterbox.canvas, options.classification,
                       stream(classification_augmentation_identity(
                           options.classification)),
                       stop)
                 : letterbox_sample(sample, letterbox, stop);
    if (options.augmentation.projective.degrees > 0 ||
        options.augmentation.projective.shear_degrees > 0 ||
        options.augmentation.projective.perspective > 0 ||
        options.augmentation.projective.scale_gain > 0 ||
        options.augmentation.projective.translate_fraction > 0) {
      output = augment_projective(
          output, options.augmentation.projective,
          stream(projective_identity(options.augmentation.projective)),
          letterbox.padding_value, stop);
    }
    if (options.augmentation.spatial.scale_gain > 0 ||
        options.augmentation.spatial.translate_fraction > 0) {
      const auto random =
          stream(scale_translate_identity(options.augmentation.spatial));
      output = augment_scale_translate(output, options.augmentation.spatial,
                                       random, letterbox.geometry, stop);
    }
    if (options.augmentation.horizontal_flip_probability > 0 ||
        options.augmentation.vertical_flip_probability > 0) {
      image::RandomGeometryOptions geometry;
      geometry.horizontal_flip_probability =
          options.augmentation.horizontal_flip_probability;
      geometry.vertical_flip_probability =
          options.augmentation.vertical_flip_probability;
      const auto random =
          stream(image::joint_transform_identity(geometry, letterbox.geometry));
      const auto steps = image::sample_geometry({output.width, output.height},
                                                geometry, random);
      if (!steps.empty())
        output =
            image::transform_sample(output, steps, letterbox.geometry, stop);
    }
    if (PhotoEnabled(options.augmentation.photo)) {
      auto photo = options.augmentation.photo;
      if (!photo.resource_budget)
        photo.resource_budget = letterbox.geometry.resource_budget;
      const auto random = stream(photometric_identity(photo));
      output = augment_photometric(output, photo, random, stop);
    }
    prepared.push_back(std::move(output));
  }
  image::ImageCollateOptions collate;
  collate.mask_padding_id=options.letterbox.geometry.mask_padding_id;
  collate.limits = options.limits;
  collate.resource_budget = options.letterbox.geometry.resource_budget;
  auto batch = image::collate_images(prepared, keys, collate, stop);
  image::ImageBatchTransformOptions numeric;
  numeric.limits = options.limits;
  numeric.resource_budget = options.letterbox.geometry.resource_budget;
  numeric.image.dtype = torch::kFloat32;
  numeric.image.scale = image::BatchPixelScale::to_unit;
  numeric.image.mean = options.normalization_mean;
  numeric.image.std = options.normalization_std;
  return image::transform_image_batch(batch, numeric, stop);
}

Tensor build_yolo_targets(const image::ImageBatch& batch,
                          const YoloDataOptions& options) {
  CheckTargets(options);
  if (!batch.images.defined() || batch.images.dim() != 4 ||
      batch.images.size(0) != static_cast<std::int64_t>(batch.items.size())) {
    throw InvalidArgument(
        "YOLO target packing requires a valid packed ImageBatch");
  }
  std::vector<std::array<float, 6>> rows;
  for (std::size_t index = 0; index < batch.items.size(); ++index) {
    const auto& annotation = batch.items[index].annotation;
    if (!annotation) {
      if (!options.allow_unannotated_negatives) {
        throw InvalidArgument(
            "Missing YOLO annotation is not a declared negative sample");
      }
      continue;
    }
    for (const auto& object : annotation->objects) {
      if (object.ignore || (options.exclude_crowd && object.crowd) ||
          (options.exclude_difficult && object.difficult)) {
        continue;
      }
      if (!object.box ||
          object.category >= static_cast<std::size_t>(options.num_classes)) {
        throw InvalidArgument(
            "YOLO target needs an axis-aligned box and a valid contiguous "
            "class");
      }
      if (rows.size() >= static_cast<std::size_t>(options.max_targets)) {
        throw ResourceExhausted("YOLO target packing row bound exceeded");
      }
      const auto& box = *object.box;
      rows.push_back(
          {static_cast<float>(index), static_cast<float>(object.category),
           static_cast<float>(box.x_min), static_cast<float>(box.y_min),
           static_cast<float>(box.x_max), static_cast<float>(box.y_max)});
    }
  }
  auto target = torch::empty({static_cast<std::int64_t>(rows.size()), 6},
                             torch::TensorOptions().dtype(torch::kFloat32));
  auto* output = target.data_ptr<float>();
  for (const auto& row : rows) {
    output = std::copy(row.begin(), row.end(), output);
  }
  return target.to(batch.images.device());
}

training::BatchMapper make_yolo_image_mapper() {
  return [](const data::BatchEnvelope& envelope) {
    auto batch = image::image_batch_from_value(envelope.value);
    if (!batch.images.defined()) {
      throw InvalidArgument("YOLO images cannot be ragged");
    }
    return nn::Invocation{{batch.images}, {}};
  };
}

training::TargetMapper make_yolo_target_mapper(const YoloDataOptions& options) {
  CheckTargets(options);
  return
      [options](
          const data::BatchEnvelope& envelope) -> std::optional<c10::IValue> {
        return build_yolo_targets(image::image_batch_from_value(envelope.value),
                                  options);
      };
}

std::string yolo_data_identity(const YoloDataOptions& options) {
  CheckProjection(options);
  if (!options.size_divisor || options.size_divisor > 256 ||
      !std::isfinite(options.multiscale_min) ||
      !std::isfinite(options.multiscale_max) || options.multiscale_min <= 0 ||
      options.multiscale_min > options.multiscale_max ||
      options.multiscale_max > 4 ||
      options.multiscale_max_dimension < options.size_divisor ||
      options.multiscale_max_dimension > 8192)
    throw InvalidArgument("Invalid rectangular/multiscale bounds");
  image::RandomGeometryOptions random;
  random.horizontal_flip_probability =
      options.augmentation.horizontal_flip_probability;
  random.vertical_flip_probability =
      options.augmentation.vertical_flip_probability;
  image::ImageCollateOptions collate;
  collate.limits = options.limits;
  image::ImageBatchTransformOptions numeric;
  numeric.limits = options.limits;
  numeric.image.mean = options.normalization_mean;
  numeric.image.std = options.normalization_std;
  std::ostringstream identity;
  identity.imbue(std::locale::classic());
  identity << std::setprecision(17)
           << "liuding.yolo-data/v3:" << options.rectangular << ':'
           << options.multiscale << ':' << options.multiscale_min << ':'
           << options.multiscale_max << ':' << options.size_divisor << ':'
           << options.multiscale_max_dimension << ':'
           << options.letterbox.canvas.width << ':'
           << options.letterbox.canvas.height << ':'
           << options.letterbox.scale_up << ':' << options.letterbox.center
           << ':' << options.letterbox.padding_value << ':'
           << image::joint_transform_identity(random,
                                              options.letterbox.geometry)
           << ':' << scale_translate_identity(options.augmentation.spatial)
           << ':' << photometric_identity(options.augmentation.photo) << ':'
           << projective_identity(options.augmentation.projective) << ':'
           << multisource_identity(options.augmentation.multi_source) << ':'
           << classification_augmentation_identity(options.classification)
           << ':' << image::image_batch_identity(collate, numeric) << ':'
           << options.num_classes << ':' << options.max_targets << ':'
           << options.allow_unannotated_negatives << ':'
           << options.exclude_crowd << ':' << options.exclude_difficult << ':'
           << options.augmentation.seed << ':'
           << options.augmentation.dataset_id.size() << ':'
           << options.augmentation.dataset_id;
  const auto text = identity.str();
  return sha256_hex(std::as_bytes(std::span(text)));
}

YoloDataOptions yolo_evaluation_options(const YoloDataOptions& training) {
  auto result = training;
  result.multiscale = false;
  result.rectangular = false;
  result.classification.training = false;
  result.augmentation.horizontal_flip_probability = 0;
  result.augmentation.vertical_flip_probability = 0;
  result.augmentation.spatial = {};
  result.augmentation.projective = {};
  result.augmentation.multi_source = {};
  result.augmentation.multi_source.total_epochs =
      training.augmentation.multi_source.total_epochs;
  result.augmentation.multi_source.close_epochs = 0;
  auto& photo = result.augmentation.photo;
  photo.hue_shift = photo.saturation_gain = photo.value_gain = 0;
  photo.grayscale_probability = photo.blur_probability =
      photo.median_blur_probability = photo.clahe_probability = 0;
  return result;
}

std::shared_ptr<data::DataPipeline> make_yolo_data_pipeline(
    std::shared_ptr<const image::ImageDataset> dataset,
    data::DataPipelineOptions pipeline, YoloDataOptions options,
    std::shared_ptr<data::samplers::Sampler> sampler) {
  if (!dataset || dataset->empty() || pipeline.dataset_version.empty() ||
      (pipeline.dataset_size != 0 && pipeline.dataset_size != dataset->size()))
    throw InvalidArgument(
        "YOLO pipeline requires a nonempty dataset and consistent "
        "lineage/size");
  pipeline.dataset_size = dataset->size();
  if (options.augmentation.dataset_id.empty())
    options.augmentation.dataset_id = pipeline.dataset_version;
  pipeline.batch_factory_id = "vision.yolo/" + yolo_data_identity(options);
  const auto batch_size = pipeline.batch_size;
  if (!sampler)
    sampler =
        std::make_shared<data::samplers::SequentialSampler>(dataset->size());
  return std::make_shared<data::DataPipeline>(
      std::move(sampler), std::move(pipeline),
      [dataset = std::move(dataset), options, batch_size](
          std::span<const std::size_t> indices,
          const data::DataWorkerContext& context, std::stop_token stop) {
        std::vector<image::VisionSample> samples;
        std::vector<image::ImageBatchKey> keys;
        for (const auto index : indices) {
          auto sample = dataset->get(index, stop);
          image::SampleRandom random(
              {.dataset_id = options.augmentation.dataset_id,
               .sample_id = sample.image_path,
               .transform_id =
                   multisource_identity(options.augmentation.multi_source),
               .seed = options.augmentation.seed,
               .epoch = context.epoch,
               .occurrence = context.sequence * batch_size + keys.size()});
          sample = augment_multisource(
              sample, dataset->size(),
              [dataset](std::size_t i, std::stop_token token) {
                return dataset->get(i, token);
              },
              options.augmentation.multi_source, options.letterbox, random,
              context.epoch, stop);
          samples.push_back(std::move(sample));
          keys.push_back(
              {.id = samples.back().image_path,
               .occurrence = context.sequence * batch_size + keys.size()});
        }
        auto current = options;
        current.augmentation.epoch = context.epoch;
        const auto divisor = options.size_divisor;
        if (options.rectangular) {
          std::uint32_t width = divisor, height = divisor;
          for (const auto& sample : samples) {
            auto scale = (std::min)(
                static_cast<double>(options.letterbox.canvas.width) /
                    sample.width,
                static_cast<double>(options.letterbox.canvas.height) /
                    sample.height);
            width = (std::max)(
                width, static_cast<std::uint32_t>(std::ceil(
                           std::round(sample.width * scale) / divisor)) *
                           divisor);
            height = (std::max)(
                height, static_cast<std::uint32_t>(std::ceil(
                            std::round(sample.height * scale) / divisor)) *
                            divisor);
          }
          current.letterbox.canvas = {width, height};
        }
        if (options.multiscale) {
          image::SampleRandom random(
              {.dataset_id = options.augmentation.dataset_id,
               .sample_id = std::to_string(context.sequence),
               .transform_id = "vision.multiscale/v1",
               .seed = options.augmentation.seed,
               .epoch = context.epoch});
          const auto factor = options.multiscale_min +
                              random.unit(0) * (options.multiscale_max -
                                                options.multiscale_min);
          auto aligned = [&](std::uint32_t side) {
            return static_cast<std::uint32_t>(
                (std::max)(double(divisor),
                           std::round(side * factor / divisor) * divisor));
          };
          auto canvas =
              image::ImageExtent{aligned(current.letterbox.canvas.width),
                                 aligned(current.letterbox.canvas.height)};
          if (canvas.width > options.multiscale_max_dimension ||
              canvas.height > options.multiscale_max_dimension)
            throw ResourceExhausted(
                "Multiscale candidate exceeds declared maximum dimension");
          current.letterbox.canvas = canvas;
        }
        return image::image_batch_value(
            prepare_yolo_batch(samples, keys, current, stop), options.limits);
      });
}

}  // namespace jyppx::liuding::vision::tasks
