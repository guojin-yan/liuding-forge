// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "../training/class_weights.hpp"

#include <jyppx/liuding/data_image/image_batch.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/vision/tasks/data/classification.hpp>
#include <jyppx/liuding/vision/tasks/task.hpp>

#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>
#include <numeric>
namespace jyppx::liuding::vision::tasks {
namespace {
namespace im = data::image;
using I = std::int64_t;
void Stop(std::stop_token s) {
  if (s.stop_requested())
    throw Cancelled("Classification augmentation cancelled");
}
void Probability(double p) {
  if (!std::isfinite(p) || p < 0 || p > 1)
    throw InvalidArgument("Classification probability/magnitude outside [0,1]");
}
Tensor Chw(const im::VisionSample& s) {
  return s.pixels.layout == im::ImageLayout::chw ? s.image
                                                 : s.image.permute({2, 0, 1});
}
Tensor Gray(const Tensor& x) {
  return (x[0] * .299 + x[1] * .587 + x[2] * .114).unsqueeze(0);
}
}  // namespace
std::string classification_augmentation_identity(
    const ClassificationAugmentationOptions& o) {
  for (auto p :
       {o.crop_min_area, o.crop_max_area, o.validation_crop_fraction,
        o.brightness, o.contrast, o.saturation, o.randaugment_magnitude,
        o.erasing_probability, o.mixup_probability, o.cutmix_probability})
    Probability(p);
  if (o.crop_min_area <= 0 || o.crop_min_area > o.crop_max_area ||
      !std::isfinite(o.crop_min_aspect) || !std::isfinite(o.crop_max_aspect) ||
      o.crop_min_aspect <= 0 || o.crop_min_aspect > o.crop_max_aspect ||
      o.crop_max_aspect > 16 || o.validation_crop_fraction <= 0 ||
      o.randaugment_operations > 8)
    throw InvalidArgument("Invalid classification crop/RandAugment profile");
  const auto text =
      nlohmann::json{"classification/native-rrc-color-ra-erasing/v1",
                     o.enabled,
                     o.training,
                     o.crop_min_area,
                     o.crop_max_area,
                     o.crop_min_aspect,
                     o.crop_max_aspect,
                     o.validation_crop_fraction,
                     o.brightness,
                     o.contrast,
                     o.saturation,
                     o.randaugment_operations,
                     o.randaugment_magnitude,
                     o.erasing_probability,
                     o.mixup_probability,
                     o.cutmix_probability}
          .dump();
  return sha256_hex(std::as_bytes(std::span(text)));
}
im::VisionSample prepare_classification_sample(
    const im::VisionSample& source, im::ImageExtent output,
    const ClassificationAugmentationOptions& o, const im::SampleRandom& random,
    std::stop_token stop) {
  static_cast<void>(classification_augmentation_identity(o));
  Stop(stop);
  if (!source.width || !source.height || !source.image.is_cpu() ||
      source.image.scalar_type() != torch::kUInt8 ||
      source.pixels.color != im::PixelColorModel::rgb ||
      (source.annotation && !source.annotation->objects.empty()) ||
      source.paired_target || source.mask_target)
    throw InvalidArgument(
        "Classification augmentation expects CPU RGB uint8 classification "
        "photos");
  if (!output.width || !output.height || output.width > 8192 ||
      output.height > 8192)
    throw InvalidArgument("Invalid classification output size");
  std::vector<im::ImageGeometryStep> steps;
  std::uint32_t cw = source.width, ch = source.height, x = 0, y = 0;
  if (o.training) {
    bool accepted = false;
    const double area = static_cast<double>(cw) * ch;
    for (std::uint64_t attempt = 0; attempt < 10; ++attempt) {
      const double target =
          area * (o.crop_min_area + random.unit(4 * attempt) *
                                        (o.crop_max_area - o.crop_min_area));
      const double aspect = std::exp(
          std::log(o.crop_min_aspect) +
          random.unit(4 * attempt + 1) *
              (std::log(o.crop_max_aspect) - std::log(o.crop_min_aspect)));
      const auto w = static_cast<std::uint32_t>(
                     std::llround(std::sqrt(target * aspect))),
                 h = static_cast<std::uint32_t>(
                     std::llround(std::sqrt(target / aspect)));
      if (w && h && w <= cw && h <= ch) {
        x = static_cast<std::uint32_t>(
            random.bounded(cw - w + 1, 4 * attempt + 2));
        y = static_cast<std::uint32_t>(
            random.bounded(ch - h + 1, 4 * attempt + 3));
        cw = w;
        ch = h;
        accepted = true;
        break;
      }
    }
    if (!accepted) {
      const double aspect = static_cast<double>(cw) / ch;
      if (aspect < o.crop_min_aspect)
        ch = static_cast<std::uint32_t>(cw / o.crop_min_aspect);
      else if (aspect > o.crop_max_aspect)
        cw = static_cast<std::uint32_t>(ch * o.crop_max_aspect);
      cw = (std::max)(cw, 1U);
      ch = (std::max)(ch, 1U);
      x = (source.width - cw) / 2;
      y = (source.height - ch) / 2;
    }
    steps.push_back({.kind = im::ImageGeometryKind::crop,
                     .extent = {cw, ch},
                     .left = x,
                     .top = y});
    steps.push_back({.kind = im::ImageGeometryKind::resize, .extent = output});
  } else {
    const double scale =
        (std::max)(static_cast<double>(output.width) / source.width,
                   static_cast<double>(output.height) / source.height) /
        o.validation_crop_fraction;
    cw = static_cast<std::uint32_t>(std::ceil(source.width * scale));
    ch = static_cast<std::uint32_t>(std::ceil(source.height * scale));
    steps.push_back(
        {.kind = im::ImageGeometryKind::resize, .extent = {cw, ch}});
    steps.push_back({.kind = im::ImageGeometryKind::crop,
                     .extent = output,
                     .left = (cw - output.width) / 2,
                     .top = (ch - output.height) / 2});
  }
  im::JointTransformOptions geometry;
  geometry.interpolation = im::ImageInterpolation::bilinear;
  auto result = im::transform_sample(source, steps, geometry, stop);
  if (!o.training) return result;
  auto photo = Chw(result).to(torch::kFloat32);
  const double bright = 1 + (2 * random.unit(50) - 1) * o.brightness,
               contrast = 1 + (2 * random.unit(51) - 1) * o.contrast,
               saturation = 1 + (2 * random.unit(52) - 1) * o.saturation;
  photo = photo * bright;
  auto gray = Gray(photo);
  photo = (photo - gray) * saturation + gray;
  auto mean = Gray(photo).mean();
  photo = (photo - mean) * contrast + mean;
  for (std::uint32_t i = 0; i < o.randaugment_operations; ++i) {
    Stop(stop);
    const auto operation = random.bounded(6, 60 + 3 * i);
    const double magnitude = o.randaugment_magnitude,
                 sign = random.unit(61 + 3 * i) < .5 ? -1 : 1;
    if (operation == 0)
      photo = photo * (1 + sign * .9 * magnitude);
    else if (operation == 1) {
      auto g = Gray(photo).mean();
      photo = (photo - g) * (1 + sign * .9 * magnitude) + g;
    } else if (operation == 2) {
      auto g = Gray(photo);
      photo = (photo - g) * (1 + sign * .9 * magnitude) + g;
    } else if (operation == 3)
      photo =
          torch::where(photo >= 255 * (1 - .8 * magnitude), 255 - photo, photo);
    else if (operation == 4) {
      const auto step = std::pow(2., std::round(4 * magnitude));
      photo = torch::floor(photo / step) * step;
    } else {
      const auto low = std::get<0>(photo.flatten(1).min(1)).view({3, 1, 1}),
                 high = std::get<0>(photo.flatten(1).max(1)).view({3, 1, 1});
      photo = (photo - low) * 255 / (high - low).clamp_min(1);
    }
    photo = photo.clamp(0, 255);
  }
  if (random.unit(90) < o.erasing_probability) {
    const I w = output.width, h = output.height;
    const auto rw = (std::max)(
                   I{1}, static_cast<I>(w * (.05 + .25 * random.unit(91)))),
               rh = (std::max)(
                   I{1}, static_cast<I>(h * (.05 + .25 * random.unit(92))));
    const auto ex = static_cast<I>(random.bounded(w - rw + 1, 93)),
               ey = static_cast<I>(random.bounded(h - rh + 1, 94));
    photo.slice(1, ey, ey + rh).slice(2, ex, ex + rw).fill_(127.5);
  }
  result.image = photo.round().clamp(0, 255).to(torch::kUInt8);
  result.pixels.layout = im::ImageLayout::chw;
  result.image_resource_leases.reset();
  return result;
}
training::StepFunction make_augmented_classification_step(
    I classes, const ClassificationAugmentationOptions& o, std::uint64_t seed,
    double smoothing, const std::vector<double>& class_weights) {
  detail::check_class_weights(class_weights, classes);
  auto identity = classification_augmentation_identity(o);
  if (classes < 1 || classes > 10000)
    throw InvalidArgument("Invalid mixed classification class count");
  return [classes, o, seed, smoothing, class_weights,
          identity](const training::StepInput& input) {
    if (!input.target || !input.target->isTensor())
      throw InvalidArgument("Classification requires labels");
    auto invocation = input.invocation;
    auto targets = input.target->toTensor();
    auto images = invocation.positional.at(0).toTensor();
    if (input.context.phase == training::StepPhase::training &&
        (o.mixup_probability > 0 || o.cutmix_probability > 0)) {
      const I n = images.size(0);
      auto batch = im::image_batch_from_value(input.batch.value);
      std::string sample_id;
      for (const auto& item : batch.items) {
        sample_id +=
            item.key.id + ":" + std::to_string(item.key.occurrence) + ";";
      }
      auto digest = sha256_hex(std::as_bytes(std::span(sample_id)));
      im::SampleRandom random({.dataset_id = identity,
                               .sample_id = digest,
                               .transform_id = "classification-batch-mix/v1",
                               .seed = seed,
                               .epoch = input.context.epoch,
                               .occurrence = input.context.batch_sequence});
      const bool cut = random.unit(0) < o.cutmix_probability,
                 mix = random.unit(1) < o.mixup_probability;
      if (cut || mix) {
        std::vector<I> indices(static_cast<std::size_t>(n));
        std::iota(indices.begin(), indices.end(), 0);
        for (I i = n; i > 1; --i)
          std::swap(
              indices[static_cast<std::size_t>(i - 1)],
              indices[static_cast<std::size_t>(random.bounded(i, 10 + i))]);
        auto partners =
            torch::tensor(indices, torch::kInt64).to(images.device());
        auto encoded =
            torch::zeros({n, classes}, images.options().dtype(torch::kFloat32))
                .scatter_(1, targets.to(torch::kInt64).view({n, 1}), 1.);
        double fraction = .35 + .3 * random.unit(2);
        auto mixed = images.clone();
        if (cut) {
          const I h = images.size(2), w = images.size(3),
                  rw = (std::max)(I{1},
                                  static_cast<I>(w * std::sqrt(1 - fraction))),
                  rh = (std::max)(I{1},
                                  static_cast<I>(h * std::sqrt(1 - fraction)));
          const I x = static_cast<I>(random.bounded(w - rw + 1, 3)),
                  y = static_cast<I>(random.bounded(h - rh + 1, 4));
          mixed.slice(2, y, y + rh)
              .slice(3, x, x + rw)
              .copy_(images.index_select(0, partners)
                         .slice(2, y, y + rh)
                         .slice(3, x, x + rw));
          fraction = 1 - static_cast<double>(rw * rh) / (w * h);
        } else
          mixed = images * fraction +
                  images.index_select(0, partners) * (1 - fraction);
        invocation.positional[0] = mixed;
        targets = encoded * fraction +
                  encoded.index_select(0, partners) * (1 - fraction);
      }
    }
    auto prediction = input.model.invoke(invocation);
    auto values =
        prediction.value.isTensor()
            ? prediction.value.toTensor()
            : prediction.value.toTupleRef().elements().at(0).toTensor();
    training::StepOutput result;
    result.loss = classification_loss(
        values, targets, smoothing,
        input.context.phase == training::StepPhase::training
            ? detail::class_weight_tensor(class_weights, values)
            : Tensor{});
    result.sample_count = static_cast<std::uint64_t>(images.size(0));
    result.prediction = std::move(prediction);
    return result;
  };
}
}  // namespace jyppx::liuding::vision::tasks
