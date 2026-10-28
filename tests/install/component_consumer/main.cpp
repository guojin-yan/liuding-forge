// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#if defined(JYPPX_CONSUMER_FOUNDATION)
#include <jyppx/liuding/foundation/version.hpp>
#elif defined(JYPPX_CONSUMER_CORE)
#include <jyppx/liuding/core/device.hpp>
#elif defined(JYPPX_CONSUMER_INTEROP)
#include <jyppx/liuding/interop/checkpoint_manifest.hpp>
#elif defined(JYPPX_CONSUMER_DATA)
#include <jyppx/liuding/data/samplers/sequential_sampler.hpp>
#elif defined(JYPPX_CONSUMER_DATA_IMAGE)
#include <jyppx/liuding/data/data_pipeline.hpp>
#include <jyppx/liuding/data/samplers/sequential_sampler.hpp>
#include <jyppx/liuding/data_image/annotation_files.hpp>
#include <jyppx/liuding/data_image/batch_geometry.hpp>
#include <jyppx/liuding/data_image/batch_identity.hpp>
#include <jyppx/liuding/data_image/batch_transform.hpp>
#include <jyppx/liuding/data_image/geometry_trace.hpp>
#include <jyppx/liuding/data_image/image_dataset.hpp>
#include <jyppx/liuding/data_image/image_decoder.hpp>
#include <jyppx/liuding/data_image/image_encoder.hpp>
#include <jyppx/liuding/data_image/image_output_sink.hpp>
#include <jyppx/liuding/data_image/joint_transform.hpp>
#include <jyppx/liuding/data_image/manifests.hpp>
#include <jyppx/liuding/data_image/mask_decoder.hpp>
#include <jyppx/liuding/data_image/panoptic.hpp>
#elif defined(JYPPX_CONSUMER_VISION_RENDER)
#include <jyppx/liuding/visualization/render.hpp>
#elif defined(JYPPX_CONSUMER_NN)
#include <jyppx/liuding/nn/loss/label_smoothing_cross_entropy.hpp>
#elif defined(JYPPX_CONSUMER_OPTIM)
#include <jyppx/liuding/optim/sgd_adapter.hpp>
#elif defined(JYPPX_CONSUMER_TRAINING)
#include <jyppx/liuding/training/trainer.hpp>
#elif defined(JYPPX_CONSUMER_EXPERIMENT)
#include <jyppx/liuding/experiment/config.hpp>
#elif defined(JYPPX_CONSUMER_MODELS)
#include <jyppx/liuding/models/registry.hpp>
#elif defined(JYPPX_CONSUMER_QUANTIZATION)
#include <jyppx/liuding/quantization/quantization.hpp>
#elif defined(JYPPX_CONSUMER_SERVING)
#include <jyppx/liuding/serving/serving.hpp>
#elif defined(JYPPX_CONSUMER_ALL)
#include <jyppx/liuding/foundation/version.hpp>
#elif defined(JYPPX_CONSUMER_SDK)
#include <jyppx/liuding/liuding.hpp>
#else
#error "Exactly one installed component consumer must be selected"
#endif

#include <jyppx/liuding/foundation/capability.hpp>

#if defined(JYPPX_CONSUMER_INTEROP) && JYPPX_EXPECT_COMPONENT_NN
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/interop/model_weight_import.hpp>
#include <jyppx/liuding/nn/model_adapter.hpp>

#include <memory>
#include <span>
#endif

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

#if defined(JYPPX_CONSUMER_INTEROP) && JYPPX_EXPECT_COMPONENT_NN
bool InstalledModelWeightImportAppliesVerifiedState() {
  const auto module = std::make_shared<torch::nn::Module>();
  const auto weight =
      module->register_parameter("weight", torch::zeros({2}, torch::kFloat64));
  const auto* identity = weight.unsafeGetTensorImpl();
  const std::array<double, 2> values{1.25, -2.5};
  const std::array<jyppx::liuding::interop::SafeTensorSource, 1> tensors{
      {{.name = "weight",
        .dtype = jyppx::liuding::interop::SafeTensorDtype::f64,
        .shape = {2U},
        .data = std::as_bytes(std::span(values))}}};
  const auto bytes = jyppx::liuding::interop::serialize_safetensors(tensors);
  jyppx::liuding::nn::ModelStateView view(module);
  const auto result = jyppx::liuding::interop::import_model_weights(
      view, bytes, {.hex = jyppx::liuding::sha256_hex(bytes)});
  return result.applied && result.applied_tensors == 1U &&
         weight.unsafeGetTensorImpl() == identity && weight.requires_grad() &&
         torch::equal(weight, torch::tensor({1.25, -2.5}, torch::kFloat64));
}
#endif

bool BuildCapabilityInventoryMatchesPackage() {
  constexpr std::array<std::pair<std::string_view, bool>, 13U> kExpected{{
      {"core", JYPPX_EXPECT_COMPONENT_CORE != 0},
      {"data", JYPPX_EXPECT_COMPONENT_DATA != 0},
      {"data_image", JYPPX_EXPECT_COMPONENT_DATA_IMAGE != 0},
      {"experiment", JYPPX_EXPECT_COMPONENT_EXPERIMENT != 0},
      {"foundation", JYPPX_EXPECT_COMPONENT_FOUNDATION != 0},
      {"interop", JYPPX_EXPECT_COMPONENT_INTEROP != 0},
      {"models", JYPPX_EXPECT_COMPONENT_MODELS != 0},
      {"nn", JYPPX_EXPECT_COMPONENT_NN != 0},
      {"optim", JYPPX_EXPECT_COMPONENT_OPTIM != 0},
      {"quantization", JYPPX_EXPECT_COMPONENT_QUANTIZATION != 0},
      {"serving", JYPPX_EXPECT_COMPONENT_SERVING != 0},
      {"training", JYPPX_EXPECT_COMPONENT_TRAINING != 0},
      {"vision_render", JYPPX_EXPECT_COMPONENT_VISION_RENDER != 0},
  }};
  const auto manifest = jyppx::liuding::current_capability_manifest();
  if (manifest.features.size() != kExpected.size()) return false;
  for (std::size_t index = 0; index < kExpected.size(); ++index) {
    const auto& [component, enabled] = kExpected[index];
    const auto& feature = manifest.features[index];
    const bool preview = component == "models" || component == "quantization" ||
                         component == "serving" || component == "data_image" ||
                         component == "vision_render";
    const auto expected_status =
        enabled ? (preview ? jyppx::liuding::CapabilityStatus::experimental
                           : jyppx::liuding::CapabilityStatus::available)
                : jyppx::liuding::CapabilityStatus::unsupported;
    if (feature.feature_id != "jyppx.component." + std::string(component) ||
        feature.status != expected_status ||
        (feature.implementation ==
         jyppx::liuding::CapabilityImplementation::native) != enabled) {
      return false;
    }
  }
  return !jyppx::liuding::current_capability_manifest_json().empty();
}

}  // namespace

int main() {
  if (!BuildCapabilityInventoryMatchesPackage()) return 1;
#if defined(JYPPX_CONSUMER_FOUNDATION)
  const auto fingerprint = jyppx::liuding::current_build_fingerprint();
  return !std::string(jyppx::liuding::version()).empty() &&
                 !fingerprint.git_commit.empty()
             ? 0
             : 1;
#elif defined(JYPPX_CONSUMER_CORE)
  return jyppx::liuding::parse_device("cpu").is_cpu() ? 0 : 1;
#elif defined(JYPPX_CONSUMER_INTEROP)
  const auto capability =
      jyppx::liuding::interop::make_checkpoint_manifest_capability();
  if (capability.feature_id != "interop.checkpoint_manifest") return 1;
#if JYPPX_EXPECT_COMPONENT_NN
  return InstalledModelWeightImportAppliesVerifiedState() ? 0 : 1;
#else
  return 0;
#endif
#elif defined(JYPPX_CONSUMER_DATA)
  const auto indices =
      jyppx::liuding::data::samplers::SequentialSampler(3).indices();
  return indices == std::vector<std::size_t>({0, 1, 2}) ? 0 : 1;
#elif defined(JYPPX_CONSUMER_DATA_IMAGE)
  const auto metadata =
      jyppx::liuding::data::image::parse_yolo_dataset_yaml("names: [cat]");
  jyppx::liuding::data::image::AnnotationReadOptions options;
  options.categories = metadata.categories;
  options.image_path = "a.jpg";
  options.width = 20;
  options.height = 10;
  const auto index = jyppx::liuding::data::image::parse_annotations(
      "0 .5 .5 1 1",
      jyppx::liuding::data::image::AnnotationFormat::yolo_detection, options);
  jyppx::liuding::data::image::ManifestReadOptions manifest_options;
  manifest_options.categories = metadata.categories;
  const auto manifest = jyppx::liuding::data::image::parse_manifest(
      "path,labels\na.jpg,cat\n",
      jyppx::liuding::data::image::ManifestFormat::csv,
      jyppx::liuding::data::image::ManifestTask::classification,
      manifest_options);
  constexpr std::array<unsigned char, 12> pgm{'P',  '5', '\n', '1', ' ',  '1',
                                              '\n', '2', '5',  '5', '\n', 127};
  const auto image =
      jyppx::liuding::data::image::decode_image(std::as_bytes(std::span(pgm)));
  jyppx::liuding::data::image::validate_pixels(image.tensor, image.pixels);
  auto tensor_record = jyppx::liuding::data::image::TensorImageRecord{
      image.tensor, image.pixels, "installed-tensor", {}};
  tensor_record.paired_target = image.tensor;
  tensor_record.paired_target_pixels = image.pixels;
  tensor_record.target_id = "installed-paired";
  tensor_record.geometry =
      jyppx::liuding::data::image::GeometryTrace({2, 2}).resized({1, 1});
  const jyppx::liuding::data::image::TensorImageDataset tensor_source(
      {std::move(tensor_record)});
  const auto tensor_sample = tensor_source.get(0);
  if (tensor_sample.image.data_ptr() != image.tensor.data_ptr() ||
      tensor_sample.image_path != "installed-tensor" ||
      tensor_sample.width != 1 || tensor_sample.height != 1 ||
      !tensor_sample.paired_target ||
      tensor_sample.paired_target->data_ptr() != image.tensor.data_ptr() ||
      tensor_sample.target_path != "installed-paired" ||
      !tensor_sample.geometry ||
      tensor_sample.geometry->source_extent().width != 2)
    return 1;
  const auto encoded =
      jyppx::liuding::data::image::encode_image(image.tensor, image.pixels);
  const auto trace = jyppx::liuding::data::image::GeometryTrace({20, 10})
                         .resized({40, 20})
                         .padded(0, 10, 0, 10);
  const auto restored = trace.inverse_box({0, 10, 40, 30});
  constexpr std::array<unsigned char, 68> mask_png{
      137, 80,  78,  71,  13, 10, 26, 10, 0,   0,   0,   13,  73,  72,
      68,  82,  0,   0,   0,  1,  0,  0,  0,   1,   16,  0,   0,   0,
      0,   106, 238, 71,  22, 0,  0,  0,  11,  73,  68,  65,  84,  120,
      218, 99,  248, 255, 31, 0,  3,  0,  1,   255, 111, 129, 171, 182,
      0,   0,   0,   0,   73, 69, 78, 68, 174, 66,  96,  130};
  const auto mask = jyppx::liuding::data::image::decode_mask(
      std::as_bytes(std::span(mask_png)));
  using namespace jyppx::liuding::data::image;
  VisionSample sample;
  sample.image = image.tensor;
  sample.pixels = image.pixels;
  sample.width = image.width;
  sample.height = image.height;
  sample.mask_target = mask;
  RandomGeometryOptions geometry_options;
  geometry_options.resize = ImageExtent{3, 2};
  const SampleRandom random(
      {.dataset_id = "installed-fixture-v1",
       .sample_id = "a.png",
       .transform_id = joint_transform_identity(geometry_options, {})});
  const auto transformed = transform_sample(
      sample, sample_geometry({1, 1}, geometry_options, random));
  JointTransformOptions cubic_options;
  cubic_options.interpolation = ImageInterpolation::bicubic;
  const auto cubic = transform_sample(
      sample, sample_geometry({1, 1}, geometry_options, random), cubic_options);
  if (cubic.width != 3 || cubic.height != 2 ||
      !torch::equal(cubic.mask_target->ids, transformed.mask_target->ids))
    return 1;
  VisionAnnotationRecord segments;
  segments.width = segments.height = 1;
  segments.panoptic_mask_path = "source.png";
  segments.objects.push_back({.source_id = 65535, .area = 1});
  auto panoptic = mask;
  panoptic.encoding = MaskEncoding::panoptic_rgb;
  panoptic.void_id = 0;
  const auto bound = bind_panoptic_mask(segments, panoptic, 1);
  sample.categories = std::make_shared<const std::vector<VisionCategory>>(
      std::vector<VisionCategory>{{1, "region"}});
  auto regions = std::make_shared<VisionAnnotationRecord>();
  regions->width = regions->height = 1;
  regions->objects.push_back({.rle = ImageMaskRle{1, 1, {0, 1}}});
  sample.annotation = regions;
  const auto rich = transform_sample(
      sample, sample_geometry({1, 1}, geometry_options, random));
  const auto batch = collate_images(
      std::array{rich}, std::array{ImageBatchKey{"installed-image", 7, 1}},
      {.layout = ImageBatchLayout::pad,
       .padded_extent = ImageExtent{4, 3},
       .mask_padding_id = -1});
  const auto numeric = transform_image_batch(batch);
  const auto content_identity = image_batch_content_identity(batch);
  const std::array<std::vector<ImageGeometryStep>, 1> batch_steps{
      {{{.kind = ImageGeometryKind::horizontal_flip}}}};
  ImageBatchGeometryOptions batch_geometry_options;
  batch_geometry_options.joint.mask_padding_id = -1;
  batch_geometry_options.output_padded_extent = ImageExtent{4, 3};
  const auto geometric = transform_image_batch_geometry(batch, batch_steps,
                                                        batch_geometry_options);
  jyppx::liuding::data::DataPipelineOptions pipeline_options;
  pipeline_options.pipeline_id = "installed-image-precommit";
  pipeline_options.dataset_version = "installed-image/v1";
  pipeline_options.dataset_size = 1;
  pipeline_options.batch_factory_id = "installed-collate/v1";
  pipeline_options.schema = image_batch_schema(numeric, torch::kCPU);
  pipeline_options.consumer_transform =
      jyppx::liuding::data::BatchConsumerTransform{
          .id = "image-numeric/" +
                image_batch_identity({}, ImageBatchTransformOptions{}),
          .input_schema = image_batch_schema(batch, torch::kCPU),
          .apply = [](const c10::IValue& value,
                      const jyppx::liuding::data::BatchConsumerContext&,
                      std::stop_token stop) {
            return image_batch_value(
                transform_image_batch(image_batch_from_value(value), {}, stop));
          }};
  jyppx::liuding::data::DataPipeline pipeline(
      std::make_shared<jyppx::liuding::data::samplers::SequentialSampler>(1),
      pipeline_options,
      [batch](std::span<const std::size_t>,
              const jyppx::liuding::data::DataWorkerContext&,
              std::stop_token) { return image_batch_value(batch); });
  pipeline.set_epoch(0);
  const auto envelope = pipeline.next();
  if (!envelope || !envelope->transformed_memory ||
      pipeline.state().next_cursor != 1 ||
      !torch::equal(image_batch_from_value(envelope->value).images,
                    numeric.images))
    return 1;
  const auto decoded = image_batch_from_value(envelope->value);
  jyppx::liuding::data::validate_batch_value(
      image_batch_value(decoded), image_batch_schema(decoded, torch::kCPU));
  return index.images.size() == 1U && index.images[0].objects.size() == 1U &&
                 index.images[0].objects[0].box->x_max == 20 &&
                 manifest.records().size() == 1U && image.width == 1U &&
                 image.height == 1U &&
                 encoded.format ==
                     jyppx::liuding::data::image::ImageFormat::png &&
                 !encoded.bytes.empty() &&
                 image.tensor.item<std::uint8_t>() == 127 &&
                 image.pixels.color ==
                     jyppx::liuding::data::image::PixelColorModel::gray &&
                 restored && restored->x_max == 20 && restored->y_max == 10 &&
                 !trace.inverse_point({0, 0}) &&
                 mask.ids.scalar_type() == torch::kInt64 &&
                 mask.ids.dim() == 2 &&
                 mask.ids.item<std::int64_t>() == 65535 &&
                 bound.panoptic_mask_materialized &&
                 bound.objects[0].mask_pixel_area == 1 &&
                 rich.annotation->objects[0].mask_pixel_area == 6 &&
                 rich.annotation->objects[0].rle->width == 3 &&
                 geometric.items[0].key.id == "installed-image" &&
                 content_identity.size() == 64 &&
                 geometric.items[0].annotation->objects[0].rle->width == 4 &&
                 decoded.images.sizes() == torch::IntArrayRef({1, 1, 3, 4}) &&
                 decoded.items[0].key.occurrence == 7 &&
                 decoded.items[0].annotation->objects[0].rle->width == 4 &&
                 decoded.items[0].annotation->objects[0].mask_pixel_area == 6 &&
                 std::abs(decoded.images[0][0][0][0].item<float>() -
                          127.0F / 255) < 1e-6F &&
                 transformed.width == 3 && transformed.height == 2 &&
                 transformed.image[0][1][2].item<int>() == 127 &&
                 transformed.mask_target->ids[1][2].item<std::int64_t>() ==
                     65535
             ? 0
             : 1;
#elif defined(JYPPX_CONSUMER_VISION_RENDER)
  using namespace jyppx::liuding::data::image;
  using namespace jyppx::liuding::visualization;
  PixelSpec spec;
  const auto image = torch::zeros({3, 2, 2}, torch::kUInt8);
  VisionAnnotationRecord record{.width = 2, .height = 2};
  record.objects.push_back({.category = 0, .box = ImageBox{0, 0, 2, 2}});
  const auto rendered = render_annotations(image, spec, record);
  return rendered.pixels.sizes() == torch::IntArrayRef({2, 2, 3}) ? 0 : 1;
#elif defined(JYPPX_CONSUMER_NN)
  const auto loss =
      jyppx::liuding::nn::functional::label_smoothing_cross_entropy(
          torch::tensor({{2.0, -1.0}}, torch::kFloat64),
          torch::tensor({0}, torch::kLong), 0.1);
  return loss.numel() == 1 && std::isfinite(loss.item<double>()) ? 0 : 1;
#elif defined(JYPPX_CONSUMER_OPTIM)
  auto parameter = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
  jyppx::liuding::optim::SgdOptimizerAdapter optimizer(
      "optimizer.consumer",
      {{.group_id = "main",
        .parameters = {{.parameter_id = "weight", .parameter = parameter}},
        .options = {.learning_rate = 0.01}}});
  return optimizer.optimizer_id() == "optimizer.consumer" &&
                 optimizer.parameter_count() == 1
             ? 0
             : 1;
#elif defined(JYPPX_CONSUMER_TRAINING)
  jyppx::liuding::training::Trainer trainer(
      {.trainer_id = "trainer.consumer",
       .checkpoint = {.max_participants = 64,
                      .max_rollback_bytes = 3ULL << 30U}});
  return trainer.phase() == jyppx::liuding::training::TrainerPhase::created &&
                 trainer.options().trainer_id == "trainer.consumer" &&
                 trainer.options().checkpoint.max_participants == 64 &&
                 trainer.options().checkpoint.max_rollback_bytes ==
                     (3ULL << 30U)
             ? 0
             : 1;
#elif defined(JYPPX_CONSUMER_EXPERIMENT)
  return jyppx::liuding::experiment::config::is_secret_reference(
             "secret://env/consumer/token")
             ? 0
             : 1;
#elif defined(JYPPX_CONSUMER_MODELS)
  return jyppx::liuding::models::list_models().empty() ? 1 : 0;
#elif defined(JYPPX_CONSUMER_QUANTIZATION)
  jyppx::liuding::quantization::validate_format_descriptor({});
  return 0;
#elif defined(JYPPX_CONSUMER_SERVING)
  return jyppx::liuding::serving::serving_preview_manifest_json().empty() ? 1
                                                                          : 0;
#elif defined(JYPPX_CONSUMER_ALL)
  return std::string(jyppx::liuding::version()).empty() ? 1 : 0;
#elif defined(JYPPX_CONSUMER_SDK)
  return !std::string(jyppx::liuding::version()).empty() &&
                 jyppx::liuding::parse_device("cpu").is_cpu()
             ? 0
             : 1;
#endif
}
