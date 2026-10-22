// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#if defined(JYPPX_CONSUMER_FOUNDATION)
#include <jyppx/foundation/version.hpp>
#elif defined(JYPPX_CONSUMER_CORE)
#include <jyppx/core/device.hpp>
#elif defined(JYPPX_CONSUMER_INTEROP)
#include <jyppx/interop/checkpoint_manifest.hpp>
#elif defined(JYPPX_CONSUMER_DATA)
#include <jyppx/data/samplers/sequential_sampler.hpp>
#elif defined(JYPPX_CONSUMER_DATA_IMAGE)
#include <jyppx/data_image/annotation_files.hpp>
#include <jyppx/data_image/image_decoder.hpp>
#include <jyppx/data_image/manifests.hpp>
#elif defined(JYPPX_CONSUMER_NN)
#include <jyppx/nn/loss/label_smoothing_cross_entropy.hpp>
#elif defined(JYPPX_CONSUMER_OPTIM)
#include <jyppx/optim/sgd_adapter.hpp>
#elif defined(JYPPX_CONSUMER_TRAINING)
#include <jyppx/training/trainer.hpp>
#elif defined(JYPPX_CONSUMER_EXPERIMENT)
#include <jyppx/experiment/config.hpp>
#elif defined(JYPPX_CONSUMER_MODELS)
#include <jyppx/models/registry.hpp>
#elif defined(JYPPX_CONSUMER_QUANTIZATION)
#include <jyppx/quantization/quantization.hpp>
#elif defined(JYPPX_CONSUMER_SERVING)
#include <jyppx/serving/serving.hpp>
#elif defined(JYPPX_CONSUMER_ALL)
#include <jyppx/foundation/version.hpp>
#elif defined(JYPPX_CONSUMER_SDK)
#include <jyppx/jyppx.hpp>
#else
#error "Exactly one installed component consumer must be selected"
#endif

#include <jyppx/foundation/capability.hpp>

#if defined(JYPPX_CONSUMER_INTEROP) && JYPPX_EXPECT_COMPONENT_NN
#include <jyppx/foundation/sha256.hpp>
#include <jyppx/interop/model_weight_import.hpp>
#include <jyppx/nn/model_adapter.hpp>

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
  const std::array<jyppx::interop::SafeTensorSource, 1> tensors{
      {{.name = "weight",
        .dtype = jyppx::interop::SafeTensorDtype::f64,
        .shape = {2U},
        .data = std::as_bytes(std::span(values))}}};
  const auto bytes = jyppx::interop::serialize_safetensors(tensors);
  jyppx::nn::ModelStateView view(module);
  const auto result = jyppx::interop::import_model_weights(
      view, bytes, {.hex = jyppx::sha256_hex(bytes)});
  return result.applied && result.applied_tensors == 1U &&
         weight.unsafeGetTensorImpl() == identity && weight.requires_grad() &&
         torch::equal(weight, torch::tensor({1.25, -2.5}, torch::kFloat64));
}
#endif

bool BuildCapabilityInventoryMatchesPackage() {
  constexpr std::array<std::pair<std::string_view, bool>, 12U> kExpected{{
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
  }};
  const auto manifest = jyppx::current_capability_manifest();
  if (manifest.features.size() != kExpected.size()) return false;
  for (std::size_t index = 0; index < kExpected.size(); ++index) {
    const auto& [component, enabled] = kExpected[index];
    const auto& feature = manifest.features[index];
    const bool preview = component == "models" || component == "quantization" ||
                         component == "serving" || component == "data_image";
    const auto expected_status =
        enabled ? (preview ? jyppx::CapabilityStatus::experimental
                           : jyppx::CapabilityStatus::available)
                : jyppx::CapabilityStatus::unsupported;
    if (feature.feature_id != "jyppx.component." + std::string(component) ||
        feature.status != expected_status ||
        (feature.implementation == jyppx::CapabilityImplementation::native) !=
            enabled) {
      return false;
    }
  }
  return !jyppx::current_capability_manifest_json().empty();
}

}  // namespace

int main() {
  if (!BuildCapabilityInventoryMatchesPackage()) return 1;
#if defined(JYPPX_CONSUMER_FOUNDATION)
  const auto fingerprint = jyppx::current_build_fingerprint();
  return !std::string(jyppx::version()).empty() &&
                 !fingerprint.git_commit.empty()
             ? 0
             : 1;
#elif defined(JYPPX_CONSUMER_CORE)
  return jyppx::parse_device("cpu").is_cpu() ? 0 : 1;
#elif defined(JYPPX_CONSUMER_INTEROP)
  const auto capability = jyppx::interop::make_checkpoint_manifest_capability();
  if (capability.feature_id != "interop.checkpoint_manifest") return 1;
#if JYPPX_EXPECT_COMPONENT_NN
  return InstalledModelWeightImportAppliesVerifiedState() ? 0 : 1;
#else
  return 0;
#endif
#elif defined(JYPPX_CONSUMER_DATA)
  const auto indices = jyppx::data::samplers::SequentialSampler(3).indices();
  return indices == std::vector<std::size_t>({0, 1, 2}) ? 0 : 1;
#elif defined(JYPPX_CONSUMER_DATA_IMAGE)
  const auto metadata =
      jyppx::data::image::parse_yolo_dataset_yaml("names: [cat]");
  jyppx::data::image::AnnotationReadOptions options;
  options.categories = metadata.categories;
  options.image_path = "a.jpg";
  options.width = 20;
  options.height = 10;
  const auto index = jyppx::data::image::parse_annotations(
      "0 .5 .5 1 1", jyppx::data::image::AnnotationFormat::yolo_detection,
      options);
  jyppx::data::image::ManifestReadOptions manifest_options;
  manifest_options.categories = metadata.categories;
  const auto manifest = jyppx::data::image::parse_manifest(
      "path,labels\na.jpg,cat\n", jyppx::data::image::ManifestFormat::csv,
      jyppx::data::image::ManifestTask::classification, manifest_options);
  constexpr std::array<unsigned char, 12> pgm{'P',  '5', '\n', '1', ' ',  '1',
                                              '\n', '2', '5',  '5', '\n', 127};
  const auto image =
      jyppx::data::image::decode_image(std::as_bytes(std::span(pgm)));
  return index.images.size() == 1U && index.images[0].objects.size() == 1U &&
                 index.images[0].objects[0].box->x_max == 20 &&
                 manifest.records().size() == 1U && image.width == 1U &&
                 image.height == 1U && image.tensor.item<std::uint8_t>() == 127
             ? 0
             : 1;
#elif defined(JYPPX_CONSUMER_NN)
  const auto loss = jyppx::nn::functional::label_smoothing_cross_entropy(
      torch::tensor({{2.0, -1.0}}, torch::kFloat64),
      torch::tensor({0}, torch::kLong), 0.1);
  return loss.numel() == 1 && std::isfinite(loss.item<double>()) ? 0 : 1;
#elif defined(JYPPX_CONSUMER_OPTIM)
  auto parameter = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
  jyppx::optim::SgdOptimizerAdapter optimizer(
      "optimizer.consumer",
      {{.group_id = "main",
        .parameters = {{.parameter_id = "weight", .parameter = parameter}},
        .options = {.learning_rate = 0.01}}});
  return optimizer.optimizer_id() == "optimizer.consumer" &&
                 optimizer.parameter_count() == 1
             ? 0
             : 1;
#elif defined(JYPPX_CONSUMER_TRAINING)
  jyppx::training::Trainer trainer({.trainer_id = "trainer.consumer"});
  return trainer.phase() == jyppx::training::TrainerPhase::created &&
                 trainer.options().trainer_id == "trainer.consumer"
             ? 0
             : 1;
#elif defined(JYPPX_CONSUMER_EXPERIMENT)
  return jyppx::experiment::config::is_secret_reference(
             "secret://env/consumer/token")
             ? 0
             : 1;
#elif defined(JYPPX_CONSUMER_MODELS)
  return jyppx::models::list_models().empty() ? 1 : 0;
#elif defined(JYPPX_CONSUMER_QUANTIZATION)
  jyppx::quantization::validate_format_descriptor({});
  return 0;
#elif defined(JYPPX_CONSUMER_SERVING)
  return jyppx::serving::serving_preview_manifest_json().empty() ? 1 : 0;
#elif defined(JYPPX_CONSUMER_ALL)
  return std::string(jyppx::version()).empty() ? 1 : 0;
#elif defined(JYPPX_CONSUMER_SDK)
  return !std::string(jyppx::version()).empty() &&
                 jyppx::parse_device("cpu").is_cpu()
             ? 0
             : 1;
#endif
}
