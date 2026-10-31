// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "transfer_recipe.hpp"

#if defined(JYPPX_LIUDING_CLI_IMAGE_RECIPE)
#include "recipe_support.hpp"

#include <jyppx/liuding/models/heads.hpp>
#include <jyppx/liuding/models/registry.hpp>

#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/linear.h>
#include <torch/utils.h>

#include <array>
#include <fstream>
#include <utility>

namespace jyppx::liuding::cli {
namespace {
using namespace recipe_support;

std::string text(const Json& value) {
  require_recipe(value.is_string(), "Weight descriptor requires a string");
  return value.get<std::string>();
}

class TransferModel final : public torch::nn::Module {
 public:
  TransferModel(nn::ModelAdapter backbone, std::int64_t features,
                std::int64_t classes, bool frozen, bool bn_eval)
      : backbone_(std::move(backbone)), frozen_(frozen), bn_eval_(bn_eval) {
    register_module("backbone", backbone_.state().shared_module());
    head_ = register_module(
        "head", std::make_shared<torch::nn::LinearImpl>(features, classes));
    for (auto parameter : backbone_.state().parameters())
      parameter.set_requires_grad(!frozen_);
    train();
  }

  void train(bool on = true) override {
    torch::nn::Module::train(on);
    if (frozen_) {
      backbone_.state().eval();
    } else if (bn_eval_) {
      for (const auto& module : backbone_.state().module().modules()) {
        if (std::dynamic_pointer_cast<torch::nn::BatchNorm1dImpl>(module) ||
            std::dynamic_pointer_cast<torch::nn::BatchNorm2dImpl>(module) ||
            std::dynamic_pointer_cast<torch::nn::BatchNorm3dImpl>(module))
          module->eval();
      }
    }
  }

  Tensor forward(const Tensor& image) {
    Tensor embedding;
    const auto invoke = [&] {
      return backbone_.invoke({.positional = {image}, .keyword = {}})
          .value.toTensor();
    };
    if (frozen_) {
      torch::NoGradGuard guard;
      embedding = invoke();
    } else {
      embedding = invoke();
    }
    return head_->forward(embedding);
  }

 private:
  nn::ModelAdapter backbone_;
  std::shared_ptr<torch::nn::LinearImpl> head_;
  bool frozen_;
  bool bn_eval_;
};
}  // namespace

bool is_transfer_recipe(const Json& config) {
  return config.is_object() &&
         config.value("recipe", Json{}) == std::string(kTransferRecipe);
}

TransferInput inspect_transfer_input(const Json& config) {
  const auto descriptor_path =
      std::filesystem::u8path(config.at("weights_metadata").get<std::string>());
  // Validate paths before opening; callers keep both inputs quiescent.
  const auto before = file_digest(descriptor_path, 1U << 20U);
  auto descriptor = parse_json(read_text(descriptor_path, 1U << 20U));
  require_recipe(file_digest(descriptor_path, 1U << 20U).hex == before.hex,
                 "Weight descriptor changed while reading");
  fields(descriptor,
         {"schema_version", "id", "model_name", "source_uri", "source_revision",
          "license_spdx", "license_uri", "sha256", "size_bytes",
          "source_num_classes", "state_mapping_version", "reference_uri",
          "reference_revision", "position_antialias", "image"});
  require_recipe(descriptor.at("position_antialias").is_boolean(),
                 "position_antialias must be an explicit boolean");
  require_recipe(unsigned_number(descriptor.at("schema_version"), 1) == 1,
                 "Unsupported weight descriptor version");
  const auto name = text(descriptor.at("model_name"));
  require_recipe(name == "resnet50" || name == "vit_small_patch14_dinov2",
                 "Transfer recipe currently supports resnet50 and "
                 "vit_small_patch14_dinov2");
  require_recipe(config.at("model") == name,
                 "Weight descriptor and configured architecture differ");
  const auto classes = static_cast<std::int64_t>(
      unsigned_number(descriptor.at("source_num_classes"), 1000000));
  require_recipe((name == "resnet50" && classes > 0) ||
                     (name == "vit_small_patch14_dinov2" && classes == 0),
                 "Source head must match the original weight topology");
  const auto& image = descriptor.at("image");
  fields(image, {"input_size", "resize", "resize_shorter_side", "interpolation",
                 "color", "mean", "std"});
  require_recipe(
      image.at("input_size").is_array() && image.at("input_size").size() == 2,
      "Weight input_size requires [height,width]");
  const auto height = static_cast<std::uint32_t>(
      unsigned_number(image.at("input_size").at(0), 1024));
  const auto width = static_cast<std::uint32_t>(
      unsigned_number(image.at("input_size").at(1), 1024));
  require_recipe(height == width && height == config.at("image_size"),
                 "Transfer image_size must match the square weight input");
  require_recipe(image.at("color") == "rgb" &&
                     image.at("interpolation") == "bicubic" &&
                     image.at("resize") == "shortest_edge_center_crop",
                 "Transfer preprocessing requires explicit RGB, antialiased "
                 "bicubic shortest-edge resize and center crop");
  const auto vector = [&](const char* key) {
    const auto& values = image.at(key);
    require_recipe(values.is_array() && values.size() == 3,
                   "Weight normalization requires three numeric values");
    std::vector<double> result;
    for (const auto& value : values) {
      require_recipe(value.is_number(), "Weight normalization must be numeric");
      result.push_back(value.get<double>());
    }
    return result;
  };
  models::WeightRecord record{
      .id = text(descriptor.at("id")),
      .model_name = name,
      .source_uri = text(descriptor.at("source_uri")),
      .license_spdx = text(descriptor.at("license_spdx")),
      .license_uri = text(descriptor.at("license_uri")),
      .availability = models::WeightAvailability::pinned_external,
      .artifact =
          ArtifactRef{.digest = {.hex = text(descriptor.at("sha256"))},
                      .size_bytes = unsigned_number(descriptor.at("size_bytes"),
                                                    kArtifactLimit),
                      .logical_name = "source-weights.safetensors",
                      .media_type = "application/x-safetensors",
                      .producer = text(descriptor.at("state_mapping_version"))},
      .status_reason =
          "Explicit local input; provenance is declared by caller, "
          "not authenticated by metadata validation."};
  record.input_spec = models::WeightInputSpec{
      .source_revision = text(descriptor.at("source_revision")),
      .state_mapping_version = text(descriptor.at("state_mapping_version")),
      .reference_uri = text(descriptor.at("reference_uri")),
      .reference_revision = text(descriptor.at("reference_revision")),
      .image = {.input_size = {height, width},
                .resize = models::PreprocessResize::shortest_edge_center_crop,
                .resize_shorter_side = static_cast<std::uint32_t>(
                    unsigned_number(image.at("resize_shorter_side"), 65536)),
                .interpolation = models::PreprocessInterpolation::bicubic,
                .color = models::PreprocessColor::rgb,
                .mean = vector("mean"),
                .std = vector("std")},
      .categories = std::nullopt};
  models::validate_weight_record(record);
  const auto identity = models::weight_input_identity(record);
  const auto path =
      std::filesystem::u8path(config.at("weights_file").get<std::string>());
  const auto actual_digest = file_digest(path);
  require_recipe(
      std::filesystem::file_size(path) == record.artifact->size_bytes &&
          actual_digest.hex == record.artifact->digest.hex,
      "Local weights do not match the declared SHA-256 and size");
  return {std::move(record), std::move(descriptor), path, classes, identity};
}

std::shared_ptr<nn::ModelAdapter> make_transfer_model(
    const TransferInput& input, const Json& config, std::int64_t classes,
    bool load_pretrained) {
  models::ModelOptions options;
  options.resnet.num_classes = input.source_classes;
  options.dinov2.num_classes = input.source_classes;
  options.dinov2.position_antialias =
      input.descriptor.at("position_antialias").get<bool>();
  auto backbone = models::create_model(input.record.model_name, options);
  if (load_pretrained) {
    std::vector<std::byte> encoded(input.record.artifact->size_bytes);
    std::ifstream stream(input.path, std::ios::binary);
    stream.read(reinterpret_cast<char*>(encoded.data()),
                static_cast<std::streamsize>(encoded.size()));
    require_recipe(stream && stream.peek() == std::ifstream::traits_type::eof(),
                   "Weights changed while reading");
    require_recipe(
        models::load_weights(backbone, input.record, encoded).applied,
        "Strict pretrained weight import failed");
  }
  const auto features = models::classifier_info(backbone).embedding_features;
  models::reset_classifier(backbone, 0);
  auto concrete =
      std::make_shared<TransferModel>(std::move(backbone), features, classes,
                                      config.at("backbone_mode") == "frozen",
                                      config.at("batch_norm") == "eval");
  return std::make_shared<nn::ModelAdapter>(nn::adapt_tensor_module(concrete));
}

Json transfer_state(const nn::ModelAdapter& model) {
  Sha256 backbone, head;
  std::uint64_t trainable = 0, frozen = 0, buffers = 0;
  const auto append = [&](const auto& entries, std::string_view kind) {
    for (const auto& entry : entries) {
      const auto tensor = entry.value().detach().to(torch::kCPU).contiguous();
      auto& hash = entry.key().starts_with("backbone.") ? backbone : head;
      hash.update(bytes(std::string(kind) + entry.key() + "\n"));
      hash.update({static_cast<const std::byte*>(tensor.const_data_ptr()),
                   tensor.nbytes()});
    }
  };
  append(model.state().named_parameters(), "parameter.");
  append(model.state().named_buffers(), "buffer.");
  for (const auto& parameter : model.state().parameters()) {
    auto& count = parameter.requires_grad() ? trainable : frozen;
    count += static_cast<std::uint64_t>(parameter.numel());
  }
  for (const auto& buffer : model.state().buffers())
    buffers += static_cast<std::uint64_t>(buffer.numel());
  return {{"backbone_sha256", sha256_digest_hex(backbone.finish())},
          {"head_sha256", sha256_digest_hex(head.finish())},
          {"trainable_parameters", trainable},
          {"frozen_parameters", frozen},
          {"buffer_elements", buffers}};
}
}  // namespace jyppx::liuding::cli
#endif
