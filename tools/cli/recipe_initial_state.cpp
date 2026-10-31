// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "recipe_initial_state.hpp"

#if defined(JYPPX_LIUDING_CLI_IMAGE_RECIPE)
#include "recipe_support.hpp"

#include <jyppx/liuding/interop/model_weight_import.hpp>
#include <jyppx/liuding/interop/safetensors.hpp>

#include <fstream>
#include <map>

namespace jyppx::liuding::cli {
namespace {
using namespace recipe_support;
constexpr std::string_view kFormat = "jyppx.liuding.recipe.initial-state/v1";

std::map<std::string, std::string> metadata(const Json& config,
                                            const Json& classes) {
  return {{"format", std::string(kFormat)},
          {"recipe", config.at("recipe").get<std::string>()},
          {"architecture", config.value("model", "resnet18")},
          {"classes", classes.dump()}};
}

interop::SafeTensorsLimits tensor_limits() {
  interop::SafeTensorsLimits result;
  result.max_file_bytes = kArtifactLimit;
  result.max_payload_bytes = kArtifactLimit;
  result.max_tensor_bytes = kArtifactLimit;
  return result;
}
}  // namespace

std::optional<RecipeInitialState> inspect_recipe_initial_state(
    const Json& config, const Json& classes) {
  if (!config.contains("initial_state_file")) return std::nullopt;
  const auto path = std::filesystem::u8path(
      config.at("initial_state_file").get<std::string>());
  const auto actual = file_digest(path);
  require_recipe(
      actual.hex == config.at("initial_state_sha256").get<std::string>(),
      "Initial state does not match its declared SHA-256");
  const auto size = std::filesystem::file_size(path);
  require_recipe(size <= kArtifactLimit, "Initial state exceeds 256 MiB");
  std::vector<std::byte> encoded(size);
  std::ifstream stream(path, std::ios::binary);
  stream.read(reinterpret_cast<char*>(encoded.data()),
              static_cast<std::streamsize>(encoded.size()));
  require_recipe(stream && stream.peek() == std::ifstream::traits_type::eof() &&
                     sha256_hex(encoded) == actual.hex,
                 "Initial state changed while reading");
  const interop::SafeTensorsView view(encoded, tensor_limits());
  require_recipe(
      view.directory().metadata == metadata(config, classes),
      "Initial state recipe, architecture or ordered classes differ");
  Json descriptor{{"format", kFormat},
                  {"sha256", actual.hex},
                  {"size_bytes", encoded.size()},
                  {"tensors", view.directory().tensors.size()},
                  {"recipe", config.at("recipe")},
                  {"architecture", config.value("model", "resnet18")},
                  {"classes", classes}};
  return RecipeInitialState{path, actual, std::move(descriptor),
                            std::move(encoded)};
}

Json recipe_model_state_identity(const nn::ModelAdapter& model) {
  Sha256 hash;
  std::uint64_t count = 0, payload = 0;
  const auto add = [&](const auto& entries, std::string_view kind) {
    // Sort independently of module registration order for a portable identity.
    std::map<std::string, Tensor> sorted;
    for (const auto& entry : entries)
      sorted.emplace(entry.key(), entry.value());
    for (const auto& [name, value] : sorted) {
      const auto tensor = value.detach().to(torch::kCPU).contiguous();
      hash.update(bytes(Json{{"kind", kind},
                             {"name", name},
                             {"dtype", c10::toString(tensor.scalar_type())},
                             {"shape", tensor.sizes().vec()}}
                            .dump() +
                        "\n"));
      hash.update({static_cast<const std::byte*>(tensor.const_data_ptr()),
                   tensor.nbytes()});
      ++count;
      payload += tensor.nbytes();
    }
  };
  add(model.state().named_parameters(), "parameter");
  add(model.state().named_buffers(), "buffer");
  return {{"schema_version", 1},
          {"state_sha256", sha256_digest_hex(hash.finish())},
          {"tensors", count},
          {"tensor_bytes", payload}};
}

Json apply_recipe_initial_state(nn::ModelAdapter& model,
                                RecipeInitialState& input) {
  interop::ModelWeightImportOptions options;
  options.tensor_limits = tensor_limits();
  options.max_staging_bytes = 3 * kArtifactLimit;
  const auto applied = interop::import_model_weights(
      model.state(), input.encoded, input.digest, options);
  require_recipe(applied.applied && applied.plan.ready &&
                     applied.plan.missing.empty() &&
                     applied.plan.unexpected.empty(),
                 "Initial state must match every parameter and buffer exactly");
  // Release the full input before optimizer states and training batches exist.
  std::vector<std::byte>().swap(input.encoded);
  return {{"method", "strict_complete_safetensors/v1"},
          {"source", input.descriptor},
          {"applied_tensors", applied.applied_tensors},
          {"charged_staging_bytes", applied.charged_staging_bytes},
          {"model", recipe_model_state_identity(model)}};
}

std::vector<std::byte> encode_recipe_initial_state(
    const nn::ModelAdapter& model, const Json& config, const Json& classes) {
  std::vector<Tensor> owners;
  std::vector<interop::SafeTensorSource> sources;
  const auto add = [&](const auto& entries) {
    for (const auto& entry : entries) {
      owners.push_back(entry.value().detach().to(torch::kCPU).contiguous());
      const auto& tensor = owners.back();
      require_recipe(tensor.scalar_type() == torch::kFloat32 ||
                         tensor.scalar_type() == torch::kInt64,
                     "Recipe initial state requires FP32 or int64 tensors");
      std::vector<std::uint64_t> shape;
      for (const auto extent : tensor.sizes())
        shape.push_back(static_cast<std::uint64_t>(extent));
      sources.push_back(
          {.name = entry.key(),
           .dtype = tensor.scalar_type() == torch::kFloat32
                        ? interop::SafeTensorDtype::f32
                        : interop::SafeTensorDtype::i64,
           .shape = std::move(shape),
           .data = {static_cast<const std::byte*>(tensor.const_data_ptr()),
                    tensor.nbytes()}});
    }
  };
  add(model.state().named_parameters());
  add(model.state().named_buffers());
  return interop::serialize_safetensors(sources, metadata(config, classes),
                                        tensor_limits());
}
}  // namespace jyppx::liuding::cli
#endif
