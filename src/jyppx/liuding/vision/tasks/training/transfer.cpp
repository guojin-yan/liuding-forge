// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "../workflow/artifact_io.hpp"

#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/interop/model_weight_import.hpp>
#include <jyppx/liuding/interop/safetensors.hpp>
#include <jyppx/liuding/vision/tasks/training/transfer.hpp>

#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>
#include <set>
namespace jyppx::liuding::vision::tasks {
namespace {
using J = nlohmann::json;
namespace io = interop;
bool Under(std::string_view name, std::string_view prefix) {
  return name == prefix ||
         (name.size() > prefix.size() && name.starts_with(prefix) &&
          name[prefix.size()] == '.');
}
bool Any(std::string_view name, const std::vector<std::string>& prefixes) {
  return std::any_of(prefixes.begin(), prefixes.end(),
                     [&](const auto& p) { return Under(name, p); });
}
io::SafeTensorDtype Dtype(torch::ScalarType type) {
  switch (type) {
    case torch::kBool:
      return io::SafeTensorDtype::boolean;
    case torch::kUInt8:
      return io::SafeTensorDtype::u8;
    case torch::kInt8:
      return io::SafeTensorDtype::i8;
    case torch::kInt16:
      return io::SafeTensorDtype::i16;
    case torch::kInt32:
      return io::SafeTensorDtype::i32;
    case torch::kInt64:
      return io::SafeTensorDtype::i64;
    case torch::kFloat16:
      return io::SafeTensorDtype::f16;
    case torch::kBFloat16:
      return io::SafeTensorDtype::bf16;
    case torch::kFloat32:
      return io::SafeTensorDtype::f32;
    case torch::kFloat64:
      return io::SafeTensorDtype::f64;
    default:
      throw CompatibilityError(
          "Vision weight dtype has no native safetensors encoding");
  }
}
std::map<std::string, Tensor> State(const nn::ModelAdapter& model) {
  std::map<std::string, Tensor> state;
  for (const auto& p : model.state().named_parameters())
    state.emplace(p.key(), p.value());
  for (const auto& p : model.state().named_buffers())
    if (!state.emplace(p.key(), p.value()).second)
      throw InvalidArgument("Duplicate model state key");
  return state;
}
void Prefixes(const std::vector<std::string>& prefixes,
              const std::map<std::string, Tensor>& state) {
  std::set<std::string> seen;
  for (const auto& p : prefixes) {
    if (p.empty() || p.size() > 4096 || !seen.insert(p).second ||
        !std::any_of(state.begin(), state.end(),
                     [&](const auto& v) { return Under(v.first, p); }))
      throw InvalidArgument("Missing, duplicate or empty model subtree: " + p);
  }
}
}  // namespace
std::string save_vision_weights(const nn::ModelAdapter& model,
                                const std::filesystem::path& path,
                                std::map<std::string, std::string> metadata,
                                std::uint64_t maximum) {
  if (!maximum || maximum > (4ULL << 30U))
    throw InvalidArgument("Invalid native weight byte bound");
  auto state = State(model);
  std::vector<Tensor> copies;
  std::vector<io::SafeTensorSource> sources;
  std::uint64_t total = 0;
  copies.reserve(state.size());
  sources.reserve(state.size());
  for (const auto& [name, value] : state) {
    if (value.nbytes() > maximum - total)
      throw ResourceExhausted("Native weights exceed aggregate byte bound");
    total += value.nbytes();
    copies.push_back(value.detach().to(torch::kCPU).contiguous());
    const auto& cpu = copies.back();
    std::vector<std::uint64_t> shape(cpu.sizes().begin(), cpu.sizes().end());
    sources.push_back({name,
                       Dtype(cpu.scalar_type()),
                       std::move(shape),
                       {static_cast<const std::byte*>(cpu.const_data_ptr()),
                        static_cast<std::size_t>(cpu.nbytes())}});
  }
  metadata["format"] = "liuding.vision.weights/v1";
  if (!metadata.contains("role")) metadata["role"] = "raw";
  if (metadata["role"] != "raw" && metadata["role"] != "ema" &&
      metadata["role"] != "fused")
    throw InvalidArgument("Unknown native weight role");
  io::SafeTensorsLimits limits;
  limits.max_file_bytes = maximum;
  limits.max_payload_bytes = maximum;
  limits.max_tensor_bytes = maximum;
  auto bytes = io::serialize_safetensors(sources, metadata, limits);
  const auto digest = sha256_hex(bytes);
  detail::publish_artifact(path, bytes);
  return digest;
}
std::string save_simmim_encoder(const nn::ModelAdapter& model,const std::filesystem::path& path,std::string_view architecture,std::uint64_t maximum){
  std::string encoder_name;if(architecture=="simmim_swin_tiny")encoder_name="swin_tiny_patch4_window7_224";else if(architecture=="simmim_vit_tiny")encoder_name="vit_tiny_patch16_224";else if(architecture=="simmim_vit_base")encoder_name="vit_base_patch16_224";else throw InvalidArgument("Encoder extraction requires a declared SimMIM architecture");
  auto modules=model.state().shared_module()->named_modules();if(!modules.contains("encoder"))throw CompatibilityError("SimMIM encoder subtree missing");
  auto encoder=nn::adapt_module(modules["encoder"],[](const nn::Invocation& x)->c10::IValue{return x.positional.at(0);},{.positional={ValueSchema::tensor({})},.keyword={},.allow_extra_keywords=false},ValueSchema::tensor({}));
  return save_vision_weights(encoder,path,{{"architecture",encoder_name},{"source_architecture",std::string(architecture)},{"artifact","encoder_initialization"},{"role","raw"}},maximum);
}
std::string initialize_vision(nn::ModelAdapter& model,
                              const InitializationOptions& o,
                              std::string_view architecture,
                              std::stop_token stop) {
  auto state = State(model);
  Prefixes(o.freeze_prefixes, state);
  Prefixes(o.replace_head_prefixes, state);
  J loaded = J::array(), retained = J::array(), ignored = J::array();
  std::string digest;
  std::set<std::string> ignored_keys(o.ignored_source_keys.begin(),
                                     o.ignored_source_keys.end()),
      retained_buffers(o.retain_buffer_keys.begin(),
                       o.retain_buffer_keys.end());
  if (ignored_keys.size() != o.ignored_source_keys.size() ||
      retained_buffers.size() != o.retain_buffer_keys.size())
    throw InvalidArgument("Duplicate explicit weight exception");
  const auto buffers = model.state().named_buffers();
  for (const auto& key : retained_buffers)
    if (!buffers.contains(key))
      throw InvalidArgument("Retained key is not an exact model buffer: " +
                            key);
  if (!o.weights.empty()) {
    if (o.sha256.size() != 64 ||
        !std::all_of(o.sha256.begin(), o.sha256.end(), [](unsigned char c) {
          return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        }))
      throw InvalidArgument(
          "Weight initialization requires a lowercase SHA-256");
    auto bytes = detail::read_artifact(o.weights, o.max_bytes);
    digest = sha256_hex(bytes);
    if (digest != o.sha256)
      throw CompatibilityError("Vision weight digest mismatch");
    io::SafeTensorsView view(bytes);
    const auto& directory = view.directory();
    if (auto it = directory.metadata.find("architecture");
        it != directory.metadata.end() && it->second != architecture)
      throw CompatibilityError(
          "Weight architecture metadata differs from selected model");
    std::set<std::string> targets, used_mapping;
    std::vector<io::SafeTensorSource> mapped;
    for (const auto& source : directory.tensors) {
      if (ignored_keys.contains(source.name)) {
        if (o.key_mapping.contains(source.name))
          throw InvalidArgument("Ignored source also has a mapping");
        ignored.push_back(source.name);
        continue;
      }
      auto found = o.key_mapping.find(source.name);
      const auto name =
          found == o.key_mapping.end() ? source.name : found->second;
      if (found != o.key_mapping.end()) used_mapping.insert(source.name);
      if (!targets.insert(name).second)
        throw InvalidArgument("Weight mapping is not injective");
      if (retained_buffers.contains(name))
        throw InvalidArgument("Retained buffer must not also be imported: " +
                              name);
      if (Any(name, o.replace_head_prefixes)) {
        retained.push_back(name);
        continue;
      }
      const auto target = state.find(name);
      if (target == state.end() ||
          source.dtype != Dtype(target->second.scalar_type()) ||
          source.shape !=
              std::vector<std::uint64_t>(target->second.sizes().begin(),
                                         target->second.sizes().end()))
        throw CompatibilityError(
            "Weight key/shape/dtype mismatch outside replaced head: " + name);
      mapped.push_back(
          {name, source.dtype, source.shape, view.tensor_bytes(source.name)});
      loaded.push_back(name);
    }
    if (used_mapping.size() != o.key_mapping.size())
      throw InvalidArgument("Weight mapping refers to absent source keys");
    if (ignored.size() != ignored_keys.size())
      throw InvalidArgument("Ignored source key is absent");
    for (const auto& [name, value] : state)
      if (!targets.contains(name) && !Any(name, o.replace_head_prefixes) &&
          !retained_buffers.contains(name))
        throw CompatibilityError("Missing required weight: " + name);
    if (mapped.empty())
      throw InvalidArgument("Weight initialization matched no model state");
    io::SafeTensorsLimits limits;
    limits.max_file_bytes = o.max_bytes;
    limits.max_payload_bytes = o.max_bytes;
    limits.max_tensor_bytes = o.max_bytes;
    auto canonical = io::serialize_safetensors(mapped, {}, limits);
    io::ModelWeightImportOptions policy;
    policy.mapping.mode =
        o.replace_head_prefixes.empty() && retained_buffers.empty()
            ? io::WeightImportMode::strict
            : io::WeightImportMode::partial;
    if (policy.mapping.mode == io::WeightImportMode::partial)
      policy.mapping.minimum_tensor_coverage =
          static_cast<double>(mapped.size()) / state.size();
    policy.max_staging_bytes = o.max_bytes;
    policy.tensor_limits = limits;
    auto result = io::import_model_weights(
        model.state(), canonical, {.hex = sha256_hex(canonical)}, policy, stop);
    if (!result.applied)
      throw CompatibilityError(
          "Native weight import rejected the complete staged mapping");
  } else if (!o.sha256.empty() || !o.key_mapping.empty() ||
             !o.replace_head_prefixes.empty() || !ignored_keys.empty() ||
             !retained_buffers.empty())
    throw InvalidArgument("Weight mapping policy requires an input artifact");
  for (const auto& p : model.state().named_parameters())
    if (Any(p.key(), o.freeze_prefixes)) p.value().set_requires_grad(false);
  enforce_frozen_modules(model, o);
  return J{{"schema", "vision.initialization/v1"},
           {"source_sha256", digest},
           {"architecture", architecture},
           {"loaded", loaded},
           {"retained_source_head", retained},
           {"ignored_source_keys", ignored},
           {"retained_buffers", o.retain_buffer_keys},
           {"frozen", o.freeze_prefixes},
           {"random_initialization", o.weights.empty()}}
      .dump(2);
}
std::string propose_vision_weight_mapping(const nn::ModelAdapter& model,
                                          const std::filesystem::path& path,
                                          std::string_view expected,
                                          std::string_view architecture,
                                          std::uint64_t maximum) {
  if (expected.size() != 64 || !maximum || maximum > (4ULL << 30U))
    throw InvalidArgument(
        "Mapping proposal requires a bounded artifact and SHA-256");
  auto bytes = detail::read_artifact(path, maximum);
  if (sha256_hex(bytes) != expected)
    throw CompatibilityError("Mapping source digest mismatch");
  io::SafeTensorsView view(bytes);
  auto state = State(model);
  J mapping = J::object(), missing = J::array(), unexpected = J::array(),
    mismatched = J::array();
  std::set<std::string> used;
  auto rename = [](std::string value, std::string_view from,
                   std::string_view to) {
    if (value.starts_with(from)) value.replace(0, from.size(), to);
    return value;
  };
  for (const auto& source : view.directory().tensors) {
    std::string key = source.name;
    if (architecture.starts_with("yolo")) {
      for (const auto& [from, to] :
           std::vector<std::pair<std::string, std::string>>{
               {".one2one_cv2.", ".one_to_one.cv2."},
               {".one2one_cv3.", ".one_to_one.cv3."},
               {".one2one_cv4_kpts.", ".one_to_one.cv4_kpts."},
               {".one2one_cv4_sigma.", ".one_to_one.cv4_sigma."},
               {".one2one_cv4.", ".one_to_one.cv4."}})
        if (auto at = key.find(from); at != std::string::npos)
          key.replace(at, from.size(), to);
      if (!state.contains(key) && key.starts_with("model.")) {
        auto end = key.find('.', 6);
        if (end != std::string::npos) {
          auto tail = key.substr(end + 1);
          if (tail.starts_with("cv2.") || tail.starts_with("cv3.") ||
              tail.starts_with("cv4.") || tail.starts_with("cv4_kpts.") ||
              tail.starts_with("cv4_sigma."))
            key.insert(end + 1, "one_to_many.");
        }
      }
      if (auto at = key.find(".dfl.conv."); at != std::string::npos)
        key.replace(at, 10, ".dfl.");
      if (!state.contains(key)) {
        auto dot = key.rfind('.');
        if (dot != std::string::npos &&
            (key.substr(dot) == ".weight" || key.substr(dot) == ".bias")) {
          auto candidate = key;
          candidate.insert(dot, ".output");
          if (state.contains(candidate)) key = std::move(candidate);
        }
      }
    } else if (architecture == "faster_rcnn_resnet50_fpn") {
      key = rename(key, "backbone.body.", "backbone.");
      for (const auto& [from, to] :
           std::vector<std::pair<std::string, std::string>>{
               {"rpn.head.conv.0.0.", "rpn_conv."},
               {"rpn.head.conv.", "rpn_conv."},
               {"rpn.head.cls_logits.", "rpn_objectness."},
               {"rpn.head.bbox_pred.", "rpn_bbox_pred."},
               {"roi_heads.box_head.fc6.", "roi_fc6."},
               {"roi_heads.box_head.fc7.", "roi_fc7."},
               {"roi_heads.box_predictor.cls_score.", "roi_cls_score."},
               {"roi_heads.box_predictor.bbox_pred.", "roi_bbox_pred."}})
        key = rename(key, from, to);
      for (std::size_t i = 0; i < 4; ++i) {
        const auto n = std::to_string(i);
        key = rename(key, "backbone.fpn.inner_blocks." + n + ".0.",
                     "fpn_inner." + n + ".");
        key = rename(key, "backbone.fpn.layer_blocks." + n + ".0.",
                     "fpn_layer." + n + ".");
      }
    }
    const auto target = state.find(key);
    if (target == state.end()) {
      unexpected.push_back(source.name);
      continue;
    }
    if (source.dtype != Dtype(target->second.scalar_type()) ||
        source.shape !=
            std::vector<std::uint64_t>(target->second.sizes().begin(),
                                       target->second.sizes().end())) {
      mismatched.push_back(
          {{"source", source.name},
           {"target", key},
           {"source_shape", source.shape},
           {"target_shape",
            std::vector<std::int64_t>(target->second.sizes().begin(),
                                      target->second.sizes().end())}});
      continue;
    }
    if (!used.insert(key).second)
      throw CompatibilityError("Declared aliases collide at target: " + key);
    mapping[source.name] = key;
  }
  for (const auto& [key, value] : state)
    if (!used.contains(key)) missing.push_back(key);
  return J{
      {"schema", "vision.weight_mapping/v1"},
      {"source_sha256", expected},
      {"architecture", architecture},
      {"alias_policy", "declared-names/v1"},
      {"key_mapping", mapping},
      {"missing_target", missing},
      {"unexpected_source", unexpected},
      {"shape_or_dtype_mismatch", mismatched},
      {"complete", missing.empty() && unexpected.empty() && mismatched.empty()},
      {"applied", false}}
      .dump(2);
}
void enforce_frozen_modules(nn::ModelAdapter& model,
                            const InitializationOptions& options) {
  for (const auto& module : model.state().module().named_modules("", false))
    if (Any(module.key(), options.freeze_prefixes)) module.value()->eval();
}
void apply_parameter_rates(DetectionSgdGroups& groups,
                           std::span<const ParameterRateRule> rules) {
  std::set<std::string> prefixes;
  std::vector<bool> used(rules.size(), false);
  for (const auto& r : rules)
    if (r.prefix.empty() || !prefixes.insert(r.prefix).second ||
        !std::isfinite(r.multiplier) || r.multiplier <= 0 ||
        r.multiplier > 1000)
      throw InvalidArgument(
          "Invalid differential parameter learning-rate rule");
  std::vector<optim::SgdParameterGroup> result;
  for (auto& group : groups.groups) {
    std::map<std::size_t, optim::SgdParameterGroup> split;
    for (auto& parameter : group.parameters) {
      std::size_t index = rules.size(), longest = 0;
      for (std::size_t i = 0; i < rules.size(); ++i)
        if (Under(parameter.parameter_id, rules[i].prefix) &&
            rules[i].prefix.size() > longest) {
          index = i;
          longest = rules[i].prefix.size();
        }
      auto [it, inserted] = split.try_emplace(index);
      auto& output = it->second;
      if (inserted) {
        output.options = group.options;
        output.group_id =
            group.group_id + (index == rules.size()
                                  ? ".base"
                                  : ".rate-" + std::to_string(index));
        if (index != rules.size())
          output.options.learning_rate *= rules[index].multiplier;
      }
      if (index != rules.size()) used[index] = true;
      output.parameters.push_back(std::move(parameter));
    }
    for (auto& [id, g] : split) result.push_back(std::move(g));
  }
  if (std::find(used.begin(), used.end(), false) != used.end())
    throw InvalidArgument(
        "Differential learning-rate rule matches no trainable parameter");
  groups.groups = std::move(result);
}
}  // namespace jyppx::liuding::vision::tasks
