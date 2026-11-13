// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "artifact_io.hpp"
#include "extended_io.hpp"

#include <jyppx/liuding/foundation/random.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/vision/tasks/original_v5.hpp>
#include <jyppx/liuding/vision/tasks/workflow/export.hpp>
#if defined(JYPPX_VISION_WITH_ONNX_EXPORT)
#include <jyppx/liuding/deployment/onnx_export/exporter.hpp>
#endif
#include <chrono>
#include <map>
#include <nlohmann/json.hpp>
namespace jyppx::liuding::vision::tasks {
namespace detail {
std::map<std::string, onnx_export::OperatorLowering> vision_onnx_lowerings();
onnx_export::ForwardFunction make_rcnn_onnx_forward(
    const std::shared_ptr<nn::ModelAdapter>&, const VisionExportOptions&);
void add_rcnn_onnx_lowerings(
    std::map<std::string, onnx_export::OperatorLowering>&);
}  // namespace detail
namespace {
using J = nlohmann::json;
using I = std::int64_t;
namespace image = data::image;
void CopyState(const nn::ModelAdapter& from, nn::ModelAdapter& to) {
  torch::NoGradGuard guard;
  auto params = from.state().named_parameters(),
       buffers = from.state().named_buffers();
  for (auto& p : to.state().named_parameters()) {
    if (!params.contains(p.key()) ||
        params[p.key()].sizes() != p.value().sizes())
      throw CompatibilityError("Export model parameter topology changed");
    p.value().copy_(params[p.key()].detach().to(torch::kCPU));
  }
  for (auto& p : to.state().named_buffers()) {
    if (!buffers.contains(p.key()) ||
        buffers[p.key()].sizes() != p.value().sizes())
      throw CompatibilityError("Export model buffer topology changed");
    p.value().copy_(buffers[p.key()].detach().to(torch::kCPU));
  }
}
}  // namespace
std::string export_vision(const TaskBinding& binding, const RecipeSpec& recipe,
                          const std::filesystem::path& output,
                          const VisionExportOptions& profile) {
#if defined(JYPPX_VISION_WITH_ONNX_EXPORT)
  check_recipe(recipe);
  if (output.empty() ||
      (std::filesystem::exists(output) && !std::filesystem::is_empty(output)))
    throw InvalidArgument("ONNX output must be a new or empty directory");
  if (profile.batch_size < 1 || profile.batch_size > 64 ||
      profile.opset != 17 || profile.max_bytes < 1 ||
      profile.max_bytes > (4ULL << 30U) || profile.rcnn_proposals < 1 ||
      profile.rcnn_proposals > 2048 ||
      (profile.weight_role != "raw" && profile.weight_role != "ema"))
    throw InvalidArgument("Invalid vision ONNX fixed profile");
  if (binding.capability.architecture != recipe.model.architecture ||
      binding.capability.task != recipe.model.task ||
      binding.capability.variant != recipe.model.variant)
    throw CompatibilityError(
        "Export binding differs from recipe model identity");
  const auto rng = core::get_rng_state(false);
  try {
    auto r = recipe;
    r.device = Device(torch::kCPU);
    r.precision = training::PrecisionMode::full;
    r.extension.teacher_architecture.clear();
    r.extension.teacher_initialization={};
    r.extension.feature_outputs=false;r.extension.feature_projection=0;
    r.extension.activation_recomputation=false;
    r.extension.tokenizer_merges.clear();r.extension.tokenizer_sha256.clear();
    r.extension.text_encoder_weights.clear();r.extension.text_encoder_sha256.clear();
    auto cloned = bind_task(r);
    CopyState(*binding.model, *cloned.model);
    cloned.model->state().eval();
    std::size_t fused_blocks = 0;
    if (profile.fuse) {
      if (!cloned.fuse_inference)
        throw CompatibilityError("This model has no native inference fusion");
      fused_blocks = cloned.fuse_inference();
    }
    const auto exported_role =
        profile.fuse ? std::string("fused") : profile.weight_role;
    const I h = r.data.letterbox.canvas.height,
            w = r.data.letterbox.canvas.width;
    if (profile.batch_size >
        static_cast<I>(profile.max_bytes / sizeof(float)) / 3 / h / w)
      throw ResourceExhausted("Export example exceeds profile byte bound");
    auto example = torch::zeros({profile.batch_size, 3, h, w});
    onnx_export::ExportOptions options;
    options.opset = profile.opset;
    options.dynamic_batch = false;
    options.max_output_bytes = profile.max_bytes;
    options.input_names = {"images"};
    options.custom_lowerings = detail::vision_onnx_lowerings();
    onnx_export::ForwardFunction forward = [model = cloned.model](
                                               std::span<const Tensor> inputs) {
      return raw_tensors(model->invoke(nn::Invocation{{inputs[0]}, {}}).value);
    };
    if (binding.capability.component == "models_rcnn") {
#if defined(JYPPX_VISION_WITH_RCNN)
      options.output_names = {"proposals", "class_logits", "box_deltas",
                              "valid_proposals"};
      detail::add_rcnn_onnx_lowerings(options.custom_lowerings);
      forward = detail::make_rcnn_onnx_forward(cloned.model, profile);
#else
      throw CompatibilityError("The RCNN model pack is disabled");
#endif
    }
    const bool extended=detail::extended_bundle(r);
    std::vector<Tensor> examples{example};
    if(extended && binding.capability.component!="models_rcnn") {
      auto e=detail::extended_export(cloned,r,profile,example);
      examples=std::move(e.examples);options.input_names=std::move(e.input_names);
      options.output_names=std::move(e.output_names);forward=std::move(e.forward);
      detail::add_extended_onnx_lowerings(options.custom_lowerings);
    }
    torch::NoGradGuard no_grad;
    J input_types=J::array(),output_types=J::array();
    auto dtype=[](const Tensor& t)->int {if(t.scalar_type()==torch::kFloat32)return 1;if(t.scalar_type()==torch::kInt64)return 7;if(t.scalar_type()==torch::kInt32)return 6;throw CompatibilityError("Bundle I/O only supports float32/int32/int64");};
    for(const auto& t:examples)input_types.push_back(dtype(t));
    auto raw_forward=std::move(forward);
    // Some symbolics (RPN/NMS/ROIAlign) need an active tracing scope. Record
    // actual output dtypes during that invocation instead of a separate run.
    forward=[raw_forward,&output_types,dtype](std::span<const Tensor> inputs){auto outputs=raw_forward(inputs);output_types=J::array();for(auto& t:outputs){output_types.push_back(dtype(t));t=t.clone();}return outputs;};
    auto artifact = onnx_export::export_module(cloned.model->state().module(),
                                               forward, examples, options);
    auto portable = r;
    portable.initialization = {};
    portable.cache = {};
    portable.validation.reset();
    portable.classification_validation_root.clear();
    if(!portable.extension.training_manifest.empty())portable.extension.training_manifest="__unused__";
    portable.extension.validation_manifest.clear();
    if (r.model.task == TaskKind::classify)
      portable.classification_root = "__unused__";
    else
      portable.training.source = "__unused__";
    portable.data = yolo_evaluation_options(r.data);
    const auto portable_json = recipe_json(portable);
    static_cast<void>(
        parse_recipe(portable_json, std::filesystem::absolute(output)));
    J manifest = {
        {"schema", extended?"vision.onnx_bundle/v3":"vision.onnx_bundle/v2"},
        {"model_file", "model.onnx"},
        {"sha256", artifact.digest.hex},
        {"io_profile", "cpu.dense_typed/v1"},
        {"architecture_revision", binding.capability.revision},
        {"size_divisor", binding.capability.input_divisor},
        {"weight_role", exported_role},
        {"input_weight_role", profile.weight_role},
        {"opset", profile.opset},
        {"fusion",
         {{"enabled", profile.fuse}, {"conv_bn_blocks", fused_blocks}}},
        {"fixed_batch", profile.batch_size},
        {"width", w},
        {"height", h},
        {"input_names", artifact.input_names},
        {"output_names", artifact.output_names},
        {"input_shapes", artifact.input_shapes},
        {"input_types",input_types},{"output_types",output_types},
        {"output_shapes", artifact.output_shapes},
        {"operators", artifact.operators},
        {"decoder", binding.capability.component == "models_rcnn"
                        ? "rcnn.fixed_proposals/v2"
                        : (extended?"vision.extended/v1":"vision.raw/v1")},
        {"recipe", J::parse(portable_json)},
        {"pixel_contract",
         {{"color", "RGB"},
          {"layout", "NCHW"},
          {"dtype", "float32"},
          {"range", "unit_then_recipe_normalization"}}},
        {"geometry", (r.model.task == TaskKind::classify||r.model.task==TaskKind::classify_multilabel||r.model.task==TaskKind::reid)
                         ? "deterministic_center_crop"
                         : "letterbox_with_inverse_trace"},
        {"class_count", r.model.num_classes},
        {"end_to_end", binding.capability.end_to_end},
        {"qualification",
         {{"architecture_reference", binding.capability.reference_qualified},
          {"task_quality", binding.capability.quality_qualified},
          {"runtime_verified", false}}}};
    detail::ArtifactDirectory pending(output);
    const auto& destination = pending.path();
    detail::publish_artifact(destination / "model.onnx", artifact.onnx);
    if (artifact.report)
      detail::publish_text(
          destination / "export.json",
          onnx_export::serialize_export_report(*artifact.report));
    manifest["native_weights"] = {
        {"file", "model.safetensors"},
        {"sha256",
         save_vision_weights(*cloned.model, destination / "model.safetensors",
                             {{"architecture", r.model.architecture},
                              {"variant", r.model.variant},
                              {"role", exported_role}},
                             profile.max_bytes)}};
    detail::publish_text(destination / "manifest.json", manifest.dump(2));
    pending.commit();
    core::set_rng_state(rng, false);
    return manifest.dump(2);
  } catch (...) {
    core::set_rng_state(rng, false);
    throw;
  }
#else
  static_cast<void>(binding);
  static_cast<void>(recipe);
  static_cast<void>(output);
  static_cast<void>(profile);
  throw CompatibilityError(
      "Vision ONNX export requires the onnx_export component");
#endif
}
}  // namespace jyppx::liuding::vision::tasks
