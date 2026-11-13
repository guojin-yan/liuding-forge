// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/foundation/random.hpp>
#include <jyppx/liuding/optim/adamw_adapter.hpp>
#include <jyppx/liuding/training/amp/autocast.hpp>
#include <jyppx/liuding/training/amp/grad_scaler.hpp>
#include <jyppx/liuding/vision/tasks/training/musgd.hpp>
#include <jyppx/liuding/vision/tasks/workflow/preflight.hpp>
#if defined(JYPPX_VISION_WITH_CUDA)
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#endif
#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>
namespace jyppx::liuding::vision::tasks {
namespace {
using J = nlohmann::json;
using I = std::int64_t;
std::uint64_t Free(const Device& device) {
  if (!device.is_cuda()) return 0;
#if defined(JYPPX_VISION_WITH_CUDA)
  c10::cuda::CUDAGuard guard(device);
  std::size_t free = 0, total = 0;
  if (cudaDeviceSynchronize() != cudaSuccess ||
      cudaMemGetInfo(&free, &total) != cudaSuccess)
    throw DeviceError("CUDA batch probe could not query free memory");
  return free;
#else
  throw DeviceError("This build has no CUDA batch probe");
#endif
}
std::unique_ptr<training::amp::AutocastGuard> Autocast(const RecipeSpec& r) {
  if (r.precision == training::PrecisionMode::full) return {};
  return std::make_unique<training::amp::AutocastGuard>(
      r.device.type(), r.precision == training::PrecisionMode::cuda_float16
                           ? torch::kFloat16
                           : torch::kBFloat16);
}
std::shared_ptr<optim::OptimizerAdapter> Optimizer(nn::ModelAdapter& model,
                                                   const RecipeSpec& r) {
  auto options = r.optimizer;
  options.effective_batch = r.batch_size * r.accumulate_steps;
  auto groups = make_detection_sgd_groups(model.state().module(), options);
  apply_parameter_rates(groups, r.initialization.learning_rates);
  const auto name = resolve_vision_optimizer(r);
  if (name == "sgd")
    return std::make_shared<optim::SgdOptimizerAdapter>(
        "preflight.optimizer", std::move(groups.groups));
  if (name == "musgd")
    return std::make_shared<MuSgdOptimizer>("preflight.optimizer",
                                            std::move(groups.groups));
  std::vector<optim::AdamWParameterGroup> converted;
  for (auto& g : groups.groups) {
    optim::AdamWParameterGroup v;
    v.group_id = g.group_id;
    v.parameters = std::move(g.parameters);
    v.options.learning_rate = g.options.learning_rate;
    v.options.weight_decay = g.options.weight_decay;
    v.options.beta1 = r.optimizer.momentum;
    converted.push_back(std::move(v));
  }
  return std::make_shared<optim::AdamWOptimizerAdapter>("preflight.optimizer",
                                                        std::move(converted));
}
}  // namespace
std::string resolve_vision_optimizer(const RecipeSpec& r) {
  if (r.optimizer_name != "auto") return r.optimizer_name;
  return r.model.architecture == "faster_rcnn_resnet50_fpn" ? "sgd"
         : r.model.architecture.starts_with("yolo26")       ? "musgd"
                                                            : "adamw";
}
std::string preflight_vision(const RecipeSpec& recipe,
                             std::span<const data::image::VisionSample> source,
                             const PreflightOptions& o, std::stop_token stop) {
  check_recipe(recipe);
  if (source.empty() || o.batch_candidates.empty() ||
      o.batch_candidates.size() > 64 || !o.maximum_probes ||
      o.maximum_probes > 64 || !o.max_tensor_bytes ||
      !std::isfinite(o.cuda_memory_fraction) || o.cuda_memory_fraction <= 0 ||
      o.cuda_memory_fraction > 1 || !std::isfinite(o.amp_absolute_tolerance) ||
      o.amp_absolute_tolerance < 0 ||
      !std::isfinite(o.amp_relative_tolerance) || o.amp_relative_tolerance < 0)
    throw InvalidArgument("Invalid preflight candidates/budget/tolerances");
  auto candidates = o.batch_candidates;
  std::sort(candidates.begin(), candidates.end());
  if (std::adjacent_find(candidates.begin(), candidates.end()) !=
          candidates.end() ||
      candidates.front() < 1 || candidates.back() > 4096 ||
      candidates.size() > o.maximum_probes)
    throw InvalidArgument(
        "Preflight candidates must be unique, positive and within probe "
        "budget");
  const auto rng = core::get_rng_state(recipe.device.is_cuda());
  J probes = J::array();
  std::size_t recommended = 0;
  try {
    const auto available = Free(recipe.device);
    for (auto count : candidates) {
      if (stop.stop_requested()) throw Cancelled("Vision preflight cancelled");
      core::manual_seed(recipe.seed, recipe.device.is_cuda());
      J row = {{"batch", count},
               {"success", false},
               {"amp_compatible",
                recipe.precision == training::PrecisionMode::full}};
      try {
        auto r = recipe;
        r.batch_size = count;
        r.data.augmentation.dataset_id = "vision.preflight";
        r.data.num_classes = r.model.num_classes;
        r.data.allow_unannotated_negatives = r.model.task == TaskKind::classify || static_cast<unsigned>(r.model.task)>4;
        r.data.classification.enabled = r.model.task == TaskKind::classify || r.model.task==TaskKind::classify_multilabel||r.model.task==TaskKind::reid;
        auto binding = bind_task(r);
        r.data.size_divisor = static_cast<std::uint32_t>(binding.capability.input_divisor);
        binding.model->state().to(r.device);
        initialize_vision(*binding.model, r.initialization,
                          r.model.architecture, stop);
        std::uint64_t state_bytes = 0;
        for (const auto& p : binding.model->state().parameters())
          state_bytes += p.nbytes();
        for (const auto& p : binding.model->state().buffers())
          state_bytes += p.nbytes();
        const auto pixels = std::uint64_t(count) * 3 *
                            r.data.letterbox.canvas.width *
                            r.data.letterbox.canvas.height;
        if (state_bytes > o.max_tensor_bytes ||
            pixels > (o.max_tensor_bytes - state_bytes) / sizeof(float))
          throw ResourceExhausted("Probe model/input tensor budget exceeded");
        std::vector<data::image::VisionSample> samples;
        std::vector<data::image::ImageBatchKey> keys;
        for (std::size_t i = 0; i < count; ++i) {
          samples.push_back(source[i % source.size()]);
          keys.push_back({.id = samples.back().image_path, .occurrence = i});
        }
        auto batch = prepare_yolo_batch(samples, keys,
                                        yolo_evaluation_options(r.data), stop);
        batch.images = batch.images.to(r.device);
        if (batch.labels.defined()) batch.labels = batch.labels.to(r.device);
        data::BatchEnvelope envelope;
        envelope.value = data::image::image_batch_value(batch, r.data.limits);
        const auto invocation = binding.image_mapper(envelope);
        const auto target = binding.target_mapper(envelope);
        if (r.precision != training::PrecisionMode::full) {
          torch::NoGradGuard guard;
          binding.model->state().eval();
          auto reference = raw_tensors(binding.model->invoke(invocation).value);
          auto cast = Autocast(r);
          auto reduced = raw_tensors(binding.model->invoke(invocation).value);
          cast.reset();
          bool compatible = reference.size() == reduced.size();
          double maximum_error = 0;
          for (std::size_t i = 0; i < reference.size() && compatible; ++i) {
            if (reference[i].sizes() != reduced[i].sizes()) {
              compatible = false;
              break;
            }
            if (!reference[i].is_floating_point()) continue;
            auto fp32 = reference[i].to(torch::kFloat32),
                 value = reduced[i].to(torch::kFloat32);
            if (value.numel()) {
              auto error = (fp32 - value).abs();
              maximum_error =
                  (std::max)(maximum_error, error.max().item<double>());
              compatible = torch::isfinite(value).all().item<bool>() &&
                           (error <= o.amp_absolute_tolerance +
                                         o.amp_relative_tolerance * fp32.abs())
                               .all()
                               .item<bool>();
            }
          }
          row["amp_compatible"] = compatible;
          row["maximum_output_error"] = maximum_error;
          if (!compatible)
            throw NumericalError(
                "AMP outputs exceed the explicit preflight tolerance");
        }
        binding.model->state().train();
        enforce_frozen_modules(*binding.model, r.initialization);
        training::StepContext context;
        training::StepOutput result;
        {
          auto cast = Autocast(r);
          result = binding.step(
              {*binding.model, envelope, invocation, target, context});
        }
        if (!torch::isfinite(result.loss).all().item<bool>())
          throw NumericalError("Preflight task loss is nonfinite");
        auto optimizer = Optimizer(*binding.model, r);
        training::amp::GradScaler scaler(
            {.init_scale = 128.,
             .enabled = r.precision == training::PrecisionMode::cuda_float16});
        scaler.scale(result.loss).backward();
        scaler.unscale(optimizer);
        std::size_t gradients = 0;
        for (const auto& p : binding.model->state().parameters())
          if (p.grad().defined()) {
            ++gradients;
            if (!torch::isfinite(p.grad()).all().item<bool>())
              throw NumericalError("Preflight task gradient is nonfinite");
          }
        if (!gradients)
          throw NumericalError("Preflight produced no trainable gradients");
        if (!scaler.step_allowed(optimizer))
          throw NumericalError("Preflight scaled gradients overflowed");
        optimizer->step();
        scaler.mark_step(optimizer, true);
        scaler.update();
        for (const auto& p : binding.model->state().parameters())
          if (!torch::isfinite(p).all().item<bool>())
            throw NumericalError(
                "Preflight optimizer produced nonfinite parameters");
        const auto after = Free(r.device),
                   used = available > after ? available - after : 0;
        if (r.device.is_cuda() &&
            used >
                static_cast<std::uint64_t>(available * o.cuda_memory_fraction))
          throw ResourceExhausted(
              "Candidate exceeds explicit CUDA free-memory fraction");
        row["loss"] = result.loss.detach().item<double>();
        row["gradient_tensors"] = gradients;
        row["live_device_delta_bytes"] = used;
        row["success"] = true;
        recommended = count;
      } catch (const Cancelled&) {
        throw;
      } catch (const c10::Error& error) {
        row["reason"] = error.what_without_backtrace();
      } catch (const Error& error) {
        row["reason"] = error.what();
      }
      probes.push_back(row);
      if (!row["success"].get<bool>()) break;
    }
    J report = {
        {"schema", "vision.preflight/v1"},
        {"recipe_identity", recipe_identity(recipe)},
        {"device", recipe.device.str()},
        {"optimizer", resolve_vision_optimizer(recipe)},
        {"optimizer_policy", "native-vision-auto/v1"},
        {"cuda_initial_free_bytes", available},
        {"cuda_memory_fraction", o.cuda_memory_fraction},
        {"recommended_batch", recommended ? J(recommended) : J(nullptr)},
        {"measurement", "successful_disposable_step_and_live_device_delta"},
        {"probes", probes},
        {"advisory", true}};
    core::set_rng_state(rng, recipe.device.is_cuda());
    return report.dump(2);
  } catch (...) {
    core::set_rng_state(rng, recipe.device.is_cuda());
    throw;
  }
}
}  // namespace jyppx::liuding::vision::tasks
