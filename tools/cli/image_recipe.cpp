// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "cli.hpp"
#include "recipe_support.hpp"
#include "transfer_recipe.hpp"

#if defined(JYPPX_LIUDING_CLI_IMAGE_RECIPE)
#include <jyppx/liuding/core/device.hpp>
#include <jyppx/liuding/data/data_pipeline.hpp>
#include <jyppx/liuding/data/samplers/random_sampler.hpp>
#include <jyppx/liuding/data_image/batch_transform.hpp>
#include <jyppx/liuding/data_image/image_batch.hpp>
#include <jyppx/liuding/data_image/joint_transform.hpp>
#include <jyppx/liuding/experiment/config.hpp>
#include <jyppx/liuding/experiment/evaluation.hpp>
#include <jyppx/liuding/experiment/run_store.hpp>
#include <jyppx/liuding/foundation/random_checkpoint_participant.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/foundation/version.hpp>
#include <jyppx/liuding/interop/model_bundle.hpp>
#include <jyppx/liuding/models/registry.hpp>
#include <jyppx/liuding/nn/model_checkpoint_participant.hpp>
#include <jyppx/liuding/optim/composite_lr_scheduler.hpp>
#include <jyppx/liuding/optim/linear_lr_scheduler.hpp>
#include <jyppx/liuding/optim/sgd_adapter.hpp>
#include <jyppx/liuding/optim/step_lr_scheduler.hpp>
#include <jyppx/liuding/training/classification_evaluation.hpp>
#include <jyppx/liuding/training/diagnostics.hpp>
#include <jyppx/liuding/training/file_checkpoint_store.hpp>
#include <jyppx/liuding/training/scheduler_adapter.hpp>
#include <jyppx/liuding/training/single_device_strategy.hpp>
#include <jyppx/liuding/training/trainer.hpp>

#include <torch/cuda.h>
#include <torch/nn/functional/loss.h>

#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <memory>
#include <numeric>
#include <stop_token>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif
#endif

namespace jyppx::liuding::cli {
#if defined(JYPPX_LIUDING_CLI_IMAGE_RECIPE)
namespace {
namespace image = data::image;
namespace tr = training;
namespace ex = experiment;
constexpr std::string_view kRecipe = "jyppx.liuding.imagefolder.resnet18/v1";
constexpr std::string_view kRun = "run.image-classification";

using namespace recipe_support;

bool is_group_rate(std::string_view name) {
  return name == "backbone_learning_rate" || name == "head_learning_rate";
}

bool is_schedule_field(std::string_view name) {
  return name == "lr_schedule" || name == "warmup_updates" ||
         name == "warmup_start_factor" || name == "lr_end_factor";
}

bool uses_update_schedule(const Json& config) {
  return config.value("lr_schedule", "epoch_step") == "warmup_linear";
}

const std::map<std::string, ValueKind>& field_kinds(bool transfer = false) {
  static const std::map<std::string, ValueKind> fields{
      {"schema_version", ValueKind::integer},
      {"recipe", ValueKind::string},
      {"train_root", ValueKind::string},
      {"validation_root", ValueKind::string},
      {"data_license", ValueKind::string},
      {"device", ValueKind::string},
      {"seed", ValueKind::integer},
      {"epochs", ValueKind::integer},
      {"batch_size", ValueKind::integer},
      {"image_size", ValueKind::integer},
      {"workers", ValueKind::integer},
      {"prefetch_batches", ValueKind::integer},
      {"learning_rate", ValueKind::floating},
      {"momentum", ValueKind::floating},
      {"weight_decay", ValueKind::floating},
      {"lr_gamma", ValueKind::floating},
      {"lr_schedule", ValueKind::string},
      {"warmup_updates", ValueKind::integer},
      {"warmup_start_factor", ValueKind::floating},
      {"lr_end_factor", ValueKind::floating},
      {"checkpoint_every", ValueKind::integer},
      {"minimum_accuracy", ValueKind::floating},
      {"max_image_bytes", ValueKind::integer},
      {"max_data_bytes", ValueKind::integer},
      {"max_samples", ValueKind::integer},
      {"image_budget_bytes", ValueKind::integer}};
  if (!transfer) return fields;
  static const auto transfer_fields = [] {
    auto result = field_kinds(false);
    for (const auto* name : {"model", "weights_file", "weights_metadata",
                             "backbone_mode", "batch_norm"})
      result.emplace(name, ValueKind::string);
    for (const auto* name : {"backbone_learning_rate", "head_learning_rate"})
      result.emplace(name, ValueKind::floating);
    return result;
  }();
  return transfer_fields;
}
ex::config::ConfigSchema config_schema(bool transfer) {
  ex::config::ConfigSchema result;
  result.schema_id = std::string(transfer ? kTransferRecipe : kRecipe);
  result.root.kind = ValueKind::dictionary;
  for (const auto& [name, kind] : field_kinds(transfer)) {
    ValueSchema field;
    field.kind = kind;
    field.name = name;
    field.required = !is_group_rate(name) && !is_schedule_field(name);
    result.root.fields.emplace(name, std::move(field));
  }
  return result;
}
Json schema_json(bool transfer = false) {
  Json fields = Json::object();
  for (const auto& [name, kind] : field_kinds(transfer)) {
    fields[name] = {{"kind", value_kind_name(kind)}};
    if (is_group_rate(name) || is_schedule_field(name))
      fields[name]["required"] = false;
  }
  return {{"schema_version", 1},
          {"schema_id", transfer ? kTransferRecipe : kRecipe},
          {"version", 1},
          {"root", {{"kind", "dictionary"}, {"fields", fields}}}};
}
Json resolve_config(const std::filesystem::path& path) {
  const auto transfer = is_transfer_recipe(parse_json(read_text(path)));
  ex::config::ConfigResolver resolver(config_schema(transfer));
  resolver.add_json_file(path);
  const auto resolved = resolver.resolve();
  auto config = parse_json(resolved.canonical_json());
  const auto integer = [&](const char* name, std::int64_t minimum,
                           std::int64_t maximum) {
    const auto value = resolved.integer(std::string("/") + name);
    require_recipe(value >= minimum && value <= maximum,
                   std::string("Recipe field out of range: ") + name);
  };
  integer("schema_version", 1, 1);
  require_recipe(config.at("recipe").get<std::string>() ==
                     (transfer ? kTransferRecipe : kRecipe),
                 "Unsupported recipe identity");
  require_recipe(config.at("device") == "cpu" || config.at("device") == "cuda",
                 "Recipe device must be cpu or cuda");
  integer("seed", 0, 2147483647);
  integer("epochs", 1, 1000);
  integer("batch_size", 2, 1024);
  integer("image_size", 32, 1024);
  integer("workers", 0, 8);
  integer("prefetch_batches", 1, 8);
  integer("checkpoint_every", 1, 1000000);
  integer("max_image_bytes", 1024, 256LL << 20);
  integer("max_data_bytes", 1024, 1LL << 40);
  integer("max_samples", 2, 1000000);
  integer("image_budget_bytes", 1LL << 20, 4LL << 30);
  for (const auto* key : {"learning_rate", "momentum", "weight_decay",
                          "lr_gamma", "minimum_accuracy"}) {
    const auto value = resolved.floating(std::string("/") + key);
    require_recipe(
        std::isfinite(value) && value >= 0.0 && value <= 1.0,
        std::string("Recipe floating field must be in [0,1]: ") + key);
  }
  require_recipe(config.at("learning_rate").get<double>() > 0 &&
                     config.at("lr_gamma").get<double>() > 0 &&
                     config.at("momentum").get<double>() < 1 &&
                     config.at("minimum_accuracy").get<double>() > 0,
                 "Recipe learning rate, gamma and accuracy threshold must be "
                 "positive; momentum < 1");
  require_recipe(!config.at("data_license").get<std::string>().empty(),
                 "Declare the dataset license or terms in data_license");
  const auto schedule = config.value("lr_schedule", "epoch_step");
  require_recipe(schedule == "epoch_step" || schedule == "warmup_linear",
                 "lr_schedule must be epoch_step or warmup_linear");
  for (const auto* key :
       {"warmup_updates", "warmup_start_factor", "lr_end_factor"})
    require_recipe(
        config.contains(key) == uses_update_schedule(config),
        std::string(key) +
            " must be supplied exactly when lr_schedule=warmup_linear");
  if (uses_update_schedule(config)) {
    integer("warmup_updates", 1, 999999);
    for (const auto* key : {"warmup_start_factor", "lr_end_factor"}) {
      const auto factor = resolved.floating(std::string("/") + key);
      require_recipe(std::isfinite(factor) && factor > 0.0 && factor <= 1.0,
                     std::string(key) + " must be finite and in (0,1]");
    }
    require_recipe(
        config.at("lr_gamma").get<double>() == 1.0,
        "warmup_linear requires lr_gamma=1; decay uses lr_end_factor");
    const auto start = config.at("warmup_start_factor").get<double>();
    const auto warmup = config.at("warmup_updates").get<double>();
    require_recipe(std::isfinite((start + (1.0 - start) / warmup) / start),
                   "warmup_start_factor produces an unrepresentable first "
                   "learning-rate transition");
  }
  for (const auto* key : {"train_root", "validation_root"}) {
    require_recipe(!config.at(key).get<std::string>().empty(),
                   "Recipe dataset path must not be empty");
    auto root = std::filesystem::u8path(config.at(key).get<std::string>());
    if (root.is_relative()) root = path.parent_path() / root;
    config[key] = path_text(std::filesystem::absolute(root).lexically_normal());
  }
  require_recipe(config.at("train_root") != config.at("validation_root"),
                 "Training and validation directories must differ");
  if (transfer) {
    for (const auto* name : {"backbone_learning_rate", "head_learning_rate"}) {
      if (!config.contains(name)) continue;
      const auto rate = resolved.floating(std::string("/") + name);
      require_recipe(std::isfinite(rate) && rate > 0.0 && rate <= 1.0,
                     std::string(name) + " must be finite and in (0,1]");
    }
    require_recipe(config.at("backbone_mode") == "frozen" ||
                       config.at("backbone_mode") == "finetune",
                   "backbone_mode must be frozen or finetune");
    require_recipe(
        config.at("batch_norm") == "eval" || config.at("batch_norm") == "train",
        "batch_norm must be eval or train");
    require_recipe(config.at("backbone_mode") != "frozen" ||
                       config.at("batch_norm") == "eval",
                   "Frozen backbones require batch_norm=eval");
    require_recipe(config.at("backbone_mode") != "frozen" ||
                       !config.contains("backbone_learning_rate"),
                   "backbone_learning_rate requires backbone_mode=finetune");
    for (const auto* key : {"weights_file", "weights_metadata"}) {
      const auto name = config.at(key).get<std::string>();
      require_recipe(!name.empty(), "Weight paths must not be empty");
      auto input = std::filesystem::u8path(name);
      if (input.is_relative()) input = path.parent_path() / input;
      config[key] =
          path_text(std::filesystem::absolute(input).lexically_normal());
    }
  }
  if (uses_update_schedule(config)) {
    for (const auto* key :
         {"learning_rate", "backbone_learning_rate", "head_learning_rate"}) {
      if (!config.contains(key)) continue;
      for (const auto* factor : {"warmup_start_factor", "lr_end_factor"})
        require_recipe(
            config.at(key).get<double>() * config.at(factor).get<double>() >
                0.0,
            "Scheduled learning rate underflows to zero");
    }
  }
  return config;
}

bool uses_group_rates(const Json& config) {
  return is_transfer_recipe(config) &&
         (config.contains("backbone_learning_rate") ||
          config.contains("head_learning_rate"));
}

Json sgd_group_plan(const Json& config) {
  const auto shared = config.at("learning_rate").get<double>();
  if (!uses_group_rates(config))
    return Json::array({{{"group_id", "main"}, {"learning_rate", shared}}});
  Json groups = Json::array();
  for (const auto* id : {"backbone", "head"}) {
    if (std::string_view(id) == "backbone" &&
        config.at("backbone_mode") == "frozen")
      continue;
    groups.push_back(
        {{"group_id", id},
         {"learning_rate",
          config.value(std::string(id) + "_learning_rate", shared)}});
  }
  return groups;
}

std::shared_ptr<optim::SgdOptimizerAdapter> make_sgd(
    const Json& config, const nn::ModelAdapter& model) {
  std::vector<optim::SgdParameterGroup> groups;
  const auto plan = sgd_group_plan(config);
  for (const auto& entry : plan)
    groups.push_back(
        {.group_id = entry.at("group_id").get<std::string>(),
         .parameters = {},
         .options = {.learning_rate = entry.at("learning_rate").get<double>(),
                     .momentum = config.at("momentum").get<double>(),
                     .weight_decay = config.at("weight_decay").get<double>()}});
  const bool shared = groups.front().group_id == "main";
  for (const auto& parameter : model.state().named_parameters()) {
    if (!parameter.value().requires_grad()) continue;
    const auto id = shared                                     ? "main"
                    : parameter.key().starts_with("backbone.") ? "backbone"
                    : parameter.key().starts_with("head.")     ? "head"
                                                               : "";
    const auto group =
        std::find_if(groups.begin(), groups.end(),
                     [&](const auto& value) { return value.group_id == id; });
    require_recipe(group != groups.end(),
                   "Trainable parameter has no declared SGD group");
    group->parameters.push_back({parameter.key(), parameter.value()});
  }
  return std::make_shared<optim::SgdOptimizerAdapter>("recipe.sgd",
                                                      std::move(groups));
}

std::shared_ptr<optim::LrScheduler> make_scheduler(
    const std::shared_ptr<optim::SgdOptimizerAdapter>& optimizer,
    const Json& config, std::uint64_t total_updates) {
  if (!uses_update_schedule(config))
    return std::make_shared<optim::StepLrScheduler>(
        optimizer, optim::StepLrSchedulerOptions{
                       .scheduler_id = "recipe.lr",
                       .step_size = 1,
                       .gamma = config.at("lr_gamma").get<double>()});
  const auto warmup = config.at("warmup_updates").get<std::uint64_t>();
  std::vector<std::unique_ptr<optim::LrScheduler>> children;
  children.push_back(std::make_unique<optim::WarmupLrScheduler>(
      optimizer,
      optim::WarmupLrSchedulerOptions{
          .scheduler_id = "recipe.lr.warmup",
          .warmup_iters = warmup,
          .start_factor = config.at("warmup_start_factor").get<double>(),
          .step_mode = optim::SchedulerStepMode::batch}));
  // Linear starts with factor one, preserving the warmup's construction-time
  // rate. Its transitions scale the current rates after warmup reaches one.
  children.push_back(std::make_unique<optim::LinearLrScheduler>(
      optimizer, optim::LinearLrSchedulerOptions{
                     .scheduler_id = "recipe.lr.linear",
                     .start_factor = 1.0,
                     .end_factor = config.at("lr_end_factor").get<double>(),
                     .total_iters = total_updates - warmup,
                     .step_mode = optim::SchedulerStepMode::batch}));
  return std::make_shared<optim::SequentialLrScheduler>(
      optimizer, std::move(children),
      optim::SequentialLrSchedulerOptions{
          .scheduler_id = "recipe.lr",
          .milestones = {static_cast<std::int64_t>(warmup)}});
}

Json describe_optimization(const optim::SgdOptimizerAdapter& optimizer,
                           const optim::LrScheduler& scheduler,
                           const Json& config, std::uint64_t total_updates) {
  Json groups = Json::array();
  const auto rates = optimizer.learning_rates();
  std::size_t index = 0;
  for (const auto& group : optimizer.groups()) {
    Json names = Json::array();
    for (const auto& parameter : group.parameters)
      names.push_back(parameter.parameter_id);
    groups.push_back({{"group_id", group.group_id},
                      {"parameter_ids", std::move(names)},
                      {"initial_learning_rate", group.options.learning_rate},
                      {"final_learning_rate", rates.at(index++)}});
    if (uses_update_schedule(config))
      groups.back()["first_learning_rate"] =
          group.options.learning_rate *
          config.at("warmup_start_factor").get<double>();
  }
  if (uses_update_schedule(config))
    return {{"optimizer_id", optimizer.optimizer_id()},
            {"groups", std::move(groups)},
            {"scheduler",
             {{"scheduler_id", "recipe.lr"},
              {"kind", "warmup_linear"},
              {"step_mode", "batch"},
              {"warmup_updates", config.at("warmup_updates")},
              {"warmup_start_factor", config.at("warmup_start_factor")},
              {"lr_end_factor", config.at("lr_end_factor")},
              {"total_updates", total_updates},
              {"completed_steps", scheduler.completed_steps()}}}};
  return {{"optimizer_id", optimizer.optimizer_id()},
          {"groups", std::move(groups)},
          {"scheduler",
           {{"scheduler_id", "recipe.lr"},
            {"kind", "step_lr"},
            {"step_size", 1},
            {"gamma", config.at("lr_gamma")},
            {"completed_steps", scheduler.completed_steps()}}}};
}

struct Inputs {
  image::ManifestIndex train;
  image::ManifestIndex validation;
  Json manifest;
  ArtifactDigest identity;
  std::uint64_t total_steps;
};
Inputs inspect_inputs(const Json& config) {
  image::ImageFolderOptions options;
  options.extensions = {".jpg", ".jpeg", ".png"};
  options.limits.max_images = config.at("max_samples").get<std::size_t>();
  options.limits.max_categories = 1024;
  const auto train_root =
      std::filesystem::u8path(config.at("train_root").get<std::string>());
  const auto val_root =
      std::filesystem::u8path(config.at("validation_root").get<std::string>());
  auto train = image::index_image_folder(train_root, options);
  auto validation = image::index_image_folder(val_root, options);
  require_recipe(train.categories().size() >= 2,
                 "Classification requires at least two classes");
  const auto classes = [](const image::ManifestIndex& index) {
    Json result = Json::array();
    for (const auto& category : index.categories())
      result.push_back(category.name);
    return result;
  };
  require_recipe(classes(train) == classes(validation),
                 "Training and validation class maps must match exactly");
  const auto batch = config.at("batch_size").get<std::size_t>();
  if (!is_transfer_recipe(config) || config.at("batch_norm") == "train")
    require_recipe(
        train.records().size() >= 2 && train.records().size() % batch != 1,
        "Training BatchNorm requires batches of at least two; choose "
        "another batch_size or use the transfer recipe with batch_norm=eval");
  const auto total_steps = ((train.records().size() + batch - 1) / batch) *
                           config.at("epochs").get<std::uint64_t>();
  require_recipe(total_steps <= 1000000,
                 "Recipe exceeds one million optimizer updates");
  require_recipe(
      !uses_update_schedule(config) ||
          config.at("warmup_updates").get<std::uint64_t>() < total_steps,
      "warmup_updates must precede the final optimizer update");
  std::uint64_t data_bytes = 0;
  const auto split = [&](const image::ManifestIndex& index,
                         const std::filesystem::path& root) {
    Sha256 hash;
    hash.update(
        bytes(config.at("recipe").get<std::string>() + "/content/v1\n"));
    for (const auto& record : index.records()) {
      const auto file = root / std::filesystem::u8path(record.path);
      const auto size = std::filesystem::file_size(file);
      const auto maximum = config.at("max_data_bytes").get<std::uint64_t>();
      require_recipe(size <= maximum - data_bytes,
                     "Dataset exceeds max_data_bytes");
      data_bytes += size;
      const auto content =
          file_digest(file, config.at("max_image_bytes").get<std::uint64_t>());
      hash.update(bytes(Json{{"path", record.path},
                             {"labels", record.labels},
                             {"bytes", size},
                             {"sha256", content.hex}}
                            .dump() +
                        "\n"));
    }
    return Json{{"samples", index.records().size()},
                {"inventory_sha256", index.source_sha256()},
                {"content_sha256", sha256_digest_hex(hash.finish())}};
  };
  const auto train_manifest = split(train, train_root);
  const auto validation_manifest = split(validation, val_root);
  Json manifest{{"schema_version", 1},
                {"classes", classes(train)},
                {"train", train_manifest},
                {"validation", validation_manifest},
                {"data_license", config.at("data_license")},
                {"image_bytes", data_bytes},
                {"identity_scope",
                 "ordered paths, labels and complete encoded image bytes"}};
  const auto identity = digest(manifest.dump());
  return {std::move(train), std::move(validation), std::move(manifest),
          identity, total_steps};
}

std::shared_ptr<data::DataPipeline> make_pipeline(
    const Json& config, image::ManifestIndex index, bool validation,
    const ArtifactDigest& identity, const Device& device,
    const std::shared_ptr<image::ImageResourceBudget>& budget,
    const TransferInput* transfer) {
  image::ImageDatasetOptions source_options;
  source_options.resource_budget = budget;
  source_options.image.channels = image::ImageChannels::rgb;
  source_options.image.dtype = image::ImageDtype::float32_normalized;
  source_options.image.limits.max_input_bytes =
      config.at("max_image_bytes").get<std::size_t>();
  source_options.image.limits.max_output_bytes =
      config.at("image_budget_bytes").get<std::size_t>();
  auto source = std::make_shared<image::ImageDataset>(
      std::filesystem::u8path(
          config.at(validation ? "validation_root" : "train_root")
              .get<std::string>()),
      std::move(index), source_options);
  const auto size = config.at("image_size").get<std::uint32_t>();
  const image::ImageGeometryStep resize{
      .kind = image::ImageGeometryKind::resize, .extent = {size, size}};
  image::JointTransformOptions transform;
  transform.interpolation = image::ImageInterpolation::bilinear;
  if (transfer) transform.interpolation = image::ImageInterpolation::bicubic;
  transform.resource_budget = budget;
  image::ImageCollateOptions collate;
  collate.resource_budget = budget;
  const auto per_batch =
      config.at("image_budget_bytes").get<std::uint64_t>() /
      (config.at("prefetch_batches").get<std::uint64_t>() + 2);
  collate.limits.max_metadata_bytes = static_cast<std::size_t>(
      std::min<std::uint64_t>(1U << 20U, per_batch / 4));
  collate.limits.max_output_bytes =
      per_batch - collate.limits.max_metadata_bytes;
  collate.limits.max_working_bytes = collate.limits.max_output_bytes;
  const auto preprocess =
      transfer ? std::optional(transfer->record.input_spec->image)
               : std::nullopt;
  image::ImageBatchTransformOptions numeric;
  numeric.resource_budget = budget;
  numeric.limits = collate.limits;
  if (preprocess) {
    numeric.image.mean = preprocess->mean;
    numeric.image.std = preprocess->std;
  }
  const auto factory = [source, resize, transform, collate, preprocess,
                        numeric](std::span<const std::size_t> indices,
                                 const data::DataWorkerContext& context,
                                 std::stop_token stop) {
    std::vector<image::VisionSample> samples;
    std::vector<image::ImageBatchKey> keys;
    samples.reserve(indices.size());
    keys.reserve(indices.size());
    for (const auto index : indices) {
      auto sample = source->get(index, stop);
      keys.push_back({sample.image_path, context.epoch, 0});
      if (preprocess) {
        const auto geometry = models::resolve_image_preprocess_geometry(
            *preprocess, {sample.height, sample.width});
        const std::array steps{
            image::ImageGeometryStep{
                .kind = image::ImageGeometryKind::resize,
                .extent = {geometry.resized_size[1], geometry.resized_size[0]}},
            image::ImageGeometryStep{
                .kind = image::ImageGeometryKind::crop,
                .extent = {geometry.output_size[1], geometry.output_size[0]},
                .left = geometry.crop_left,
                .top = geometry.crop_top}};
        samples.push_back(
            image::transform_sample(sample, steps, transform, stop));
      } else {
        samples.push_back(image::transform_sample(sample, std::span(&resize, 1),
                                                  transform, stop));
      }
    }
    auto batch = image::collate_images(samples, keys, collate, stop);
    if (preprocess) batch = image::transform_image_batch(batch, numeric, stop);
    return image::image_batch_value(batch, collate.limits);
  };
  data::DataPipelineOptions options;
  options.pipeline_id = validation ? "recipe.validation" : "recipe.training";
  options.dataset_version = identity.hex;
  options.dataset_size = source->size();
  options.batch_factory_id = config.at("recipe").get<std::string>();
  if (transfer) options.batch_factory_id += "/" + transfer->input_identity;
  options.batch_size = config.at("batch_size").get<std::size_t>();
  options.num_workers = config.at("workers").get<std::size_t>();
  options.prefetch_batches = config.at("prefetch_batches").get<std::size_t>();
  options.worker_seed = config.at("seed").get<std::uint64_t>();
  if (device.is_cuda()) {
    options.memory.pin_memory = true;
    options.memory.target_device = device;
    options.memory.non_blocking = true;
  }
  {
    const std::array<std::size_t, 1> first{0};
    const auto exemplar = image::image_batch_from_value(factory(first, {}, {}));
    options.schema = image::image_batch_schema(exemplar, device);
  }
  options.admission_factory =
      image::make_image_batch_byte_admission_factory(budget, collate.limits);
  options.resource_lease_factory =
      image::make_image_batch_resource_lease_factory(budget, collate.limits);
  return std::make_shared<data::DataPipeline>(
      std::make_shared<data::samplers::RandomSampler>(
          source->size(), config.at("seed").get<std::int64_t>()),
      options, factory);
}

class Checkpoints final : public tr::CallbackHandler {
 public:
  Checkpoints(std::uint64_t every, std::uint64_t total)
      : every_(every), total_(total) {
    descriptor_.type_id = "liuding.recipe.image-classification.checkpoint-v1";
    descriptor_.callback_id = "recipe.checkpoint";
    descriptor_.subscriptions = {tr::CallbackEventKind::batch_finished};
    descriptor_.allowed_commands = {
        tr::CallbackCommandKind::request_checkpoint};
  }
  const tr::CallbackDescriptor& descriptor() const noexcept override {
    return descriptor_;
  }
  tr::CallbackCommandRequest handle(const tr::CallbackEvent& event) override {
    if (event.phase == tr::CallbackPhase::training &&
        (event.global_step % every_ == 0 || event.global_step == total_))
      return {tr::CallbackCommandKind::request_checkpoint,
              "recipe safe boundary"};
    return {};
  }

 private:
  tr::CallbackDescriptor descriptor_;
  std::uint64_t every_, total_;
};

std::vector<std::byte> encode_model(const nn::ModelAdapter& model,
                                    const std::string& recipe) {
  std::vector<Tensor> owners;
  std::vector<interop::SafeTensorSource> tensors;
  const auto append = [&](const auto& values, const std::string& prefix) {
    for (const auto& value : values) {
      owners.push_back(value.value().detach().to(torch::kCPU).contiguous());
      const auto& tensor = owners.back();
      const bool floating = tensor.scalar_type() == torch::kFloat32;
      require_recipe(floating || tensor.scalar_type() == torch::kInt64,
                     "Unexpected ResNet18 state dtype");
      std::vector<std::uint64_t> shape;
      for (const auto extent : tensor.sizes())
        shape.push_back(static_cast<std::uint64_t>(extent));
      tensors.push_back(
          {.name = prefix + value.key(),
           .dtype = floating ? interop::SafeTensorDtype::f32
                             : interop::SafeTensorDtype::i64,
           .shape = std::move(shape),
           .data = {static_cast<const std::byte*>(tensor.const_data_ptr()),
                    tensor.nbytes()}});
    }
  };
  append(model.state().named_parameters(), "parameter.");
  append(model.state().named_buffers(), "buffer.");
  interop::SafeTensorsLimits limits;
  limits.max_file_bytes = kArtifactLimit;
  return interop::serialize_safetensors(tensors, {{"recipe", recipe}}, limits);
}

void attach(ex::SQLiteRunStore& store, const std::filesystem::path& path,
            const std::string& id, const std::string& role, std::uint64_t time,
            ex::RunArtifactKind kind, std::string_view recipe) {
  const auto hash = file_digest(path);
  if (const auto previous = store.find_reference(kRun, id)) {
    require_recipe(previous->artifact.digest.hex == hash.hex,
                   "Recipe artifact retry conflicts with stored content");
    return;
  }
  store.put_artifact_file(
      kRun, id, kind, role, path,
      {.logical_name = path_text(path.filename()),
       .media_type = "application/octet-stream",
       .producer = std::string(recipe),
       .expected_size_bytes = std::filesystem::file_size(path)},
      time, hash);
}

Json publish_bundle(const std::filesystem::path& root, const Json& config,
                    const Json& data, const Json& contract,
                    const Json& evaluation, const Json& diagnostic,
                    const BuildFingerprint& build,
                    std::span<const std::byte> weights, const Json& lineage,
                    const Json& optimization) {
  interop::ModelBundleManifest manifest;
  const bool transfer = is_transfer_recipe(config);
  const auto architecture = contract.at("architecture").get<std::string>();
  manifest.bundle_id =
      transfer ? "imagefolder-transfer-v1" : "imagefolder-resnet18-v1";
  manifest.build = build;
  const auto weight_hash = sha256_hex(weights);
  manifest.files.push_back(
      {.id = "weights",
       .relative_path = "weights.safetensors",
       .role = interop::BundleFileRole::weights,
       .artifact = {.digest = {.hex = weight_hash},
                    .size_bytes = weights.size(),
                    .logical_name = "weights",
                    .media_type = "application/safetensors",
                    .producer = config.at("recipe").get<std::string>()}});
  interop::BundleModel model;
  model.id = architecture;
  model.file_ids = {"weights"};
  const interop::SafeTensorsView view(weights);
  for (const auto& tensor : view.directory().tensors)
    model.tensors.push_back({.name = tensor.name,
                             .dtype = tensor.dtype,
                             .shape = tensor.shape,
                             .device_intent = "cpu",
                             .shard_file_id = "weights"});
  manifest.models.push_back(std::move(model));
  std::vector<interop::BundlePublishSource> sources{
      {"weights", root / "model.safetensors"}};
  std::filesystem::create_directories(root / "descriptors");
  const auto descriptor = [&](std::string id, interop::BundleFileRole role,
                              Json content) {
    content["schema_version"] = 1;
    const auto path = root / "descriptors" / (id + ".json");
    write_json(path, content);
    manifest.files.push_back(
        {.id = id,
         .relative_path = "metadata/" + id + ".json",
         .role = role,
         .artifact = {.digest = file_digest(path),
                      .size_bytes = std::filesystem::file_size(path),
                      .logical_name = id,
                      .media_type = "application/json",
                      .producer = config.at("recipe").get<std::string>()}});
    sources.push_back({id, path});
    return interop::BundleDescriptorRef{
        .schema_id = "liuding.recipe.image-classification." + id,
        .file_id = id};
  };
  manifest.input_schema = descriptor(
      "input", interop::BundleFileRole::schema,
      {{"shape",
        {nullptr, 3, config.at("image_size"), config.at("image_size")}},
       {"dtype", "float32"},
       {"layout", "nchw"}});
  manifest.output_schema =
      descriptor("output", interop::BundleFileRole::schema,
                 {{"shape", {nullptr, data.at("classes").size()}},
                  {"dtype", "float32"},
                  {"meaning", "logits"}});
  const Json preprocessing =
      transfer ? contract.at("transfer").at("descriptor").at("image")
               : Json{{"channels", "rgb"},
                      {"range", {0, 1}},
                      {"resize", config.at("image_size")},
                      {"interpolation", "bilinear"}};
  manifest.preprocessing =
      descriptor("preprocess", interop::BundleFileRole::prepost, preprocessing);
  manifest.postprocessing =
      descriptor("postprocess", interop::BundleFileRole::prepost,
                 {{"operation", "argmax"}, {"axis", 1}});
  manifest.labels = descriptor("labels", interop::BundleFileRole::labels,
                               {{"classes", data.at("classes")}});
  manifest.evaluation =
      descriptor("evaluation", interop::BundleFileRole::evaluation, evaluation);
  manifest.sbom = descriptor(
      "sbom", interop::BundleFileRole::sbom,
      {{"scope",
        "direct runtime declarations; transitive SBOM remains incomplete"},
       {"complete", false},
       {"components",
        {{{"name", "liuding-forge"}, {"version", build.jyppx_version}},
         {{"name", "libtorch"}, {"version", build.libtorch_version}}}}});
  Json provenance{
      {"lineage", lineage},
      {"config", config},
      {"dataset", data},
      {"model", contract},
      {"initialization",
       transfer ? "strict local pretrained backbone; seeded new linear head"
                : "random; no pretrained weights"},
      {"diagnostics", diagnostic}};
  if (!optimization.is_null()) provenance["optimization"] = optimization;
  manifest.provenance = descriptor(
      "provenance", interop::BundleFileRole::provenance, std::move(provenance));
  auto os_arch = build.os_arch;
  for (auto& value : os_arch)
    if (value >= 'A' && value <= 'Z')
      value = static_cast<char>(value - 'A' + 'a');
  manifest.runtime = {.backend_id = "libtorch." + architecture,
                      .backend_version = build.jyppx_version,
                      .environment_lock = digest(build_fingerprint_json(build)),
                      .dependencies = {{"libtorch", build.libtorch_version}},
                      .devices = {config.at("device").get<std::string>()},
                      .dtypes = {"float32", "int64"},
                      .os_arch = {os_arch},
                      .dynamic_shapes = true};
  if (std::filesystem::exists(root / "bundle")) {
    const auto existing =
        interop::verify_model_bundle_directory(root / "bundle");
    require_recipe(
        interop::serialize_model_bundle_manifest(existing.manifest) ==
            interop::serialize_model_bundle_manifest(manifest),
        "Existing recipe bundle has different content");
  } else {
    interop::publish_model_bundle(root / "bundle", manifest, sources);
  }
  return {
      {"manifest_sha256", file_digest(root / "bundle" / "manifest.json").hex},
      {"signed", false},
      {"transitive_sbom_complete", false}};
}

Result execute(const Arguments& arguments, bool resume) {
  const auto config = resolve_config(std::filesystem::absolute(
      std::filesystem::u8path(arguments.require("file"))));
  auto inputs = inspect_inputs(config);
  const auto transfer = is_transfer_recipe(config)
                            ? std::optional(inspect_transfer_input(config))
                            : std::nullopt;
  const Device device(config.at("device") == "cuda" ? "cuda:0" : "cpu");
  require_recipe(!device.is_cuda() || torch::cuda::is_available(),
                 "CUDA was requested but is unavailable");
  const auto pause = arguments.number("pause-after", 0, inputs.total_steps);
  require_recipe(pause == 0 || pause < inputs.total_steps,
                 "pause-after must precede the final optimizer update");
  const auto root = std::filesystem::absolute(
                        std::filesystem::u8path(arguments.require("root")))
                        .lexically_normal();
  if (resume)
    require_recipe(std::filesystem::is_directory(root),
                   "Resume requires an existing recipe directory");
  else
    require_recipe(
        std::filesystem::create_directory(root),
        "Recipe run requires a new directory; use recipe resume to continue");
  RunLock lock(root);
  const auto build = current_build_fingerprint();
  const auto build_json = parse_json(build_fingerprint_json(build));
  const auto config_hash = digest(config.dump());
  const Json numerics{{"deterministic_algorithms", true},
                      {"cudnn_benchmark", false},
                      {"tf32_cudnn", false},
                      {"tf32_cublas", false},
                      {"float32_matmul_precision", "highest"},
                      {"intraop_threads", 1},
                      {"interop_threads", 1}};
  Json contract{
      {"architecture", transfer ? config.at("model") : Json("resnet18")},
      {"num_classes", inputs.train.categories().size()},
      {"state_names", "parameter.<name>,buffer.<name>"},
      {"dtype", "float32"},
      {"numerics", numerics},
      {"recipe", config.at("recipe")}};
  if (transfer) {
    contract["transfer"] = {
        {"descriptor", transfer->descriptor},
        {"weight_input_sha256", transfer->input_identity},
        {"backbone_mode", config.at("backbone_mode")},
        {"batch_norm", config.at("batch_norm")},
        {"position_antialias", transfer->descriptor.at("position_antialias")},
        {"head", "seeded linear with bias"},
        {"dropout", "eval when frozen; follows mode when finetuning"}};
    if (uses_group_rates(config))
      contract["transfer"]["optimizer_groups"] = sgd_group_plan(config);
  }
  const Json lineage{{"config_sha256", config_hash.hex},
                     {"dataset_sha256", inputs.identity.hex},
                     {"model_contract_sha256", digest(contract.dump()).hex},
                     {"build", build_json}};
  if (resume) {
    require_recipe(parse_json(read_text(root / "lineage.json")) == lineage,
                   "Recipe resume rejected changed configuration, dataset "
                   "content, classes or build");
    require_recipe(
        parse_json(read_text(root / "config.json")) == config &&
            parse_json(read_text(root / "dataset.json")) == inputs.manifest,
        "Recipe input evidence has changed");
    if (transfer)
      require_recipe(parse_json(read_text(root / "weight-input.json")) ==
                         transfer->descriptor,
                     "Saved source weight descriptor has changed");
  } else {
    write_json(root / "lineage.json", lineage);
    write_json(root / "config.json", config);
    write_json(root / "dataset.json", inputs.manifest);
    write_json(root / "build.json", build_json);
    if (transfer) write_json(root / "weight-input.json", transfer->descriptor);
  }
  ex::SQLiteRunStore store(root / "run-store");
  if (!resume) {
    const auto time = now_ms();
    store.start({.run_id = std::string(kRun),
                 .project_id = "liuding-forge",
                 .experiment_id = "image-classification",
                 .display_name = transfer ? "ImageFolder transfer learning"
                                          : "ImageFolder ResNet18",
                 .status = ex::RunStatus::running,
                 .created_at_unix_ms = time,
                 .updated_at_unix_ms = time,
                 .lineage = {.parent_run_id = {},
                             .source_digest = build.git_commit,
                             .build_fingerprint = build_fingerprint_json(build),
                             .config_digest = config_hash.hex,
                             .data_manifest_digest = inputs.identity.hex},
                 .tags = {}});
  }
  const auto metadata = store.find(kRun);
  require_recipe(
      metadata.has_value() &&
          metadata->lineage.config_digest == config_hash.hex &&
          metadata->lineage.data_manifest_digest == inputs.identity.hex,
      "Recipe run lineage is missing or inconsistent");
  const auto time = metadata->created_at_unix_ms;
  if (metadata->status == ex::RunStatus::completed) {
    require_recipe(resume && pause == 0,
                   "Completed recipe accepts only resume without pause-after");
    const auto stored = store.find_reference(kRun, "result");
    require_recipe(stored && stored->artifact.digest.hex ==
                                 file_digest(root / "result.json").hex,
                   "Completed recipe result does not match CAS evidence");
    const auto bundle = interop::verify_model_bundle_directory(root / "bundle");
    const auto result = parse_json(read_text(root / "result.json"));
    require_recipe(
        result.at("bundle").at("manifest_sha256") == bundle.manifest_digest.hex,
        "Completed recipe bundle identity differs");
    return {result, result.at("quality_passed").get<bool>() ? 0 : 4};
  }
  require_recipe(metadata->status == ex::RunStatus::running,
                 "Recipe cannot resume a run already marked terminal; retain "
                 "its evidence and start a new run");
  for (const auto* id : {"lineage", "config", "dataset", "build"})
    attach(store, root / (std::string(id) + ".json"), id, id, time,
           ex::RunArtifactKind::artifact,
           config.at("recipe").get<std::string>());
  if (!transfer) {
    const Json definition{{"format", "jyppx.liuding.onnx_model_definition"},
                          {"schema_version", 1},
                          {"participant_id", "model.main"},
                          {"factory",
                           {{"id", "jyppx.liuding.models"},
                            {"name", "resnet18"},
                            {"num_classes", inputs.train.categories().size()},
                            {"input_channels", 3}}}};
    const auto path = root / "model-definition.json";
    if (std::filesystem::exists(path))
      require_recipe(parse_json(read_text(path)) == definition,
                     "Recipe native model definition differs");
    else
      write_json(path, definition);
    attach(store, path, "model-definition", "model.definition", time,
           ex::RunArtifactKind::artifact,
           config.at("recipe").get<std::string>());
  }
  if (transfer)
    attach(store, root / "weight-input.json", "weight-input",
           "source_weight_contract", time, ex::RunArtifactKind::artifact,
           config.at("recipe").get<std::string>());

  at::set_num_threads(1);
  at::set_num_interop_threads(1);
  at::globalContext().setDeterministicAlgorithms(true, false);
  at::globalContext().setDeterministicCuDNN(true);
  at::globalContext().setBenchmarkCuDNN(false);
  at::globalContext().setAllowTF32CuDNN(false);
  at::globalContext().setAllowTF32CuBLAS(false);
  at::globalContext().setFloat32MatmulPrecision("highest");
  require_recipe(!at::globalContext().allowTF32CuDNN() &&
                     !at::globalContext().allowTF32CuBLAS(),
                 "Recipe requires TF32 disabled for cuDNN and cuBLAS");
  core::RngGuard rng(config.at("seed").get<std::uint64_t>(), device.is_cuda());
  image::ImageResourceBudgetLimits limits;
  const auto budget_bytes =
      config.at("image_budget_bytes").get<std::uint64_t>();
  limits.max_encoded_bytes = budget_bytes;
  limits.max_decoded_bytes = budget_bytes;
  limits.max_codec_scratch_bytes = budget_bytes;
  limits.max_pinned_bytes = budget_bytes;
  limits.max_gpu_bytes = budget_bytes;
  limits.max_pipeline_inflight_bytes = budget_bytes;
  limits.max_pipeline_inflight_batches =
      config.at("prefetch_batches").get<std::uint64_t>() + 2;
  auto budget = std::make_shared<image::ImageResourceBudget>(limits);
  auto training_data =
      make_pipeline(config, inputs.train, false, inputs.identity, device,
                    budget, transfer ? &*transfer : nullptr);
  auto validation_data =
      make_pipeline(config, inputs.validation, true, inputs.identity, device,
                    budget, transfer ? &*transfer : nullptr);
  models::ModelOptions model_options;
  model_options.resnet.num_classes =
      static_cast<std::int64_t>(inputs.train.categories().size());
  auto model = transfer ? make_transfer_model(*transfer, config,
                                              model_options.resnet.num_classes)
                        : std::make_shared<nn::ModelAdapter>(
                              models::create_model("resnet18", model_options));
  const auto initial_state = transfer ? transfer_state(*model) : Json{};
  auto optimizer = make_sgd(config, *model);
  auto scheduler = make_scheduler(optimizer, config, inputs.total_steps);
  auto diagnostics = std::make_shared<tr::TrainingDiagnostics>();
  tr::Trainer trainer({.trainer_id = "recipe.trainer",
                       .run_id = std::string(kRun),
                       .trace_id = "recipe.image-classification",
                       .max_epochs = config.at("epochs").get<std::uint64_t>(),
                       .gradient_clip_norm = {},
                       .checkpoint = {}});
  const tr::BatchMapper mapper = [](const data::BatchEnvelope& batch) {
    return nn::Invocation{
        .positional = {image::image_batch_from_value(batch.value).images},
        .keyword = {}};
  };
  const tr::TargetMapper target =
      [](const data::BatchEnvelope& batch) -> std::optional<c10::IValue> {
    return image::image_batch_from_value(batch.value).labels;
  };
  const tr::StepFunction step = [](const tr::StepInput& input) {
    const auto logits = input.model.invoke(input.invocation).value.toTensor();
    const auto labels = input.target->toTensor().to(torch::kLong);
    return tr::StepOutput{
        .loss = torch::nn::functional::cross_entropy(logits, labels),
        .prediction = {},
        .metrics = {{"accuracy", logits.argmax(1)
                                     .eq(labels)
                                     .to(torch::kFloat64)
                                     .mean()
                                     .item<double>()}},
        .sample_count = input.batch.sample_ids.size()};
  };
  trainer.prepare(
      {.model = model,
       .optimizers =
           {{.optimizer = optimizer,
             .checkpoint =
                 std::make_shared<optim::SgdOptimizerCheckpointParticipant>(
                     optimizer),
             .scheduler = std::make_shared<tr::SchedulerAdapter>(scheduler)}},
       .training_data = training_data,
       .training_batch_mapper = mapper,
       .training_target_mapper = target,
       .training_step = step,
       .validation_data = validation_data,
       .validation_batch_mapper = mapper,
       .validation_target_mapper = target,
       .validation_step = step,
       .strategy = std::make_shared<tr::SingleDeviceStrategy>(
           tr::SingleDeviceStrategyOptions{.device = device}),
       .gradient_scaler = {},
       .gradient_scaler_checkpoint = {},
       .callbacks = std::make_shared<tr::CallbackDispatcher>(
           std::vector<std::shared_ptr<tr::CallbackHandler>>{
               std::make_shared<Checkpoints>(
                   config.at("checkpoint_every").get<std::uint64_t>(),
                   inputs.total_steps)}),
       .event_outbox = {},
       .model_checkpoint =
           std::make_shared<nn::ModelCheckpointParticipant>(*model),
       .rng_checkpoint = std::make_shared<core::RngCheckpointParticipant>(
           core::RngCheckpointParticipantOptions{.include_cuda =
                                                     device.is_cuda()}),
       .diagnostics = diagnostics,
       .additional_participants = {},
       .checkpoint_action = [&](const tr::CheckpointCoordinator& coordinator) {
         tr::FileCheckpointStore checkpoint(root / "latest.jyck");
         coordinator.save(checkpoint);
       }});
  if (resume) {
    tr::FileCheckpointStore checkpoint(root / "latest.jyck");
    trainer.restore_checkpoint(checkpoint);
    require_recipe(pause == 0 || pause > trainer.global_step(),
                   "pause-after must exceed the restored optimizer step");
    store.heartbeat(kRun, now_ms());
  }
  const auto fit = trainer.fit_until(pause);
  if (fit.status == tr::FitStatus::paused) {
    require_recipe(fit.global_step == pause,
                   "Recipe did not pause at its durable boundary");
    tr::FileCheckpointStore checkpoint(root / "latest.jyck");
    trainer.save_checkpoint(checkpoint);
    // A pause ends this process, not the logical run. Terminal run-store states
    // are immutable and cannot be reused as a resumable process marker.
    store.heartbeat(kRun, now_ms());
    Json paused{{"schema_version", 1},
                {"status", "paused"},
                {"global_step", fit.global_step},
                {"checkpoint_sha256", file_digest(root / "latest.jyck").hex},
                {"lineage", lineage}};
    if (uses_group_rates(config) || uses_update_schedule(config))
      paused["optimization"] = describe_optimization(
          *optimizer, *scheduler, config, inputs.total_steps);
    trainer.close();
    return {std::move(paused), 0};
  }
  require_recipe(fit.status == tr::FitStatus::completed &&
                     fit.global_step == inputs.total_steps,
                 "Recipe did not complete its configured training budget");
  tr::FileCheckpointStore final_checkpoint(root / "final.jyck");
  trainer.save_checkpoint(final_checkpoint);
  const auto final_transfer_state = transfer ? transfer_state(*model) : Json{};
  if (transfer && config.at("backbone_mode") == "frozen")
    require_recipe(initial_state.at("backbone_sha256") ==
                       final_transfer_state.at("backbone_sha256"),
                   "Frozen backbone parameters or buffers changed");
  const auto weights =
      encode_model(*model, config.at("recipe").get<std::string>());
  const auto model_hash = ArtifactDigest{.hex = sha256_hex(weights)};
  write_file(root / "model.safetensors", weights);
  validation_data->set_epoch(0);
  const auto scores = tr::evaluate_classification(
      *model,
      [&](std::stop_token stop)
          -> std::optional<tr::ClassificationEvaluationBatch> {
        const auto batch = validation_data->next(stop);
        if (!batch) return std::nullopt;
        return tr::ClassificationEvaluationBatch{
            .invocation = mapper(*batch),
            .targets = target(*batch)->toTensor(),
            .source_indices = {}};
      },
      {.num_classes = inputs.validation.categories().size(),
       .ignore_index = {},
       .sources = {},
       .max_samples = config.at("max_samples").get<std::uint64_t>()});
  require_recipe(
      scores.overall.samples == inputs.validation.records().size() &&
          scores.overall.accuracy.has_value(),
      "Recipe evaluation did not consume the entire validation split");
  const auto accuracy = std::find_if(
      scores.observations.begin(), scores.observations.end(),
      [](const auto& point) {
        return point.definition.metric_id == "classification.accuracy" &&
               point.context.at("slice") == "overall" &&
               point.context.at("source") == "all";
      });
  require_recipe(accuracy != scores.observations.end(),
                 "Classifier did not emit the overall accuracy observation");
  ex::EvaluationSuite suite{
      .suite_id = "recipe.image-classification",
      .task = "classification",
      .dataset_digest = inputs.identity,
      .environment_id = contract.at("architecture").get<std::string>() + "-" +
                        device.str() + "-" + build.os_arch,
      .seed = config.at("seed").get<std::uint64_t>(),
      .rules = {{.rule_id = "validation-accuracy",
                 .definition = accuracy->definition,
                 .context = accuracy->context,
                 .minimum_samples = scores.overall.samples,
                 .minimum = config.at("minimum_accuracy").get<double>(),
                 .maximum = 1.0,
                 .max_absolute_regression = {},
                 .max_relative_regression = {}}}};
  const ex::EvaluationIdentity evaluation_id{
      .evaluation_id = "evaluation.image-classification",
      .run_id = std::string(kRun),
      .model_digest = model_hash,
      .build_fingerprint = build_fingerprint_json(build),
      .adapter_id = "liuding.recipe.image-classification",
      .config_digest = config_hash};
  const std::array observations{*accuracy};
  const auto report =
      ex::evaluate_observations(suite, evaluation_id, observations);
  const auto quality = parse_json(ex::quality_report_json(report));
  write_file(root / "evaluation-suite.json",
             bytes(ex::evaluation_suite_json(suite)));
  write_json(root / "evaluation.json", quality);
  const auto diagnostic = parse_json(tr::training_diagnostics_report_json(
      tr::capture_training_diagnostics_report(
          *diagnostics,
          {.report_id = "diagnostic.image-classification",
           .run_id = std::string(kRun),
           .attempt_id = "attempt.final",
           .model = model_hash,
           .configuration = config_hash,
           .dataset = inputs.identity,
           .checkpoint = file_digest(root / "final.jyck")},
          tr::TrainingDiagnosticOutcome::completed)));
  write_json(root / "diagnostics.json", diagnostic);
  // Recheck before publishing success. Concurrent mutation during training is
  // still forbidden: this second pass cannot turn the input tree into a
  // snapshot.
  require_recipe(inspect_inputs(config).identity.hex == inputs.identity.hex,
                 "Dataset changed while the recipe was running");
  if (transfer)
    require_recipe(
        inspect_transfer_input(config).descriptor == transfer->descriptor,
        "Weight descriptor changed while training");
  const auto optimization =
      uses_group_rates(config) || uses_update_schedule(config)
          ? describe_optimization(*optimizer, *scheduler, config,
                                  inputs.total_steps)
          : Json{};
  const auto bundle =
      publish_bundle(root, config, inputs.manifest, contract, quality,
                     diagnostic, build, weights, lineage, optimization);
  const auto next_rng = torch::rand({16}, torch::TensorOptions().device(device))
                            .to(torch::kCPU)
                            .contiguous();
  const auto next_cpu_rng = torch::rand({16}).contiguous();
  Json resources = Json::object();
  for (const auto& [name, kind] :
       std::map<std::string, image::ImageResourceKind>{
           {"decoded_bytes", image::ImageResourceKind::decoded_bytes},
           {"pinned_bytes", image::ImageResourceKind::pinned_bytes},
           {"gpu_bytes", image::ImageResourceKind::gpu_bytes},
           {"pipeline_inflight_bytes",
            image::ImageResourceKind::pipeline_inflight_bytes}}) {
    const auto usage = budget->usage(kind);
    resources[name] = {{"peak", usage.peak}, {"limit", usage.limit}};
  }
  const bool passed = report.verdict == ex::QualityVerdict::pass;
  Json result{
      {"schema_version", 1},
      {"recipe", config.at("recipe")},
      {"status", "completed"},
      {"run_id", kRun},
      {"global_step", fit.global_step},
      {"completed_epochs", fit.completed_epochs},
      {"optimizer_steps", optimizer->completed_steps()},
      {"scheduler_steps", scheduler->completed_steps()},
      {"training_metrics", fit.training_metrics},
      {"validation_metrics", fit.validation_metrics},
      {"accuracy", *scores.overall.accuracy},
      {"confusion", scores.overall.confusion},
      {"validation_samples", scores.overall.samples},
      {"quality_passed", passed},
      {"model_sha256", model_hash.hex},
      {"numerics", numerics},
      {"lineage", lineage},
      {"bundle", bundle},
      {"next_rng_sha256",
       sha256_hex({static_cast<const std::byte*>(next_rng.const_data_ptr()),
                   next_rng.nbytes()})},
      {"next_cpu_rng_sha256",
       sha256_hex({static_cast<const std::byte*>(next_cpu_rng.const_data_ptr()),
                   next_cpu_rng.nbytes()})},
      {"image_resource_payload", resources}};
  if (transfer) {
    result["transfer"] = {{"initial", initial_state},
                          {"final", final_transfer_state},
                          {"backbone_mode", config.at("backbone_mode")},
                          {"batch_norm", config.at("batch_norm")},
                          {"weight_input_sha256", transfer->input_identity}};
  }
  if (!optimization.is_null()) result["optimization"] = optimization;
  write_json(root / "result.json", result);
  for (const auto* id :
       {"model.safetensors", "evaluation-suite.json", "evaluation.json",
        "diagnostics.json", "result.json"}) {
    const std::filesystem::path path(id);
    attach(store, root / path, path.stem().string(), path.stem().string(), time,
           ex::RunArtifactKind::artifact,
           config.at("recipe").get<std::string>());
  }
  attach(store, root / "final.jyck", "final-checkpoint", "final", time,
         ex::RunArtifactKind::checkpoint,
         config.at("recipe").get<std::string>());
  attach(store, root / "bundle" / "manifest.json", "bundle", "model_bundle",
         time, ex::RunArtifactKind::artifact,
         config.at("recipe").get<std::string>());
  store.transition(kRun, ex::RunStatus::completed, now_ms());
  trainer.close();
  return {result, passed ? 0 : 4};
}
}  // namespace
#endif

Result recipe_command(const Arguments& arguments) {
  arguments.allow({"file", "root", "pause-after", "name"});
  if (arguments.words().size() != 2U)
    usage("recipe", "Supply exactly one recipe subcommand.");
  const auto& action = arguments.words()[1];
  if (action == "schema" && arguments.optional("name") == "paired-affine")
    return paired_recipe_command(arguments);
  const bool transfer_schema =
      action == "schema" && arguments.optional("name") == "transfer";
  recipe_support::require_recipe(
      !arguments.optional("name") || transfer_schema,
      "recipe schema accepts --name paired-affine or --name transfer");
  if (action == "check" || action == "run" || action == "resume") {
    const auto config = parse_json(read_text(arguments.require("file")));
    if (config.is_object() &&
        config.value("recipe", Json{}) == "jyppx.liuding.paired-affine/v1")
      return paired_recipe_command(arguments);
  }
#if defined(JYPPX_LIUDING_CLI_IMAGE_RECIPE)
  if (action == "schema") {
    require_recipe(!arguments.optional("file") && !arguments.optional("root") &&
                       !arguments.optional("pause-after"),
                   "recipe schema accepts no file, root or pause-after");
    return {schema_json(transfer_schema), 0};
  }
  if (action == "check") {
    require_recipe(
        !arguments.optional("root") && !arguments.optional("pause-after"),
        "recipe check accepts only --file");
    const auto config = resolve_config(std::filesystem::absolute(
        std::filesystem::u8path(arguments.require("file"))));
    const auto inputs = inspect_inputs(config);
    Json result{{"valid", true},
                {"config", config},
                {"config_sha256", digest(config.dump()).hex},
                {"dataset", inputs.manifest},
                {"dataset_sha256", inputs.identity.hex},
                {"total_steps", inputs.total_steps}};
    if (is_transfer_recipe(config)) {
      const auto transfer = inspect_transfer_input(config);
      result["transfer"] = {{"descriptor", transfer.descriptor},
                            {"weight_input_sha256", transfer.input_identity}};
    }
    return {result, 0};
  }
  if (action == "run" || action == "resume")
    return execute(arguments, action == "resume");
  usage("recipe", "Choose schema, check, run or resume.");
#else
  throw NotImplemented(
      "Image classification recipes require the models and data_image "
      "components");
#endif
}
}  // namespace jyppx::liuding::cli
