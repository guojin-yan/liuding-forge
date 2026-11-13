// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "artifact_io.hpp"
#include "evaluation_diagnostics.hpp"
#include "history.hpp"
#include "run_ledger.hpp"

#include <jyppx/liuding/data/samplers/random_sampler.hpp>
#include <jyppx/liuding/foundation/random.hpp>
#include <jyppx/liuding/foundation/random_checkpoint_participant.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/nn/model_checkpoint_participant.hpp>
#include <jyppx/liuding/optim/adamw_adapter.hpp>
#include <jyppx/liuding/training/amp/grad_scaler.hpp>
#include <jyppx/liuding/training/amp/grad_scaler_checkpoint_participant.hpp>
#include <jyppx/liuding/training/callbacks/early_stopping_callback.hpp>
#include <jyppx/liuding/training/file_checkpoint_store.hpp>
#include <jyppx/liuding/training/scheduler_adapter.hpp>
#include <jyppx/liuding/training/single_device_strategy.hpp>
#include <jyppx/liuding/vision/tasks/data/targets.hpp>
#include <jyppx/liuding/vision/tasks/metrics.hpp>
#include <jyppx/liuding/vision/tasks/training/ema.hpp>
#include <jyppx/liuding/vision/tasks/training/distillation.hpp>
#include <jyppx/liuding/vision/tasks/training/model_selection.hpp>
#include <jyppx/liuding/vision/tasks/training/musgd.hpp>
#include <jyppx/liuding/vision/tasks/workflow/preflight.hpp>
#include <jyppx/liuding/vision/tasks/workflow/suite.hpp>

#include <torch/cuda.h>

#include <algorithm>
#include <jyppx/liuding/vision/tasks/data/extended.hpp>
#include <jyppx/liuding/vision/tasks/data/sampling.hpp>
#include <jyppx/liuding/vision/tasks/extended.hpp>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
namespace jyppx::liuding::vision::tasks {
namespace {
namespace image = data::image;
using J = nlohmann::json;
using I = std::int64_t;
std::string Hash(const std::string& s) {
  return sha256_hex(std::as_bytes(std::span(s)));
}
void Save(const std::filesystem::path& p, std::string_view bytes) {
  detail::publish_text(p, bytes);
}
struct Modes {
  std::vector<std::pair<std::shared_ptr<torch::nn::Module>, bool>> modes;
  explicit Modes(nn::ModelAdapter& m) {
    for (auto& v : m.state().module().modules())
      modes.emplace_back(v, v->is_training());
    m.state().eval();
  }
  ~Modes() {
    for (auto& [v, mode] : modes) v->train(mode);
  }
};
void Move(image::ImageBatch& batch, const Device& device) {
  batch.images = batch.images.to(device);
  if (batch.labels.defined()) batch.labels = batch.labels.to(device);
}
void SameClasses(const std::vector<image::VisionCategory>& a,
                 const std::vector<image::VisionCategory>& b) {
  if (a.size() != b.size())
    throw InvalidArgument("Training and validation class maps differ");
  for (std::size_t i = 0; i < a.size(); ++i)
    if (a[i].name != b[i].name || a[i].source_id != b[i].source_id)
      throw InvalidArgument("Training and validation class order differs");
}
struct Sources {
  std::shared_ptr<image::ImageDataset> train, val;
  std::string train_id, val_id;
  std::vector<image::VisionCategory> classes;
  std::shared_ptr<ImageDecodeCache> cache;
  std::vector<image::ImageExtent> extents;
  std::vector<std::int64_t> identities;
  std::filesystem::path train_root, val_root;
};
std::string ContentIdentity(const image::ImageDataset& dataset,
                            const std::filesystem::path& root,
                            std::string metadata, std::set<std::string>& seen,
                            bool validation, std::stop_token stop) {
  Sha256 identity;
  identity.update(std::as_bytes(std::span(metadata)));
  std::array<std::byte, 65536> chunk;
  for (const auto& record : dataset.index().records()) {
    const auto path = root / std::filesystem::u8path(record.path);
    const auto size = std::filesystem::file_size(path);
    if (size > (1ULL << 30U))
      throw ResourceExhausted("Vision image hash exceeds 1 GiB per source");
    std::ifstream input(path, std::ios::binary);
    if (!input) throw IoError("Cannot hash vision image");
    Sha256 digest;
    std::uint64_t read = 0;
    while (input) {
      if (stop.stop_requested())
        throw Cancelled("Vision content indexing cancelled");
      input.read(reinterpret_cast<char*>(chunk.data()), chunk.size());
      const auto count = static_cast<std::size_t>(input.gcount());
      digest.update(std::span(chunk).first(count));
      read += count;
    }
    if (!input.eof() || read != size)
      throw IoError("Vision image changed while hashing");
    const auto hex = sha256_digest_hex(digest.finish());
    if (validation && seen.contains(hex))
      throw InvalidArgument(
          "Training and validation splits share identical image content");
    if (!validation) seen.insert(hex);
    const auto item = std::to_string(record.path.size()) + ":" + record.path +
                      ":" + hex + "\n";
    identity.update(std::as_bytes(std::span(item)));
  }
  return sha256_digest_hex(identity.finish());
}
Sources Read(RecipeSpec& r, std::stop_token stop) {
  Sources s;
  image::ImageDatasetOptions decoding;
  decoding.decoded_sample_transform = [policy = r.exif_policy](
                                          image::VisionSample sample,
                                          const auto& path, auto stop) {
    return orient_vision_sample(std::move(sample), path, policy, stop);
  };
  decoding.image.channels = image::ImageChannels::rgb;
  if (r.cache.ram_bytes || r.cache.disk_bytes) {
    s.cache = std::make_shared<ImageDecodeCache>(r.cache);
    decoding.image_reader = [cache = s.cache](const auto& path,
                                              const auto& options, auto token) {
      return cache->read(path, options, token);
    };
  }
  if(!r.extension.training_manifest.empty()) {
    auto train=read_task_dataset(r.extension.training_manifest,decoding,stop);
    auto metadata=[&](const TaskDatasetImport& d){if(r.model.task==TaskKind::depth && d.depth_representation!=(r.extension.relative_inverse_depth?image::DepthRepresentation::relative_inverse_depth:image::DepthRepresentation::depth_meters))throw InvalidArgument("Dataset depth representation differs from recipe loss/metric profile");if(r.model.task==TaskKind::semantic_segment&&d.semantic_ignore!=r.extension.semantic_ignore)throw InvalidArgument("Semantic dataset/recipe ignore IDs differ");};metadata(train);
    s.train=train.dataset;s.train_root=train.image_root;s.train_id=train.identity;s.classes=train.categories;s.identities=train.identity_labels;
    if(r.rectangular_batches)for(std::size_t i=0;i<s.train->size();++i){auto sample=s.train->get(i,stop);s.extents.push_back({sample.width,sample.height});}
    if(!r.extension.validation_manifest.empty()) {
      auto val=read_task_dataset(r.extension.validation_manifest,decoding,stop);metadata(val);if(r.model.task!=TaskKind::reid)SameClasses(s.classes,val.categories);else{std::set<std::int64_t> ids;for(const auto& c:train.categories)ids.insert(c.source_id);for(const auto& c:val.categories)if(ids.contains(c.source_id))throw InvalidArgument("ReID train/validation identity inventories overlap");}
      std::set<std::string> train_paths,sequence_ids(train.sequence_ids.begin(),train.sequence_ids.end());
      for(const auto& a:train.dataset->index().records())train_paths.insert(std::filesystem::weakly_canonical(train.image_root/std::filesystem::u8path(a.path)).generic_string());
      for(const auto& a:val.dataset->index().records())if(train_paths.contains(std::filesystem::weakly_canonical(val.image_root/std::filesystem::u8path(a.path)).generic_string()))throw InvalidArgument("Task dataset train/validation paths overlap");
      for(const auto& id:val.sequence_ids)if(sequence_ids.contains(id))throw InvalidArgument("Task datasets share sequence IDs across splits");
      s.val=val.dataset;s.val_root=val.image_root;s.val_id=val.identity;
    }
  } else if (r.model.task == TaskKind::classify) {
    s.train_root = r.classification_root;
    s.val_root = r.classification_validation_root;
    auto index = image::index_image_folder(r.classification_root, {}, stop);
    s.train_id = index.source_sha256();
    s.classes = index.categories();
    s.train = std::make_shared<image::ImageDataset>(r.classification_root,
                                                    std::move(index), decoding);
    if (!r.classification_validation_root.empty()) {
      auto val =
          image::index_image_folder(r.classification_validation_root, {}, stop);
      SameClasses(s.classes, val.categories());
      if (std::filesystem::equivalent(r.classification_root,
                                      r.classification_validation_root))
        throw InvalidArgument("Training and validation roots overlap");
      // Reuse canonical split-path validation for all samples, including linked
      // roots.
      DetectionDatasetImport train_paths, val_paths;
      train_paths.image_root = r.classification_root;
      val_paths.image_root = r.classification_validation_root;
      for (const auto& record : s.train->index().records()) {
        image::VisionAnnotationRecord v;
        v.image_path = record.path;
        train_paths.annotations.images.push_back(v);
      }
      for (const auto& record : val.records()) {
        image::VisionAnnotationRecord v;
        v.image_path = record.path;
        val_paths.annotations.images.push_back(v);
      }
      require_disjoint_detection_splits(train_paths, val_paths, stop);
      s.val_id = val.source_sha256();
      s.val = std::make_shared<image::ImageDataset>(
          r.classification_validation_root, std::move(val), decoding);
    }
  } else {
    auto dialect = [&](DetectionDatasetOptions& source) {
      source.task =
          r.model.task == TaskKind::instance_segment
              ? DatasetAnnotationTask::instance_segment
          : r.model.task == TaskKind::pose ? DatasetAnnotationTask::pose
          : r.model.task == TaskKind::obb  ? DatasetAnnotationTask::obb
                                           : DatasetAnnotationTask::detect;
      if (r.model.task == TaskKind::pose) {
        source.keypoint_count = r.model.keypoints;
        source.keypoint_dimensions =
            static_cast<std::uint8_t>(r.model.keypoint_dimensions);
      }
    };
    dialect(r.training);
    if (r.validation) dialect(*r.validation);
    auto train = read_detection_dataset(r.training, stop);
    s.train_root = train.image_root;
    s.classes = train.annotations.categories;
    s.train_id = train.metadata_identity;
    if (r.model.task == TaskKind::pose && !train.flip_indices.empty() &&
        r.data.letterbox.geometry.horizontal_keypoint_permutations.empty())
      r.data.letterbox.geometry.horizontal_keypoint_permutations.assign(
          s.classes.size(), train.flip_indices);
    for (const auto& record : train.annotations.images) {
      image::ImageExtent extent{record.width, record.height};
      if (r.exif_policy == ExifPolicy::encoded_coordinates &&
          image_exif_orientation(train.image_root /
                                 std::filesystem::u8path(record.image_path)) >=
              5)
        std::swap(extent.width, extent.height);
      s.extents.push_back(extent);
    }
    if (r.validation) {
      auto source = *r.validation;
      if (!source.categories.empty()) SameClasses(s.classes, source.categories);
      source.categories = s.classes;
      auto val = read_detection_dataset(source, stop);
      s.val_root = val.image_root;
      require_disjoint_detection_splits(train, val, stop);
      s.val_id = val.metadata_identity;
      s.val = std::make_shared<image::ImageDataset>(
          val.image_root, std::move(val.annotations), decoding);
    }
    s.train = std::make_shared<image::ImageDataset>(
        train.image_root, std::move(train.annotations), decoding);
  }
  if (s.classes.size() != static_cast<std::size_t>(r.model.num_classes))
    throw InvalidArgument(
        "Recipe class count differs from imported training data");
  if (r.verify_image_contents) {
    std::set<std::string> seen;
    s.train_id =
        ContentIdentity(*s.train, s.train_root, s.train_id, seen, false, stop);
    if (s.val)
      s.val_id =
          ContentIdentity(*s.val, s.val_root, s.val_id, seen, true, stop);
  }
  return s;
}
}  // namespace
VisionDataSplits load_vision_data(RecipeSpec& recipe, std::stop_token stop) {
  check_recipe(recipe);
  auto source = Read(recipe, stop);
  return {source.train, source.val, source.train_id, source.val_id,
          source.classes};
}
TrainResult train_vision(const TrainRequest& request) {
  auto r = request.recipe;
  check_recipe(r);
  if (request.output.empty())
    throw InvalidArgument("Training output directory is required");
  if (std::filesystem::exists(request.output) &&
      !std::filesystem::is_empty(request.output))
    throw InvalidArgument(
        "Training needs a new or empty output directory, including resume");
  if (request.stop.stop_requested())
    throw Cancelled("Vision training cancelled before initialization");
  auto source = Read(r, request.stop);
  r.data.size_divisor =
      static_cast<std::uint32_t>(describe_binding(r.model).input_divisor);
  r.data.rectangular = r.rectangular_batches;
  r.data.num_classes = r.model.num_classes;
  r.data.allow_unannotated_negatives = r.model.task != TaskKind::detect;
  r.data.classification.enabled = r.model.task == TaskKind::classify || r.model.task == TaskKind::classify_multilabel || r.model.task == TaskKind::reid;
  if(r.model.task==TaskKind::semantic_segment)r.data.letterbox.geometry.mask_padding_id=r.extension.semantic_ignore;
  r.data.augmentation.multi_source.total_epochs = r.epochs;
  r.data.augmentation.multi_source.close_epochs =
      (std::min)(r.data.augmentation.multi_source.close_epochs, r.epochs);
  r.data.augmentation.seed = r.seed;
  r.data.augmentation.dataset_id = source.train_id;
  r.optimizer_name = resolve_vision_optimizer(r);
  // Resolution happens before run identity, optimizer creation and checkpoint
  // binding.
  if (r.selection_metric == "auto") {
    r.selection_metric =
        r.full_validation_metrics
            ? (r.model.task == TaskKind::classify ? "validation.top1" : r.model.task==TaskKind::classify_multilabel?"validation.map":r.model.task==TaskKind::semantic_segment?"validation.miou":r.model.task==TaskKind::depth?"validation.rmse":r.model.task==TaskKind::reid?"validation.rank1":r.model.task==TaskKind::pretrain_simmim?"validation.loss":"validation.map")
            : "validation.loss";
    r.selection_maximize = r.selection_metric != "validation.loss"&&r.selection_metric!="validation.rmse"&&r.selection_metric!="validation.mae"&&r.selection_metric!="validation.absrel";
  }
  const auto recipe_id = recipe_identity(r);
  const auto run_id =
      Hash(recipe_id + ":" + source.train_id + ":" + source.val_id + ":" +
           describe_binding(r.model).revision);
  torch::manual_seed(r.seed);
  auto binding = bind_task(r);
  binding.model->state().to(r.device);
  auto initialization = r.initialization;
  if (!request.resume.empty()) {
    initialization.weights.clear();
    initialization.sha256.clear();
    initialization.key_mapping.clear();
    initialization.ignored_source_keys.clear();
    initialization.retain_buffer_keys.clear();
    initialization.replace_head_prefixes.clear();
  }
  const auto initialization_report = initialize_vision(
      *binding.model, initialization, r.model.architecture, request.stop);
  std::shared_ptr<nn::ModelAdapter> teacher;
  if(!r.extension.teacher_architecture.empty()) {
    auto teacher_recipe=r;teacher_recipe.model.architecture=r.extension.teacher_architecture;teacher_recipe.extension.teacher_architecture.clear();teacher_recipe.extension.feature_projection=0;
    teacher_recipe.extension.feature_outputs=r.extension.feature_distillation_weight>0;
    teacher=bind_task(teacher_recipe).model;teacher->state().to(r.device);
    initialize_vision(*teacher,r.extension.teacher_initialization,teacher_recipe.model.architecture,request.stop);
    binding.step=make_distillation_step(binding.step,teacher,{.temperature=r.extension.distillation_temperature,.logits_weight=r.extension.logits_distillation_weight,.feature_weight=r.extension.feature_distillation_weight,.multilabel=r.model.task==TaskKind::classify_multilabel});
  }
  struct Performance {
    std::uint64_t samples = 0, batches = 0, peak_batch_bytes = 0;
  };
  auto performance = std::make_shared<Performance>();
  const auto training_step = binding.step;
  binding.step = [training_step, policy = r.initialization, performance,
                  measured = r.diagnostics](const training::StepInput& input) {
    enforce_frozen_modules(input.model, policy);
    auto output = training_step(input);
    if (measured && input.context.phase == training::StepPhase::training) {
      performance->samples += output.sample_count;
      ++performance->batches;
      const auto bytes = input.batch.transformed_memory
                             ? input.batch.transformed_memory->tensor_bytes
                             : input.batch.memory.tensor_bytes;
      performance->peak_batch_bytes =
          (std::max)(performance->peak_batch_bytes, bytes);
    }
    return output;
  };
  auto eval_binding = binding;
  std::shared_ptr<ModelEma> ema;
  if (r.ema) {
    const auto rng = core::get_rng_state(r.device.is_cuda());
    try {
      eval_binding = bind_task(r);
    } catch (...) {
      core::set_rng_state(rng, r.device.is_cuda());
      throw;
    }
    core::set_rng_state(rng, r.device.is_cuda());
    eval_binding.model->state().to(r.device);
    ema = std::make_shared<ModelEma>(
        binding.model, eval_binding.model,
        ModelEmaOptions{
            .source_model_id =
                "model." + Hash(J::parse(recipe_json(r)).at("model").dump() +
                                ":" + binding.capability.revision),
            .decay = r.ema_decay,
            .ramp_updates = r.ema_ramp_updates});
  }
  r.optimizer.effective_batch = r.batch_size * r.accumulate_steps;
  auto groups =
      make_detection_sgd_groups(binding.model->state().module(), r.optimizer);
  apply_parameter_rates(groups, r.initialization.learning_rates);
  std::shared_ptr<optim::OptimizerAdapter> optimizer;
  std::shared_ptr<CheckpointParticipant> optimizer_checkpoint;
  if (r.optimizer_name == "sgd") {
    auto concrete = std::make_shared<optim::SgdOptimizerAdapter>(
        "optimizer.vision", std::move(groups.groups));
    optimizer = concrete;
    optimizer_checkpoint =
        std::make_shared<optim::SgdOptimizerCheckpointParticipant>(concrete);
  } else if (r.optimizer_name == "adamw") {
    std::vector<optim::AdamWParameterGroup> converted;
    for (auto& g : groups.groups) {
      optim::AdamWParameterGroup group;
      group.group_id = g.group_id;
      group.parameters = std::move(g.parameters);
      group.options.learning_rate = g.options.learning_rate;
      group.options.weight_decay = g.options.weight_decay;
      group.options.beta1 = r.optimizer.momentum;
      converted.push_back(std::move(group));
    }
    auto concrete = std::make_shared<optim::AdamWOptimizerAdapter>(
        "optimizer.vision", std::move(converted));
    optimizer = concrete;
    optimizer_checkpoint =
        std::make_shared<optim::AdamWOptimizerCheckpointParticipant>(concrete);
  } else if (r.optimizer_name == "musgd") {
    auto concrete = std::make_shared<MuSgdOptimizer>("optimizer.vision",
                                                     std::move(groups.groups));
    optimizer = concrete;
    optimizer_checkpoint = concrete;
  } else
    throw InvalidArgument("Optimizer was not resolved before native binding");
  const auto batches = (source.train->size() + r.batch_size - 1) / r.batch_size;
  const auto updates = (batches + r.accumulate_steps - 1) / r.accumulate_steps;
  if (!updates ||
      r.epochs >
          static_cast<std::uint64_t>((std::numeric_limits<I>::max)()) / updates)
    throw ResourceExhausted("Planned optimizer update count exceeds int64");
  auto schedule = r.scheduler;
  schedule.total_updates = r.epochs * updates;
  auto scheduler = std::make_shared<training::SchedulerAdapter>(
      std::make_shared<WarmupCosineLrScheduler>(optimizer, schedule));
  data::DataPipelineOptions p;
  p.pipeline_id = "vision.training";
  p.dataset_version = source.train_id;
  p.batch_size = r.batch_size;
  p.num_workers = r.workers;
  p.prefetch_batches = r.prefetch_batches;
  p.memory.target_device = r.device;
  p.memory.pin_memory = r.pin_memory;
  std::shared_ptr<data::samplers::Sampler> sampler =
      std::make_shared<data::samplers::RandomSampler>(source.train->size(),
                                                      r.seed);
  if(r.model.task==TaskKind::reid) sampler=make_identity_sampler(source.identities,r.batch_size,2,r.seed);
  if (r.rectangular_batches) {
    if (r.model.task == TaskKind::classify)
      throw InvalidArgument(
          "Rectangular ordering requires indexed detection extents");
    sampler = make_rect_sampler(source.extents, r.batch_size, r.seed);
  }
  auto train_data =
      make_yolo_data_pipeline(source.train, p, r.data, std::move(sampler));
  auto val_options = yolo_evaluation_options(r.data);
  std::shared_ptr<data::DataPipeline> validation;
  std::shared_ptr<ModelSelectionCallback> selection;
  std::vector<std::shared_ptr<training::CallbackHandler>> callbacks;
  if (source.val) {
    p.pipeline_id = "vision.validation";
    p.dataset_version = source.val_id;
    p.dataset_size = 0;
    val_options.augmentation.dataset_id = source.val_id;
    validation = make_yolo_data_pipeline(source.val, p, val_options);
    selection = std::make_shared<ModelSelectionCallback>(
        ModelSelectionOptions{.monitor = r.selection_metric,
                              .maximize = r.selection_maximize,
                              .save_interval_epochs = r.save_interval_epochs});
    callbacks.push_back(selection);
    if (r.early_stopping_patience)
      callbacks.push_back(
          std::make_shared<training::callbacks::EarlyStoppingCallback>(
              training::callbacks::EarlyStoppingOptions{
                  .monitor = r.selection_metric,
                  .patience = r.early_stopping_patience,
                  .mode =
                      r.selection_maximize
                          ? training::callbacks::EarlyStoppingMode::maximize
                          : training::callbacks::EarlyStoppingMode::minimize}));
  }
  std::filesystem::create_directories(request.output);
  Save(request.output / "recipe.json", recipe_json(r));
  Save(request.output / "initialization.json", initialization_report);
  Save(request.output / "identity.json",
       J{{"schema", "vision.run/v1"},
         {"run_identity", run_id},
         {"recipe_identity", recipe_id},
         {"training_data", source.train_id},
         {"validation_data", source.val_id},
         {"architecture_revision", binding.capability.revision},
         {"evaluation_model", r.ema ? "ema" : "raw"}}
           .dump(2));
  detail::RunLedger ledger(
      request.output / "experiment", "vision." + run_id, recipe_id,
      Hash(source.train_id + ":" + source.val_id),
      {{"architecture", r.model.architecture},
       {"resume", request.resume.empty() ? "false" : "true"}});
  ledger.file(request.output / "recipe.json", "recipe");
  ledger.file(request.output / "identity.json", "identity");
  training::TrainerOptions options;
  options.trainer_id = "trainer.vision";
  options.run_id = "vision." + run_id;
  options.trace_id = options.run_id;
  options.max_epochs = r.epochs;
  options.accumulate_steps = r.accumulate_steps;
  options.precision = r.precision;
  options.gradient_clip_norm = r.gradient_clip_norm;
  training::FitRequest fit;
  fit.model = binding.model;
  fit.optimizers = {{optimizer, optimizer_checkpoint, scheduler}};
  std::shared_ptr<training::TrainingDiagnostics> diagnostics;
  if (r.diagnostics) {
    diagnostics =
        std::make_shared<training::TrainingDiagnostics>(r.diagnostics_options);
    fit.diagnostics = diagnostics;
  }
  fit.training_data = train_data;
  fit.training_batch_mapper = binding.image_mapper;
  fit.training_target_mapper = binding.target_mapper;
  fit.training_step = binding.step;
  if (validation) {
    fit.validation_data = validation;
    fit.validation_batch_mapper = eval_binding.image_mapper;
    fit.validation_target_mapper = eval_binding.target_mapper;
    fit.validation_step = [eval_binding](
                              const training::StepInput& input) mutable {
      Modes eval_modes(*eval_binding.model);
      return eval_binding.step({*eval_binding.model, input.batch,
                                input.invocation, input.target, input.context});
    };
  }
  if (validation && r.full_validation_metrics)
    fit.validation_summary = [eval_binding, dataset = source.val,
                              identity = source.val_id, r,
                              path = request.output](
                                 std::stop_token stop) mutable {
      auto report =
          J::parse(evaluate_vision(eval_binding, *dataset, r, identity, stop));
      Save(path / "validation.json", report.dump(2));
      std::map<std::string, double> metrics;
      for (auto key : {"map", "ap50", "top1", "top5", "macro_f1", "micro_f1", "miou", "pixel_accuracy", "mae", "rmse", "absrel", "delta1", "rank1"})
        if (report.contains(key) && report[key].is_number())
          metrics[std::string("validation.") + key] = report[key].get<double>();
      return metrics;
    };
  callbacks.push_back(std::make_shared<detail::History>(
      request.output, optimizer, r.tensorboard));
  fit.strategy = std::make_shared<training::SingleDeviceStrategy>(
      training::SingleDeviceStrategyOptions{.device = r.device});
  if (r.precision == training::PrecisionMode::cuda_float16) {
    fit.gradient_scaler = std::make_shared<training::amp::GradScaler>();
    fit.gradient_scaler_checkpoint =
        std::make_shared<training::amp::GradScalerCheckpointParticipant>(
            fit.gradient_scaler);
  }
  fit.callbacks =
      std::make_shared<training::CallbackDispatcher>(std::move(callbacks));
  fit.model_checkpoint =
      std::make_shared<nn::ModelCheckpointParticipant>(*binding.model);
  fit.rng_checkpoint = std::make_shared<core::RngCheckpointParticipant>(
      core::RngCheckpointParticipantOptions{.include_cuda =
                                                r.device.is_cuda()});
  if(teacher)fit.additional_participants.push_back(std::make_shared<nn::ModelCheckpointParticipant>(*teacher,nn::ModelCheckpointParticipantOptions{.participant_id="model.teacher"}));
  if (ema) fit.optimizer_update_participants.push_back(ema);
  if (selection)
    fit.checkpoint_action = [selection, path = request.output, binding,
                             eval_binding,
                             r](const training::CheckpointCoordinator& c) {
      if (selection->state().last_improved) {
        training::FileCheckpointStore store(path / "best.checkpoint");
        c.save(store);
        save_vision_weights(*eval_binding.model, path / "best.safetensors",
                            {{"architecture", r.model.architecture},
                             {"variant", r.model.variant},
                             {"role", r.ema ? "ema" : "raw"}},
                            r.initialization.max_bytes);
      }
      if (r.save_interval_epochs &&
          (selection->state().observations % r.save_interval_epochs) == 0) {
        training::FileCheckpointStore periodic(
            path / ("epoch-" + std::to_string(selection->state().observations) +
                    ".checkpoint"));
        c.save(periodic);
      }
      training::FileCheckpointStore store(path / "last.checkpoint");
      c.save(store);
    };
  training::Trainer trainer(options);
  trainer.prepare(std::move(fit));
  if (!request.resume.empty()) {
    training::FileCheckpointStore store(request.resume);
    trainer.restore_checkpoint(store);
  }
  if (r.diagnostics && r.device.is_cuda())
    torch::cuda::synchronize(r.device.index());
  const auto fit_start = r.diagnostics
                             ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
  auto result =
      request.pause_after_updates
          ? trainer.fit_until(request.pause_after_updates, request.stop)
          : trainer.fit(request.stop);
  if (r.diagnostics && r.device.is_cuda())
    torch::cuda::synchronize(r.device.index());
  const auto fit_seconds =
      r.diagnostics ? std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - fit_start)
                          .count()
                    : 0.;
  auto last = request.output / "last.checkpoint";
  training::FileCheckpointStore store(last);
  trainer.save_checkpoint(store);
  trainer.close();
  Save(request.output / "training.json",
       J{{"schema", "vision.training_result/v1"},
         {"run_identity", run_id},
         {"status", static_cast<int>(result.status)},
         {"reason", result.reason},
         {"epochs", result.completed_epochs},
         {"updates", result.global_step},
         {"training", result.training_metrics},
         {"validation", result.validation_metrics}}
           .dump(2));
  if (diagnostics) {
    const auto outcome = result.status == training::FitStatus::completed ||
                                 result.status == training::FitStatus::stopped
                             ? training::TrainingDiagnosticOutcome::completed
                         : result.status == training::FitStatus::cancelled
                             ? training::TrainingDiagnosticOutcome::cancelled
                             : training::TrainingDiagnosticOutcome::incomplete;
    auto report = training::capture_training_diagnostics_report(
        *diagnostics,
        {.report_id = "vision.diagnostics",
         .run_id = "vision." + run_id,
         .attempt_id = Hash(request.output.generic_string()),
         .model = {.hex = Hash(J::parse(recipe_json(r)).at("model").dump() +
                               ":" + binding.capability.revision)},
         .configuration = {.hex = recipe_id},
         .dataset = {.hex = Hash(source.train_id + ":" + source.val_id)}},
        outcome);
    Save(request.output / "diagnostics.json",
         training::training_diagnostics_report_json(report));
    Save(
        request.output / "performance.json",
        J{{"schema", "vision.segment_performance/v1"},
          {"run_identity", run_id},
          {"measurement", "fit_wall_with_validation_and_checkpoint_actions"},
          {"segment_training_samples", performance->samples},
          {"segment_training_batches", performance->batches},
          {"fit_seconds", fit_seconds},
          {"training_samples_per_fit_second",
           fit_seconds > 0 ? J(performance->samples / fit_seconds)
                           : J(nullptr)},
          {"maximum_logical_batch_tensor_bytes", performance->peak_batch_bytes},
          {"device_synchronized", r.device.is_cuda()},
          {"process_rss_measured", false}}
            .dump(2));
  }
  std::map<std::string, std::string> metadata{
      {"architecture", r.model.architecture},
      {"variant", r.model.variant},
      {"task", std::to_string(static_cast<int>(r.model.task))},
      {"num_classes", std::to_string(r.model.num_classes)},
      {"run_identity", run_id},
      {"role", "raw"}};
  const auto raw_digest =
      save_vision_weights(*binding.model, request.output / "last.safetensors",
                          metadata, r.initialization.max_bytes);
  J artifacts = {
      {"raw", {{"file", "last.safetensors"}, {"sha256", raw_digest}}}};
  if (ema) {
    metadata["role"] = "ema";
    artifacts["ema"] = {
        {"file", "ema.safetensors"},
        {"sha256", save_vision_weights(*eval_binding.model,
                                       request.output / "ema.safetensors",
                                       metadata, r.initialization.max_bytes)}};
  }
  Save(request.output / "weights.json", artifacts.dump(2));
  if (!request.stop.stop_requested()) {
    for (auto name :
         {"training.json", "initialization.json", "weights.json",
          "validation.json", "metrics.json", "metrics.csv", "report.html",
          "diagnostics.json", "performance.json", "last.safetensors",
          "ema.safetensors", "best.safetensors"})
      ledger.file(request.output / name, name, false, request.stop);
    ledger.file(last, "last.checkpoint", true, request.stop);
    ledger.file(request.output / "best.checkpoint", "best.checkpoint", true,
                request.stop);
  }
  ledger.finish(result.status == training::FitStatus::completed ||
                        result.status == training::FitStatus::stopped
                    ? experiment::RunStatus::completed
                : result.status == training::FitStatus::cancelled
                    ? experiment::RunStatus::cancelled
                    : experiment::RunStatus::interrupted);
  if (source.cache) {
    auto stats = source.cache->stats();
    Save(request.output / "cache.json", J{{"ram_hits", stats.ram_hits},
                                          {"disk_hits", stats.disk_hits},
                                          {"misses", stats.misses},
                                          {"invalid", stats.invalid},
                                          {"ram_bytes", stats.ram_bytes},
                                          {"disk_bytes", stats.disk_bytes}}
                                            .dump(2));
  }
  return {std::move(result), run_id, last, binding.model, eval_binding.model};
}
std::vector<ImageResult> predict_vision(
    TaskBinding& binding, std::span<const image::VisionSample> samples,
    const RecipeSpec& recipe, bool restore, std::stop_token stop) {
  return predict_vision_profiled(binding, samples, recipe, 0, restore, stop)
      .results;
}
ProfiledPrediction predict_vision_profiled(
    TaskBinding& binding, std::span<const image::VisionSample> samples,
    const RecipeSpec& recipe, std::size_t warmup, bool restore,
    std::stop_token stop) {
  if (warmup > 100)
    throw InvalidArgument("Inference warmup exceeds 100 iterations");
  if (samples.empty()) return {};
  torch::NoGradGuard guard;
  Modes modes(*binding.model);
  auto sync = [&]() {
    if (recipe.device.is_cuda())
      torch::cuda::synchronize(recipe.device.index());
  };
  sync();
  const auto start = std::chrono::steady_clock::now();
  auto data = yolo_evaluation_options(recipe.data);
  data.size_divisor = static_cast<std::uint32_t>(binding.capability.input_divisor);
  data.num_classes = recipe.model.num_classes;
  data.allow_unannotated_negatives = true;
  if(data.augmentation.dataset_id.empty())data.augmentation.dataset_id="vision.predict:"+recipe_identity(recipe);
  data.classification.enabled = recipe.model.task == TaskKind::classify || recipe.model.task == TaskKind::classify_multilabel || recipe.model.task == TaskKind::reid;
  std::vector<image::ImageBatchKey> keys;
  for (const auto& s : samples) keys.push_back({.id = s.image_path});
  auto batch = prepare_yolo_batch(samples, keys, data, stop);
  Move(batch, recipe.device);
  data::BatchEnvelope envelope;
  envelope.value = image::image_batch_value(batch, data.limits);
  nn::Invocation invocation;
  if (binding.capability.component == "models_rcnn") {
    auto sizes =
        torch::tensor(
            std::vector<float>{static_cast<float>(batch.images.size(2)),
                               static_cast<float>(batch.images.size(3))},
            batch.images.options())
            .view({1, 2})
            .expand({batch.images.size(0), 2});
    invocation.positional = {
        batch.images, torch::empty({0, 6}, batch.images.options()), sizes};
  } else
    invocation = binding.image_mapper(envelope);
  sync();
  const auto prepared = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < warmup; ++i) {
    if (stop.stop_requested()) throw Cancelled("Inference warmup cancelled");
    binding.model->invoke(invocation);
  }
  sync();
  const auto ready = std::chrono::steady_clock::now();
  auto raw = binding.model->invoke(invocation);
  sync();
  const auto executed = std::chrono::steady_clock::now();
  auto result =
      binding.decode(raw.value, {batch.images.size(2), batch.images.size(3)},
                     recipe.prediction);
  for (std::size_t i = 0; i < result.size(); ++i) {
    result[i].source_id = samples[i].image_path;
    if (restore && (recipe.model.task==TaskKind::detect || recipe.model.task==TaskKind::instance_segment || recipe.model.task==TaskKind::pose || recipe.model.task==TaskKind::obb || is_dense_task(recipe.model.task))) {
      if (!batch.items[i].geometry)
        throw InvalidArgument("Predictive inverse geometry is missing");
      result[i] = restore_result(result[i], *batch.items[i].geometry,
                                 recipe.prediction.max_mask_elements);
    } else
      result[i].extent = {samples[i].width, samples[i].height};
  }
  sync();
  const auto finished = std::chrono::steady_clock::now();
  auto ms = [](auto duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
  };
  return {std::move(result),
          {ms(prepared - start), ms(executed - ready), ms(finished - executed),
           ms(ready - prepared), samples.size(), warmup}};
}
std::string evaluate_vision(TaskBinding& binding,
                            const image::ImageDataset& dataset,
                            const RecipeSpec& recipe, std::string dataset_id,
                            std::stop_token stop) {
  if (dataset.empty() || dataset_id.empty())
    throw InvalidArgument(
        "Evaluation needs a complete nonempty identified split");
  torch::NoGradGuard guard;
  Modes modes(*binding.model);
  auto r = recipe;
  if(static_cast<unsigned>(r.model.task)>4)return evaluate_extended_task(binding,dataset,r,std::move(dataset_id),stop);
  r.prediction = r.evaluation;
  const auto class_id =
      Hash(std::to_string(r.model.num_classes) + ":" + dataset_id);
  std::unique_ptr<training::metrics::DetectionAveragePrecisionMetric> boxes;
  std::unique_ptr<TaskMetric> rich;
  if (r.model.task == TaskKind::detect)
    boxes =
        std::make_unique<training::metrics::DetectionAveragePrecisionMetric>(
            training::metrics::DetectionMetricOptions{
                .dataset_id = dataset_id,
                .class_vocabulary_id = class_id,
                .num_classes = static_cast<std::size_t>(r.model.num_classes),
                .max_detections_per_image_class =
                    static_cast<std::size_t>(r.evaluation.max_detections)});
  else if (r.model.task != TaskKind::classify)
    rich = std::make_unique<TaskMetric>(TaskMetricOptions{
        .task = r.model.task,
        .dataset_id = dataset_id,
        .classes_id = class_id,
        .num_classes = static_cast<std::size_t>(r.model.num_classes),
        .keypoint_sigmas = r.keypoint_sigmas,
        .max_detections_per_image_class =
            static_cast<std::size_t>(r.evaluation.max_detections),
        .max_mask_elements = r.evaluation.max_mask_elements});
  detail::EvaluationDiagnostics diagnostics;
  J per_class = J::array();
  std::uint64_t correct1 = 0, correct5 = 0;
  J confusion = J::array(), failures = J::array();
  std::map<std::pair<I, I>, std::uint64_t> cells;
  if (r.model.task == TaskKind::classify) {
    r.prediction.confidence = 0;
    r.prediction.classes.clear();
    r.prediction.max_candidates = (std::max)(I{5}, r.prediction.max_candidates);
    r.prediction.max_detections = (std::max)(I{5}, r.prediction.max_detections);
  }
  std::size_t maximum_targets = 0, capacity_images = 0;
  std::vector<std::uint64_t> support(
      static_cast<std::size_t>(r.model.num_classes)),
      correct(support.size());
  for (std::size_t i = 0; i < dataset.size(); ++i) {
    if (stop.stop_requested()) throw Cancelled("Vision evaluation cancelled");
    auto sample = dataset.get(i, stop);
    const std::array<image::VisionSample, 1> samples{sample};
    auto predicted = predict_vision(binding, samples, r, true, stop).at(0);
    if (r.model.task != TaskKind::classify) {
      if (!sample.annotation)
        throw InvalidArgument("Evaluation needs annotations");
      diagnostics.update(predicted, *sample.annotation);
      const auto targets = static_cast<std::size_t>(std::count_if(
          sample.annotation->objects.begin(), sample.annotation->objects.end(),
          [](const auto& o) { return !o.ignore && !o.difficult && !o.crowd; }));
      maximum_targets = (std::max)(maximum_targets, targets);
      capacity_images +=
          targets > static_cast<std::size_t>(r.evaluation.max_detections);
    }
    if (r.model.task == TaskKind::classify) {
      if (sample.labels.size() != 1)
        throw InvalidArgument(
            "Classification evaluation requires one target label");
      // AP thresholds must not truncate classifier top-k.
      auto& result = predicted;
      auto label = sample.labels.front();
      ++support.at(label);
      auto chosen =
          result.classes.numel() ? result.classes[0].item<I>() : I{-1};
      ++cells[{static_cast<I>(label), chosen}];
      if (chosen != static_cast<I>(label) && failures.size() < 256)
        failures.push_back({{"source", sample.image_path},
                            {"target", label},
                            {"prediction", chosen}});
      if (result.classes.numel() &&
          result.classes[0].item<I>() == static_cast<I>(label)) {
        ++correct1;
        ++correct[label];
      }
      if ((result.classes.slice(0, 0,
                                (std::min)(I{5}, result.classes.numel())) ==
           static_cast<I>(label))
              .any()
              .item<bool>())
        ++correct5;
    } else if (boxes) {
      training::metrics::DetectionImage item;
      item.image_id = i + 1;
      if (!sample.annotation)
        throw InvalidArgument("Evaluation requires explicit annotations");
      for (const auto& object : sample.annotation->objects)
        if (object.box) {
          const auto& b = *object.box;
          item.targets.push_back({{b.x_min, b.y_min, b.x_max, b.y_max},
                                  object.category,
                                  object.crowd,
                                  object.ignore || object.difficult,
                                  object.area});
        }
      for (I row = 0; row < predicted.boxes.size(0); ++row)
        item.predictions.push_back(
            {{predicted.boxes[row][0].item<double>(),
              predicted.boxes[row][1].item<double>(),
              predicted.boxes[row][2].item<double>(),
              predicted.boxes[row][3].item<double>()},
             predicted.scores[row].item<double>(),
             static_cast<std::size_t>(predicted.classes[row].item<I>())});
      boxes->update(std::span(&item, 1));
    } else {
      if (!sample.annotation)
        throw InvalidArgument("Rich evaluation requires annotations");
      auto annotation =
          std::make_shared<image::VisionAnnotationRecord>(*sample.annotation);
      std::vector<bool> ignored, crowd;
      std::vector<float> areas;
      for (auto& object : annotation->objects) {
        ignored.push_back(object.ignore || object.difficult);
        crowd.push_back(object.crowd);
        const auto& b = *object.box;
        areas.push_back(static_cast<float>(
            object.area.value_or((b.x_max - b.x_min) * (b.y_max - b.y_min))));
        object.ignore = false;
        object.crowd = false;
        object.difficult = false;
      }
      sample.annotation = annotation;
      const std::array<image::VisionSample, 1> originals{sample};
      const std::array<image::ImageBatchKey, 1> key{
          {{.id = sample.image_path}}};
      auto original_batch = image::collate_images(originals, key);
      auto options = r.data;
      options.exclude_crowd = false;
      options.exclude_difficult = false;
      auto targets =
          project_task_targets(original_batch, binding.yolo, options);
      ImageResult target;
      target.task = r.model.task;
      target.extent = {sample.width, sample.height};
      target.boxes = targets.boxes.slice(1, 2, 6).to(torch::kCPU);
      target.classes =
          targets.boxes.select(1, 1).to(torch::kCPU, torch::kInt64);
      target.masks = targets.masks;
      target.keypoints = targets.keypoints;
      if (r.model.task == TaskKind::obb) {
        std::vector<float> corners;
        for (const auto& object : annotation->objects)
          for (const auto& p : *object.oriented_box) {
            corners.push_back(static_cast<float>(p.x));
            corners.push_back(static_cast<float>(p.y));
          }
        target.oriented_corners = torch::tensor(corners).view({-1, 4, 2});
      }
      auto flags = [](const std::vector<bool>& v) {
        auto t = torch::empty({static_cast<I>(v.size())}, torch::kBool);
        for (std::size_t j = 0; j < v.size(); ++j) t[static_cast<I>(j)] = v[j];
        return t;
      };
      rich->update(i + 1, predicted, target, torch::tensor(areas),
                   flags(ignored), flags(crowd));
    }
  }
  J report = {{"schema", "vision.evaluation/v1"},
              {"dataset_identity", dataset_id},
              {"images", dataset.size()},
              {"model", binding.capability.architecture},
              {"protocol", nullptr},
              {"map", nullptr},
              {"ap50", nullptr}};
  if (boxes) {
    auto summary = boxes->summarize();
    report["protocol"] = "axis_aligned_box_ap.r101/v1";
    if (summary.mean_average_precision)
      report["map"] = *summary.mean_average_precision;
    double ap50 = 0;
    std::size_t included = 0;
    for (const auto& c : summary.per_class)
      if (!c.ap_by_iou.empty() && c.ap_by_iou.front()) {
        ap50 += *c.ap_by_iou.front();
        ++included;
      }
    if (included) report["ap50"] = ap50 / included;
    for (const auto& c : summary.per_class) {
      per_class.push_back(
          {{"targets", c.ground_truth},
           {"predictions", c.predictions},
           {"ap", c.average_precision ? J(*c.average_precision) : J(nullptr)},
           {"precision_by_iou", c.precision_by_iou},
           {"ap_by_iou", J::array()},
           {"recall_by_iou", J::array()}});
      for (auto value : c.ap_by_iou)
        per_class.back()["ap_by_iou"].push_back(value ? J(*value) : J(nullptr));
      for (auto value : c.recall_by_iou)
        per_class.back()["recall_by_iou"].push_back(value ? J(*value)
                                                          : J(nullptr));
    }
  } else if (rich) {
    auto summary = rich->compute();
    report["protocol"] = summary.protocol;
    if (summary.mean_average_precision)
      report["map"] = *summary.mean_average_precision;
    if (summary.ap50) report["ap50"] = *summary.ap50;
    for (const auto& c : summary.classes)
      per_class.push_back(
          {{"targets", c.targets},
           {"predictions", c.predictions},
           {"ap", c.average_precision ? J(*c.average_precision) : J(nullptr)},
           {"ap_by_iou", c.ap_by_threshold},
           {"precision", c.precision},
           {"recall", c.recall}});
  } else {
    report["protocol"] = "single_label_top1_top5/v1";
    report["top1"] = static_cast<double>(correct1) / dataset.size();
    report["top5"] = static_cast<double>(correct5) / dataset.size();
    for (std::size_t i = 0; i < support.size(); ++i)
      per_class.push_back(
          {{"targets", support[i]},
           {"top1", support[i] ? J(static_cast<double>(correct[i]) / support[i])
                               : J(nullptr)}});
  }
  if (r.model.task == TaskKind::classify) {
    for (auto [cell, count] : cells)
      confusion.push_back({{"target", cell.first},
                           {"prediction", cell.second},
                           {"count", count}});
    report["confusion"] = confusion;
    report["failure_samples"] = failures;
  } else {
    report["diagnostics"] = diagnostics.report();
    report["evaluation_contract"] = {
        {"thresholds", {.5, .55, .6, .65, .7, .75, .8, .85, .9, .95}},
        {"score_threshold", r.evaluation.confidence},
        {"max_candidates", r.evaluation.max_candidates},
        {"max_detections", r.evaluation.max_detections},
        {"metric_max_detections_per_image_class", r.evaluation.max_detections},
        {"missing_class", "exclude"},
        {"keypoint_sigmas", r.keypoint_sigmas},
        {"empty_keypoint_sigmas", "uniform_1_over_K"}};
    report["capacity"] = {
        {"maximum_eligible_targets_per_image", maximum_targets},
        {"images_exceeding_max_detections", capacity_images}};
  }
  report["classes"] = std::move(per_class);
  return report.dump(2);
}
}  // namespace jyppx::liuding::vision::tasks
