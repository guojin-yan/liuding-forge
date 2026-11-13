// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
/** @file yolo_training.cpp @brief Native YOLOv8 training over canonical
 * datasets. */

#include <jyppx/liuding/data/samplers/random_sampler.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/random.hpp>
#include <jyppx/liuding/foundation/random_checkpoint_participant.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/nn/model_checkpoint_participant.hpp>
#include <jyppx/liuding/optim/sgd_adapter.hpp>
#include <jyppx/liuding/training/callbacks/early_stopping_callback.hpp>
#include <jyppx/liuding/training/file_checkpoint_store.hpp>
#include <jyppx/liuding/training/scheduler_adapter.hpp>
#include <jyppx/liuding/training/single_device_strategy.hpp>
#include <jyppx/liuding/vision/models/yolo/yolov8.hpp>
#include <jyppx/liuding/vision/tasks/data/dataset.hpp>
#include <jyppx/liuding/vision/tasks/data/preparation.hpp>
#include <jyppx/liuding/vision/tasks/evaluation.hpp>
#include <jyppx/liuding/vision/tasks/prediction.hpp>
#include <jyppx/liuding/vision/tasks/training/ema.hpp>
#include <jyppx/liuding/vision/tasks/training/model_selection.hpp>
#include <jyppx/liuding/vision/tasks/training/strategy.hpp>

#include <ATen/Parallel.h>
#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <string_view>

namespace {
namespace forge = jyppx::liuding;
namespace vision = forge::vision::tasks;
namespace image = forge::data::image;
namespace fs = std::filesystem;
using Json = nlohmann::json;

void CheckKeys(const Json& object,
               std::initializer_list<std::string_view> allowed,
               const char* name) {
  if (!object.is_object())
    throw forge::InvalidArgument(std::string(name) + " must be an object");
  for (const auto& item : object.items())
    if (std::find(allowed.begin(), allowed.end(),
                  std::string_view(item.key())) == allowed.end())
      throw forge::InvalidArgument(std::string("Unknown ") + name +
                                   " field: " + item.key());
}

std::string Utf8(const fs::path& path) {
  const auto bytes = path.generic_u8string();
  return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

void SaveJson(const fs::path& path, const Json& value) {
  std::ofstream stream(path, std::ios::binary);
  stream << value.dump(2) << '\n';
  stream.close();
  if (!stream.good()) throw forge::IoError("Cannot write example JSON output");
}

vision::DetectionDatasetOptions DatasetOptions(const Json& config,
                                               const fs::path& directory,
                                               vision::DatasetSplit split) {
  CheckKeys(config,
            {"format", "source", "image_root", "categories",
             "allow_missing_labels", "split"},
            "dataset");
  const auto resolve = [&](const std::string& text) {
    auto path = fs::u8path(text);
    return path.is_absolute() ? path : directory / path;
  };
  vision::DetectionDatasetOptions source;
  const auto format = config.at("format").get<std::string>();
  if (format == "coco")
    source.format = vision::DetectionDatasetFormat::coco;
  else if (format == "voc")
    source.format = vision::DetectionDatasetFormat::voc;
  else if (format == "yolo")
    source.format = vision::DetectionDatasetFormat::yolo;
  else
    throw forge::InvalidArgument("Dataset format must be coco, voc or yolo");
  source.source = resolve(config.at("source").get<std::string>());
  if (config.contains("image_root"))
    source.image_root = resolve(config.at("image_root").get<std::string>());
  source.split = split;
  if (config.contains("split")) {
    const auto name = config.at("split").get<std::string>();
    if (name == "train")
      source.split = vision::DatasetSplit::train;
    else if (name == "validation")
      source.split = vision::DatasetSplit::validation;
    else if (name == "test")
      source.split = vision::DatasetSplit::test;
    else
      throw forge::InvalidArgument(
          "Dataset split must be train, validation or test");
  }
  if (config.contains("categories")) {
    std::int64_t id = 0;
    for (const auto& name : config.at("categories"))
      source.categories.push_back(
          {.source_id = id++, .name = name.get<std::string>()});
  }
  if (config.value("allow_missing_labels", false))
    source.missing_labels = vision::MissingLabelPolicy::empty_negative;
  return source;
}

Json SmokeConfiguration(const fs::path& output) {
  const auto root = output / "smoke-data";
  fs::create_directory(root);
  Json images = Json::array(), annotations = Json::array();
  Json val_images = Json::array(), val_annotations = Json::array();
  for (int index = 0; index < 6; ++index) {
    const auto name = "image-" + std::to_string(index) + ".ppm";
    std::ofstream stream(root / name, std::ios::binary);
    stream << "P6\n64 48\n255\n";
    for (int y = 0; y < 48; ++y)
      for (int x = 0; x < 64; ++x) {
        const bool inside = x >= 16 && x < 40 && y >= 8 && y < 32;
        const char pixel[3]{static_cast<char>(inside ? 220 : 30),
                            static_cast<char>(inside ? 40 + index * 20 : 80),
                            static_cast<char>(20)};
        stream.write(pixel, 3);
      }
    stream.close();
    if (!stream.good()) throw forge::IoError("Cannot create example image");
    auto& image_split = index < 4 ? images : val_images;
    auto& annotation_split = index < 4 ? annotations : val_annotations;
    image_split.push_back({{"id", index + 1},
                           {"file_name", name},
                           {"width", 64},
                           {"height", 48}});
    annotation_split.push_back({{"id", index + 1},
                                {"image_id", index + 1},
                                {"category_id", 1},
                                {"bbox", {16, 8, 24, 24}},
                                {"area", 576},
                                {"iscrowd", 0}});
  }
  SaveJson(root / "annotations.json",
           {{"images", images},
            {"annotations", annotations},
            {"categories", Json::array({{{"id", 1}, {"name", "object"}}})}});
  SaveJson(root / "validation.json",
           {{"images", val_images},
            {"annotations", val_annotations},
            {"categories", Json::array({{{"id", 1}, {"name", "object"}}})}});
  return {
      {"dataset",
       {{"format", "coco"},
        {"source", Utf8(root / "annotations.json")},
        {"image_root", Utf8(root)}}},
      {"validation",
       {{"dataset",
         {{"format", "coco"},
          {"source", Utf8(root / "validation.json")},
          {"image_root", Utf8(root)}}},
        {"confidence", 0.001},
        {"early_stopping_patience", 10}}},
      {"ema", {{"enabled", true}, {"decay", 0.9999}, {"ramp_updates", 2000.0}}},
      {"capacity", "n"},
      {"image_size", 64},
      {"batch_size", 2},
      {"epochs", 3},
      {"accumulate_steps", 2},
      {"gradient_clip_norm", 10.0},
      {"warmup_updates", 1},
      {"learning_rate", 0.001},
      {"horizontal_flip", 0.5},
      {"augmentation",
       {{"hsv_h", 0.015},
        {"hsv_s", 0.4},
        {"hsv_v", 0.2},
        {"grayscale", 1.0},
        {"blur", 1.0},
        {"median_blur", 1.0},
        {"scale", 0.2},
        {"translate", 0.1}}},
      {"prediction_confidence", 0.001},
      {"device", "cpu"},
      {"seed", 7}};
}

forge::models::yolo::YoloV8Capacity Capacity(const std::string& name) {
  using Capacity = forge::models::yolo::YoloV8Capacity;
  if (name == "n") return Capacity::nano_capacity;
  if (name == "s") return Capacity::small_capacity;
  if (name == "m") return Capacity::medium_capacity;
  if (name == "l") return Capacity::large_capacity;
  if (name == "x") return Capacity::extra_large_capacity;
  throw forge::InvalidArgument("YOLOv8 capacity must be n, s, m, l or x");
}

void Run(const Json& config, const fs::path& config_directory,
         const fs::path& output, const fs::path& resume,
         std::uint64_t pause_at) {
  CheckKeys(config,
            {"dataset",
             "capacity",
             "image_size",
             "batch_size",
             "workers",
             "epochs",
             "seed",
             "device",
             "learning_rate",
             "momentum",
             "weight_decay",
             "nesterov",
             "scale_weight_decay",
             "nominal_batch_size",
             "accumulate_steps",
             "gradient_clip_norm",
             "schedule",
             "warmup_updates",
             "warmup_start_factor",
             "final_lr_factor",
             "horizontal_flip",
             "vertical_flip",
             "augmentation",
             "prediction_confidence",
             "validation",
             "ema"},
            "configuration");
  if (config.contains("augmentation"))
    CheckKeys(config.at("augmentation"),
              {"scale", "translate", "hsv_h", "hsv_s", "hsv_v", "grayscale",
               "blur", "blur_kernel", "blur_sigma", "median_blur",
               "median_kernel", "min_box_area", "min_box_visibility"},
              "augmentation");
  const auto source = DatasetOptions(config.at("dataset"), config_directory,
                                     vision::DatasetSplit::train);
  auto imported = vision::read_detection_dataset(source);
  const auto categories = imported.annotations.categories;
  const auto classes =
      static_cast<std::int64_t>(imported.annotations.categories.size());
  image::ImageDatasetOptions decoding;
  decoding.image.channels = image::ImageChannels::rgb;
  const auto validation = config.value("validation", Json::object());
  CheckKeys(validation,
            {"dataset", "evaluate_ap", "confidence", "nms_iou",
             "max_candidates", "max_detections", "ignore_difficult",
             "early_stopping_patience", "min_delta", "save_interval_epochs"},
            "validation");
  const auto ema_config = config.value("ema", Json::object());
  CheckKeys(ema_config, {"enabled", "decay", "ramp_updates"}, "ema");
  std::optional<vision::DetectionDatasetImport> validation_import;
  std::shared_ptr<image::ImageDataset> validation_dataset;
  if (config.contains("validation")) {
    auto val_source = DatasetOptions(validation.at("dataset"), config_directory,
                                     vision::DatasetSplit::validation);
    if (!val_source.categories.empty()) {
      if (val_source.categories.size() != categories.size())
        throw forge::InvalidArgument(
            "Validation categories differ from training");
      for (std::size_t i = 0; i < categories.size(); ++i)
        if (val_source.categories[i].name != categories[i].name)
          throw forge::InvalidArgument(
              "Validation category order differs from training");
    }
    val_source.categories = categories;
    validation_import = vision::read_detection_dataset(val_source);
    vision::require_disjoint_detection_splits(imported, *validation_import);
    validation_dataset = std::make_shared<image::ImageDataset>(
        validation_import->image_root,
        std::move(validation_import->annotations), decoding);
  }
  auto dataset = std::make_shared<image::ImageDataset>(
      imported.image_root, std::move(imported.annotations), decoding);
  const auto size = config.value("image_size", std::uint32_t{640});
  const auto batch_size = config.value("batch_size", std::size_t{4});
  const auto workers = config.value("workers", std::size_t{0});
  const auto accumulation = config.value("accumulate_steps", std::size_t{1});
  const auto epochs = config.value("epochs", std::uint64_t{1});
  if (size < 64 || size > 4096 || size % 32 != 0 || batch_size == 0 ||
      batch_size > 256 || workers > 16 || accumulation == 0 ||
      accumulation > 256 || epochs == 0)
    throw forge::InvalidArgument(
        "Example needs image_size in [64,4096] divisible by 32, batch 1..256 "
        "workers 0..16, accumulation 1..256 and positive epochs");
  const auto batches =
      dataset->size() / batch_size + (dataset->size() % batch_size != 0);
  const auto updates_per_epoch =
      batches / accumulation + (batches % accumulation != 0);
  if (epochs >
      static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)()) /
          updates_per_epoch)
    throw forge::ResourceExhausted(
        "Example planned update count exceeds int64");
  const auto planned_updates = epochs * updates_per_epoch;
  const auto seed = config.value("seed", std::int64_t{0});
  torch::manual_seed(static_cast<std::uint64_t>(seed));
  const forge::Device device(config.value("device", std::string("cpu")));
  forge::models::yolo::YoloV8Options architecture;
  architecture.capacity = Capacity(config.value("capacity", std::string("n")));
  architecture.num_classes = classes;
  auto model = std::make_shared<forge::nn::ModelAdapter>(
      forge::models::yolo::create_yolov8(architecture));
  // Placement precedes optimizer binding so adapters retain the final tensors.
  model->state().to(device);
  std::shared_ptr<vision::ModelEma> ema;
  auto evaluation_model = model;
  if (ema_config.value("enabled", true)) {
    const auto rng = forge::core::get_rng_state(false);
    std::shared_ptr<forge::nn::ModelAdapter> shadow;
    try {
      shadow = std::make_shared<forge::nn::ModelAdapter>(
          forge::models::yolo::create_yolov8(architecture));
    } catch (...) {
      forge::core::set_rng_state(rng, false);
      throw;
    }
    forge::core::set_rng_state(rng, false);
    shadow->state().to(device);
    vision::ModelEmaOptions averaging;
    averaging.source_model_id =
        forge::models::yolo::yolov8_model_id(architecture.capacity);
    averaging.decay = ema_config.value("decay", 0.9999);
    averaging.ramp_updates = ema_config.value("ramp_updates", 2000.0);
    ema = std::make_shared<vision::ModelEma>(model, shadow, averaging);
    evaluation_model = std::move(shadow);
  }
  vision::DetectionSgdOptions sgd;
  sgd.learning_rate = config.value("learning_rate", 0.01);
  sgd.momentum = config.value("momentum", 0.9);
  sgd.weight_decay = config.value("weight_decay", 0.0005);
  sgd.nesterov = config.value("nesterov", false);
  sgd.scale_weight_decay = config.value("scale_weight_decay", true);
  sgd.effective_batch = batch_size * accumulation;
  sgd.nominal_batch = config.value("nominal_batch_size", std::uint64_t{64});
  auto grouping =
      vision::make_detection_sgd_groups(model->state().module(), sgd);
  const auto parameter_count = grouping.parameter_elements;
  Json group_report = Json::array();
  for (const auto& group : grouping.groups)
    group_report.push_back(
        {{"id", group.group_id},
         {"parameter_tensors", group.parameters.size()},
         {"weight_decay", group.options.weight_decay},
         {"base_learning_rate", group.options.learning_rate}});
  auto optimizer = std::make_shared<forge::optim::SgdOptimizerAdapter>(
      "optimizer.yolo", std::move(grouping.groups));
  const auto schedule = config.value("schedule", std::string("cosine"));
  vision::WarmupCosineOptions cosine;
  cosine.total_updates = planned_updates;
  cosine.warmup_updates = config.value("warmup_updates", std::uint64_t{0});
  cosine.warmup_start_factor = config.value("warmup_start_factor", 0.1);
  cosine.final_factor = config.value("final_lr_factor", 0.01);
  std::shared_ptr<forge::training::SchedulerAdapter> scheduler;
  if (schedule == "cosine")
    scheduler = std::make_shared<forge::training::SchedulerAdapter>(
        std::make_shared<vision::WarmupCosineLrScheduler>(optimizer, cosine));
  else if (schedule != "constant" || cosine.warmup_updates != 0)
    throw forge::InvalidArgument(
        "Schedule must be cosine, or constant without warmup");
  vision::YoloDataOptions data;
  data.num_classes = classes;
  data.letterbox.canvas = {size, size};
  data.augmentation.seed = static_cast<std::uint64_t>(seed);
  data.augmentation.dataset_id = imported.metadata_identity;
  data.augmentation.horizontal_flip_probability =
      config.value("horizontal_flip", 0.5);
  data.augmentation.vertical_flip_probability =
      config.value("vertical_flip", 0.0);
  const auto augmentation = config.value("augmentation", Json::object());
  data.augmentation.spatial.scale_gain = augmentation.value("scale", 0.0);
  data.augmentation.spatial.translate_fraction =
      augmentation.value("translate", 0.0);
  auto& photo = data.augmentation.photo;
  photo.hue_shift = augmentation.value("hsv_h", 0.0);
  photo.saturation_gain = augmentation.value("hsv_s", 0.0);
  photo.value_gain = augmentation.value("hsv_v", 0.0);
  photo.grayscale_probability = augmentation.value("grayscale", 0.0);
  photo.blur_probability = augmentation.value("blur", 0.0);
  photo.blur_kernel = augmentation.value("blur_kernel", std::uint32_t{3});
  photo.blur_sigma = augmentation.value("blur_sigma", 1.0);
  photo.median_blur_probability = augmentation.value("median_blur", 0.0);
  photo.median_kernel = augmentation.value("median_kernel", std::uint32_t{3});
  data.letterbox.geometry.min_box_area =
      augmentation.value("min_box_area", 0.0);
  data.letterbox.geometry.min_box_visibility =
      augmentation.value("min_box_visibility", 0.0);
  forge::data::DataPipelineOptions pipeline;
  pipeline.pipeline_id = "vision.yolo.training-data";
  pipeline.dataset_version = imported.metadata_identity;
  pipeline.batch_size = batch_size;
  pipeline.num_workers = workers;
  pipeline.memory.target_device = device;
  auto training_data = vision::make_yolo_data_pipeline(
      dataset, pipeline, data,
      std::make_shared<forge::data::samplers::RandomSampler>(dataset->size(),
                                                             seed));
  auto evaluation_data = vision::yolo_evaluation_options(data);
  std::shared_ptr<forge::data::DataPipeline> validation_data;
  std::shared_ptr<vision::ModelSelectionCallback> selection;
  std::vector<std::shared_ptr<forge::training::CallbackHandler>> callbacks;
  vision::ModelSelectionOptions selection_options;
  selection_options.min_delta = validation.value("min_delta", 0.0);
  selection_options.save_interval_epochs =
      validation.value("save_interval_epochs", std::uint64_t{1});
  const auto patience =
      validation.value("early_stopping_patience", std::uint64_t{0});
  if (validation_dataset) {
    evaluation_data.augmentation.dataset_id =
        validation_import->metadata_identity;
    auto validation_pipeline = pipeline;
    validation_pipeline.pipeline_id = "vision.yolo.validation-data";
    validation_pipeline.dataset_version = validation_import->metadata_identity;
    validation_data = vision::make_yolo_data_pipeline(
        validation_dataset, validation_pipeline, evaluation_data);
    selection =
        std::make_shared<vision::ModelSelectionCallback>(selection_options);
    callbacks.push_back(selection);
    if (patience != 0)
      callbacks.push_back(
          std::make_shared<forge::training::callbacks::EarlyStoppingCallback>(
              forge::training::callbacks::EarlyStoppingOptions{
                  .patience = patience,
                  .min_delta = selection_options.min_delta}));
  }
  Json vocabulary = Json::array();
  for (const auto& category : categories)
    vocabulary.push_back(
        {{"source_id", category.source_id}, {"name", category.name}});
  const auto vocabulary_text = vocabulary.dump();
  const auto vocabulary_id =
      forge::sha256_hex(std::as_bytes(std::span(vocabulary_text)));
  vision::YoloEvaluationOptions ap_options;
  ap_options.data = evaluation_data;
  ap_options.prediction.detection.num_classes = classes;
  ap_options.prediction.confidence_threshold =
      validation.value("confidence", 0.001);
  ap_options.prediction.iou_threshold = validation.value("nms_iou", 0.7);
  ap_options.prediction.max_candidates =
      validation.value("max_candidates", std::int64_t{4096});
  ap_options.prediction.max_detections =
      validation.value("max_detections", std::int64_t{300});
  ap_options.ignore_difficult = validation.value("ignore_difficult", true);
  ap_options.batch_size = batch_size;
  ap_options.device = device;
  ap_options.evaluation.metric.num_classes = static_cast<std::size_t>(classes);
  ap_options.evaluation.metric.class_vocabulary_id = vocabulary_id;
  if (validation_import)
    ap_options.evaluation.metric.dataset_id =
        validation_import->metadata_identity;
  const Json validation_contract =
      validation_dataset
          ? Json{{"metadata_identity", validation_import->metadata_identity},
                 {"data_identity", vision::yolo_data_identity(evaluation_data)},
                 {"images", validation_dataset->size()},
                 {"model", ema ? "ema" : "training"},
                 {"selection_metric", selection_options.monitor},
                 {"min_delta", selection_options.min_delta},
                 {"save_interval_epochs",
                  selection_options.save_interval_epochs},
                 {"early_stopping_patience", patience},
                 {"evaluate_ap", validation.value("evaluate_ap", true)},
                 {"confidence", ap_options.prediction.confidence_threshold},
                 {"nms_iou", ap_options.prediction.iou_threshold},
                 {"max_candidates", ap_options.prediction.max_candidates},
                 {"max_detections", ap_options.prediction.max_detections},
                 {"ignore_difficult", ap_options.ignore_difficult},
                 {"class_vocabulary_id", vocabulary_id},
                 {"ap_policy",
                  "box-ap-101pt-iou50:95-maxdet100-per-image-class/v1"}}
          : Json(nullptr);
  forge::training::TrainerOptions options;
  options.trainer_id = "trainer.yolo";
  options.run_id = "vision.yolo.example";
  options.trace_id = "vision.yolo.example";
  options.max_epochs = epochs;
  options.accumulate_steps = accumulation;
  if (config.contains("gradient_clip_norm"))
    options.gradient_clip_norm = config.at("gradient_clip_norm").get<double>();
  const Json contract{
      {"schema", "vision.yolov8.training/v3"},
      {"model", forge::models::yolo::yolov8_model_id(architecture.capacity)},
      {"classes", classes},
      {"metadata_identity", imported.metadata_identity},
      {"data_identity", vision::yolo_data_identity(data)},
      {"categories", vocabulary},
      {"ema", ema ? Json{{"decay", ema->options().decay},
                         {"ramp_updates", ema->options().ramp_updates}}
                  : Json(nullptr)},
      {"validation", validation_contract},
      {"image_size", size},
      {"reg_max", architecture.reg_max},
      {"letterbox",
       {{"scale_up", data.letterbox.scale_up},
        {"center", data.letterbox.center},
        {"padding", data.letterbox.padding_value}}},
      {"augmentation",
       {{"horizontal_flip", data.augmentation.horizontal_flip_probability},
        {"vertical_flip", data.augmentation.vertical_flip_probability},
        {"scale", data.augmentation.spatial.scale_gain},
        {"translate", data.augmentation.spatial.translate_fraction},
        {"hsv_h", photo.hue_shift},
        {"hsv_s", photo.saturation_gain},
        {"hsv_v", photo.value_gain},
        {"grayscale", photo.grayscale_probability},
        {"blur", photo.blur_probability},
        {"blur_kernel", photo.blur_kernel},
        {"blur_sigma", photo.blur_sigma},
        {"median_blur", photo.median_blur_probability},
        {"median_kernel", photo.median_kernel},
        {"min_box_area", data.letterbox.geometry.min_box_area},
        {"min_box_visibility", data.letterbox.geometry.min_box_visibility}}},
      {"epochs", epochs},
      {"batch_size", batch_size},
      {"workers", workers},
      {"seed", seed},
      {"device", device.str()},
      {"precision", "full"},
      {"accumulate_steps", accumulation},
      {"accumulation_tail", "flush"},
      {"gradient_clip_norm", options.gradient_clip_norm
                                 ? Json(*options.gradient_clip_norm)
                                 : Json(nullptr)},
      {"optimizer",
       {{"type", "sgd"},
        {"momentum", sgd.momentum},
        {"nesterov", sgd.nesterov},
        {"effective_batch", sgd.effective_batch},
        {"nominal_batch", sgd.nominal_batch},
        {"scale_weight_decay", sgd.scale_weight_decay},
        {"nominal_decay", sgd.weight_decay},
        {"groups", group_report}}},
      {"scheduler",
       {{"type", schedule},
        {"total_updates", planned_updates},
        {"warmup_updates", cosine.warmup_updates},
        {"warmup_start_factor", cosine.warmup_start_factor},
        {"final_factor", cosine.final_factor}}},
      {"criterion", "yolo-tal-ciou-bce-dfl/v1"}};
  const auto contract_text = contract.dump();
  const auto recipe_identity =
      forge::sha256_hex(std::as_bytes(std::span(contract_text)));
  options.run_id = "vision.yolo." + recipe_identity;
  options.trace_id = options.run_id;
  SaveJson(output / "resolved.json", {{"recipe_identity", recipe_identity},
                                      {"training_contract", contract},
                                      {"configuration", config}});
  vision::YoloDetectionOptions loss;
  loss.num_classes = classes;
  forge::training::StepFunction validation_step;
  if (validation_dataset)
    validation_step = [evaluation_model,
                       step = vision::make_yolo_detection_step(loss)](
                          const forge::training::StepInput& input) {
      return step({*evaluation_model, input.batch, input.invocation,
                   input.target, input.context});
    };
  forge::training::CheckpointAction checkpoint_action;
  if (selection)
    checkpoint_action =
        [selection,
         output](const forge::training::CheckpointCoordinator& coordinator) {
          if (selection->state().last_improved) {
            forge::training::FileCheckpointStore best(output /
                                                      "best.checkpoint");
            coordinator.save(best);
          }
          forge::training::FileCheckpointStore last(output / "last.checkpoint");
          coordinator.save(last);
        };
  std::vector<std::shared_ptr<forge::training::OptimizerUpdateParticipant>>
      updates;
  if (ema) updates.push_back(ema);
  forge::training::Trainer trainer(options);
  trainer.prepare(
      {.model = model,
       .optimizers = {{.optimizer = optimizer,
                       .checkpoint = std::make_shared<
                           forge::optim::SgdOptimizerCheckpointParticipant>(
                           optimizer),
                       .scheduler = scheduler}},
       .training_data = training_data,
       .training_batch_mapper = vision::make_yolo_image_mapper(),
       .training_target_mapper = vision::make_yolo_target_mapper(data),
       .training_step = vision::make_yolo_detection_step(loss),
       .validation_data = validation_data,
       .validation_batch_mapper = validation_data
                                      ? vision::make_yolo_image_mapper()
                                      : forge::training::BatchMapper{},
       .validation_target_mapper =
           validation_data ? vision::make_yolo_target_mapper(evaluation_data)
                           : forge::training::TargetMapper{},
       .validation_step = validation_step,
       .strategy = std::make_shared<forge::training::SingleDeviceStrategy>(
           forge::training::SingleDeviceStrategyOptions{.device = device}),
       .callbacks = std::make_shared<forge::training::CallbackDispatcher>(
           std::move(callbacks)),
       .model_checkpoint =
           std::make_shared<forge::nn::ModelCheckpointParticipant>(*model),
       .rng_checkpoint =
           std::make_shared<forge::core::RngCheckpointParticipant>(
               forge::core::RngCheckpointParticipantOptions{
                   .include_cuda = device.is_cuda()}),
       .checkpoint_action = checkpoint_action,
       .optimizer_update_participants = std::move(updates)});
  if (!resume.empty()) {
    if (!fs::is_regular_file(resume))
      throw forge::IoError("Resume checkpoint does not exist");
    forge::training::FileCheckpointStore previous(resume);
    trainer.restore_checkpoint(previous);
  }
  const auto result = pause_at ? trainer.fit_until(pause_at) : trainer.fit();
  forge::training::FileCheckpointStore checkpoint(output / "last.checkpoint");
  trainer.save_checkpoint(checkpoint);
  trainer.close();
  Json ap_report = nullptr;
  if (validation_dataset && validation.value("evaluate_ap", true) &&
      result.status != forge::training::FitStatus::paused) {
    const auto evaluated = vision::evaluate_yolo_detection(
        *evaluation_model, *validation_dataset, ap_options);
    Json per_class = Json::array();
    for (std::size_t i = 0; i < evaluated.overall.per_class.size(); ++i) {
      const auto& values = evaluated.overall.per_class[i];
      Json ap_by_iou = Json::array(), recall_by_iou = Json::array();
      for (const auto& value : values.ap_by_iou)
        ap_by_iou.push_back(value ? Json(*value) : Json(nullptr));
      for (const auto& value : values.recall_by_iou)
        recall_by_iou.push_back(value ? Json(*value) : Json(nullptr));
      per_class.push_back(
          {{"name", categories[i].name},
           {"ground_truth", values.ground_truth},
           {"predictions", values.predictions},
           {"ap", values.average_precision ? Json(*values.average_precision)
                                           : Json(nullptr)},
           {"ap_by_iou", ap_by_iou},
           {"recall_by_iou", recall_by_iou}});
    }
    ap_report = {
        {"model", ema ? "ema" : "training"},
        {"weights", "last"},
        {"images", evaluated.overall.images},
        {"batches", evaluated.batches},
        {"iou_thresholds", ap_options.evaluation.metric.iou_thresholds},
        {"map", evaluated.overall.mean_average_precision
                    ? Json(*evaluated.overall.mean_average_precision)
                    : Json(nullptr)},
        {"per_class", per_class}};
  }
  Json predictions = Json::array();
  {
    const torch::NoGradGuard no_grad;
    evaluation_model->state().eval();
    data = vision::yolo_evaluation_options(data);
    const std::vector<image::VisionSample> samples{dataset->get(0)};
    const std::vector<image::ImageBatchKey> keys{{.id = samples[0].image_path}};
    const auto batch = vision::prepare_yolo_batch(samples, keys, data);
    const auto raw =
        evaluation_model->invoke({.positional = {batch.images.to(device)}})
            .value.toTensor();
    vision::YoloPredictionOptions prediction;
    prediction.detection = loss;
    prediction.confidence_threshold =
        config.value("prediction_confidence", 0.25);
    const auto decoded =
        vision::postprocess_yolo_detection(raw, {size, size}, prediction);
    const auto source_boxes = vision::restore_detection_coordinates(
        decoded[0], *batch.items[0].geometry);
    for (const auto& item : source_boxes)
      predictions.push_back({{"box", item.box},
                             {"score", item.score},
                             {"class_index", item.class_index}});
  }
  const Json report{
      {"model", forge::models::yolo::yolov8_model_id(architecture.capacity)},
      {"parameters", parameter_count},
      {"images", dataset->size()},
      {"classes", classes},
      {"metadata_identity", imported.metadata_identity},
      {"recipe_identity", recipe_identity},
      {"optimizer_groups", group_report},
      {"planned_updates", planned_updates},
      {"scheduler_updates",
       scheduler ? scheduler->scheduler().completed_steps() : 0},
      {"learning_rates", optimizer->learning_rates()},
      {"completed_epochs", result.completed_epochs},
      {"global_step", result.global_step},
      {"ema_updates", ema ? Json(ema->completed_updates()) : Json(nullptr)},
      {"ema_decay", ema ? Json(ema->current_decay()) : Json(nullptr)},
      {"paused", result.status == forge::training::FitStatus::paused},
      {"training_metrics", result.training_metrics},
      {"validation_metrics", result.validation_metrics},
      {"validation_images",
       validation_dataset ? validation_dataset->size() : 0},
      {"selection", selection && selection->state().observations > 0
                        ? Json{{"monitor", "validation.loss"},
                               {"best_value", selection->state().best_value},
                               {"best_epoch", selection->state().best_epoch},
                               {"best_step", selection->state().best_step}}
                        : Json(nullptr)},
      {"best_checkpoint", fs::exists(output / "best.checkpoint")
                              ? Json("best.checkpoint")
                              : Json(nullptr)},
      {"evaluation", ap_report},
      {"prediction_model", ema ? "ema" : "training"},
      {"first_image_predictions", predictions},
      {"checkpoint", "last.checkpoint"}};
  SaveJson(output / "report.json", report);
  std::cout << report.dump(2) << '\n';
}

template <class Character>
int Main(int argc, Character* argv[]) {
  constexpr auto usage =
      "usage: liuding-forge-example-yolo-training CONFIG.json "
      "NEW_OUTPUT_DIRECTORY [--resume CHECKPOINT] [--pause-at STEP]\n"
      "       liuding-forge-example-yolo-training --smoke "
      "NEW_OUTPUT_DIRECTORY\n";
  try {
    if (argc == 2 && fs::path(argv[1]) == fs::path("--help")) {
      std::cout << usage;
      return 0;
    }
    if (argc < 3 || argc % 2 == 0) {
      std::cerr << usage;
      return 2;
    }
    fs::path resume;
    std::uint64_t pause_at = 0;
    for (int index = 3; index < argc; index += 2) {
      if (fs::path(argv[index]) == fs::path("--resume"))
        resume = fs::path(argv[index + 1]);
      else if (fs::path(argv[index]) == fs::path("--pause-at")) {
        const auto text = Utf8(fs::path(argv[index + 1]));
        const auto parsed =
            std::from_chars(text.data(), text.data() + text.size(), pause_at);
        if (parsed.ec != std::errc{} ||
            parsed.ptr != text.data() + text.size() || pause_at == 0)
          throw forge::InvalidArgument(
              "--pause-at requires a positive absolute optimizer step");
      } else
        throw forge::InvalidArgument("Unknown example argument");
    }
    const auto output = fs::absolute(fs::path(argv[2]));
    if (fs::exists(output))
      throw forge::InvalidArgument("Example output directory must be new");
    fs::create_directories(output);
    at::set_num_threads(2);
    at::set_num_interop_threads(1);
    if (fs::path(argv[1]) == fs::path("--smoke"))
      Run(SmokeConfiguration(output), output, output, resume, pause_at);
    else {
      const auto config_path = fs::absolute(fs::path(argv[1]));
      if (fs::file_size(config_path) > (1U << 20U))
        throw forge::ResourceExhausted("Example configuration exceeds 1 MiB");
      std::ifstream stream(config_path, std::ios::binary);
      if (!stream) throw forge::IoError("Cannot read example configuration");
      const auto config = Json::parse(stream);
      Run(config, config_path.parent_path(), output, resume, pause_at);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) { return Main(argc, argv); }
#else
int main(int argc, char* argv[]) { return Main(argc, argv); }
#endif
