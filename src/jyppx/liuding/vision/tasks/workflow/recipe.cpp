// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "selection_metrics.hpp"

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/vision/tasks/workflow/recipe.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
namespace jyppx::liuding::vision::tasks {
namespace {
using J = nlohmann::json;
void Keys(const J& value, std::initializer_list<std::string_view> keys) {
  if (!value.is_object())
    throw InvalidArgument("Recipe section must be an object");
  for (const auto& entry : value.items())
    if (std::find(keys.begin(), keys.end(), entry.key()) == keys.end())
      throw InvalidArgument("Unknown vision recipe field: " + entry.key());
}
std::string Path(const std::filesystem::path& p) {
  auto s = p.generic_u8string();
  return {reinterpret_cast<const char*>(s.data()), s.size()};
}
std::filesystem::path Resolve(const J& value,
                              const std::filesystem::path& root) {
  auto p = std::filesystem::u8path(value.get<std::string>());
  return p.empty() ? p
                   : std::filesystem::absolute(p.is_absolute() ? p : root / p)
                         .lexically_normal();
}
constexpr std::array<std::string_view, 4> precisions{
    "full", "cpu_bfloat16", "cuda_float16", "cuda_bfloat16"};
DetectionDatasetOptions Source(const J& v, const std::filesystem::path& root,
                               DatasetSplit split, const ModelSpec& model) {
  Keys(v, {"format", "source", "image_root", "categories",
           "allow_missing_labels", "split"});
  DetectionDatasetOptions d;
  d.task = model.task == TaskKind::instance_segment
               ? DatasetAnnotationTask::instance_segment
           : model.task == TaskKind::pose ? DatasetAnnotationTask::pose
           : model.task == TaskKind::obb  ? DatasetAnnotationTask::obb
                                          : DatasetAnnotationTask::detect;
  d.keypoint_count = model.task == TaskKind::pose
                         ? static_cast<std::size_t>(model.keypoints)
                         : 0;
  d.keypoint_dimensions = static_cast<std::uint8_t>(model.keypoint_dimensions);
  auto f = v.at("format").get<std::string>();
  if (f == "coco")
    d.format = DetectionDatasetFormat::coco;
  else if (f == "voc")
    d.format = DetectionDatasetFormat::voc;
  else if (f == "yolo")
    d.format = DetectionDatasetFormat::yolo;
  else if (f == "labelme")
    d.format = DetectionDatasetFormat::labelme;
  else
    throw InvalidArgument("Dataset format must be coco/voc/yolo/labelme");
  d.source = Resolve(v.at("source"), root);
  if (v.contains("image_root")) d.image_root = Resolve(v["image_root"], root);
  d.split = split;
  if (v.contains("split")) {
    const auto s = v["split"].get<std::string>();
    if (s == "train")
      d.split = DatasetSplit::train;
    else if (s == "validation")
      d.split = DatasetSplit::validation;
    else if (s == "test")
      d.split = DatasetSplit::test;
    else
      throw InvalidArgument("Unknown dataset split");
  }
  if (v.value("allow_missing_labels", false))
    d.missing_labels = MissingLabelPolicy::empty_negative;
  if (v.contains("categories"))
    for (const auto& c : v["categories"]) {
      if (c.is_string())
        d.categories.push_back(
            {.source_id = static_cast<std::int64_t>(d.categories.size()),
             .name = c.get<std::string>()});
      else {
        Keys(c, {"source_id", "name"});
        d.categories.push_back(
            {.source_id = c.at("source_id").get<std::int64_t>(),
             .name = c.at("name").get<std::string>()});
      }
    }
  return d;
}
J SourceJson(const DetectionDatasetOptions& d) {
  J categories = J::array();
  for (const auto& c : d.categories)
    categories.push_back({{"source_id", c.source_id}, {"name", c.name}});
  return {{"format", d.format == DetectionDatasetFormat::coco      ? "coco"
                     : d.format == DetectionDatasetFormat::voc     ? "voc"
                     : d.format == DetectionDatasetFormat::labelme ? "labelme"
                                                                   : "yolo"},
          {"source", Path(d.source)},
          {"image_root", Path(d.image_root)},
          {"categories", categories},
          {"split", d.split == DatasetSplit::train        ? "train"
                    : d.split == DatasetSplit::validation ? "validation"
                                                          : "test"},
          {"allow_missing_labels",
           d.missing_labels == MissingLabelPolicy::empty_negative}};
}
void Results(const J& v, ResultOptions& o) {
  Keys(v, {"confidence", "nms_iou", "mask_threshold", "max_candidates",
           "max_detections", "class_agnostic", "multi_label", "classes",
           "max_mask_elements"});
  o.confidence = v.value("confidence", o.confidence);
  o.nms_iou = v.value("nms_iou", o.nms_iou);
  o.mask_threshold = v.value("mask_threshold", o.mask_threshold);
  o.max_candidates = v.value("max_candidates", o.max_candidates);
  o.max_detections = v.value("max_detections", o.max_detections);
  o.class_agnostic = v.value("class_agnostic", o.class_agnostic);
  o.multi_label = v.value("multi_label", o.multi_label);
  o.classes = v.value("classes", o.classes);
  o.max_mask_elements = v.value("max_mask_elements", o.max_mask_elements);
}
J ResultJson(const ResultOptions& o) {
  return {{"confidence", o.confidence},
          {"nms_iou", o.nms_iou},
          {"mask_threshold", o.mask_threshold},
          {"max_candidates", o.max_candidates},
          {"max_detections", o.max_detections},
          {"class_agnostic", o.class_agnostic},
          {"multi_label", o.multi_label},
          {"classes", o.classes},
          {"max_mask_elements", o.max_mask_elements}};
}
}  // namespace
RecipeSpec load_recipe(const std::filesystem::path& path) {
  std::error_code ec;
  auto bytes = std::filesystem::file_size(path, ec);
  if (ec) throw IoError("Cannot read vision recipe");
  if (bytes > 4U << 20U) throw ResourceExhausted("Vision recipe exceeds 4 MiB");
  std::ifstream stream(path, std::ios::binary);
  std::string text(static_cast<std::size_t>(bytes), '\0');
  stream.read(text.data(), static_cast<std::streamsize>(text.size()));
  if (!stream) throw IoError("Cannot read vision recipe bytes");
  return parse_recipe(text, std::filesystem::absolute(path).parent_path());
}
RecipeSpec parse_recipe(std::string_view text,
                        const std::filesystem::path& base_directory) {
  if (text.size() > (4U << 20U))
    throw ResourceExhausted("Vision recipe exceeds 4 MiB");
  const auto j = J::parse(text);
  Keys(j, {"schema_version",
           "model",
           "extension",
           "dataset",
           "validation",
           "classification_root",
           "classification_validation_root",
           "image_size",
           "batch_size",
           "workers",
           "epochs",
           "seed",
           "deterministic",
           "initial_loss_scale",
           "device",
           "precision",
           "accumulate_steps",
           "gradient_clip_norm",
           "label_smoothing",
           "optimizer",
           "scheduler",
           "augmentation",
           "ema",
           "early_stopping_patience",
           "save_interval_epochs",
           "prediction",
           "evaluation",
           "classification_augmentation",
           "exif_policy",
           "multiscale",
           "class_weights",
           "keypoint_sigmas",
           "full_validation_metrics",
           "selection_metric",
           "selection_maximize",
           "tensorboard",
           "diagnostics",
           "verify_image_contents",
           "prefetch_batches",
           "pin_memory",
           "rectangular_batches",
           "cache",
           "initialization",
           "normalization",
           "preprocessing",
           "horizontal_keypoint_permutations",
           "vertical_keypoint_permutations"});
  RecipeSpec r;
  const auto root = std::filesystem::absolute(base_directory);
  const auto exif = j.value("exif_policy", std::string("reject"));
  if (exif == "reject")
    r.exif_policy = ExifPolicy::reject_non_identity;
  else if (exif == "encoded")
    r.exif_policy = ExifPolicy::encoded_coordinates;
  else if (exif == "already_oriented")
    r.exif_policy = ExifPolicy::already_oriented_coordinates;
  else if (exif == "ignore")
    r.exif_policy = ExifPolicy::ignore;
  else
    throw InvalidArgument("Unknown exif_policy");
  if (j.contains("multiscale")) {
    const auto& m = j["multiscale"];
    Keys(m, {"enabled", "min_factor", "max_factor", "max_dimension"});
    r.data.multiscale = m.value("enabled", false);
    r.data.multiscale_min = m.value("min_factor", .5);
    r.data.multiscale_max = m.value("max_factor", 1.5);
    r.data.multiscale_max_dimension = m.value("max_dimension", 2048U);
  }
  r.schema_version = j.value("schema_version", 1U);
  if(j.contains("extension")) {
    const auto& e=j["extension"];
    Keys(e,{"training_manifest","validation_manifest","multilabel_thresholds","threshold_validation_identity","asymmetric_multilabel","semantic_ignore","relative_inverse_depth","depth_gradient_weight","model_patch_size","mask_patch_size","mask_ratio","triplet_margin","activation_recomputation","feature_outputs","feature_projection","teacher_architecture","teacher_weights","teacher_sha256","distillation_temperature","logits_distillation_weight","feature_distillation_weight","tokenizer_merges","tokenizer_sha256","text_encoder_weights","text_encoder_sha256"});
    if(e.contains("training_manifest"))r.extension.training_manifest=Resolve(e["training_manifest"],root);
    if(e.contains("validation_manifest"))r.extension.validation_manifest=Resolve(e["validation_manifest"],root);
    r.extension.multilabel_thresholds=e.value("multilabel_thresholds",std::vector<double>{});r.extension.threshold_validation_identity=e.value("threshold_validation_identity",std::string{});
    r.extension.asymmetric_multilabel=e.value("asymmetric_multilabel",false);
    r.extension.semantic_ignore=e.value("semantic_ignore",255LL);
    r.extension.relative_inverse_depth=e.value("relative_inverse_depth",false);
    r.extension.depth_gradient_weight=e.value("depth_gradient_weight",0.);
    r.extension.model_patch_size=e.value("model_patch_size",4LL);
    r.extension.mask_patch_size=e.value("mask_patch_size",32LL);
    r.extension.mask_ratio=e.value("mask_ratio",.6);
    r.extension.triplet_margin=e.value("triplet_margin",.3);
    auto& x=r.extension;if(e.contains("tokenizer_merges"))x.tokenizer_merges=Resolve(e["tokenizer_merges"],root);if(e.contains("text_encoder_weights"))x.text_encoder_weights=Resolve(e["text_encoder_weights"],root);x.tokenizer_sha256=e.value("tokenizer_sha256",std::string{});x.text_encoder_sha256=e.value("text_encoder_sha256",std::string{});x.activation_recomputation=e.value("activation_recomputation",false);x.feature_outputs=e.value("feature_outputs",false);x.feature_projection=e.value("feature_projection",0LL);
    x.teacher_architecture=e.value("teacher_architecture",std::string{});if(e.contains("teacher_weights"))x.teacher_initialization.weights=Resolve(e["teacher_weights"],root);x.teacher_initialization.sha256=e.value("teacher_sha256",std::string{});
    x.distillation_temperature=e.value("distillation_temperature",2.);x.logits_distillation_weight=e.value("logits_distillation_weight",1.);x.feature_distillation_weight=e.value("feature_distillation_weight",0.);
  }
  const auto& m = j.at("model");
  Keys(m, {"architecture", "task", "variant", "num_classes", "keypoints",
           "keypoint_dimensions", "mask_channels"});
  r.model.architecture = m.at("architecture").get<std::string>();
  r.model.task = parse_task(m.value("task", std::string("detect")));
  r.model.variant = m.value("variant", r.model.variant);
  r.model.num_classes = m.value("num_classes", r.model.num_classes);
  r.model.keypoints = m.value("keypoints", r.model.keypoints);
  r.model.keypoint_dimensions =
      m.value("keypoint_dimensions", r.model.keypoint_dimensions);
  r.model.mask_channels = m.value("mask_channels", r.model.mask_channels);
  r.keypoint_sigmas = j.value("keypoint_sigmas", r.keypoint_sigmas);
  if (j.contains("dataset"))
    r.training = Source(j["dataset"], root, DatasetSplit::train, r.model);
  if (j.contains("validation"))
    r.validation =
        Source(j["validation"], root, DatasetSplit::validation, r.model);
  if (j.contains("normalization")) {
    const auto& n = j["normalization"];
    Keys(n, {"mean", "std"});
    r.data.normalization_mean = n.at("mean").get<std::vector<double>>();
    r.data.normalization_std = n.at("std").get<std::vector<double>>();
  }
  if (j.contains("preprocessing")) {
    const auto& p = j["preprocessing"];
    Keys(p, {"scale_up", "center", "padding_value", "interpolation",
             "min_box_area", "min_box_visibility", "min_region_area",
             "min_region_visibility", "exclude_crowd", "exclude_difficult",
             "max_targets"});
    auto& b = r.data.letterbox;
    b.scale_up = p.value("scale_up", b.scale_up);
    b.center = p.value("center", b.center);
    b.padding_value = p.value("padding_value", b.padding_value);
    auto mode = p.value("interpolation", std::string("bilinear"));
    if (mode == "nearest_exact")
      b.geometry.interpolation = data::image::ImageInterpolation::nearest_exact;
    else if (mode == "bilinear")
      b.geometry.interpolation = data::image::ImageInterpolation::bilinear;
    else if (mode == "bicubic")
      b.geometry.interpolation = data::image::ImageInterpolation::bicubic;
    else
      throw InvalidArgument("Unknown preprocessing interpolation");
    b.geometry.min_box_area = p.value("min_box_area", b.geometry.min_box_area);
    b.geometry.min_box_visibility =
        p.value("min_box_visibility", b.geometry.min_box_visibility);
    b.geometry.min_region_area =
        p.value("min_region_area", b.geometry.min_region_area);
    b.geometry.min_region_visibility =
        p.value("min_region_visibility", b.geometry.min_region_visibility);
    r.data.exclude_crowd = p.value("exclude_crowd", r.data.exclude_crowd);
    r.data.exclude_difficult =
        p.value("exclude_difficult", r.data.exclude_difficult);
    r.data.max_targets = p.value("max_targets", r.data.max_targets);
  }
  if (j.contains("horizontal_keypoint_permutations"))
    r.data.letterbox.geometry.horizontal_keypoint_permutations =
        j["horizontal_keypoint_permutations"]
            .get<std::vector<std::vector<std::size_t>>>();
  if (j.contains("vertical_keypoint_permutations"))
    r.data.letterbox.geometry.vertical_keypoint_permutations =
        j["vertical_keypoint_permutations"]
            .get<std::vector<std::vector<std::size_t>>>();
  if (j.contains("classification_root"))
    r.classification_root = Resolve(j["classification_root"], root);
  if (j.contains("classification_validation_root"))
    r.classification_validation_root =
        Resolve(j["classification_validation_root"], root);
  if (j.contains("image_size")) {
    const auto& v = j["image_size"];
    if (v.is_array()) {
      if (v.size() != 2)
        throw InvalidArgument("image_size must be [width,height]");
      r.data.letterbox.canvas = {v[0].get<std::uint32_t>(),
                                 v[1].get<std::uint32_t>()};
    } else {
      auto n = v.get<std::uint32_t>();
      r.data.letterbox.canvas = {n, n};
    }
  }
  r.batch_size = j.value("batch_size", r.batch_size);
  r.workers = j.value("workers", r.workers);
  r.epochs = j.value("epochs", r.epochs);
  r.data.augmentation.multi_source.total_epochs = r.epochs;
  r.data.augmentation.multi_source.close_epochs =
      (std::min)(std::uint64_t{10}, r.epochs);
  r.prefetch_batches = j.value("prefetch_batches", r.prefetch_batches);
  r.pin_memory = j.value("pin_memory", r.pin_memory);
  r.rectangular_batches = j.value("rectangular_batches", r.rectangular_batches);
  r.data.rectangular = r.rectangular_batches;
  if (j.contains("cache")) {
    const auto& c = j["cache"];
    Keys(c, {"ram_bytes", "disk_bytes", "directory", "decoder_revision",
             "reject_corrupt"});
    r.cache.ram_bytes = c.value("ram_bytes", r.cache.ram_bytes);
    r.cache.disk_bytes = c.value("disk_bytes", r.cache.disk_bytes);
    if (c.contains("directory"))
      r.cache.directory = Resolve(c["directory"], root);
    r.cache.decoder_revision =
        c.value("decoder_revision", r.cache.decoder_revision);
    r.cache.reject_corrupt = c.value("reject_corrupt", r.cache.reject_corrupt);
  }
  if (j.contains("initialization")) {
    const auto& v = j["initialization"];
    Keys(v, {"weights", "sha256", "key_mapping", "ignored_source_keys",
             "retain_buffer_keys", "replace_head_prefixes", "freeze_prefixes",
             "learning_rates", "max_bytes"});
    auto& o = r.initialization;
    if (v.contains("weights")) o.weights = Resolve(v["weights"], root);
    o.sha256 = v.value("sha256", o.sha256);
    o.key_mapping = v.value("key_mapping", o.key_mapping);
    o.ignored_source_keys =
        v.value("ignored_source_keys", o.ignored_source_keys);
    o.retain_buffer_keys = v.value("retain_buffer_keys", o.retain_buffer_keys);
    o.replace_head_prefixes =
        v.value("replace_head_prefixes", o.replace_head_prefixes);
    o.freeze_prefixes = v.value("freeze_prefixes", o.freeze_prefixes);
    o.max_bytes = v.value("max_bytes", o.max_bytes);
    if (v.contains("learning_rates"))
      for (const auto& rule : v["learning_rates"]) {
        Keys(rule, {"prefix", "multiplier"});
        o.learning_rates.push_back({rule.at("prefix").get<std::string>(),
                                    rule.at("multiplier").get<double>()});
      }
  }
  r.seed = j.value("seed", r.seed);
  r.deterministic = j.value("deterministic", r.deterministic);
  r.initial_loss_scale = j.value("initial_loss_scale", r.initial_loss_scale);
  r.device = Device(j.value("device", std::string("cpu")));
  r.accumulate_steps = j.value("accumulate_steps", r.accumulate_steps);
  if (j.contains("precision")) {
    auto v = j["precision"].get<std::string>();
    auto i = std::find(precisions.begin(), precisions.end(), v);
    if (i == precisions.end())
      throw InvalidArgument("Unknown precision profile");
    r.precision =
        static_cast<training::PrecisionMode>(1 + i - precisions.begin());
  }
  if (j.contains("gradient_clip_norm"))
    r.gradient_clip_norm =
        j["gradient_clip_norm"].is_null()
            ? std::nullopt
            : std::optional<double>(j["gradient_clip_norm"].get<double>());
  r.class_weights = j.value("class_weights", r.class_weights);
  r.full_validation_metrics =
      j.value("full_validation_metrics", r.full_validation_metrics);
  r.tensorboard = j.value("tensorboard", r.tensorboard);
  if (j.contains("diagnostics")) {
    const auto& d = j["diagnostics"];
    Keys(d,
         {"enabled", "measure_data_wait", "gradient_norm_every", "max_tensors",
          "max_elements", "tile_elements", "max_scratch_bytes"});
    r.diagnostics = d.value("enabled", false);
    auto& o = r.diagnostics_options;
    o.measure_data_wait = d.value("measure_data_wait", o.measure_data_wait);
    o.gradient_norm_every =
        d.value("gradient_norm_every", o.gradient_norm_every);
    o.gradient_norm.max_tensors =
        d.value("max_tensors", o.gradient_norm.max_tensors);
    o.gradient_norm.max_elements =
        d.value("max_elements", o.gradient_norm.max_elements);
    o.gradient_norm.tile_elements =
        d.value("tile_elements", o.gradient_norm.tile_elements);
    o.gradient_norm.max_scratch_bytes =
        d.value("max_scratch_bytes", o.gradient_norm.max_scratch_bytes);
  }
  r.verify_image_contents =
      j.value("verify_image_contents", r.verify_image_contents);
  r.selection_metric = j.value("selection_metric", r.selection_metric);
  r.selection_maximize = j.value("selection_maximize", r.selection_maximize);
  r.label_smoothing = j.value("label_smoothing", r.label_smoothing);
  r.early_stopping_patience =
      j.value("early_stopping_patience", r.early_stopping_patience);
  r.save_interval_epochs =
      j.value("save_interval_epochs", r.save_interval_epochs);
  if (j.contains("optimizer")) {
    const auto& v = j["optimizer"];
    Keys(v, {"name", "learning_rate", "momentum", "weight_decay", "nesterov",
             "nominal_batch", "scale_weight_decay"});
    r.optimizer_name = v.value("name", r.optimizer_name);
    auto& o = r.optimizer;
    o.learning_rate = v.value("learning_rate", o.learning_rate);
    o.momentum = v.value("momentum", o.momentum);
    o.weight_decay = v.value("weight_decay", o.weight_decay);
    o.nesterov = v.value("nesterov", o.nesterov);
    o.nominal_batch = v.value("nominal_batch", o.nominal_batch);
    o.scale_weight_decay = v.value("scale_weight_decay", o.scale_weight_decay);
  }
  if (j.contains("scheduler")) {
    const auto& v = j["scheduler"];
    Keys(v, {"warmup_updates", "warmup_start_factor", "final_factor"});
    r.scheduler.warmup_updates =
        v.value("warmup_updates", r.scheduler.warmup_updates);
    r.scheduler.warmup_start_factor =
        v.value("warmup_start_factor", r.scheduler.warmup_start_factor);
    r.scheduler.final_factor =
        v.value("final_factor", r.scheduler.final_factor);
  }
  if (j.contains("ema")) {
    const auto& v = j["ema"];
    Keys(v, {"enabled", "decay", "ramp_updates"});
    r.ema = v.value("enabled", r.ema);
    r.ema_decay = v.value("decay", r.ema_decay);
    r.ema_ramp_updates = v.value("ramp_updates", r.ema_ramp_updates);
  }
  if (j.contains("augmentation")) {
    const auto& v = j["augmentation"];
    Keys(v, {"horizontal_flip",
             "vertical_flip",
             "scale",
             "translate",
             "hsv_h",
             "hsv_s",
             "hsv_v",
             "grayscale",
             "blur",
             "median_blur",
             "clahe",
             "clahe_tiles",
             "clahe_clip",
             "mosaic",
             "mixup",
             "cutmix",
             "copypaste",
             "close_epochs",
             "max_occlusion",
             "degrees",
             "shear",
             "perspective",
             "projective_min_box_area",
             "projective_min_visibility",
             "blur_kernel",
             "blur_sigma",
             "median_kernel"});
    auto& a = r.data.augmentation;
    a.horizontal_flip_probability = v.value("horizontal_flip", 0.);
    a.vertical_flip_probability = v.value("vertical_flip", 0.);
    a.spatial.scale_gain = v.value("scale", 0.);
    a.spatial.translate_fraction = v.value("translate", 0.);
    a.photo.hue_shift = v.value("hsv_h", 0.);
    a.photo.saturation_gain = v.value("hsv_s", 0.);
    a.photo.value_gain = v.value("hsv_v", 0.);
    a.photo.grayscale_probability = v.value("grayscale", 0.);
    a.photo.blur_probability = v.value("blur", 0.);
    a.photo.median_blur_probability = v.value("median_blur", 0.);
    a.photo.blur_kernel = v.value("blur_kernel", a.photo.blur_kernel);
    a.photo.blur_sigma = v.value("blur_sigma", a.photo.blur_sigma);
    a.photo.median_kernel = v.value("median_kernel", a.photo.median_kernel);
    a.photo.clahe_probability = v.value("clahe", 0.);
    a.photo.clahe_tiles = v.value("clahe_tiles", 8U);
    a.photo.clahe_clip = v.value("clahe_clip", 2.);
    a.multi_source.mosaic_probability = v.value("mosaic", 0.);
    a.multi_source.mixup_probability = v.value("mixup", 0.);
    a.multi_source.cutmix_probability = v.value("cutmix", 0.);
    a.multi_source.copypaste_probability = v.value("copypaste", 0.);
    a.multi_source.total_epochs = r.epochs;
    a.multi_source.close_epochs =
        v.value("close_epochs", (std::min)(std::uint64_t{10}, r.epochs));
    a.multi_source.max_occlusion =
        v.value("max_occlusion", a.multi_source.max_occlusion);
    a.projective.degrees = v.value("degrees", 0.);
    a.projective.shear_degrees = v.value("shear", 0.);
    a.projective.perspective = v.value("perspective", 0.);
    a.projective.min_box_area =
        v.value("projective_min_box_area", a.projective.min_box_area);
    a.projective.min_visibility =
        v.value("projective_min_visibility", a.projective.min_visibility);
  }
  if (j.contains("classification_augmentation")) {
    const auto& v = j["classification_augmentation"];
    Keys(v, {"crop_min_area", "crop_max_area", "crop_min_aspect",
             "crop_max_aspect", "validation_crop_fraction", "brightness",
             "contrast", "saturation", "randaugment_operations",
             "randaugment_magnitude", "erasing", "mixup", "cutmix"});
    auto& a = r.data.classification;
    a.crop_min_area = v.value("crop_min_area", a.crop_min_area);
    a.crop_max_area = v.value("crop_max_area", a.crop_max_area);
    a.crop_min_aspect = v.value("crop_min_aspect", a.crop_min_aspect);
    a.crop_max_aspect = v.value("crop_max_aspect", a.crop_max_aspect);
    a.validation_crop_fraction =
        v.value("validation_crop_fraction", a.validation_crop_fraction);
    a.brightness = v.value("brightness", 0.);
    a.contrast = v.value("contrast", 0.);
    a.saturation = v.value("saturation", 0.);
    a.randaugment_operations = v.value("randaugment_operations", 0U);
    a.randaugment_magnitude = v.value("randaugment_magnitude", .5);
    a.erasing_probability = v.value("erasing", 0.);
    a.mixup_probability = v.value("mixup", 0.);
    a.cutmix_probability = v.value("cutmix", 0.);
  }
  if (j.contains("prediction")) Results(j["prediction"], r.prediction);
  if (j.contains("evaluation")) Results(j["evaluation"], r.evaluation);
  r.data.num_classes = r.model.num_classes;
  r.data.augmentation.seed = r.seed;
  r.data.classification.enabled = r.model.task == TaskKind::classify;
  check_recipe(r);
  return r;
}
void check_recipe(const RecipeSpec& r) {
  if (!std::isfinite(r.initial_loss_scale) || r.initial_loss_scale <= 0)
    throw InvalidArgument(
        "Initial FP16 loss scale must be finite and positive");
  if ((r.schema_version != 1 && r.schema_version != 2) || r.model.architecture.empty() ||
      r.model.architecture.size() > 256 || r.model.num_classes < 1 ||
      r.model.num_classes > 10000 || static_cast<unsigned>(r.model.task) > 9 ||
      r.model.keypoints < 1 || r.model.keypoints > 256 ||
      r.model.mask_channels < 1 || r.model.mask_channels > 256 ||
      (r.model.keypoint_dimensions != 2 && r.model.keypoint_dimensions != 3))
    throw InvalidArgument("Invalid model/task recipe");
  if(!r.extension.multilabel_thresholds.empty()){if(r.schema_version<2||r.model.task!=TaskKind::classify_multilabel||r.extension.multilabel_thresholds.size()!=static_cast<std::size_t>(r.model.num_classes)||r.extension.threshold_validation_identity.empty())throw InvalidArgument("Calibrated thresholds need schema2 multilabel vocabulary and validation identity");for(double value:r.extension.multilabel_thresholds)if(!std::isfinite(value)||value<0||value>1)throw InvalidArgument("Invalid calibrated multilabel threshold");}
  if(!r.extension.tokenizer_merges.empty()||!r.extension.text_encoder_weights.empty()){if(r.schema_version<2||r.extension.tokenizer_merges.empty()||r.extension.text_encoder_weights.empty()||r.extension.tokenizer_sha256.size()!=64||r.extension.text_encoder_sha256.size()!=64)throw InvalidArgument("Native raw-text assets need tokenizer and text encoder with explicit digests");}
  if(!r.extension.teacher_architecture.empty()) {
    const auto& e=r.extension;
    if(r.schema_version<2||(r.model.task!=TaskKind::classify&&r.model.task!=TaskKind::classify_multilabel)||e.teacher_initialization.weights.empty()||e.teacher_initialization.sha256.size()!=64||!std::isfinite(e.distillation_temperature)||e.distillation_temperature<=0||!std::isfinite(e.logits_distillation_weight)||e.logits_distillation_weight<0||!std::isfinite(e.feature_distillation_weight)||e.feature_distillation_weight<0||e.logits_distillation_weight+e.feature_distillation_weight<=0)
      throw InvalidArgument("Distillation needs schema2 classification and a verified local teacher asset");
    if(r.data.classification.mixup_probability||r.data.classification.cutmix_probability)throw InvalidArgument("Same-view distillation currently requires batch mixing disabled");
    if(e.feature_distillation_weight&&(!e.feature_outputs||e.feature_projection<1))throw InvalidArgument("Feature distillation requires declared student projection");
  }
  if(r.extension.feature_projection<0||r.extension.feature_projection>4096)throw InvalidArgument("Invalid distillation projection dimension");
  if((static_cast<unsigned>(r.model.task)>4 || !r.extension.training_manifest.empty()) && r.schema_version<2)
    throw InvalidArgument("Supplementary tasks/manifests require recipe schema_version=2");
  if(static_cast<unsigned>(r.model.task)>4 && r.extension.training_manifest.empty())
    throw InvalidArgument("This task requires extension.training_manifest");
  if(!r.extension.validation_manifest.empty() && r.extension.training_manifest.empty())
    throw InvalidArgument("Validation task manifest requires training task manifest");
  if(!std::isfinite(r.extension.depth_gradient_weight)||r.extension.depth_gradient_weight<0||
     !std::isfinite(r.extension.triplet_margin)||r.extension.triplet_margin<0||
     !std::isfinite(r.extension.mask_ratio)||r.extension.mask_ratio<=0||r.extension.mask_ratio>1||
     r.extension.model_patch_size<1||r.extension.model_patch_size>32||r.extension.mask_patch_size<r.extension.model_patch_size||
     r.extension.mask_patch_size>128||r.extension.mask_patch_size%r.extension.model_patch_size)
    throw InvalidArgument("Invalid supplementary loss/mask policy");
  if(!r.extension.training_manifest.empty()) {
    const auto& a=r.data.augmentation;const auto& p=a.projective;const auto& m=a.multi_source;
    if(p.degrees||p.shear_degrees||p.perspective||p.scale_gain||p.translate_fraction||m.mosaic_probability||m.mixup_probability||m.cutmix_probability||m.copypaste_probability||r.data.classification.mixup_probability||r.data.classification.cutmix_probability)
      throw InvalidArgument("Supplementary targets require shared axis-aligned geometry and single-source augmentation");
  }
  if(r.model.task==TaskKind::pretrain_simmim &&
     (r.data.letterbox.canvas.height%r.extension.mask_patch_size||r.data.letterbox.canvas.width%r.extension.mask_patch_size))
    throw InvalidArgument("SimMIM image size must align to mask_patch_size");
  if (r.epochs < 1 || r.epochs > 1000000 || r.batch_size < 1 ||
      r.batch_size > 4096 || r.workers > 64 || r.accumulate_steps < 1 ||
      r.accumulate_steps > 4096 || r.prefetch_batches < 1 ||
      r.prefetch_batches > 256 || r.data.letterbox.canvas.width < 32 ||
      r.data.letterbox.canvas.width > 8192 ||
      r.data.letterbox.canvas.height < 32 ||
      r.data.letterbox.canvas.height > 8192)
    throw InvalidArgument("Invalid training/data resource bounds");
  if (!std::isfinite(r.label_smoothing) || r.label_smoothing < 0 ||
      r.label_smoothing >= 1 || !std::isfinite(r.ema_decay) ||
      r.ema_decay < 0 || r.ema_decay >= 1 ||
      !std::isfinite(r.ema_ramp_updates) || r.ema_ramp_updates <= 0)
    throw InvalidArgument("Invalid smoothing/EMA policy");
  if (r.optimizer_name != "sgd" && r.optimizer_name != "adamw" &&
      r.optimizer_name != "musgd" && r.optimizer_name != "auto")
    throw InvalidArgument("Optimizer must be sgd/adamw/musgd/auto");
  if (!std::isfinite(r.optimizer.learning_rate) ||
      r.optimizer.learning_rate <= 0 || !std::isfinite(r.optimizer.momentum) ||
      r.optimizer.momentum < 0 || r.optimizer.momentum >= 1 ||
      !std::isfinite(r.optimizer.weight_decay) ||
      r.optimizer.weight_decay < 0 || !r.optimizer.nominal_batch ||
      !std::isfinite(r.scheduler.final_factor) ||
      r.scheduler.final_factor < 0 || r.scheduler.final_factor > 1 ||
      !std::isfinite(r.scheduler.warmup_start_factor) ||
      r.scheduler.warmup_start_factor <= 0 ||
      r.scheduler.warmup_start_factor > 1)
    throw InvalidArgument("Invalid optimizer/scheduler recipe");
  if (r.cache.ram_bytes > (64ULL << 30U) ||
      r.cache.disk_bytes > (1ULL << 40U) ||
      (r.cache.disk_bytes && r.cache.directory.empty()) ||
      r.cache.decoder_revision.empty())
    throw InvalidArgument("Invalid decode cache recipe");
  if (r.initialization.max_bytes < 1 ||
      r.initialization.max_bytes > (4ULL << 30U) ||
      r.initialization.freeze_prefixes.size() > 4096 ||
      r.initialization.learning_rates.size() > 256)
    throw InvalidArgument("Invalid model initialization bounds");
  if (r.diagnostics_options.participant_id != "training.diagnostics")
    throw InvalidArgument(
        "Vision diagnostics uses the fixed Trainer participant identity");
  static_cast<void>(training::TrainingDiagnostics(r.diagnostics_options));
  if (r.class_weights.size() != 0 &&
      r.class_weights.size() != static_cast<std::size_t>(r.model.num_classes))
    throw InvalidArgument("class_weights must match the class vocabulary");
  double weight_sum = 0;
  for (auto value : r.class_weights) {
    if (!std::isfinite(value) || value < 0)
      throw InvalidArgument("Class weights must be finite nonnegative");
    weight_sum += value;
  }
  if (!r.class_weights.empty() &&
      (!std::isfinite(weight_sum) || weight_sum <= 0))
    throw InvalidArgument("At least one class weight must be positive");
  if (!r.keypoint_sigmas.empty() &&
      (r.model.task != TaskKind::pose ||
       r.keypoint_sigmas.size() != static_cast<std::size_t>(r.model.keypoints)))
    throw InvalidArgument(
        "keypoint_sigmas requires one value per pose keypoint");
  for (auto sigma : r.keypoint_sigmas)
    if (!std::isfinite(sigma) || sigma <= 0)
      throw InvalidArgument("Keypoint sigmas must be positive finite");
  if (r.selection_metric != "auto" &&
      !detail::selection_metric_available(r, r.selection_metric))
    throw InvalidArgument(
        "Vision selection metric is unavailable for this task");
  if (!r.full_validation_metrics && r.selection_metric != "auto" &&
      r.selection_metric != "validation.loss")
    throw InvalidArgument(
        "Task metric selection requires full_validation_metrics");
  if (r.data.multiscale &&
      std::ceil((std::max)(r.data.letterbox.canvas.width,
                           r.data.letterbox.canvas.height) *
                r.data.multiscale_max / 64.) *
              64. >
          r.data.multiscale_max_dimension)
    throw InvalidArgument(
        "Multiscale upper canvas exceeds its dimension budget");
  if (r.exif_policy == ExifPolicy::already_oriented_coordinates &&
      (r.training.format == DetectionDatasetFormat::yolo ||
       (r.validation && r.validation->format == DetectionDatasetFormat::yolo)))
    throw InvalidArgument(
        "YOLO normalized labels require encoded EXIF coordinates; use COCO for "
        "already-oriented coordinates");
  if (r.gradient_clip_norm &&
      (!std::isfinite(*r.gradient_clip_norm) || *r.gradient_clip_norm <= 0))
    throw InvalidArgument("Invalid gradient clipping norm");
  if (r.model.task == TaskKind::classify && r.classification_root.empty() && r.extension.training_manifest.empty())
    throw InvalidArgument("Classification needs classification_root");
  if (r.model.task != TaskKind::classify && r.training.source.empty() && r.extension.training_manifest.empty())
    throw InvalidArgument("Training needs an explicit dataset source");
  if (r.precision == training::PrecisionMode::cpu_bfloat16 &&
      !r.device.is_cpu())
    throw InvalidArgument("CPU bfloat16 requires CPU");
  if ((r.precision == training::PrecisionMode::cuda_float16 ||
       r.precision == training::PrecisionMode::cuda_bfloat16) &&
      !r.device.is_cuda())
    throw InvalidArgument("CUDA precision requires CUDA");
  auto checked_data = r.data;
  if (checked_data.augmentation.dataset_id.empty())
    checked_data.augmentation.dataset_id = "recipe.preflight";
  static_cast<void>(yolo_data_identity(checked_data));
  if (r.model.task == TaskKind::classify &&
      (r.data.multiscale || r.rectangular_batches))
    throw InvalidArgument(
        "Classification uses its fixed crop profile, not "
        "rectangular/multiscale batches");
  const auto& multi = r.data.augmentation.multi_source;
  if (r.rectangular_batches &&
      (multi.mosaic_probability > 0 || multi.mixup_probability > 0 ||
       multi.cutmix_probability > 0 || multi.copypaste_probability > 0))
    throw InvalidArgument(
        "Rectangular batches require multi-source augmentation disabled");
  if ((r.model.task == TaskKind::pose || r.model.task == TaskKind::obb) &&
      (multi.cutmix_probability > 0 || multi.copypaste_probability > 0))
    throw InvalidArgument("CutMix/CopyPaste has no pose/OBB recipe binding");
  if (r.model.task != TaskKind::instance_segment &&
      multi.copypaste_probability > 0)
    throw InvalidArgument("CopyPaste requires instance segmentation");
  if (r.model.task == TaskKind::classify &&
      (multi.mosaic_probability > 0 || multi.mixup_probability > 0 ||
       multi.cutmix_probability > 0 || multi.copypaste_probability > 0))
    throw InvalidArgument(
        "Classification mixing must use classification_augmentation soft-label "
        "recipe");
  if (r.model.task == TaskKind::obb &&
      (r.data.augmentation.projective.shear_degrees > 0 ||
       r.data.augmentation.projective.perspective > 0))
    throw InvalidArgument(
        "OBB recipe rejects shear/perspective rectangle changes");
  for (const auto* p : {&r.prediction, &r.evaluation}) {
    for (auto v : {p->confidence, p->nms_iou, p->mask_threshold})
      if (!std::isfinite(v) || v < 0 || v > 1)
        throw InvalidArgument("Invalid result threshold");
    if (p->max_candidates < 1 || p->max_candidates > 65536 ||
        p->max_detections < 1 || p->max_detections > p->max_candidates ||
        p->max_mask_elements < 1 || p->max_mask_elements > (1LL << 30U))
      throw InvalidArgument("Invalid result capacity");
    for (auto c : p->classes)
      if (c < 0 || c >= r.model.num_classes)
        throw InvalidArgument("Result class filter out of range");
  }
}
std::string recipe_json(const RecipeSpec& r) {
  const auto& a = r.data.augmentation;
  const auto& o = r.optimizer;
  J j = {
      {"schema_version", r.schema_version},
      {"model",
       {{"architecture", r.model.architecture},
        {"task", task_name(r.model.task)},
        {"variant", r.model.variant},
        {"num_classes", r.model.num_classes},
        {"keypoints", r.model.keypoints},
        {"keypoint_dimensions", r.model.keypoint_dimensions},
        {"mask_channels", r.model.mask_channels}}},
      {"image_size",
       {r.data.letterbox.canvas.width, r.data.letterbox.canvas.height}},
      {"batch_size", r.batch_size},
      {"workers", r.workers},
      {"epochs", r.epochs},
      {"seed", r.seed},
      {"deterministic", r.deterministic},
      {"initial_loss_scale", r.initial_loss_scale},
      {"device", r.device.str()},
      {"precision", precisions.at(static_cast<std::size_t>(r.precision) - 1)},
      {"accumulate_steps", r.accumulate_steps},
      {"exif_policy",
       r.exif_policy == ExifPolicy::reject_non_identity   ? "reject"
       : r.exif_policy == ExifPolicy::encoded_coordinates ? "encoded"
       : r.exif_policy == ExifPolicy::already_oriented_coordinates
           ? "already_oriented"
           : "ignore"},
      {"multiscale",
       {{"enabled", r.data.multiscale},
        {"min_factor", r.data.multiscale_min},
        {"max_factor", r.data.multiscale_max},
        {"max_dimension", r.data.multiscale_max_dimension}}},
      {"prefetch_batches", r.prefetch_batches},
      {"pin_memory", r.pin_memory},
      {"rectangular_batches", r.rectangular_batches},
      {"normalization",
       {{"mean", r.data.normalization_mean},
        {"std", r.data.normalization_std}}},
      {"preprocessing",
       {{"scale_up", r.data.letterbox.scale_up},
        {"center", r.data.letterbox.center},
        {"padding_value", r.data.letterbox.padding_value},
        {"interpolation", r.data.letterbox.geometry.interpolation ==
                                  data::image::ImageInterpolation::bilinear
                              ? "bilinear"
                          : r.data.letterbox.geometry.interpolation ==
                                  data::image::ImageInterpolation::bicubic
                              ? "bicubic"
                              : "nearest_exact"},
        {"min_box_area", r.data.letterbox.geometry.min_box_area},
        {"min_box_visibility", r.data.letterbox.geometry.min_box_visibility},
        {"min_region_area", r.data.letterbox.geometry.min_region_area},
        {"min_region_visibility",
         r.data.letterbox.geometry.min_region_visibility},
        {"exclude_crowd", r.data.exclude_crowd},
        {"exclude_difficult", r.data.exclude_difficult},
        {"max_targets", r.data.max_targets}}},
      {"horizontal_keypoint_permutations",
       r.data.letterbox.geometry.horizontal_keypoint_permutations},
      {"vertical_keypoint_permutations",
       r.data.letterbox.geometry.vertical_keypoint_permutations},
      {"cache",
       {{"ram_bytes", r.cache.ram_bytes},
        {"disk_bytes", r.cache.disk_bytes},
        {"directory", Path(r.cache.directory)},
        {"decoder_revision", r.cache.decoder_revision},
        {"reject_corrupt", r.cache.reject_corrupt}}},
      {"initialization",
       {{"weights", Path(r.initialization.weights)},
        {"sha256", r.initialization.sha256},
        {"key_mapping", r.initialization.key_mapping},
        {"ignored_source_keys", r.initialization.ignored_source_keys},
        {"retain_buffer_keys", r.initialization.retain_buffer_keys},
        {"replace_head_prefixes", r.initialization.replace_head_prefixes},
        {"freeze_prefixes", r.initialization.freeze_prefixes},
        {"learning_rates", J::array()},
        {"max_bytes", r.initialization.max_bytes}}},
      {"class_weights", r.class_weights},
      {"keypoint_sigmas", r.keypoint_sigmas},
      {"full_validation_metrics", r.full_validation_metrics},
      {"tensorboard", r.tensorboard},
      {"verify_image_contents", r.verify_image_contents},
      {"selection_metric", r.selection_metric},
      {"selection_maximize", r.selection_maximize},
      {"diagnostics",
       {{"enabled", r.diagnostics},
        {"measure_data_wait", r.diagnostics_options.measure_data_wait},
        {"gradient_norm_every", r.diagnostics_options.gradient_norm_every},
        {"max_tensors", r.diagnostics_options.gradient_norm.max_tensors},
        {"max_elements", r.diagnostics_options.gradient_norm.max_elements},
        {"tile_elements", r.diagnostics_options.gradient_norm.tile_elements},
        {"max_scratch_bytes",
         r.diagnostics_options.gradient_norm.max_scratch_bytes}}},
      {"gradient_clip_norm",
       r.gradient_clip_norm ? J(*r.gradient_clip_norm) : J(nullptr)},
      {"label_smoothing", r.label_smoothing},
      {"optimizer",
       {{"name", r.optimizer_name},
        {"learning_rate", o.learning_rate},
        {"momentum", o.momentum},
        {"weight_decay", o.weight_decay},
        {"nesterov", o.nesterov},
        {"nominal_batch", o.nominal_batch},
        {"scale_weight_decay", o.scale_weight_decay}}},
      {"scheduler",
       {{"warmup_updates", r.scheduler.warmup_updates},
        {"warmup_start_factor", r.scheduler.warmup_start_factor},
        {"final_factor", r.scheduler.final_factor}}},
      {"augmentation",
       {{"horizontal_flip", a.horizontal_flip_probability},
        {"vertical_flip", a.vertical_flip_probability},
        {"scale", a.spatial.scale_gain},
        {"translate", a.spatial.translate_fraction},
        {"hsv_h", a.photo.hue_shift},
        {"hsv_s", a.photo.saturation_gain},
        {"hsv_v", a.photo.value_gain},
        {"grayscale", a.photo.grayscale_probability},
        {"blur", a.photo.blur_probability},
        {"median_blur", a.photo.median_blur_probability},
        {"clahe", a.photo.clahe_probability},
        {"clahe_tiles", a.photo.clahe_tiles},
        {"clahe_clip", a.photo.clahe_clip},
        {"blur_kernel", a.photo.blur_kernel},
        {"blur_sigma", a.photo.blur_sigma},
        {"median_kernel", a.photo.median_kernel},
        {"mosaic", a.multi_source.mosaic_probability},
        {"mixup", a.multi_source.mixup_probability},
        {"cutmix", a.multi_source.cutmix_probability},
        {"copypaste", a.multi_source.copypaste_probability},
        {"close_epochs", a.multi_source.close_epochs},
        {"max_occlusion", a.multi_source.max_occlusion},
        {"degrees", a.projective.degrees},
        {"shear", a.projective.shear_degrees},
        {"perspective", a.projective.perspective},
        {"projective_min_box_area", a.projective.min_box_area},
        {"projective_min_visibility", a.projective.min_visibility}}},
      {"ema",
       {{"enabled", r.ema},
        {"decay", r.ema_decay},
        {"ramp_updates", r.ema_ramp_updates}}},
      {"early_stopping_patience", r.early_stopping_patience},
      {"save_interval_epochs", r.save_interval_epochs},
      {"prediction", ResultJson(r.prediction)},
      {"evaluation", ResultJson(r.evaluation)}};
  for (const auto& rule : r.initialization.learning_rates)
    j["initialization"]["learning_rates"].push_back(
        {{"prefix", rule.prefix}, {"multiplier", rule.multiplier}});
  if(r.schema_version>=2) j["extension"]={
    {"training_manifest",Path(r.extension.training_manifest)}, {"validation_manifest",Path(r.extension.validation_manifest)},
    {"multilabel_thresholds",r.extension.multilabel_thresholds},{"threshold_validation_identity",r.extension.threshold_validation_identity},
    {"asymmetric_multilabel",r.extension.asymmetric_multilabel},{"semantic_ignore",r.extension.semantic_ignore},
    {"relative_inverse_depth",r.extension.relative_inverse_depth},{"depth_gradient_weight",r.extension.depth_gradient_weight},
    {"model_patch_size",r.extension.model_patch_size},{"mask_patch_size",r.extension.mask_patch_size},
    {"mask_ratio",r.extension.mask_ratio},{"triplet_margin",r.extension.triplet_margin},
    {"tokenizer_merges",Path(r.extension.tokenizer_merges)},{"tokenizer_sha256",r.extension.tokenizer_sha256},{"text_encoder_weights",Path(r.extension.text_encoder_weights)},{"text_encoder_sha256",r.extension.text_encoder_sha256},
    {"activation_recomputation",r.extension.activation_recomputation},{"feature_outputs",r.extension.feature_outputs},{"feature_projection",r.extension.feature_projection},
    {"teacher_architecture",r.extension.teacher_architecture},{"teacher_weights",Path(r.extension.teacher_initialization.weights)},{"teacher_sha256",r.extension.teacher_initialization.sha256},
    {"distillation_temperature",r.extension.distillation_temperature},{"logits_distillation_weight",r.extension.logits_distillation_weight},{"feature_distillation_weight",r.extension.feature_distillation_weight}};
  if (r.model.task == TaskKind::classify) {
    j["classification_root"] = Path(r.classification_root);
    j["classification_validation_root"] =
        Path(r.classification_validation_root);
    const auto& c = r.data.classification;
    j["classification_augmentation"] = {
        {"crop_min_area", c.crop_min_area},
        {"crop_max_area", c.crop_max_area},
        {"crop_min_aspect", c.crop_min_aspect},
        {"crop_max_aspect", c.crop_max_aspect},
        {"validation_crop_fraction", c.validation_crop_fraction},
        {"brightness", c.brightness},
        {"contrast", c.contrast},
        {"saturation", c.saturation},
        {"randaugment_operations", c.randaugment_operations},
        {"randaugment_magnitude", c.randaugment_magnitude},
        {"erasing", c.erasing_probability},
        {"mixup", c.mixup_probability},
        {"cutmix", c.cutmix_probability}};
  } else {
    j["dataset"] = SourceJson(r.training);
    if (r.validation) j["validation"] = SourceJson(*r.validation);
  }
  return j.dump(2);
}
std::string recipe_identity(const RecipeSpec& r) {
  check_recipe(r);
  const auto s = recipe_json(r);
  return sha256_hex(std::as_bytes(std::span(s)));
}
}  // namespace jyppx::liuding::vision::tasks
