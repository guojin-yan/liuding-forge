// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "annotation_detail.hpp"
#include "panoptic_detail.hpp"

#include <jyppx/liuding/data_image/image_dataset.hpp>

#include <algorithm>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#if defined(_WIN32)
#include <Windows.h>
#endif

namespace jyppx::liuding::data::image {
namespace {

constexpr std::string_view kComponent = "data.image.dataset";

[[noreturn]] void Invalid(std::string_view operation, std::string_view expected,
                          std::string_view actual = {}) {
  throw InvalidArgument("Invalid image dataset configuration",
                        ErrorDetails{.component = std::string(kComponent),
                                     .operation = std::string(operation),
                                     .expected = std::string(expected),
                                     .actual = std::string(actual)});
}

[[noreturn]] void Malformed(std::string_view operation,
                            std::string_view actual = {}) {
  throw SerializationError(
      "Image dataset index is inconsistent with its source files",
      ErrorDetails{
          .component = std::string(kComponent),
          .operation = std::string(operation),
          .actual = std::string(actual),
          .hint = "Rebuild the index or verify the referenced dataset files"});
}

[[noreturn]] void Io(std::string_view operation,
                     const std::filesystem::path& path,
                     std::string_view reason = {}) {
  throw IoError(
      "Image dataset filesystem access failed",
      ErrorDetails{
          .component = std::string(kComponent),
          .operation = std::string(operation),
          .actual = reason.empty() ? path.string() : std::string(reason),
          .hint = "Check dataset permissions, file types, and links"});
}

[[noreturn]] void Cancel() {
  throw Cancelled("Image dataset read cancelled",
                  ErrorDetails{.component = std::string(kComponent),
                               .operation = "sample.cancel"});
}

void CheckStop(std::stop_token stop) {
  if (stop.stop_requested()) Cancel();
}

[[noreturn]] void TensorInvalid(std::string_view operation,
                                std::string_view expected) {
  throw InvalidArgument("Invalid Tensor image source",
                        ErrorDetails{.component = "data.image.tensor_dataset",
                                     .operation = std::string(operation),
                                     .expected = std::string(expected)});
}

[[noreturn]] void TensorLimit(std::string_view operation) {
  throw ResourceExhausted("Tensor image source limit exceeded",
                          ErrorDetails{.component = "data.image.tensor_dataset",
                                       .operation = std::string(operation)});
}

void CheckTensorStop(std::stop_token stop) {
  if (stop.stop_requested())
    throw Cancelled("Tensor image source read cancelled",
                    ErrorDetails{.component = "data.image.tensor_dataset",
                                 .operation = "sample.cancel"});
}

void CheckTensorId(std::string_view id, std::size_t maximum,
                   std::string_view operation) {
  if (id.empty() || !detail::Utf8(id))
    TensorInvalid(operation, "a non-empty UTF-8 ID without NUL");
  if (id.size() > maximum) TensorLimit(operation);
}

void ChargeTensorBytes(std::uint64_t pixels, std::uint64_t bytes_per_pixel,
                       std::uint64_t maximum, std::uint64_t& charged) {
  if (pixels > (maximum - charged) / bytes_per_pixel)
    TensorLimit("source.sample_bytes");
  charged += pixels * bytes_per_pixel;
}

std::string FoldAscii(std::string_view text) {
  std::string result(text);
  for (char& character : result) {
    const auto byte = static_cast<unsigned char>(character);
    if (byte >= static_cast<unsigned char>('A') &&
        byte <= static_cast<unsigned char>('Z')) {
      character = static_cast<char>(byte - static_cast<unsigned char>('A') +
                                    static_cast<unsigned char>('a'));
    }
  }
  return result;
}

std::filesystem::path ResolveDatasetFile(const std::filesystem::path& root,
                                         std::string_view relative) {
  if (relative.empty() || relative.front() == '/' ||
      relative.find_first_of("\\:<>\"|?*") != std::string_view::npos) {
    Malformed("path.relative", std::string(relative));
  }

  auto current = root;
  std::size_t begin = 0;
  while (begin <= relative.size()) {
    const auto end = relative.find('/', begin);
    const auto component = relative.substr(begin, end == std::string_view::npos
                                                      ? relative.size() - begin
                                                      : end - begin);
    if (component.empty() || component == "." || component == ".." ||
        component.back() == '.' || component.back() == ' ') {
      Malformed("path.component", std::string(relative));
    }
    std::u8string utf8_component;
    utf8_component.reserve(component.size());
    for (const char character : component) {
      utf8_component.push_back(static_cast<char8_t>(character));
    }
    current /= std::filesystem::path(utf8_component);

    std::error_code error;
    const auto status = std::filesystem::symlink_status(current, error);
    if (error || !std::filesystem::exists(status)) {
      Io("path.inspect", current, error ? error.message() : "missing path");
    }
    if (std::filesystem::is_symlink(status)) {
      Io("path.link", current, "symbolic links are not accepted");
    }
#if defined(_WIN32)
    const auto attributes = GetFileAttributesW(current.c_str());
    if (attributes == INVALID_FILE_ATTRIBUTES ||
        (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0U) {
      Io("path.reparse", current,
         "reparse points are not accepted in indexed paths");
    }
#endif

    if (end == std::string_view::npos) {
      if (!std::filesystem::is_regular_file(status)) {
        Io("path.file_type", current, "expected a regular file");
      }
      break;
    }
    if (!std::filesystem::is_directory(status)) {
      Io("path.parent_type", current, "expected a directory");
    }
    begin = end + 1U;
  }
  return current;
}

std::filesystem::path ValidateRoot(const std::filesystem::path& root) {
  if (root.empty()) Invalid("root.empty", "a non-empty existing directory");
  std::error_code error;
  const auto absolute =
      std::filesystem::absolute(root, error).lexically_normal();
  if (error) Io("root.absolute", root, error.message());
  auto component = absolute.root_path();
  auto relative = absolute.relative_path();
  for (const auto& part : relative) {
    component /= part;
    const auto status = std::filesystem::symlink_status(component, error);
    if (error || !std::filesystem::exists(status)) {
      Io("root.inspect", component, error ? error.message() : "missing path");
    }
    if (std::filesystem::is_symlink(status)) {
      Io("root.link", component, "root ancestors may not be symbolic links");
    }
#if defined(_WIN32)
    const auto attributes = GetFileAttributesW(component.c_str());
    if (attributes == INVALID_FILE_ATTRIBUTES ||
        (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0U) {
      Io("root.reparse", component, "root ancestors may not be reparse points");
    }
#endif
  }
  const auto root_status = std::filesystem::symlink_status(root, error);
  if (error || !std::filesystem::exists(root_status)) {
    Io("root.inspect", root, error ? error.message() : "missing directory");
  }
  if (std::filesystem::is_symlink(root_status)) {
    Io("root.link", root, "the selected root is a symbolic link");
  }
#if defined(_WIN32)
  const auto attributes = GetFileAttributesW(root.c_str());
  if (attributes == INVALID_FILE_ATTRIBUTES ||
      (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0U) {
    Io("root.reparse", root, "the selected root is a reparse point");
  }
#endif
  if (!std::filesystem::is_directory(root_status)) {
    Io("root.type", root, "expected a directory");
  }

  auto canonical = std::filesystem::canonical(root, error);
  if (error) Io("root.canonical", root, error.message());
  return canonical;
}

void ValidateIndex(const ManifestIndex& index) {
  if (index.task() != ManifestTask::classification &&
      index.task() != ManifestTask::paired) {
    Invalid("index.task", "a classification or paired-file manifest");
  }
  const auto& categories = index.categories();
  for (const auto& record : index.records()) {
    if (index.task() == ManifestTask::classification) {
      if (record.target_path || record.labels.empty()) {
        Malformed("index.classification", record.path);
      }
      for (const auto label : record.labels) {
        if (label >= categories.size()) Malformed("index.label", record.path);
      }
    } else if (!record.target_path || !record.labels.empty()) {
      Malformed("index.paired", record.path);
    }
  }
}

}  // namespace

ImageDataset::ImageDataset(std::filesystem::path root, ManifestIndex index,
                           ImageDatasetOptions options)
    : ImageDataset(std::move(root), std::move(index), nullptr, options) {}

ImageDataset::ImageDataset(std::filesystem::path root, ManifestIndex index,
                           AnnotationIndex annotations,
                           ImageDatasetOptions options)
    : ImageDataset(
          std::move(root), std::move(index),
          std::make_shared<const AnnotationIndex>(std::move(annotations)),
          options) {}

ImageDataset::ImageDataset(std::filesystem::path root, ManifestIndex index,
                           std::shared_ptr<const AnnotationIndex> annotations,
                           ImageDatasetOptions options)
    : root_(ValidateRoot(root)),
      index_(std::move(index)),
      options_(options),
      annotations_(std::move(annotations)) {
  ValidateIndex(index_);
  if (options_.resource_budget) {
    if (!options_.image.resource_budget)
      options_.image.resource_budget = options_.resource_budget;
    if (!options_.paired_target.resource_budget)
      options_.paired_target.resource_budget = options_.resource_budget;
    if (options_.mask_target && !options_.mask_target->resource_budget)
      options_.mask_target->resource_budget = options_.resource_budget;
    if (options_.panoptic_target && !options_.panoptic_target->resource_budget)
      options_.panoptic_target->resource_budget = options_.resource_budget;
  }
  if (options_.mask_target && index_.task() != ManifestTask::paired)
    Invalid("mask.task", "a paired manifest when mask_target is configured");
  if (options_.panoptic_target &&
      (options_.mask_target || index_.task() != ManifestTask::classification ||
       !annotations_ ||
       annotations_->format != AnnotationFormat::coco_panoptic ||
       options_.panoptic_target->encoding != MaskEncoding::panoptic_rgb ||
       !options_.panoptic_target->void_id))
    Invalid("panoptic.task",
            "COCO panoptic annotations, panoptic_rgb and explicit void ID");

  label_categories_ =
      std::make_shared<const std::vector<VisionCategory>>(index_.categories());

  if (annotations_ == nullptr) return;
  if (index_.task() != ManifestTask::classification ||
      annotations_->categories.empty()) {
    Invalid("annotations.task",
            "classification samples and a non-empty category map");
  }

  categories_ = std::shared_ptr<const std::vector<VisionCategory>>(
      annotations_, &annotations_->categories);
  sample_annotations_.resize(index_.records().size());

  std::unordered_map<std::string, std::size_t> sample_by_path;
  sample_by_path.reserve(index_.records().size());
  for (std::size_t i = 0; i < index_.records().size(); ++i) {
    if (!sample_by_path.emplace(index_.records()[i].path, i).second) {
      Malformed("index.path_duplicate", index_.records()[i].path);
    }
  }

  std::unordered_set<std::string> annotation_paths;
  annotation_paths.reserve(annotations_->images.size());
  for (std::size_t i = 0; i < annotations_->images.size(); ++i) {
    const auto& annotation = annotations_->images[i];
    if (annotation.image_path.empty() ||
        !annotation_paths.insert(FoldAscii(annotation.image_path)).second) {
      Malformed("annotations.path_duplicate", annotation.image_path);
    }
    const auto found = sample_by_path.find(annotation.image_path);
    if (found == sample_by_path.end()) {
      Malformed("annotations.path_unmatched", annotation.image_path);
    }
    static_cast<void>(ResolveDatasetFile(root_, annotation.image_path));
    if (options_.panoptic_target) {
      if (!annotation.panoptic_mask_path)
        Malformed("panoptic.reference", annotation.image_path);
      static_cast<void>(
          ResolveDatasetFile(root_, *annotation.panoptic_mask_path));
    }
    for (const auto& object : annotation.objects) {
      if (object.category >= annotations_->categories.size()) {
        Malformed("annotations.category", annotation.image_path);
      }
    }
    sample_annotations_[found->second] =
        std::shared_ptr<const VisionAnnotationRecord>(annotations_,
                                                      &annotation);
  }
  if (options_.panoptic_target &&
      std::any_of(sample_annotations_.begin(), sample_annotations_.end(),
                  [](const auto& record) { return !record; }))
    Invalid("panoptic.missing_annotation",
            "a panoptic record for every sample");
}

VisionSample ImageDataset::get(std::size_t index) const {
  return get(index, {});
}

VisionSample ImageDataset::get(std::size_t index, std::stop_token stop) const {
  if (index >= index_.records().size()) {
    throw InvalidArgument("Image sample index is outside the dataset",
                          ErrorDetails{.component = std::string(kComponent),
                                       .operation = "sample.index",
                                       .expected = "index < size",
                                       .actual = std::to_string(index)});
  }
  CheckStop(stop);

  const auto& record = index_.records()[index];
  const auto image_path = ResolveDatasetFile(root_, record.path);
  auto decoded = read_image_file(image_path, options_.image, stop);

  VisionSample sample;
  sample.image = std::move(decoded.tensor);
  sample.image_resource_leases = std::move(decoded.resource_leases);
  sample.labels = record.labels;
  sample.image_path = record.path;
  sample.width = decoded.width;
  sample.height = decoded.height;
  sample.pixels = decoded.pixels;
  sample.geometry.emplace(ImageExtent{decoded.width, decoded.height});
  sample.label_categories = label_categories_;
  sample.categories = categories_;
  if (index < sample_annotations_.size()) {
    sample.annotation = sample_annotations_[index];
    if (sample.annotation != nullptr && sample.annotation->width != 0U &&
        sample.annotation->height != 0U &&
        (sample.annotation->width != sample.width ||
         sample.annotation->height != sample.height)) {
      Malformed("annotations.dimensions", record.path);
    }
  }

  if (record.target_path) {
    CheckStop(stop);
    const auto target_path = ResolveDatasetFile(root_, *record.target_path);
    if (options_.mask_target) {
      auto target = read_mask_file(target_path, *options_.mask_target, stop);
      if (target.width != sample.width || target.height != sample.height)
        Malformed("mask.dimensions", *record.target_path);
      sample.mask_target = std::move(target);
    } else {
      auto target = read_image_file(target_path, options_.paired_target, stop);
      if (options_.require_matching_target_size &&
          (target.width != sample.width || target.height != sample.height)) {
        Malformed("target.dimensions", *record.target_path);
      }
      sample.paired_target = std::move(target.tensor);
      sample.paired_target_resource_leases = std::move(target.resource_leases);
      sample.paired_target_pixels = target.pixels;
    }
    sample.target_path = *record.target_path;
  }
  if (options_.panoptic_target) {
    CheckStop(stop);
    const auto reference = *sample.annotation->panoptic_mask_path;
    auto mask = read_mask_file(ResolveDatasetFile(root_, reference),
                               *options_.panoptic_target, stop);
    auto bound = detail::BindPanopticMask(
        *sample.annotation, mask, sample.categories->size(),
        options_.panoptic_validation, options_.resource_budget, stop);
    auto owner = std::make_shared<detail::PanopticRecordOwner>();
    owner->output_lease = std::move(bound.output_lease);
    owner->record = std::move(bound.record);
    sample.annotation =
        std::shared_ptr<const VisionAnnotationRecord>(owner, &owner->record);
    sample.mask_target = std::move(mask);
    sample.target_path = reference;
  }
  CheckStop(stop);
  return sample;
}

std::size_t ImageDataset::size() const { return index_.records().size(); }

const ManifestIndex& ImageDataset::index() const noexcept { return index_; }

TensorImageDataset::TensorImageDataset(std::vector<TensorImageRecord> records,
                                       std::vector<VisionCategory> categories,
                                       TensorImageDatasetLimits limits)
    : records_(std::move(records)),
      categories_(std::make_shared<const std::vector<VisionCategory>>(
          std::move(categories))) {
  if (!limits.max_records || !limits.max_dimension || !limits.max_pixels ||
      !limits.max_sample_bytes || !limits.max_id_bytes ||
      !limits.max_labels_per_sample)
    TensorInvalid("source.limits", "positive source and image limits");
  if (records_.size() > limits.max_records) TensorLimit("source.records");

  for (const auto& record : records_) {
    CheckTensorId(record.id, limits.max_id_bytes, "source.id");
    if (record.labels.size() > limits.max_labels_per_sample)
      TensorLimit("source.labels");
    validate_pixels(record.image, record.pixels);
    if (record.image.requires_grad())
      TensorInvalid("source.autograd", "image input without autograd history");
    const auto width =
        record.image.size(record.pixels.layout == ImageLayout::chw ? 2 : 1);
    const auto height =
        record.image.size(record.pixels.layout == ImageLayout::chw ? 1 : 0);
    if (width > limits.max_dimension || height > limits.max_dimension)
      TensorLimit("source.dimension");
    const auto pixels =
        static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
    if (pixels > limits.max_pixels) TensorLimit("source.pixels");
    std::uint64_t charged = 0;
    ChargeTensorBytes(pixels,
                      static_cast<std::uint64_t>(record.pixels.channels()) *
                          record.image.element_size(),
                      limits.max_sample_bytes, charged);
    const bool paired = record.paired_target.has_value();
    const bool masked = record.mask_target.has_value();
    if (paired && masked)
      TensorInvalid("source.targets", "one paired image or one discrete mask");
    if (paired != record.paired_target_pixels.has_value() ||
        (paired || masked) != record.target_id.has_value())
      TensorInvalid("source.target_metadata",
                    "a matching PixelSpec and explicit target ID");
    if (record.target_id)
      CheckTensorId(*record.target_id, limits.max_id_bytes, "source.target_id");
    if (paired) {
      validate_pixels(*record.paired_target, *record.paired_target_pixels);
      if (record.paired_target->requires_grad() ||
          record.paired_target->device() != record.image.device())
        TensorInvalid("source.paired_device",
                      "same-device paired pixels without autograd history");
      const auto pair_height = record.paired_target->size(
          record.paired_target_pixels->layout == ImageLayout::chw ? 1 : 0);
      const auto pair_width = record.paired_target->size(
          record.paired_target_pixels->layout == ImageLayout::chw ? 2 : 1);
      if (pair_width != width || pair_height != height)
        TensorInvalid("source.paired_dimensions",
                      "paired pixels aligned to the source image");
      ChargeTensorBytes(
          pixels,
          static_cast<std::uint64_t>(record.paired_target_pixels->channels()) *
              record.paired_target->element_size(),
          limits.max_sample_bytes, charged);
    }
    if (masked) {
      const auto& mask = *record.mask_target;
      if (mask.encoding != MaskEncoding::class_index || !mask.palette.empty() ||
          !mask.ids.defined() || !mask.ids.device().is_cpu() ||
          mask.ids.scalar_type() != torch::kInt64 ||
          mask.ids.layout() != torch::kStrided || mask.ids.dim() != 2 ||
          mask.ids.requires_grad() || mask.ids.size(0) != height ||
          mask.ids.size(1) != width || mask.width != width ||
          mask.height != height)
        TensorInvalid("source.mask", "aligned CPU int64 class-index labels");
      ChargeTensorBytes(pixels, sizeof(std::int64_t), limits.max_sample_bytes,
                        charged);
    }
    for (const auto label : record.labels)
      if (label >= categories_->size())
        TensorInvalid("source.label", "indices into the category map");
  }
}

VisionSample TensorImageDataset::get(std::size_t index) const {
  return get(index, {});
}

VisionSample TensorImageDataset::get(std::size_t index,
                                     std::stop_token stop) const {
  if (index >= records_.size()) TensorInvalid("sample.index", "index < size");
  CheckTensorStop(stop);
  const auto& record = records_[index];
  VisionSample sample;
  sample.image = record.image;
  sample.labels = record.labels;
  sample.image_path = record.id;
  sample.width = static_cast<std::uint32_t>(
      record.image.size(record.pixels.layout == ImageLayout::chw ? 2 : 1));
  sample.height = static_cast<std::uint32_t>(
      record.image.size(record.pixels.layout == ImageLayout::chw ? 1 : 0));
  sample.pixels = record.pixels;
  sample.paired_target = record.paired_target;
  sample.paired_target_pixels = record.paired_target_pixels;
  sample.mask_target = record.mask_target;
  sample.target_path = record.target_id;
  sample.geometry.emplace(ImageExtent{sample.width, sample.height});
  sample.label_categories = categories_;
  CheckTensorStop(stop);
  return sample;
}

std::size_t TensorImageDataset::size() const { return records_.size(); }

}  // namespace jyppx::liuding::data::image
