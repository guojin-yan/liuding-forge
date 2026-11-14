// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/vision/tasks/data/dataset.hpp>

#include <algorithm>
#include <fstream>
#include <map>
#include <set>
#include <span>
#include <string_view>
#include <utility>

namespace jyppx::liuding::vision::tasks {
namespace {
namespace fs = std::filesystem;
namespace image = data::image;

void CheckStop(std::stop_token stop) {
  if (stop.stop_requested())
    throw Cancelled("Detection dataset import cancelled");
}

std::string Utf8(const fs::path& path) {
  const auto bytes = path.generic_u8string();
  return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

std::string Lower(std::string text) {
  for (auto& ch : text)
    if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch + ('a' - 'A'));
  return text;
}

std::string Digest(const std::string& text) {
  return sha256_hex(std::as_bytes(std::span(text)));
}

fs::path Absolute(const fs::path& path) {
  std::error_code error;
  auto resolved = fs::canonical(path, error);
  if (error)
    throw IoError("Detection dataset path cannot be resolved: " + Utf8(path));
  return resolved;
}

std::string Relative(const fs::path& path, const fs::path& root) {
  const auto relative = Absolute(path).lexically_relative(root);
  if (relative.empty() || relative.is_absolute())
    throw InvalidArgument(
        "Detection image must be inside the selected image root");
  for (const auto& part : relative)
    if (part == "..")
      throw InvalidArgument("Detection image escapes the selected image root");
  return Utf8(relative);
}

bool IsImage(const fs::path& path) {
  const auto extension = Lower(Utf8(path.extension()));
  for (const auto suffix :
       {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm", ".pnm", ".tga"})
    if (extension == suffix) return true;
  return false;
}

struct ImportState {
  const DetectionDatasetOptions& options;
  std::stop_token stop;
  std::size_t entries = 0, objects = 0, document_bytes = 0;
  std::set<std::string> paths;
  std::string identity = "liuding.detection-import/v1\n";

  void ChargeDocument(const fs::path& path) {
    CheckStop(stop);
    std::error_code error;
    const auto count = fs::file_size(path, error);
    if (error) throw IoError("Cannot inspect detection annotation file");
    if (count > options.limits.max_input_bytes - document_bytes)
      throw ResourceExhausted(
          "Detection import aggregate annotation byte bound exceeded");
    document_bytes += static_cast<std::size_t>(count);
  }

  std::vector<fs::path> Scan(const fs::path& directory, bool xml,
                             std::string_view annotation_extension = ".xml") {
    if (!fs::is_directory(directory))
      throw IoError("Detection source is not a directory");
    std::vector<fs::path> files;
    for (fs::recursive_directory_iterator it(directory), end; it != end; ++it) {
      CheckStop(stop);
      if (++entries > options.max_directory_entries ||
          static_cast<std::size_t>(it.depth()) > options.max_directory_depth)
        throw ResourceExhausted(
            "Detection dataset directory scan bound exceeded");
      const auto name = Utf8(it->path().filename());
      if (fs::is_symlink(it->symlink_status()) ||
          (!name.empty() && name[0] == '.')) {
        if (it->is_directory()) it.disable_recursion_pending();
        continue;
      }
      if (it->is_regular_file() &&
          (xml ? Lower(Utf8(it->path().extension())) == annotation_extension
               : IsImage(it->path()))) {
        if (files.size() >= options.limits.max_images)
          throw ResourceExhausted(
              "Detection import image count bound exceeded");
        files.push_back(it->path());
      }
    }
    std::sort(files.begin(), files.end(),
              [](const auto& left, const auto& right) {
                return Utf8(left) < Utf8(right);
              });
    return files;
  }

  void Append(image::AnnotationIndex& output, image::AnnotationIndex input) {
    if (input.images.size() != 1)
      throw SerializationError(
          "Per-image detection label must describe one image");
    auto record = std::move(input.images.front());
    if (output.images.size() >= options.limits.max_images ||
        record.objects.size() > options.limits.max_objects - objects)
      throw ResourceExhausted(
          "Detection import aggregate image/object bound exceeded");
    if (!paths.insert(Lower(record.image_path)).second)
      throw SerializationError(
          "Duplicate or case-colliding detection image path");
    objects += record.objects.size();
    record.source_image_id = static_cast<std::int64_t>(output.images.size());
    identity += input.source_sha256 + ":" +
                std::to_string(record.image_path.size()) + ":" +
                record.image_path + ":" + std::to_string(record.width) + ":" +
                std::to_string(record.height) + "\n";
    output.images.push_back(std::move(record));
  }
};

void FreezeCategories(image::AnnotationIndex& annotations,
                      const std::vector<image::VisionCategory>& expected) {
  if (expected.empty()) return;
  if (expected.size() != annotations.categories.size())
    throw InvalidArgument(
        "Detection split category count differs from the frozen training map");
  std::map<std::string, std::size_t> by_name;
  std::set<std::int64_t> source_ids;
  for (std::size_t i = 0; i < expected.size(); ++i)
    if (expected[i].name.empty() ||
        !by_name.emplace(expected[i].name, i).second ||
        !source_ids.insert(expected[i].source_id).second)
      throw InvalidArgument(
          "Frozen detection categories must have unique names and source IDs");
  std::vector<std::size_t> remap;
  for (const auto& category : annotations.categories) {
    const auto found = by_name.find(category.name);
    if (found == by_name.end() ||
        expected[found->second].source_id != category.source_id)
      throw InvalidArgument(
          "Detection split category identity differs from the frozen training "
          "map");
    remap.push_back(found->second);
  }
  for (auto& record : annotations.images)
    for (auto& object : record.objects) {
      if (object.category >= remap.size())
        throw SerializationError("Invalid canonical detection category");
      object.category = remap[object.category];
    }
  annotations.categories = expected;
}

fs::path LabelPath(const fs::path& root, std::string_view relative) {
  auto path = fs::u8path(relative);
  std::vector<fs::path> parts;
  for (const auto& part : path) parts.push_back(part);
  auto selected = parts.size();
  for (std::size_t i = 0; i < parts.size(); ++i)
    if (parts[i] == "images") selected = i;
  if (selected == parts.size())
    throw InvalidArgument(
        "YOLO images need an explicit images path component for label "
        "matching");
  parts[selected] = "labels";
  path.clear();
  for (const auto& part : parts) path /= part;
  path.replace_extension(".txt");
  return root / path;
}

std::vector<fs::path> ReadImageList(const fs::path& list, const fs::path& root,
                                    ImportState& state) {
  state.ChargeDocument(list);
  const auto size = fs::file_size(list);
  std::string bytes(static_cast<std::size_t>(size), '\0');
  std::ifstream stream(list, std::ios::binary);
  if (!stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size())))
    throw IoError("Cannot read YOLO image list");
  state.identity += Digest(bytes) + "\n";
  if (bytes.starts_with("\xef\xbb\xbf")) bytes.erase(0, 3);
  std::vector<fs::path> files;
  for (std::size_t begin = 0; begin < bytes.size();) {
    CheckStop(state.stop);
    auto end = bytes.find('\n', begin);
    if (end == std::string::npos) end = bytes.size();
    std::string_view line(bytes.data() + begin, end - begin);
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
    if (!line.empty()) {
      if (line.size() > state.options.limits.max_string_bytes ||
          line.find('\0') != std::string_view::npos)
        throw SerializationError("Invalid YOLO image-list path");
      auto path = fs::u8path(line);
      if (!path.is_absolute()) path = list.parent_path() / path;
      if (!IsImage(path))
        throw SerializationError("Unsupported image suffix in YOLO split list");
      static_cast<void>(Relative(path, root));
      if (files.size() >= state.options.limits.max_images)
        throw ResourceExhausted("YOLO image-list record bound exceeded");
      files.push_back(std::move(path));
    }
    begin = end + 1;
  }
  return files;
}

}  // namespace

DetectionDatasetImport read_detection_dataset(
    const DetectionDatasetOptions& options, std::stop_token stop) try {
  CheckStop(stop);
  if (options.source.empty() || options.limits.max_input_bytes == 0 ||
      options.limits.max_images == 0 || options.max_directory_entries == 0 ||
      options.max_directory_depth == 0 ||
      (options.missing_labels != MissingLabelPolicy::reject &&
       options.missing_labels != MissingLabelPolicy::empty_negative))
    throw InvalidArgument(
        "Detection import requires a source and positive resource limits");
  ImportState state{options, stop};
  DetectionDatasetImport result;
  image::AnnotationReadOptions read;
  read.limits = options.limits;
  if (static_cast<unsigned>(options.task) > 3)
    throw InvalidArgument("Unknown annotation task dialect");
  if (options.format == DetectionDatasetFormat::coco) {
    if (options.image_root.empty())
      throw InvalidArgument("COCO import requires image_root");
    result.image_root = Absolute(options.image_root);
    state.ChargeDocument(options.source);
    result.annotations = image::read_annotation_file(
        options.source,
        options.task == DatasetAnnotationTask::pose
            ? image::AnnotationFormat::coco_keypoints
            : image::AnnotationFormat::coco_instances,
        read, stop);
    state.identity += result.annotations.source_sha256 + "\n";
  } else if (options.format == DetectionDatasetFormat::voc) {
    if (options.task != DatasetAnnotationTask::detect)
      throw InvalidArgument("Pascal VOC XML supplies detection boxes only");
    if (options.image_root.empty() || options.categories.empty())
      throw InvalidArgument(
          "VOC import requires image_root and a frozen category map");
    result.image_root = Absolute(options.image_root);
    read.categories = result.annotations.categories = options.categories;
    read.voc_coordinates = options.voc_coordinates;
    result.annotations.format = image::AnnotationFormat::voc;
    for (const auto& xml : state.Scan(options.source, true)) {
      state.ChargeDocument(xml);
      // The VOC document owns the image filename, including its extension and
      // any path relative to image_root. An override here would silently point
      // PNG images and nested datasets at unrelated JPEG paths.
      read.image_path.clear();
      state.Append(result.annotations,
                   image::read_annotation_file(
                       xml, image::AnnotationFormat::voc, read, stop));
    }
  } else if (options.format == DetectionDatasetFormat::labelme) {
    if (options.image_root.empty() || options.categories.empty())
      throw InvalidArgument(
          "LabelMe import requires image_root and frozen categories");
    if (options.task == DatasetAnnotationTask::pose)
      throw InvalidArgument(
          "LabelMe point groups need an explicit pose layout; use COCO "
          "keypoints or YOLO pose");
    result.image_root = Absolute(options.image_root);
    read.categories = result.annotations.categories = options.categories;
    result.annotations.format = image::AnnotationFormat::labelme;
    for (const auto& path : state.Scan(options.source, true, ".json")) {
      state.ChargeDocument(path);
      read.image_path.clear();
      auto parsed = image::read_annotation_file(
          path, image::AnnotationFormat::labelme, read, stop);
      for (auto& record : parsed.images)
        record.image_path =
            Relative(path.parent_path() / fs::u8path(record.image_path),
                     result.image_root);
      state.Append(result.annotations, std::move(parsed));
    }
  } else if (options.format == DetectionDatasetFormat::yolo) {
    state.ChargeDocument(options.source);
    const auto spec = image::read_yolo_dataset_yaml_file(options.source,
                                                         options.limits, stop);
    result.flip_indices = spec.flip_indices;
    if (options.task == DatasetAnnotationTask::pose) {
      read.keypoint_count =
          options.keypoint_count ? options.keypoint_count : spec.keypoint_count;
      read.keypoint_dimensions = spec.keypoint_count
                                     ? spec.keypoint_dimensions
                                     : options.keypoint_dimensions;
      if (!read.keypoint_count ||
          (options.keypoint_count && spec.keypoint_count &&
           options.keypoint_count != spec.keypoint_count) ||
          (spec.keypoint_count &&
           options.keypoint_dimensions != spec.keypoint_dimensions))
        throw InvalidArgument("Model and dataset pose shapes differ");
    }
    state.identity += spec.source_sha256 + "\n";
    const auto yaml = Absolute(options.source);
    result.image_root = Absolute(
        options.image_root.empty() ? yaml.parent_path() / fs::u8path(spec.root)
                                   : options.image_root);
    const std::vector<std::string>* splits = nullptr;
    switch (options.split) {
      case DatasetSplit::train:
        splits = &spec.train;
        break;
      case DatasetSplit::validation:
        splits = &spec.validation;
        break;
      case DatasetSplit::test:
        splits = &spec.test;
        break;
      default:
        throw InvalidArgument("Invalid YOLO split selection");
    }
    if (splits->empty())
      throw InvalidArgument("The requested YOLO split is not configured");
    read.categories = result.annotations.categories = spec.categories;
    result.annotations.format =
        options.task == DatasetAnnotationTask::instance_segment
            ? image::AnnotationFormat::yolo_segmentation
        : options.task == DatasetAnnotationTask::pose
            ? image::AnnotationFormat::yolo_pose
        : options.task == DatasetAnnotationTask::obb
            ? image::AnnotationFormat::yolo_obb
            : image::AnnotationFormat::yolo_detection;
    for (const auto& split : *splits) {
      auto path = fs::u8path(split);
      if (!path.is_absolute()) path = result.image_root / path;
      const auto files = fs::is_directory(path)
                             ? state.Scan(path, false)
                             : ReadImageList(path, result.image_root, state);
      for (const auto& file : files) {
        CheckStop(stop);
        read.image_path = Relative(file, result.image_root);
        if (state.paths.contains(Lower(read.image_path)))
          throw SerializationError("YOLO split contains duplicate image paths");
        auto decoded =
            image::read_image_file(file, options.dimension_decode, stop);
        read.width = decoded.width;
        read.height = decoded.height;
        const auto label = LabelPath(result.image_root, read.image_path);
        image::AnnotationIndex annotation;
        if (fs::exists(label)) {
          state.ChargeDocument(label);
          annotation = image::read_annotation_file(
              label, result.annotations.format, read, stop);
        } else if (options.missing_labels ==
                   MissingLabelPolicy::empty_negative) {
          annotation = image::parse_annotations("", result.annotations.format,
                                                read, stop);
        } else {
          throw IoError(
              "YOLO label file is missing; negative-sample policy is explicit");
        }
        state.Append(result.annotations, std::move(annotation));
      }
    }
  } else {
    throw InvalidArgument("Unknown detection dataset format");
  }
  if (result.annotations.images.empty())
    throw InvalidArgument("Detection split has no images");
  for (auto& record : result.annotations.images)
    for (auto& object : record.objects) {
      if (options.task == DatasetAnnotationTask::obb && !object.oriented_box) {
        if (object.polygons.size() != 1 || object.polygons.front().size() != 4)
          throw InvalidArgument(
              "OBB import requires exactly four ordered corners; no implicit "
              "minimum-area fitting");
        std::array<image::ImagePoint, 4> corners;
        std::copy_n(object.polygons.front().begin(), 4, corners.begin());
        object.oriented_box = corners;
      }
      if (!object.box && object.oriented_box) {
        image::ImageBox box{static_cast<double>(record.width),
                            static_cast<double>(record.height), 0, 0};
        for (const auto& p : *object.oriented_box) {
          box.x_min = (std::min)(box.x_min, p.x);
          box.y_min = (std::min)(box.y_min, p.y);
          box.x_max = (std::max)(box.x_max, p.x);
          box.y_max = (std::max)(box.y_max, p.y);
        }
        object.box = box;
      }
      if (!object.box && !object.polygons.empty()) {
        image::ImageBox box{static_cast<double>(record.width),
                            static_cast<double>(record.height), 0, 0};
        for (const auto& ring : object.polygons)
          for (const auto& p : ring) {
            box.x_min = (std::min)(box.x_min, p.x);
            box.y_min = (std::min)(box.y_min, p.y);
            box.x_max = (std::max)(box.x_max, p.x);
            box.y_max = (std::max)(box.y_max, p.y);
          }
        object.box = box;
      }
      if (options.task == DatasetAnnotationTask::instance_segment &&
          object.polygons.empty() && !object.rle)
        throw InvalidArgument(
            "Instance segmentation requires a polygon or RLE for every object");
      if (options.task == DatasetAnnotationTask::pose &&
          options.keypoint_count &&
          object.keypoints.size() != options.keypoint_count)
        throw InvalidArgument(
            "Pose import keypoint layout differs from the model");
    }
  FreezeCategories(result.annotations, options.categories);
  for (const auto& category : result.annotations.categories)
    state.identity += std::to_string(category.source_id) + ":" +
                      std::to_string(category.name.size()) + ":" +
                      category.name + "\n";
  state.identity += std::to_string(static_cast<int>(options.format)) + ":" +
                    std::to_string(static_cast<int>(options.split)) + ":" +
                    std::to_string(static_cast<int>(options.voc_coordinates)) +
                    ":" +
                    std::to_string(static_cast<int>(options.missing_labels)) +
                    ":" + std::to_string(static_cast<int>(options.task)) + ":" +
                    std::to_string(read.keypoint_count) + ":" +
                    std::to_string(read.keypoint_dimensions);
  result.metadata_identity = Digest(state.identity);
  if (options.format != DetectionDatasetFormat::coco)
    result.annotations.source_sha256 = result.metadata_identity;
  CheckStop(stop);
  return result;
} catch (const fs::filesystem_error& error) {
  throw IoError(std::string("Detection dataset filesystem operation failed: ") +
                error.what());
}

void require_disjoint_detection_splits(const DetectionDatasetImport& training,
                                       const DetectionDatasetImport& validation,
                                       std::stop_token stop) {
  const auto resolved = [](const fs::path& root, const std::string& relative) {
    std::error_code error;
    const auto path = fs::canonical(root / fs::u8path(relative), error);
    if (error)
      throw IoError("Cannot resolve an image while checking split separation");
#ifdef _WIN32
    return Lower(Utf8(path));
#else
    return Utf8(path);
#endif
  };
  std::set<std::string> paths;
  for (const auto& record : training.annotations.images) {
    CheckStop(stop);
    paths.insert(resolved(training.image_root, record.image_path));
  }
  for (const auto& record : validation.annotations.images) {
    CheckStop(stop);
    if (paths.contains(resolved(validation.image_root, record.image_path)))
      throw InvalidArgument(
          "Training and validation splits share an image path");
  }
}

}  // namespace jyppx::liuding::vision::tasks
