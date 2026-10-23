// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "annotation_detail.hpp"

#include <jyppx/liuding/foundation/sha256.hpp>

#include <charconv>
#include <new>
#include <sstream>
#include <yaml-cpp/eventhandler.h>
#include <yaml-cpp/exceptions.h>
#include <yaml-cpp/parser.h>

namespace jyppx::liuding::data::image {
namespace {

// Consume parser events directly instead of constructing YAML::Node graphs.
// In particular, no alias graph is ever expanded or recursively traversed.
class YamlData final : public YAML::EventHandler {
 public:
  YamlData(const AnnotationLimits& limits, std::stop_token stop)
      : limits_(limits), stop_(stop) {}
  void OnDocumentStart(const YAML::Mark&) override {
    if (++documents_ != 1U) detail::Malformed("yaml.single_document");
  }
  void OnDocumentEnd() override {
    if (!frames_.empty() || !root_) detail::Malformed("yaml.document");
  }
  void OnNull(const YAML::Mark&, YAML::anchor_t anchor) override {
    Event("?", anchor);
    Add(nullptr);
  }
  void OnAlias(const YAML::Mark&, YAML::anchor_t) override {
    detail::Malformed("yaml.alias");
  }
  void OnAnchor(const YAML::Mark&, const std::string&) override {
    detail::Malformed("yaml.anchor");
  }
  void OnScalar(const YAML::Mark&, const std::string& tag,
                YAML::anchor_t anchor, const std::string& value) override {
    Event(tag, anchor);
    if (value.size() > limits_.max_string_bytes)
      detail::Resource("yaml.scalar");
    Add(value);
  }
  void OnSequenceStart(const YAML::Mark&, const std::string& tag,
                       YAML::anchor_t anchor,
                       YAML::EmitterStyle::value) override {
    Start(tag, anchor, false);
  }
  void OnSequenceEnd() override { End(false); }
  void OnMapStart(const YAML::Mark&, const std::string& tag,
                  YAML::anchor_t anchor, YAML::EmitterStyle::value) override {
    Start(tag, anchor, true);
  }
  void OnMapEnd() override { End(true); }
  detail::Json Take() {
    if (!root_ || !root_->is_object()) detail::Malformed("yaml.root_map");
    return std::move(*root_);
  }

 private:
  struct Frame {
    detail::Json value;
    std::optional<std::string> key;
  };
  void Event(const std::string& tag, YAML::anchor_t anchor) {
    detail::CheckStop(stop_);
    detail::Add(nodes_, 1U, limits_.max_nodes, "yaml.nodes");
    if (anchor != 0U || (tag != "?" && tag != "!"))
      detail::Malformed("yaml.explicit_tag_or_anchor");
  }
  void Start(const std::string& tag, YAML::anchor_t anchor, bool map) {
    Event(tag, anchor);
    if (frames_.size() >= limits_.max_depth) detail::Resource("yaml.depth");
    if (!frames_.empty() && frames_.back().value.is_object() &&
        !frames_.back().key)
      detail::Malformed("yaml.scalar_keys");
    frames_.push_back(
        {map ? detail::Json::object() : detail::Json::array(), {}});
  }
  void End(bool map) {
    if (frames_.empty() || frames_.back().value.is_object() != map ||
        frames_.back().key)
      detail::Malformed("yaml.container");
    auto value = std::move(frames_.back().value);
    frames_.pop_back();
    Add(std::move(value));
  }
  void Add(detail::Json value) {
    if (frames_.empty()) {
      if (root_) detail::Malformed("yaml.multiple_roots");
      root_ = std::move(value);
    } else if (frames_.back().value.is_array()) {
      frames_.back().value.push_back(std::move(value));
    } else {
      auto& frame = frames_.back();
      if (frame.key) {
        frame.value.emplace(*frame.key, std::move(value));
        frame.key.reset();
      } else {
        if (!value.is_string()) detail::Malformed("yaml.scalar_keys");
        auto key = value.get<std::string>();
        detail::Text(key, limits_);
        if (frame.value.contains(key)) detail::Malformed("yaml.duplicate_key");
        frame.key = std::move(key);
      }
    }
  }
  const AnnotationLimits& limits_;
  std::stop_token stop_;
  std::size_t nodes_ = 0;
  std::size_t documents_ = 0;
  std::vector<Frame> frames_;
  std::optional<detail::Json> root_;
};

std::size_t Index(const detail::Json& value) {
  if (!value.is_string()) detail::Malformed("yaml.integer");
  const auto& text = value.get_ref<const std::string&>();
  std::size_t result = 0;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), result);
  if (text.empty() || parsed.ec != std::errc{} ||
      parsed.ptr != text.data() + text.size())
    detail::Malformed("yaml.integer");
  return result;
}

std::vector<std::string> Split(const detail::Json& document, const char* key,
                               const AnnotationLimits& limits) {
  const auto it = document.find(key);
  if (it == document.end() || it->is_null()) return {};
  std::vector<std::string> result;
  if (it->is_string()) {
    result.push_back(detail::String(*it, limits));
  } else {
    detail::Array(*it);
    if (it->size() > limits.max_images) detail::Resource("yaml.split_paths");
    for (const auto& path : *it) result.push_back(detail::String(path, limits));
  }
  return result;
}

}  // namespace

YoloDatasetSpec parse_yolo_dataset_yaml(std::string_view bytes,
                                        const AnnotationLimits& limits,
                                        std::stop_token stop) {
  try {
    detail::CheckStop(stop);
    if (limits.max_input_bytes == 0U ||
        limits.max_input_bytes > (1ULL << 30U) || limits.max_depth == 0U ||
        limits.max_depth > 128U || limits.max_nodes == 0U ||
        limits.max_categories == 0U || limits.max_string_bytes == 0U)
      throw InvalidArgument("Invalid YOLO YAML limits",
                            detail::Details("yaml.options"));
    if (bytes.size() > limits.max_input_bytes)
      detail::Resource("yaml.input_bytes");
    if (!detail::Utf8(bytes)) detail::Malformed("yaml.utf8");
    std::istringstream input{std::string(bytes)};
    YAML::Parser parser(input);
    YamlData handler(limits, stop);
    if (!parser.HandleNextDocument(handler))
      detail::Malformed("yaml.empty_document");
    parser.HandleNextDocument(
        handler);  // A second document is rejected by the handler.
    const auto document = handler.Take();
    YoloDatasetSpec result;
    if (document.contains("path"))
      result.root = detail::String(document.at("path"), limits);
    result.train = Split(document, "train", limits);
    result.validation = Split(document, "val", limits);
    result.test = Split(document, "test", limits);
    const auto& names = document.at("names");
    if (!names.is_array() && !names.is_object())
      detail::Malformed("yolo.names");
    if (names.empty() || names.size() > limits.max_categories)
      detail::Resource("yolo.names_count");
    if (names.is_array()) {
      for (const auto& name : names)
        result.categories.push_back(
            {static_cast<std::int64_t>(result.categories.size()),
             detail::String(name, limits)});
    } else {
      result.categories.resize(names.size());
      std::vector<bool> found(names.size(), false);
      for (auto it = names.begin(); it != names.end(); ++it) {
        const auto index = Index(it.key());
        if (index >= names.size() || found[index])
          detail::Malformed("yolo.names_contiguous");
        found[index] = true;
        result.categories[index] = {static_cast<std::int64_t>(index),
                                    detail::String(it.value(), limits)};
      }
    }
    detail::ValidateCategories(result.categories, limits);
    if (document.contains("nc") &&
        Index(document.at("nc")) != result.categories.size())
      detail::Malformed("yolo.nc_names_mismatch");
    if (document.contains("kpt_shape")) {
      const auto& shape = document.at("kpt_shape");
      detail::Array(shape);
      if (shape.size() != 2U) detail::Malformed("yolo.kpt_shape");
      result.keypoint_count = Index(shape[0]);
      const auto dimensions = Index(shape[1]);
      if (result.keypoint_count == 0U ||
          result.keypoint_count > limits.max_points ||
          (dimensions != 2U && dimensions != 3U))
        detail::Malformed("yolo.kpt_shape");
      result.keypoint_dimensions = static_cast<std::uint8_t>(dimensions);
    }
    if (document.contains("flip_idx")) {
      const auto& indices = document.at("flip_idx");
      detail::Array(indices);
      if (indices.size() != result.keypoint_count ||
          result.keypoint_count == 0U)
        detail::Malformed("yolo.flip_shape");
      std::vector<bool> found(result.keypoint_count, false);
      for (const auto& value : indices) {
        const auto i = Index(value);
        if (i >= result.keypoint_count || found[i])
          detail::Malformed("yolo.flip_permutation");
        found[i] = true;
        result.flip_indices.push_back(i);
      }
      for (std::size_t i = 0; i < result.flip_indices.size(); ++i)
        if (result.flip_indices[result.flip_indices[i]] != i)
          detail::Malformed("yolo.flip_involution");
    }
    result.has_download_instruction = document.contains("download");
    result.source_sha256 = sha256_hex(std::as_bytes(std::span(bytes)));
    detail::CheckStop(stop);
    return result;
  } catch (const YAML::Exception&) {
    detail::Malformed("yaml.parse");
  } catch (const nlohmann::json::exception&) {
    detail::Malformed("yaml.schema");
  } catch (const std::bad_alloc&) {
    detail::Resource("yaml.allocation");
  } catch (const std::length_error&) {
    detail::Resource("yaml.container");
  }
}

}  // namespace jyppx::liuding::data::image
