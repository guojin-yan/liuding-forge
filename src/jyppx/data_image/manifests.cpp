// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "annotation_detail.hpp"

#include <jyppx/data_image/manifests.hpp>
#include <jyppx/foundation/sha256.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <new>
#include <span>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace jyppx::data::image {
namespace {

namespace fs = std::filesystem;
using Json = nlohmann::json;

ErrorDetails Details(std::string_view operation) {
  return {.component = "data.image.manifests",
          .operation = std::string(operation),
          .expected = "a bounded manifest schema or portable image directory",
          .actual = "the data-source contract was not satisfied",
          .hint = "Check the schema, class map, paths, permissions and limits"};
}

[[noreturn]] void Malformed(std::string_view operation) {
  throw SerializationError("Invalid image manifest", Details(operation));
}

[[noreturn]] void Resource(std::string_view operation) {
  throw ResourceExhausted("Image manifest limit exceeded", Details(operation));
}

[[noreturn]] void Io(std::string_view operation) {
  throw IoError("Image manifest filesystem operation failed",
                Details(operation));
}

void CheckStop(std::stop_token stop) {
  if (stop.stop_requested())
    throw Cancelled("Image manifest operation cancelled",
                    Details("source.cancel"));
}

void Add(std::size_t& count, std::size_t increment, std::size_t maximum,
         std::string_view operation) {
  if (count > maximum || increment > maximum - count) Resource(operation);
  count += increment;
}

void Text(std::string_view text, const AnnotationLimits& limits) {
  if (text.size() > limits.max_string_bytes) Resource("string.bytes");
  if (text.empty() || !detail::Utf8(text) ||
      std::any_of(text.begin(), text.end(),
                  [](unsigned char c) { return c < 0x20U || c == 0x7fU; }))
    Malformed("string.utf8");
}

std::string FoldAscii(std::string_view text) {
  std::string result(text);
  for (char& c : result)
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  return result;
}

void Path(std::string_view path, const AnnotationLimits& limits) {
  Text(path, limits);
  if (path.front() == '/' ||
      path.find_first_of("\\:<>\"|?*") != std::string_view::npos)
    Malformed("path.relative");
  std::size_t begin = 0;
  while (begin <= path.size()) {
    const auto end = path.find('/', begin);
    const auto part =
        path.substr(begin, end == std::string_view::npos ? path.size() - begin
                                                         : end - begin);
    if (part.empty() || part == "." || part == ".." || part.back() == '.' ||
        part.back() == ' ')
      Malformed("path.component");
    const auto base = FoldAscii(part.substr(0, part.find('.')));
    if (base == "con" || base == "prn" || base == "aux" || base == "nul" ||
        (base.size() == 4U &&
         (base.starts_with("com") || base.starts_with("lpt")) &&
         base[3] >= '1' && base[3] <= '9'))
      Malformed("path.device");
    if (end == std::string_view::npos) break;
    begin = end + 1U;
  }
}

void ValidateLimits(const AnnotationLimits& limits, std::size_t index_bytes) {
  if (limits.max_input_bytes == 0U || limits.max_input_bytes > (1ULL << 30U) ||
      limits.max_images == 0U || limits.max_categories == 0U ||
      limits.max_string_bytes == 0U || limits.max_nodes == 0U ||
      limits.max_depth == 0U || limits.max_depth > 128U || index_bytes == 0U)
    throw InvalidArgument("Invalid image manifest limits",
                          Details("source.options"));
}

void ValidateOptions(ManifestFormat format, ManifestTask task,
                     const ManifestReadOptions& options) {
  ValidateLimits(options.limits, options.max_index_bytes);
  if ((format != ManifestFormat::csv && format != ManifestFormat::tsv &&
       format != ManifestFormat::jsonl) ||
      (task != ManifestTask::classification && task != ManifestTask::paired) ||
      options.max_record_bytes == 0U || options.max_labels_per_record == 0U ||
      (task == ManifestTask::classification && options.categories.empty()) ||
      (task == ManifestTask::paired &&
       (!options.categories.empty() || options.multi_label)))
    throw InvalidArgument("Invalid image manifest options",
                          Details("source.options"));
  detail::ValidateCategories(options.categories, options.limits);
  if (task == ManifestTask::classification && format != ManifestFormat::jsonl)
    for (const auto& category : options.categories)
      if (category.name.find('|') != std::string::npos)
        throw InvalidArgument(
            "Delimited manifest class name contains its label separator",
            Details("labels.separator"));
}

void ChargeCategories(const std::vector<VisionCategory>& categories,
                      std::size_t& bytes, std::size_t maximum) {
  for (const auto& category : categories) {
    Add(bytes, sizeof(VisionCategory), maximum, "index.bytes");
    Add(bytes, category.name.size(), maximum, "index.bytes");
    for (const auto& point : category.keypoint_names) {
      Add(bytes, sizeof(std::string), maximum, "index.bytes");
      Add(bytes, point.size(), maximum, "index.bytes");
    }
    for (const auto& edge : category.skeleton) {
      static_cast<void>(edge);
      Add(bytes, sizeof(std::array<std::size_t, 2>), maximum, "index.bytes");
    }
  }
}

struct ManifestState {
  const ManifestReadOptions& options;
  ManifestTask task;
  std::stop_token stop;
  std::vector<ManifestRecord> records = {};
  std::unordered_map<std::string, std::size_t> classes = {};
  std::unordered_set<std::string> paths = {};
  std::size_t bytes = 0;
  std::size_t nodes = 0;
  std::size_t labels = 0;

  void Initialize() {
    ChargeCategories(options.categories, bytes, options.max_index_bytes);
    for (std::size_t i = 0; i < options.categories.size(); ++i) {
      // The lookup table owns another label copy until parsing finishes.
      Add(bytes, options.categories[i].name.size() + sizeof(std::size_t),
          options.max_index_bytes, "index.bytes");
      classes.emplace(options.categories[i].name, i);
    }
  }

  void Label(ManifestRecord& record, std::string_view name) {
    Text(name, options.limits);
    if (record.labels.size() >= options.max_labels_per_record)
      Resource("labels.per_record");
    const auto found = classes.find(std::string(name));
    if (found == classes.end()) Malformed("labels.unknown");
    if (std::find(record.labels.begin(), record.labels.end(), found->second) !=
        record.labels.end())
      Malformed("labels.duplicate");
    Add(labels, 1U, options.limits.max_objects, "labels.total");
    Add(bytes, sizeof(std::size_t), options.max_index_bytes, "index.bytes");
    record.labels.push_back(found->second);
  }

  void Append(ManifestRecord record) {
    CheckStop(stop);
    Path(record.path, options.limits);
    if (records.size() >= options.limits.max_images) Resource("records.count");
    if (task == ManifestTask::classification) {
      if (record.target_path || record.labels.empty() ||
          (!options.multi_label && record.labels.size() != 1U))
        Malformed("labels.cardinality");
    } else {
      if (!record.target_path || !record.labels.empty())
        Malformed("paired.schema");
      Path(*record.target_path, options.limits);
      Add(bytes, record.target_path->size(), options.max_index_bytes,
          "index.bytes");
    }
    Add(bytes, sizeof(ManifestRecord), options.max_index_bytes, "index.bytes");
    Add(bytes, record.path.size(), options.max_index_bytes, "index.bytes");
    // Charge the temporary case-insensitive duplicate-check set's path copy.
    Add(bytes, record.path.size(), options.max_index_bytes, "index.bytes");
    if (!paths.insert(FoldAscii(record.path)).second)
      Malformed("path.duplicate");
    records.push_back(std::move(record));
  }
};

void DelimitedRow(const std::array<std::string, 2>& fields, bool header,
                  ManifestState& state) {
  Add(state.nodes, 2U, state.options.limits.max_nodes, "records.fields");
  if (header) {
    const bool valid = state.task == ManifestTask::classification
                           ? fields[0] == "path" && fields[1] == "labels"
                           : fields[0] == "input" && fields[1] == "target";
    if (!valid) Malformed("header.schema");
    return;
  }
  ManifestRecord record;
  record.path = fields[0];
  if (state.task == ManifestTask::paired) {
    record.target_path = fields[1];
  } else {
    std::size_t start = 0;
    while (start <= fields[1].size()) {
      const auto end = fields[1].find('|', start);
      state.Label(record,
                  std::string_view(fields[1]).substr(
                      start, end == std::string::npos ? fields[1].size() - start
                                                      : end - start));
      if (end == std::string::npos) break;
      start = end + 1U;
    }
  }
  state.Append(std::move(record));
}

void ReadDelimited(std::string_view bytes, char delimiter,
                   ManifestState& state) {
  std::size_t position = 0;
  bool header = true;
  while (position < bytes.size()) {
    CheckStop(state.stop);
    const auto start = position;
    std::array<std::string, 2> fields;
    std::size_t column = 0;
    auto CheckRecord = [&] {
      if (position - start > state.options.max_record_bytes)
        Resource("record.bytes");
      if (((position - start) & 0xfffU) == 0U) CheckStop(state.stop);
    };
    for (;;) {
      std::string& field = fields[column];
      if (position < bytes.size() && bytes[position] == '"') {
        ++position;
        bool closed = false;
        while (position < bytes.size()) {
          const char c = bytes[position++];
          CheckRecord();
          if (c == '"') {
            if (position < bytes.size() && bytes[position] == '"') {
              ++position;
              CheckRecord();
              field.push_back('"');
            } else {
              closed = true;
              break;
            }
          } else {
            field.push_back(c);
          }
        }
        if (!closed) Malformed("csv.unclosed_quote");
      } else {
        while (position < bytes.size() && bytes[position] != delimiter &&
               bytes[position] != '\r' && bytes[position] != '\n') {
          const char c = bytes[position++];
          CheckRecord();
          if (c == '"') Malformed("csv.quote_position");
          field.push_back(c);
        }
      }
      if (position == bytes.size()) {
        if (column != 1U) Malformed("csv.columns");
        break;
      }
      const char next = bytes[position++];
      CheckRecord();
      if (next == delimiter) {
        if (column != 0U) Malformed("csv.columns");
        ++column;
        continue;
      }
      if (next != '\r' && next != '\n') Malformed("csv.after_quote");
      if (next == '\r') {
        if (position == bytes.size() || bytes[position] != '\n')
          Malformed("csv.line_ending");
        ++position;
        CheckRecord();
      }
      if (column != 1U) Malformed("csv.columns");
      break;
    }
    DelimitedRow(fields, header, state);
    header = false;
  }
  if (header || state.records.empty()) Malformed("records.empty");
}

Json JsonLine(std::string_view line, ManifestState& state) {
  std::vector<std::unordered_set<std::string>> keys;
  auto callback = [&](int depth, Json::parse_event_t event, Json& value) {
    CheckStop(state.stop);
    if (depth < 0 ||
        static_cast<std::uint32_t>(depth) > state.options.limits.max_depth)
      Resource("json.depth");
    Add(state.nodes, 1U, state.options.limits.max_nodes, "json.nodes");
    if (event == Json::parse_event_t::object_start) keys.emplace_back();
    if (event == Json::parse_event_t::key) {
      const auto& key = value.get_ref<const std::string&>();
      Text(key, state.options.limits);
      if (keys.empty() || !keys.back().insert(key).second)
        Malformed("json.duplicate_key");
    }
    if (event == Json::parse_event_t::object_end) {
      if (keys.empty()) Malformed("json.object");
      keys.pop_back();
    }
    if (event == Json::parse_event_t::value && value.is_string() &&
        value.get_ref<const std::string&>().size() >
            state.options.limits.max_string_bytes)
      Resource("json.string_bytes");
    return true;
  };
  auto result = Json::parse(line.begin(), line.end(), callback, true, false);
  if (!result.is_object() || result.size() != 2U) Malformed("json.row_schema");
  return result;
}

std::string String(const Json& value, const AnnotationLimits& limits) {
  if (!value.is_string()) Malformed("json.string");
  const auto& result = value.get_ref<const std::string&>();
  Text(result, limits);
  return result;
}

void ReadJsonLines(std::string_view bytes, ManifestState& state) {
  std::size_t position = 0;
  while (position < bytes.size()) {
    CheckStop(state.stop);
    const auto end = bytes.find('\n', position);
    const auto length = end == std::string_view::npos ? bytes.size() - position
                                                      : end - position;
    if (length > state.options.max_record_bytes) Resource("record.bytes");
    auto line = bytes.substr(position, length);
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1U);
    if (line.empty()) Malformed("jsonl.blank_line");
    const auto row = JsonLine(line, state);
    ManifestRecord record;
    if (state.task == ManifestTask::classification) {
      if (!row.contains("path") || !row.contains("labels") ||
          !row.at("labels").is_array())
        Malformed("jsonl.classification_schema");
      record.path = String(row.at("path"), state.options.limits);
      for (const auto& label : row.at("labels"))
        state.Label(record, String(label, state.options.limits));
    } else {
      if (!row.contains("input") || !row.contains("target"))
        Malformed("jsonl.paired_schema");
      record.path = String(row.at("input"), state.options.limits);
      record.target_path = String(row.at("target"), state.options.limits);
    }
    state.Append(std::move(record));
    if (end == std::string_view::npos) break;
    position = end + 1U;
  }
  if (state.records.empty()) Malformed("records.empty");
}

std::string Utf8Path(const fs::path& path) {
  const auto value = path.generic_u8string();
  return {reinterpret_cast<const char*>(value.data()), value.size()};
}

struct EntryInfo {
  fs::file_status status;
  bool hidden = false;
};

EntryInfo Inspect(const fs::path& path) {
  std::error_code error;
  EntryInfo info{fs::symlink_status(path, error), false};
  if (error || !fs::exists(info.status)) Io("folder.inspect");
  if (fs::is_symlink(info.status)) Io("folder.link");
#ifdef _WIN32
  const DWORD attributes = ::GetFileAttributesW(path.c_str());
  if (attributes == INVALID_FILE_ATTRIBUTES) Io("folder.attributes");
  if ((attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0U) Io("folder.reparse");
  info.hidden = (attributes & FILE_ATTRIBUTE_HIDDEN) != 0U;
#endif
  return info;
}

fs::path RootPath(const fs::path& input) {
  if (input.empty() || input.native().find(fs::path::value_type{}) !=
                           fs::path::string_type::npos)
    throw InvalidArgument("ImageFolder root is empty or contains a null",
                          Details("folder.root"));
  for (const auto& component : input)
    if (component == "..")
      throw InvalidArgument("ImageFolder root contains traversal",
                            Details("folder.root"));
  std::error_code error;
  const auto root = fs::absolute(input, error).lexically_normal();
  if (error || root.empty()) Io("folder.absolute");
  fs::path ancestor = root.root_path();
  if (ancestor.empty() || !fs::is_directory(Inspect(ancestor).status))
    Io("folder.root");
  for (const auto& component : root.relative_path()) {
    ancestor /= component;
    if (!fs::is_directory(Inspect(ancestor).status)) Io("folder.ancestor");
  }
  return root;
}

bool Excluded(std::string_view name, const EntryInfo& info,
              const ImageFolderOptions& options) {
  const auto folded = FoldAscii(name);
  if (folded == ".git" || folded == ".svn") return true;
  return !options.include_hidden &&
         ((!name.empty() && name.front() == '.') || info.hidden);
}

void HashText(Sha256& hash, std::string_view text) {
  // Decimal length framing makes arbitrary UTF-8 labels/paths unambiguous.
  const auto size = std::to_string(text.size());
  hash.update(std::as_bytes(std::span(size)));
  constexpr std::string_view separator = ":";
  hash.update(std::as_bytes(std::span(separator)));
  hash.update(std::as_bytes(std::span(text)));
}

struct FolderState {
  const ImageFolderOptions& options;
  const std::unordered_set<std::string>& extensions;
  const fs::path& root;
  std::stop_token stop;
  std::size_t entries = 0;
  std::size_t bytes = 0;
  std::vector<ManifestRecord> records = {};
  std::unordered_set<std::string> names = {};

  std::pair<std::string, EntryInfo> Entry(const fs::directory_entry& entry) {
    CheckStop(stop);
    Add(entries, 1U, options.max_directory_entries, "folder.entries");
    const auto name = Utf8Path(entry.path().filename());
    const auto info = Inspect(entry.path());
    if (Excluded(name, info, options)) return {{}, info};
    const auto relative = Utf8Path(entry.path().lexically_relative(root));
    Path(relative, options.limits);
    // Includes directory and ignored-suffix names, so component collisions are
    // rejected even when the colliding directories contain different files.
    Add(bytes, relative.size() + sizeof(std::string), options.max_index_bytes,
        "index.bytes");
    if (!names.insert(FoldAscii(relative)).second)
      Malformed("folder.case_collision");
    return {name, info};
  }

  bool Matches(const fs::path& path) const {
    return extensions.contains(FoldAscii(Utf8Path(path.extension())));
  }

  void Walk(const fs::path& directory, std::size_t category,
            std::size_t depth) {
    CheckStop(stop);
    if (depth > options.max_directory_depth) Resource("folder.depth");
    std::error_code error;
    fs::directory_iterator iterator(directory, error);
    if (error) Io("folder.enumerate");
    const fs::directory_iterator end;
    while (iterator != end) {
      const auto& entry = *iterator;
      const auto [name, info] = Entry(entry);
      if (!name.empty()) {
        if (fs::is_directory(info.status)) {
          if (options.recursive) Walk(entry.path(), category, depth + 1U);
        } else if (fs::is_regular_file(info.status) && Matches(entry.path())) {
          if (records.size() >= options.limits.max_images)
            Resource("records.count");
          const auto relative = Utf8Path(entry.path().lexically_relative(root));
          Add(bytes, sizeof(ManifestRecord) + sizeof(std::size_t),
              options.max_index_bytes, "index.bytes");
          Add(bytes, relative.size(), options.max_index_bytes, "index.bytes");
          records.push_back({relative, std::nullopt, {category}});
        }
      }
      iterator.increment(error);
      if (error) Io("folder.enumerate");
    }
  }
};

}  // namespace

ManifestIndex parse_manifest(std::string_view bytes, ManifestFormat format,
                             ManifestTask task,
                             const ManifestReadOptions& options,
                             std::stop_token stop) {
  try {
    CheckStop(stop);
    ValidateOptions(format, task, options);
    if (bytes.size() > options.limits.max_input_bytes) Resource("input.bytes");
    const auto original = bytes;
    if (bytes.starts_with("\xef\xbb\xbf")) bytes.remove_prefix(3U);
    if (!detail::Utf8(bytes)) Malformed("input.utf8");
    ManifestState state{options, task, stop};
    state.Initialize();
    if (format == ManifestFormat::jsonl)
      ReadJsonLines(bytes, state);
    else
      ReadDelimited(bytes, format == ManifestFormat::csv ? ',' : '\t', state);
    CheckStop(stop);
    ManifestIndex result;
    result.task_ = task;
    result.categories_ = options.categories;
    result.records_ = std::move(state.records);
    result.source_sha256_ = sha256_hex(std::as_bytes(std::span(original)));
    CheckStop(stop);
    return result;
  } catch (const nlohmann::json::exception&) {
    Malformed("json.parse");
  } catch (const std::bad_alloc&) {
    Resource("source.allocation");
  } catch (const std::length_error&) {
    Resource("source.container");
  }
}

ManifestIndex read_manifest_file(const fs::path& path, ManifestFormat format,
                                 ManifestTask task,
                                 const ManifestReadOptions& options,
                                 std::stop_token stop) {
  try {
    CheckStop(stop);
    ValidateOptions(format, task, options);
    return parse_manifest(
        detail::ReadAnnotationBytes(path, options.limits.max_input_bytes, stop),
        format, task, options, stop);
  } catch (const std::bad_alloc&) {
    Resource("source.allocation");
  } catch (const std::length_error&) {
    Resource("source.container");
  }
}

ManifestIndex index_image_folder(const fs::path& input,
                                 const ImageFolderOptions& options,
                                 std::stop_token stop) {
  try {
    CheckStop(stop);
    ValidateLimits(options.limits, options.max_index_bytes);
    if (options.extensions.empty() || options.max_directory_entries == 0U ||
        options.max_directory_depth == 0U || options.max_directory_depth > 128U)
      throw InvalidArgument("Invalid ImageFolder options",
                            Details("folder.options"));
    std::unordered_set<std::string> extensions;
    for (const auto& extension : options.extensions) {
      if (extension.size() < 2U || extension.size() > 16U ||
          extension.front() != '.' ||
          !std::all_of(extension.begin() + 1, extension.end(),
                       [](unsigned char c) {
                         return (c >= 'a' && c <= 'z') ||
                                (c >= 'A' && c <= 'Z') ||
                                (c >= '0' && c <= '9');
                       }) ||
          !extensions.insert(FoldAscii(extension)).second)
        throw InvalidArgument("Invalid ImageFolder extension selection",
                              Details("folder.extensions"));
    }
    const auto root = RootPath(input);
    struct ClassDirectory {
      std::string name;
      fs::path path;
    };
    std::vector<ClassDirectory> classes;
    FolderState state{options, extensions, root, stop};
    std::error_code error;
    fs::directory_iterator iterator(root, error);
    if (error) Io("folder.enumerate");
    const fs::directory_iterator end;
    while (iterator != end) {
      const auto& entry = *iterator;
      auto [name, info] = state.Entry(entry);
      if (!name.empty()) {
        if (fs::is_directory(info.status)) {
          if (classes.size() >= options.limits.max_categories)
            Resource("categories.count");
          Add(state.bytes, sizeof(ClassDirectory) + name.size(),
              options.max_index_bytes, "index.bytes");
          classes.push_back({std::move(name), entry.path()});
        } else if (fs::is_regular_file(info.status) &&
                   state.Matches(entry.path())) {
          Malformed("folder.unclassified_image");
        }
      }
      iterator.increment(error);
      if (error) Io("folder.enumerate");
    }
    std::sort(classes.begin(), classes.end(),
              [](const auto& a, const auto& b) { return a.name < b.name; });
    ManifestIndex result;
    result.root_ = root;
    result.digest_scope_ = ManifestDigestScope::folder_paths_and_classes;
    for (std::size_t i = 0; i < classes.size(); ++i) {
      if (i >
          static_cast<std::size_t>((std::numeric_limits<std::int64_t>::max)()))
        Resource("categories.id");
      VisionCategory category;
      category.source_id = static_cast<std::int64_t>(i);
      category.name = classes[i].name;
      Add(state.bytes, sizeof(VisionCategory) + category.name.size(),
          options.max_index_bytes, "index.bytes");
      result.categories_.push_back(std::move(category));
      const auto before = state.records.size();
      state.Walk(classes[i].path, i, 1U);
      if (!options.allow_empty_classes && state.records.size() == before)
        Malformed("folder.empty_class");
    }
    if (state.records.empty()) Malformed("records.empty");
    std::sort(state.records.begin(), state.records.end(),
              [](const auto& a, const auto& b) { return a.path < b.path; });
    result.records_ = std::move(state.records);
    Sha256 hash;
    HashText(hash, "jyppx.imagefolder.inventory.v1");
    HashText(hash, std::to_string(result.categories_.size()));
    for (const auto& category : result.categories_)
      HashText(hash, category.name);
    HashText(hash, std::to_string(result.records_.size()));
    for (const auto& record : result.records_) {
      CheckStop(stop);
      HashText(hash, record.path);
      HashText(hash, std::to_string(record.labels.front()));
    }
    CheckStop(stop);
    result.source_sha256_ = sha256_digest_hex(hash.finish());
    return result;
  } catch (const fs::filesystem_error&) {
    Io("folder.path_conversion");
  } catch (const std::bad_alloc&) {
    Resource("source.allocation");
  } catch (const std::length_error&) {
    Resource("source.container");
  }
}

}  // namespace jyppx::data::image
