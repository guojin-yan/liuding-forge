// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_DATA_IMAGE_MANIFESTS_HPP
#define JYPPX_DATA_IMAGE_MANIFESTS_HPP

/**
 * @file manifests.hpp
 * @brief Reads bounded classification, paired-file and ImageFolder indexes.
 *
 * These readers produce owned paths and class IDs for a decoder to consume.
 * They do not decode files or establish image-content integrity. Manifest paths
 * use UTF-8, forward slashes, and dataset-relative components without
 * traversal.
 */

#include <jyppx/data_image/annotations.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::data::image {

/** @brief Explicit manifest encoding; dialect inference is not performed. */
enum class ManifestFormat : std::uint8_t {
  csv,    ///< Comma-separated records with RFC-style doubled-quote escaping.
  tsv,    ///< Tab-separated records with the same quoting rules as CSV.
  jsonl,  ///< One strict JSON object per physical line.
};

/** @brief Selects a classification or a paired-file manifest schema. */
enum class ManifestTask : std::uint8_t {
  classification,  ///< A path plus one or more contiguous class indices.
  paired,          ///< An input image path plus an external target file path.
};

/** @brief Identifies precisely what an index's source digest covers. */
enum class ManifestDigestScope : std::uint8_t {
  document_bytes,  ///< Exact input bytes, excluding external class maps/files.
  folder_paths_and_classes,  ///< Versioned sorted class/path/label inventory.
};

/** @brief One owned source record, suitable for later image decoding. */
struct ManifestRecord {
  std::string path = {};  ///< Dataset-relative input image path in UTF-8.
  std::optional<std::string> target_path =
      {};  ///< Paired target; never decoded.
  std::vector<std::size_t> labels =
      {};  ///< Source-ordered contiguous class IDs.
};

/** @brief Bounds logical manifest and directory-index storage/work. */
struct ManifestReadOptions {
  AnnotationLimits limits =
      {};  ///< Input, records, classes, strings and nodes.
  std::vector<VisionCategory> categories =
      {};                    ///< Required classification map.
  bool multi_label = false;  ///< False requires exactly one label per record.
  std::size_t max_labels_per_record = 64;    ///< Additional multi-label bound.
  std::size_t max_record_bytes = 1U << 20U;  ///< Encoded CSV record/JSONL line.
  std::size_t max_index_bytes = 256U << 20U;  ///< Logical owned payload budget.
};

/** @brief Selects files and bounds traversal for an ImageFolder index. */
struct ImageFolderOptions {
  AnnotationLimits limits = {};  ///< Record, class, path and string limits.
  std::vector<std::string> extensions = {".jpeg", ".jpg", ".png",
                                         ".webp", ".tif", ".tiff",
                                         ".bmp",  ".ppm", ".pgm"};
  ///< ASCII-insensitive suffix filter; this does not imply codec support.
  bool recursive = true;  ///< Includes nested files beneath each class folder.
  bool include_hidden =
      false;  ///< Includes dot names and Windows hidden names.
  bool allow_empty_classes = false;  ///< Otherwise every class needs an image.
  std::size_t max_directory_entries =
      2'000'000;  ///< Includes filtered entries.
  std::size_t max_directory_depth =
      32;  ///< Root is zero; class folders are one.
  std::size_t max_index_bytes = 256U << 20U;  ///< Logical owned payload budget.
};

/**
 * @brief Owned index exposing only const views of its published values.
 *
 * Concurrent const reads are supported. Assignment and destruction still need
 * external synchronization. The logical byte budget counts value payload and
 * element storage, not allocator bookkeeping or temporary parser containers.
 */
class ManifestIndex final {
 public:
  /** @brief Copies an independently owned index value. */
  ManifestIndex(const ManifestIndex&) = default;
  /** @brief Transfers ownership of an index value. */
  ManifestIndex(ManifestIndex&&) noexcept = default;
  /** @brief Replaces this index with an independently owned copy.
   * @return This index after replacement.
   */
  ManifestIndex& operator=(const ManifestIndex&) = default;
  /** @brief Replaces this index by transferring ownership.
   * @return This index after replacement.
   */
  ManifestIndex& operator=(ManifestIndex&&) noexcept = default;
  /** @brief Releases owned records and metadata. */
  ~ManifestIndex() = default;

  /** @brief Returns the schema represented by the records.
   * @return Classification or paired-file semantics.
   */
  ManifestTask task() const noexcept { return task_; }
  /** @brief Returns the frozen contiguous class map.
   * @return Class values; empty for a paired-file index.
   */
  const std::vector<VisionCategory>& categories() const noexcept {
    return categories_;
  }
  /** @brief Returns records in source or deterministic ImageFolder order.
   * @return Owned paths and labels, valid until this value is replaced.
   */
  const std::vector<ManifestRecord>& records() const noexcept {
    return records_;
  }
  /** @brief Returns the declared scope of the source digest.
   * @return Exact bytes or a versioned folder inventory, never image contents.
   */
  ManifestDigestScope digest_scope() const noexcept { return digest_scope_; }
  /** @brief Returns the lowercase SHA-256 source digest.
   * @return A 64-character digest with the scope reported by digest_scope().
   */
  const std::string& source_sha256() const noexcept { return source_sha256_; }
  /** @brief Returns the lexical absolute root used for a folder inventory.
   * @return Empty for a parsed/file manifest, or the scanned folder root.
   */
  const std::filesystem::path& root() const noexcept { return root_; }

 private:
  /** @brief Allows reader factories to construct a fully validated index. */
  ManifestIndex() = default;
  ManifestTask task_ = ManifestTask::classification;  ///< Record schema.
  std::vector<VisionCategory> categories_ = {};       ///< Frozen class mapping.
  std::vector<ManifestRecord> records_ = {};  ///< Ordered immutable records.
  ManifestDigestScope digest_scope_ = ManifestDigestScope::document_bytes;
  ///< Exact scope covered by source_sha256_.
  std::string source_sha256_ = {};   ///< Source bytes or inventory digest.
  std::filesystem::path root_ = {};  ///< Selected ImageFolder root if present.

  friend ManifestIndex parse_manifest(std::string_view bytes,
                                      ManifestFormat format, ManifestTask task,
                                      const ManifestReadOptions& options,
                                      std::stop_token stop);
  friend ManifestIndex index_image_folder(const std::filesystem::path& root,
                                          const ImageFolderOptions& options,
                                          std::stop_token stop);
};

/**
 * @brief Parses an in-memory classification or paired-file manifest.
 * @param bytes UTF-8 CSV, TSV or JSONL; an initial UTF-8 BOM is accepted.
 * @param format Explicit encoding, with no automatic dialect inference.
 * @param task Selects the strict two-field schema.
 * @param options Frozen class map and resource limits.
 * @param stop Cooperative cancellation checked between parsing events.
 * @return An owned index with the digest of the exact original bytes.
 * @throws InvalidArgument Invalid options, task, format or missing class map.
 * @throws SerializationError Malformed data, paths, duplicates or schema.
 * @throws ResourceExhausted A configured bound or allocation limit was reached.
 * @throws Cancelled Cancellation was requested before publication.
 *
 * CSV/TSV classification headers are exactly `path,labels`/`path\tlabels`;
 * labels are class names separated by `|`, without whitespace trimming. Single
 * label mode requires one name. CSV quoting supports commas and doubled quotes
 * in names; class names containing `|` require JSONL instead. JSONL uses
 * exactly
 * {"path":"...","labels":["class",...]}. Paired headers are `input,target`
 * or `input\ttarget`, and JSONL uses exactly {"input":"...","target":"..."}.
 * Target files may be masks, depth, clean images or other externally decoded
 * files. Empty records, unknown columns/keys, duplicate labels and duplicate or
 * ASCII-case-colliding input paths are rejected. Paths are not opened here.
 */
ManifestIndex parse_manifest(std::string_view bytes, ManifestFormat format,
                             ManifestTask task,
                             const ManifestReadOptions& options = {},
                             std::stop_token stop = {});

/**
 * @brief Reads a bounded local file and parses its explicit manifest schema.
 * @param path Manifest document to open, not an image root.
 * @param format Explicit CSV, TSV or JSONL encoding.
 * @param task Classification or paired-file schema.
 * @param options Frozen class map and byte/record/storage limits.
 * @param stop Cooperative file-read and parse cancellation.
 * @return Owned index, with document bytes hashed exactly as read.
 * @throws IoError The local manifest could not be safely read.
 * @throws InvalidArgument Invalid parsing options.
 * @throws SerializationError Malformed manifest or invalid relative paths.
 * @throws ResourceExhausted A configured bound or allocation failed.
 * @throws Cancelled A cooperative cancellation request was observed.
 * @note The digest excludes referenced image/target contents and class options.
 */
ManifestIndex read_manifest_file(const std::filesystem::path& path,
                                 ManifestFormat format, ManifestTask task,
                                 const ManifestReadOptions& options = {},
                                 std::stop_token stop = {});

/**
 * @brief Builds a real classification index from `root/class_name/...` files.
 * @param root Existing image dataset directory; linked ancestors are rejected.
 * @param options Extension selection, visibility and finite traversal limits.
 * @param stop Cooperative cancellation between filesystem entries.
 * @return An owned class/path index ordered by UTF-8 bytes with an inventory
 * hash.
 * @throws InvalidArgument Invalid options or an empty root path.
 * @throws IoError An entry cannot be inspected, or a link/reparse point is
 * found.
 * @throws SerializationError Invalid UTF-8/path, collision, empty class or
 * layout.
 * @throws ResourceExhausted A configured traversal/storage bound was reached.
 * @throws Cancelled Cancellation was observed before publication.
 *
 * Immediate subdirectories define class names. Nested paths inherit their
 * class. `.git` and `.svn` are always excluded; other hidden names are excluded
 * by default. Root-level matching image files are rejected as unclassified.
 * Empty class directories are retained only when explicitly allowed; at least
 * one selected file is required. All inspected symbolic links and Windows
 * reparse points are rejected. ASCII case collisions are rejected for portable
 * indexing; Unicode normalization/case folding is not performed. The snapshot
 * covers names and labels only, so later file changes are not detected by its
 * digest. Directory mutation during traversal is unsupported; the decoder must
 * revalidate paths when opening them to prevent replacement races.
 */
ManifestIndex index_image_folder(const std::filesystem::path& root,
                                 const ImageFolderOptions& options = {},
                                 std::stop_token stop = {});

}  // namespace jyppx::data::image

#endif  // JYPPX_DATA_IMAGE_MANIFESTS_HPP
