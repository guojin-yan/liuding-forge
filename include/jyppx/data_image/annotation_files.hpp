// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_DATA_IMAGE_ANNOTATION_FILES_HPP
#define JYPPX_DATA_IMAGE_ANNOTATION_FILES_HPP

/** @file annotation_files.hpp
 * @brief Reads bounded local annotation files with the built-in format parsers.
 */

#include <jyppx/data_image/annotations.hpp>

#include <filesystem>

namespace jyppx::data::image {

/**
 * @brief Reads and parses one explicitly selected local annotation file.
 * @param path Native local file path selected by the caller, not a dataset
 * root.
 * @param format Explicit COCO, YOLO, VOC or LabelMe dialect.
 * @param options Class, dimension, coordinate and resource context.
 * @param stop Cooperative cancellation between bounded read chunks and objects.
 * @return Fully owned index whose digest covers the exact file bytes.
 * @throws IoError Missing/non-regular files, read failures or detected changes.
 * @throws InvalidArgument Invalid reader options or an empty path.
 * @throws SerializationError Malformed or inconsistent annotations.
 * @throws ResourceExhausted Input, parse or allocation limits.
 * @throws Cancelled Cancellation was requested.
 * @note Symlinks in this explicitly supplied file path use normal OS
 * resolution. No referenced image or mask is opened. This is not a filesystem
 * sandbox.
 */
AnnotationIndex read_annotation_file(const std::filesystem::path& path,
                                     AnnotationFormat format,
                                     const AnnotationReadOptions& options = {},
                                     std::stop_token stop = {});

/**
 * @brief Reads a bounded, data-only local YOLO dataset YAML specification.
 * @param path Explicit local YAML path; image/split paths are not opened.
 * @param limits Input and logical parser limits.
 * @param stop Cooperative cancellation request.
 * @return Validated split, category and pose metadata; downloads never execute.
 * @throws IoError Missing/non-regular files, read failures or detected changes.
 * @throws InvalidArgument Invalid limits or an empty path.
 * @throws SerializationError Invalid YAML or dataset metadata.
 * @throws ResourceExhausted Input, parse or allocation limits.
 * @throws Cancelled Cancellation was requested.
 */
YoloDatasetSpec read_yolo_dataset_yaml_file(const std::filesystem::path& path,
                                            const AnnotationLimits& limits = {},
                                            std::stop_token stop = {});

}  // namespace jyppx::data::image

#endif  // JYPPX_DATA_IMAGE_ANNOTATION_FILES_HPP
