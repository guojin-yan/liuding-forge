// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_TOOLS_DATASETS_PAIRED_AFFINE_HPP
#define JYPPX_TOOLS_DATASETS_PAIRED_AFFINE_HPP

#include <filesystem>

namespace jyppx::liuding::datasets {

// Generates one bounded record at a time and verifies the frozen v1 split
// identities before publishing data_root/paired-affine-v1 without replacement.
// data_root must be outside the source repository. Its ancestors must contain
// no links/reparse points and remain stable for the invocation. Only staging
// created by this invocation is cleaned on failure; existing data is untouched.
// Throws foundation Error on invalid paths, I/O errors or identity mismatch.
std::filesystem::path prepare_paired_affine(
    const std::filesystem::path& data_root);

}  // namespace jyppx::liuding::datasets
#endif  // JYPPX_TOOLS_DATASETS_PAIRED_AFFINE_HPP
