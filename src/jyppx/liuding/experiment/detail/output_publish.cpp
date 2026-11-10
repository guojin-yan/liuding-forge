// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "file_publication.hpp"

namespace jyppx::liuding::experiment::detail {
std::error_code publish_output_file(const std::filesystem::path& staging,
                                    const std::filesystem::path& destination) {
  return native_publish_output_file(staging, destination);
}
}  // namespace jyppx::liuding::experiment::detail
