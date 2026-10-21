// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "artifact_publish.hpp"

namespace jyppx::experiment::detail {

std::error_code publish_artifact_file(
    const std::filesystem::path& source,
    const std::filesystem::path& destination) {
  return native_publish_artifact_file(source, destination);
}

}  // namespace jyppx::experiment::detail
