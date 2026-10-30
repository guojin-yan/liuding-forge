// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "../datasets/paired_affine.hpp"
#include "cli.hpp"

#include <cstdlib>
#include <memory>

#ifdef _WIN32
#include <cwchar>
#endif

namespace jyppx::liuding::cli {
Result data_prepare_command(const Arguments& arguments) {
  arguments.allow({"name", "data-root"});
  if (arguments.require("name") != "paired-affine")
    usage("data.prepare",
          "Use data prepare --name paired-affine --data-root DIRECTORY.");
  std::filesystem::path root;
  if (const auto explicit_root = arguments.optional("data-root")) {
    root = std::filesystem::u8path(*explicit_root);
  } else {
#ifdef _WIN32
    wchar_t* environment = nullptr;
    std::size_t length = 0U;
    if (_wdupenv_s(&environment, &length, L"JYPPX_LIUDING_DATA_ROOT") != 0)
      throw IoError("Could not read dataset root environment variable");
    const std::unique_ptr<wchar_t, decltype(&std::free)> owned(environment,
                                                               &std::free);
    if (environment && length > 1U) root = environment;
#else
    const auto* environment = std::getenv("JYPPX_LIUDING_DATA_ROOT");
    if (environment && *environment != '\0')
      root = std::filesystem::u8path(environment);
#endif
  }
  if (root.empty())
    usage("data.prepare", "Supply --data-root or set JYPPX_LIUDING_DATA_ROOT.");
  const auto destination = datasets::prepare_paired_affine(root);
  const auto path = destination.generic_u8string();
  return {Json{{"dataset_id", "jyppx.liuding.paired-affine/v1"},
               {"destination",
                std::string(reinterpret_cast<const char*>(path.data()),
                            path.size())},
               {"preparation",
                parse_json(read_text(destination / "preparation.json"))}},
          0};
}
}  // namespace jyppx::liuding::cli
