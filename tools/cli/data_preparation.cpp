// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "../datasets/cifar10.hpp"
#include "../datasets/fashion_mnist.hpp"
#include "../datasets/paired_affine.hpp"
#include "cli.hpp"

#include <cstdlib>
#include <memory>

#ifdef _WIN32
#include <cwchar>
#endif

namespace jyppx::liuding::cli {
Result data_prepare_command(const Arguments& arguments) {
  arguments.allow({"name", "data-root", "source-dir", "selection", "archive"});
  const auto name = arguments.require("name");
  const bool fashion = name == "fashion-mnist";
  const bool cifar = name == "cifar10";
  if (!fashion && !cifar && name != "paired-affine")
    usage("data.prepare",
          "Use --name paired-affine, fashion-mnist, or cifar10.");
  if (name == "paired-affine" &&
      (arguments.optional("source-dir") || arguments.optional("selection") ||
       arguments.optional("archive")))
    usage("data.prepare", "paired-affine accepts only --name and --data-root.");
  if (cifar &&
      (arguments.optional("source-dir") || arguments.optional("selection")))
    usage(
        "data.prepare",
        "cifar10 requires --archive and always prepares the complete dataset.");
  if (fashion && arguments.optional("archive"))
    usage("data.prepare",
          "fashion-mnist requires --source-dir, not --archive.");
  std::filesystem::path source;
  const auto selection =
      arguments.optional("selection").value_or("first100-per-class");
  if (fashion) {
    source = std::filesystem::u8path(arguments.require("source-dir"));
    if (selection != "first100-per-class" && selection != "full")
      usage("data.prepare",
            "Fashion-MNIST selection must be first100-per-class or full.");
  } else if (cifar) {
    source = std::filesystem::u8path(arguments.require("archive"));
  }
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
  const auto destination =
      fashion ? datasets::prepare_fashion_mnist(source, root, selection)
      : cifar ? datasets::prepare_cifar10(source, root)
              : datasets::prepare_paired_affine(root);
  const auto path = destination.generic_u8string();
  return {Json{{"dataset_id", fashion ? "jyppx.liuding.fashion-mnist/v1"
                              : cifar ? "jyppx.liuding.cifar10/v1"
                                      : "jyppx.liuding.paired-affine/v1"},
               {"destination",
                std::string(reinterpret_cast<const char*>(path.data()),
                            path.size())},
               {"preparation",
                parse_json(read_text(destination / "preparation.json"))}},
          0};
}
}  // namespace jyppx::liuding::cli
