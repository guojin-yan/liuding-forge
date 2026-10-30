// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "dinov2_conversion.hpp"

#include <cstdio>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {
constexpr std::string_view kUsage =
    "liuding-forge-convert-dinov2-hf --source OFFICIAL.safetensors "
    "--destination CONVERTED.safetensors --report conversion.json\n"
    "Offline fixed official DINOv2-S conversion. All three paths are required; "
    "outputs must be distinct new files in existing trusted directories.\n";

int run(const std::vector<std::string>& arguments) {
  if (arguments.size() == 1 && arguments.front() == "--help") {
    std::cout << kUsage;
    return 0;
  }
  if (arguments.size() != 6) {
    std::cerr << kUsage;
    return 2;
  }
  std::map<std::string, std::filesystem::path> options;
  for (std::size_t index = 0; index < arguments.size(); index += 2) {
    const auto& name = arguments[index];
    if ((name != "--source" && name != "--destination" && name != "--report") ||
        arguments[index + 1].empty() ||
        !options.emplace(name, std::filesystem::u8path(arguments[index + 1]))
             .second) {
      std::cerr << kUsage;
      return 2;
    }
  }
  try {
    const auto result = jyppx::liuding::tools::dinov2::convert(
        options.at("--source"), options.at("--destination"),
        options.at("--report"));
    std::cout << nlohmann::json{{"mapping_version",
                                 result.at("mapping_version")},
                                {"output", result.at("output")}}
                     .dump()
              << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "conversion failed: " << error.what() << '\n';
    return 1;
  }
}
}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) try {
  std::vector<std::string> arguments;
  for (int index = 1; index < argc; ++index) {
    const int size =
        WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[index], -1,
                            nullptr, 0, nullptr, nullptr);
    if (size <= 0) return 2;
    std::string value(static_cast<std::size_t>(size), '\0');
    if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[index], -1,
                            value.data(), size, nullptr, nullptr) != size)
      return 2;
    value.pop_back();
    arguments.push_back(std::move(value));
  }
  return run(arguments);
} catch (...) {
  std::fputs(
      "Conversion failed before arguments or response could be allocated.\n",
      stderr);
  return 1;
}
#else
int main(int argc, char* argv[]) try {
  std::vector<std::string> arguments;
  for (int index = 1; index < argc; ++index)
    arguments.emplace_back(argv[index]);
  return run(arguments);
} catch (...) {
  std::fputs(
      "Conversion failed before arguments or response could be allocated.\n",
      stderr);
  return 1;
}
#endif
