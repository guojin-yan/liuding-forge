// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_TEST_NATIVE_MODULES_HPP
#define JYPPX_TEST_NATIVE_MODULES_HPP
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
// clang-format off
#include <windows.h>
#include <psapi.h>
// clang-format on
#else
#include <link.h>
#endif
namespace jyppx_test {
inline void Modules(const std::filesystem::path& path) {
  std::ofstream output(path, std::ios::binary);
#ifdef _WIN32
  std::vector<HMODULE> handles(4096);
  DWORD needed = 0;
  if (!EnumProcessModulesEx(
          GetCurrentProcess(), handles.data(),
          static_cast<DWORD>(handles.size() * sizeof(HMODULE)), &needed,
          LIST_MODULES_ALL) ||
      needed > handles.size() * sizeof(HMODULE))
    throw std::runtime_error("module enumeration failed");
  handles.resize(needed / sizeof(HMODULE));
  for (auto handle : handles) {
    std::wstring name(32768, L'\0');
    const auto count = GetModuleFileNameW(handle, name.data(),
                                          static_cast<DWORD>(name.size()));
    if (count == 0 || count >= name.size())
      throw std::runtime_error("module path failed");
    const auto bytes = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                           name.data(), static_cast<int>(count),
                                           nullptr, 0, nullptr, nullptr);
    if (bytes <= 0) throw std::runtime_error("module path encoding failed");
    std::string utf8(static_cast<std::size_t>(bytes), '\0');
    if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, name.data(),
                            static_cast<int>(count), utf8.data(), bytes,
                            nullptr, nullptr) != bytes)
      throw std::runtime_error("module path conversion failed");
    output << utf8 << '\n';
  }
#else
  dl_iterate_phdr(
      [](dl_phdr_info* info, std::size_t, void* data) {
        if (info->dlpi_name != nullptr && *info->dlpi_name != '\0')
          *static_cast<std::ofstream*>(data) << info->dlpi_name << '\n';
        return 0;
      },
      &output);
#endif
  output.close();
  if (!output) throw std::runtime_error("module receipt write failure");
}

}  // namespace jyppx_test
#endif
