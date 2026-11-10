// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

// These entry points belong only to the recovery test's sink object. The normal
// library and other objects call the unmodified Windows APIs.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
std::filesystem::path directory, staging_path;
std::string boundary;
std::atomic<bool> enabled{false}, fired{false};
std::atomic<HANDLE> staged_handle{INVALID_HANDLE_VALUE};
std::atomic<unsigned> delete_opens{0}, named_opens{0}, link_calls{0};

bool staged(HANDLE handle) {
  return enabled.load() && handle == staged_handle.load();
}

void pause_at(HANDLE handle, std::string_view selected) {
  if (boundary != selected || fired.exchange(true)) return;
  BY_HANDLE_FILE_INFORMATION info{};
  FILE_ID_INFO identity{};
  if (!GetFileInformationByHandle(handle, &info) ||
      !GetFileInformationByHandleEx(handle, FileIdInfo, &identity,
                                    sizeof(identity)))
    throw std::runtime_error("could not observe staged Windows file");
  std::uint64_t low = 0, high = 0;
  std::memcpy(&low, identity.FileId.Identifier, sizeof(low));
  std::memcpy(&high, identity.FileId.Identifier + sizeof(low), sizeof(high));
  std::cout << nlohmann::json{{"phase", "ready"},
                              {"boundary", boundary},
                              {"staging_path", staging_path.string()},
                              {"inode", low},
                              {"inode_high", high},
                              {"device", identity.VolumeSerialNumber},
                              {"links", info.nNumberOfLinks},
                              {"bytes",
                               (std::uint64_t{info.nFileSizeHigh} << 32U) |
                                   info.nFileSizeLow}}
                   .dump()
            << std::endl;
  char release;
  if (!std::cin.get(release))
    throw std::runtime_error("missing native boundary release");
}

HANDLE WINAPI create_file(LPCWSTR name, DWORD access, DWORD share,
                          LPSECURITY_ATTRIBUTES security, DWORD disposition,
                          DWORD flags, HANDLE prototype) {
  static const auto actual = &::CreateFileW;
  const std::filesystem::path path(name);
  const bool selected =
      enabled.load() && path.parent_path() == directory &&
      path.filename().wstring().starts_with(L"preview.png.liuding-tmp-");
  if (selected && boundary == "fail-open") {
    SetLastError(ERROR_ACCESS_DENIED);
    return INVALID_HANDLE_VALUE;
  }
  const auto handle =
      actual(name, access, share, security, disposition, flags, prototype);
  if (selected && handle != INVALID_HANDLE_VALUE) {
    staging_path = path;
    staged_handle.store(handle);
    ++named_opens;
    if ((flags & FILE_FLAG_DELETE_ON_CLOSE) != 0) ++delete_opens;
    pause_at(handle, "staging-created");
  }
  return handle;
}

BOOL WINAPI volume_information(HANDLE handle, LPWSTR volume, DWORD volume_size,
                               LPDWORD serial, LPDWORD maximum, LPDWORD flags,
                               LPWSTR filesystem, DWORD filesystem_size) {
  static const auto actual = &::GetVolumeInformationByHandleW;
  if (staged(handle) && boundary == "volume-query-failed") {
    SetLastError(ERROR_NOT_SUPPORTED);
    return FALSE;
  }
  const auto result = actual(handle, volume, volume_size, serial, maximum,
                             flags, filesystem, filesystem_size);
  if (result && staged(handle) &&
      (boundary == "fallback" || boundary == "fallback-writing"))
    *flags &= ~FILE_SUPPORTS_HARD_LINKS;
  return result;
}

BOOL WINAPI write_file(HANDLE handle, LPCVOID bytes, DWORD count,
                       LPDWORD written, LPOVERLAPPED overlapped) {
  static const auto actual = &::WriteFile;
  if (!staged(handle)) return actual(handle, bytes, count, written, overlapped);
  if (boundary == "fail-write") {
    if (!fired.exchange(true))
      return actual(handle, bytes, std::min(count, DWORD{13}), written,
                    overlapped);
    SetLastError(ERROR_WRITE_FAULT);
    return FALSE;
  }
  if ((boundary == "staging-writing" || boundary == "fallback-writing" ||
       boundary == "replace-writing") &&
      !fired.load()) {
    const auto result =
        actual(handle, bytes, std::min(count, DWORD{13}), written, overlapped);
    if (result && *written != 0) pause_at(handle, boundary);
    return result;
  }
  return actual(handle, bytes, count, written, overlapped);
}

BOOL WINAPI flush_file(HANDLE handle) {
  static const auto actual = &::FlushFileBuffers;
  if (staged(handle) && boundary == "fail-sync") {
    SetLastError(ERROR_WRITE_FAULT);
    return FALSE;
  }
  const auto result = actual(handle);
  if (result && staged(handle)) pause_at(handle, "staging-synced");
  return result;
}

BOOL WINAPI hard_link(LPCWSTR target, LPCWSTR source,
                      LPSECURITY_ATTRIBUTES security) {
  static const auto actual = &::CreateHardLinkW;
  const bool selected = enabled.load() && std::filesystem::path(target) ==
                                              directory / "preview.png";
  if (selected) {
    ++link_calls;
    if (boundary == "fail-link") {
      SetLastError(ERROR_DISK_FULL);
      return FALSE;
    }
  }
  const auto result = actual(target, source, security);
  if (result && selected) pause_at(staged_handle.load(), "staging-linked");
  return result;
}

BOOL WINAPI close_handle(HANDLE handle) {
  static const auto actual = &::CloseHandle;
  const bool fail = staged(handle) && boundary == "fail-close";
  auto expected = handle;
  staged_handle.compare_exchange_strong(expected, INVALID_HANDLE_VALUE);
  const auto result = actual(handle);
  if (result && fail) {
    SetLastError(ERROR_WRITE_FAULT);
    return FALSE;
  }
  return result;
}
}  // namespace

bool configure_native_image_output(const std::filesystem::path& root,
                                   std::string_view selected) {
  constexpr std::array allowed{"file",
                               "evidence",
                               "none",
                               "staging-created",
                               "staging-writing",
                               "staging-synced",
                               "staging-linked",
                               "fallback",
                               "fallback-writing",
                               "volume-query-failed",
                               "replace-writing",
                               "fail-open",
                               "fail-write",
                               "fail-sync",
                               "fail-link",
                               "fail-close"};
  if (std::find(allowed.begin(), allowed.end(), selected) == allowed.end())
    return false;
  directory = root;
  boundary = selected;
  enabled.store(true);
  return true;
}

nlohmann::json native_image_output_observations() {
  return {{"delete_on_close_opens", delete_opens.load()},
          {"named_opens", named_opens.load()},
          {"link_calls", link_calls.load()}};
}

extern "C" {
HANDLE WINAPI liuding_image_test_CreateFileW(LPCWSTR name, DWORD access,
                                             DWORD share,
                                             LPSECURITY_ATTRIBUTES security,
                                             DWORD disposition, DWORD flags,
                                             HANDLE prototype) {
  return create_file(name, access, share, security, disposition, flags,
                     prototype);
}
BOOL WINAPI liuding_image_test_GetVolumeInformationByHandleW(
    HANDLE handle, LPWSTR volume, DWORD volume_size, LPDWORD serial,
    LPDWORD maximum, LPDWORD flags, LPWSTR filesystem, DWORD filesystem_size) {
  return volume_information(handle, volume, volume_size, serial, maximum, flags,
                            filesystem, filesystem_size);
}
BOOL WINAPI liuding_image_test_WriteFile(HANDLE handle, LPCVOID bytes,
                                         DWORD count, LPDWORD written,
                                         LPOVERLAPPED overlapped) {
  return write_file(handle, bytes, count, written, overlapped);
}
BOOL WINAPI liuding_image_test_FlushFileBuffers(HANDLE handle) {
  return flush_file(handle);
}
BOOL WINAPI liuding_image_test_CreateHardLinkW(LPCWSTR target, LPCWSTR source,
                                               LPSECURITY_ATTRIBUTES security) {
  return hard_link(target, source, security);
}
BOOL WINAPI liuding_image_test_CloseHandle(HANDLE handle) {
  return close_handle(handle);
}
}
