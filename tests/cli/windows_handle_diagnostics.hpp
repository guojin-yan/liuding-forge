// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_TEST_WINDOWS_HANDLE_DIAGNOSTICS_HPP
#define JYPPX_LIUDING_TEST_WINDOWS_HANDLE_DIAGNOSTICS_HPP

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <windows.h>

namespace jyppx::liuding::test {

// Test-only, self-process metadata. No process/duplicate/PSS handles are
// opened; ObjectNameInformation is deliberately never queried (it can block and
// expose paths). A handle value is a reusable slot, not durable object
// identity.
struct WindowsHandleMetadata {
  std::uintptr_t value = 0;
  ULONG type_index = 0;
  ULONG attributes = 0;
  std::string type;
};

struct WindowsHandleSnapshot {
  DWORD count = 0;
  std::map<std::uintptr_t, WindowsHandleMetadata> handles;
};

namespace windows_handle_detail {
using QueryProcess = LONG(NTAPI*)(HANDLE, ULONG, PVOID, ULONG, PULONG);
using QueryObject = LONG(NTAPI*)(HANDLE, ULONG, PVOID, ULONG, PULONG);

struct ProcessEntry {
  HANDLE value;
  ULONG_PTR handle_count;
  ULONG_PTR pointer_count;
  ULONG granted_access;
  ULONG type_index;
  ULONG attributes;
  ULONG reserved;
};
struct ProcessTable {
  ULONG_PTR count;
  ULONG_PTR reserved;
  ProcessEntry entries[1];
};
struct TypeName {
  USHORT length;
  USHORT capacity;
  PWSTR buffer;
};
static_assert(offsetof(ProcessTable, entries) == 2U * sizeof(ULONG_PTR));
static_assert(sizeof(ProcessEntry) ==
              3U * sizeof(ULONG_PTR) + 4U * sizeof(ULONG));

inline std::runtime_error failure(const char* operation, LONG status) {
  std::ostringstream message;
  message << operation << " failed: status=0x" << std::hex
          << static_cast<ULONG>(status);
  return std::runtime_error(message.str());
}

inline bool resize_status(LONG status) {
  const auto code = static_cast<ULONG>(status);
  return code == 0xC0000004U || code == 0xC0000023U || code == 0x80000005U;
}

inline std::string object_type(QueryObject query, HANDLE handle) {
  std::vector<std::uintptr_t> buffer(128U);
  constexpr ULONG kMaximumBytes = 65536U;
  for (unsigned attempt = 0; attempt < 8U; ++attempt) {
    const auto capacity = static_cast<ULONG>(buffer.size() * sizeof(buffer[0]));
    ULONG returned = 0;
    const auto status = query(handle, 2U, buffer.data(), capacity, &returned);
    if (resize_status(status)) {
      if (returned <= capacity || returned > kMaximumBytes)
        throw failure("NtQueryObject(type size)", status);
      buffer.resize((returned + sizeof(buffer[0]) - 1U) / sizeof(buffer[0]));
      continue;
    }
    if (status < 0) throw failure("NtQueryObject(type)", status);
    if (returned < sizeof(TypeName) || returned > capacity)
      throw std::runtime_error("NtQueryObject returned an invalid type size");
    TypeName name{};
    std::memcpy(&name, buffer.data(), sizeof(name));
    const auto begin = reinterpret_cast<std::uintptr_t>(buffer.data());
    const auto address = reinterpret_cast<std::uintptr_t>(name.buffer);
    if (name.length == 0U || name.length % sizeof(wchar_t) != 0U ||
        name.length > name.capacity || address % alignof(wchar_t) != 0U ||
        address < begin || address - begin > returned ||
        name.length > returned - (address - begin))
      throw std::runtime_error("NtQueryObject returned an invalid type string");
    const auto count = static_cast<int>(name.length / sizeof(wchar_t));
    const auto bytes =
        WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, name.buffer, count,
                            nullptr, 0, nullptr, nullptr);
    if (bytes <= 0) throw failure("WideCharToMultiByte(size)", GetLastError());
    std::string type(static_cast<std::size_t>(bytes), '\0');
    if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, name.buffer, count,
                            type.data(), bytes, nullptr, nullptr) != bytes)
      throw failure("WideCharToMultiByte(type)", GetLastError());
    return type;
  }
  throw std::runtime_error("NtQueryObject type query exceeded its retry bound");
}
}  // namespace windows_handle_detail

inline WindowsHandleSnapshot CaptureWindowsSelfHandles() {
  namespace detail = windows_handle_detail;
  const auto module = GetModuleHandleW(L"ntdll.dll");
  if (!module) throw detail::failure("GetModuleHandleW(ntdll)", GetLastError());
  const auto process = reinterpret_cast<detail::QueryProcess>(
      GetProcAddress(module, "NtQueryInformationProcess"));
  const auto object = reinterpret_cast<detail::QueryObject>(
      GetProcAddress(module, "NtQueryObject"));
  if (!process || !object)
    throw detail::failure("GetProcAddress(native handle metadata)",
                          GetLastError());
  WindowsHandleSnapshot snapshot;
  {
    std::vector<std::uintptr_t> buffer(8192U);
    constexpr ULONG kMaximumBytes = 16U << 20U;
    bool captured = false;
    for (unsigned attempt = 0; attempt < 8U; ++attempt) {
      const auto capacity =
          static_cast<ULONG>(buffer.size() * sizeof(buffer[0]));
      ULONG returned = 0;
      const auto status =
          process(GetCurrentProcess(), 51U, buffer.data(), capacity, &returned);
      if (detail::resize_status(status)) {
        if (returned <= capacity || returned > kMaximumBytes)
          throw detail::failure("NtQueryInformationProcess(handle size)",
                                status);
        buffer.resize((returned + sizeof(buffer[0]) - 1U) / sizeof(buffer[0]));
        continue;
      }
      if (status < 0)
        throw detail::failure("NtQueryInformationProcess(handles)", status);
      constexpr auto offset = offsetof(detail::ProcessTable, entries);
      if (returned < offset || returned > capacity)
        throw std::runtime_error(
            "Native handle snapshot returned an invalid size");
      ULONG_PTR count = 0;
      std::memcpy(&count, buffer.data(), sizeof(count));
      if (count > (returned - offset) / sizeof(detail::ProcessEntry) ||
          count > std::numeric_limits<DWORD>::max())
        throw std::runtime_error(
            "Native handle snapshot entry count is invalid");
      for (ULONG_PTR index = 0; index < count; ++index) {
        detail::ProcessEntry entry{};
        const auto* raw = reinterpret_cast<const std::byte*>(buffer.data());
        std::memcpy(&entry, raw + offset + index * sizeof(entry),
                    sizeof(entry));
        WindowsHandleMetadata metadata{
            .value = reinterpret_cast<std::uintptr_t>(entry.value),
            .type_index = entry.type_index,
            .attributes = entry.attributes,
            .type = detail::object_type(object, entry.value)};
        if (!snapshot.handles.emplace(metadata.value, std::move(metadata))
                 .second)
          throw std::runtime_error("Native handle snapshot repeated a slot");
      }
      captured = true;
      break;
    }
    if (!captured)
      throw std::runtime_error(
          "Native handle snapshot exceeded its retry bound");
  }
  // Count only after all temporary native query buffers are gone. The observer
  // keeps STL metadata but owns no kernel handles. Concurrent table churn is
  // a failed diagnostic, never silently accepted as a complete snapshot.
  if (!GetProcessHandleCount(GetCurrentProcess(), &snapshot.count))
    throw detail::failure("GetProcessHandleCount", GetLastError());
  if (snapshot.count != snapshot.handles.size())
    throw std::runtime_error(
        "Self handle table changed during its metadata snapshot");
  return snapshot;
}

inline std::string WindowsHandleDifference(
    const WindowsHandleSnapshot& baseline,
    const WindowsHandleSnapshot& current) {
  std::ostringstream message;
  message << "baseline=" << baseline.count << " current=" << current.count
          << " (slot/type differences; handle values may be reused)";
  const auto changed = [](const auto& previous, const auto& next) {
    const auto found = previous.find(next.first);
    return found == previous.end() || found->second.type != next.second.type;
  };
  const auto emit = [&](const char* prefix, const auto& entry) {
    message << ' ' << prefix << "[0x" << std::hex << entry.second.value
            << std::dec << ":" << entry.second.type
            << ",type_index=" << entry.second.type_index
            << ",attributes=" << entry.second.attributes << ']';
  };
  for (const auto& entry : current.handles)
    if (changed(baseline.handles, entry)) emit("added", entry);
  for (const auto& entry : baseline.handles)
    if (changed(current.handles, entry)) emit("removed", entry);
  return message.str();
}

}  // namespace jyppx::liuding::test
#endif  // defined(_WIN32)
#endif  // JYPPX_LIUDING_TEST_WINDOWS_HANDLE_DIAGNOSTICS_HPP
