// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#ifndef JYPPX_LIUDING_TESTS_MODELS_REFERENCE_FILE_HPP
#define JYPPX_LIUDING_TESTS_MODELS_REFERENCE_FILE_HPP

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace jyppx::liuding::models::test {

// Fixture bytes stay file-backed instead of adding a second resident copy of
// a multi-GiB model. Callers still verify the complete fixed size and digest.
class MappedReference final {
 public:
  MappedReference(const std::filesystem::path& path, std::uint64_t size)
      : storage_(std::make_unique<Storage>()) {
    if (size == 0 || size > (std::numeric_limits<std::size_t>::max)())
      throw std::runtime_error("invalid models reference mapping size");
#ifdef _WIN32
    const auto file =
        CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE)
      throw std::runtime_error("cannot open models reference artifact");
    LARGE_INTEGER actual_size{};
    if (!GetFileSizeEx(file, &actual_size) || actual_size.QuadPart < 0 ||
        static_cast<std::uint64_t>(actual_size.QuadPart) != size) {
      CloseHandle(file);
      throw std::runtime_error("models reference mapping size mismatch");
    }
    storage_->mapping =
        CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    CloseHandle(file);
    if (storage_->mapping == nullptr)
      throw std::runtime_error("cannot map models reference artifact");
    const auto mapped = MapViewOfFile(storage_->mapping, FILE_MAP_READ, 0, 0,
                                      static_cast<std::size_t>(size));
    if (mapped == nullptr)
      throw std::runtime_error("cannot view models reference artifact");
#else
    const auto file = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (file < 0)
      throw std::runtime_error("cannot open models reference artifact");
    struct stat status{};
    if (fstat(file, &status) != 0 || !S_ISREG(status.st_mode) ||
        status.st_size < 0 ||
        static_cast<std::uint64_t>(status.st_size) != size) {
      close(file);
      throw std::runtime_error("models reference mapping size mismatch");
    }
    const auto mapped = mmap(nullptr, static_cast<std::size_t>(size), PROT_READ,
                             MAP_PRIVATE, file, 0);
    close(file);
    if (mapped == MAP_FAILED)
      throw std::runtime_error("cannot map models reference artifact");
#endif
    storage_->data = static_cast<const std::byte*>(mapped);
    storage_->size = static_cast<std::size_t>(size);
  }

  MappedReference(MappedReference&&) noexcept = default;
  MappedReference& operator=(MappedReference&&) noexcept = default;
  MappedReference(const MappedReference&) = delete;
  MappedReference& operator=(const MappedReference&) = delete;

  std::size_t size() const noexcept { return storage_->size; }
  operator std::span<const std::byte>() const noexcept {
    return {storage_->data, storage_->size};
  }

 private:
  struct Storage {
    const std::byte* data = nullptr;
    std::size_t size = 0;
#ifdef _WIN32
    HANDLE mapping = nullptr;
#endif
    ~Storage() {
#ifdef _WIN32
      if (data != nullptr) UnmapViewOfFile(data);
      if (mapping != nullptr) CloseHandle(mapping);
#else
      if (data != nullptr) munmap(const_cast<std::byte*>(data), size);
#endif
    }
  };
  std::unique_ptr<Storage> storage_;
};

}  // namespace jyppx::liuding::models::test

#endif  // JYPPX_LIUDING_TESTS_MODELS_REFERENCE_FILE_HPP
