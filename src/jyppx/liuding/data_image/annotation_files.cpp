// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "annotation_detail.hpp"

#include <jyppx/liuding/data_image/annotation_files.hpp>

#include <array>
#include <cerrno>
#include <new>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace jyppx::liuding::data::image {
namespace detail {
namespace {

[[noreturn]] void FileError(std::string_view operation) {
  throw IoError("Cannot read image annotation file", Details(operation));
}

class InputFile {
 public:
  explicit InputFile(const std::filesystem::path& path) {
#if defined(_WIN32)
    handle_ = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                          OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (handle_ == INVALID_HANDLE_VALUE) FileError("file.open");
#else
    handle_ = ::open(path.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (handle_ < 0) FileError("file.open");
#endif
  }
  ~InputFile() {
#if defined(_WIN32)
    CloseHandle(handle_);
#else
    ::close(handle_);
#endif
  }
  InputFile(const InputFile&) = delete;
  InputFile& operator=(const InputFile&) = delete;

  std::uint64_t Begin() {
#if defined(_WIN32)
    if (GetFileType(handle_) != FILE_TYPE_DISK ||
        !GetFileInformationByHandle(handle_, &before_) ||
        (before_.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0U)
      FileError("file.regular");
    return (static_cast<std::uint64_t>(before_.nFileSizeHigh) << 32U) |
           before_.nFileSizeLow;
#else
    if (::fstat(handle_, &before_) != 0 || !S_ISREG(before_.st_mode) ||
        before_.st_size < 0)
      FileError("file.regular");
    return static_cast<std::uint64_t>(before_.st_size);
#endif
  }
  std::size_t Read(std::span<char> bytes, std::stop_token stop) {
    CheckStop(stop);
#if defined(_WIN32)
    DWORD count = 0;
    if (!ReadFile(handle_, bytes.data(), static_cast<DWORD>(bytes.size()),
                  &count, nullptr))
      FileError("file.read");
    return count;
#else
    while (true) {
      const auto count = ::read(handle_, bytes.data(), bytes.size());
      if (count >= 0) return static_cast<std::size_t>(count);
      if (errno != EINTR) FileError("file.read");
      CheckStop(stop);
    }
#endif
  }
  void Finish() {
#if defined(_WIN32)
    BY_HANDLE_FILE_INFORMATION after{};
    if (!GetFileInformationByHandle(handle_, &after) ||
        after.nFileSizeHigh != before_.nFileSizeHigh ||
        after.nFileSizeLow != before_.nFileSizeLow ||
        CompareFileTime(&after.ftLastWriteTime, &before_.ftLastWriteTime) != 0)
      FileError("file.changed");
#else
    struct stat after {};
    if (::fstat(handle_, &after) != 0 || after.st_size != before_.st_size ||
        after.st_mtim.tv_sec != before_.st_mtim.tv_sec ||
        after.st_mtim.tv_nsec != before_.st_mtim.tv_nsec ||
        after.st_ctim.tv_sec != before_.st_ctim.tv_sec ||
        after.st_ctim.tv_nsec != before_.st_ctim.tv_nsec)
      FileError("file.changed");
#endif
  }

 private:
#if defined(_WIN32)
  HANDLE handle_ = INVALID_HANDLE_VALUE;
  BY_HANDLE_FILE_INFORMATION before_{};
#else
  int handle_ = -1;
  struct stat before_ {};
#endif
};

}  // namespace

std::string ReadAnnotationBytes(const std::filesystem::path& path,
                                std::size_t maximum, std::stop_token stop) {
  try {
    CheckStop(stop);
    if (path.empty() ||
        path.native().find(std::filesystem::path::value_type{}) !=
            std::filesystem::path::string_type::npos ||
        maximum == 0U || maximum > (1ULL << 30U))
      throw InvalidArgument("Invalid annotation input file options",
                            Details("file.options"));
    InputFile file(path);
    const auto size = file.Begin();
    if (size > maximum) Resource("file.bytes");
    std::string result;
    result.reserve(static_cast<std::size_t>(size));
    std::array<char, 65'536> chunk{};
    while (const auto count = file.Read(chunk, stop)) {
      if (count > maximum - result.size()) Resource("file.bytes");
      result.append(chunk.data(), count);
    }
    if (result.size() != size) FileError("file.changed_size");
    file.Finish();
    CheckStop(stop);
    return result;
  } catch (const std::bad_alloc&) {
    Resource("file.allocation");
  } catch (const std::length_error&) {
    Resource("file.container");
  }
}

}  // namespace detail

AnnotationIndex read_annotation_file(const std::filesystem::path& path,
                                     AnnotationFormat format,
                                     const AnnotationReadOptions& options,
                                     std::stop_token stop) {
  return parse_annotations(
      detail::ReadAnnotationBytes(path, options.limits.max_input_bytes, stop),
      format, options, stop);
}

YoloDatasetSpec read_yolo_dataset_yaml_file(const std::filesystem::path& path,
                                            const AnnotationLimits& limits,
                                            std::stop_token stop) {
  return parse_yolo_dataset_yaml(
      detail::ReadAnnotationBytes(path, limits.max_input_bytes, stop), limits,
      stop);
}

}  // namespace jyppx::liuding::data::image
