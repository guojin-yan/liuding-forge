// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

// Linker wrappers are confined to this Linux test executable. They observe
// actual sink syscalls; the shipped library has no fault or pause hooks.
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>

namespace {
std::string directory, boundary;
std::atomic<bool> enabled{false}, fired{false};
std::atomic<bool> linked_output{false};
std::atomic<int> staged_descriptor{-1};
std::atomic<unsigned> anonymous_opens{0}, named_opens{0}, link_calls{0};

std::string descriptor_path(int descriptor) {
  std::array<char, 4096> buffer{};
  const auto name = "/proc/self/fd/" + std::to_string(descriptor);
  const auto count = ::readlink(name.c_str(), buffer.data(), buffer.size());
  return count < 0 ? std::string{} : std::string(buffer.data(), count);
}

bool staged(int descriptor) {
  if (!enabled.load()) return false;
  struct stat info{};
  if (::fstat(descriptor, &info) != 0 || !S_ISREG(info.st_mode)) return false;
  const auto path = descriptor_path(descriptor);
  return path.starts_with(directory + "/") &&
         (descriptor == staged_descriptor.load() || info.st_nlink == 0 ||
          path.starts_with(directory + "/preview.png.liuding-tmp-"));
}

void pause_at(int descriptor, std::string_view selected) {
  if (boundary != selected || fired.exchange(true)) return;
  struct stat info{};
  if (::fstat(descriptor, &info) != 0)
    throw std::runtime_error("could not observe staged descriptor");
  std::cout << nlohmann::json{{"phase", "ready"},
                              {"boundary", boundary},
                              {"descriptor", descriptor},
                              {"inode", info.st_ino},
                              {"device", info.st_dev},
                              {"links", info.st_nlink},
                              {"bytes", info.st_size}}
                   .dump()
            << std::endl;
  char release;
  if (!std::cin.get(release))
    throw std::runtime_error("missing native boundary release");
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
                               "fail-write",
                               "fail-sync",
                               "fail-link",
                               "fail-close",
                               "fail-directory",
                               "fail-open"};
  if (std::find(allowed.begin(), allowed.end(), selected) == allowed.end())
    return false;
  directory = root.string();
  boundary = selected;
  enabled.store(true);
  return true;
}

nlohmann::json native_image_output_observations() {
  return {{"anonymous_opens", anonymous_opens.load()},
          {"named_opens", named_opens.load()},
          {"link_calls", link_calls.load()}};
}

extern "C" {
int __real_open(const char*, int, ...);
ssize_t __real_write(int, const void*, size_t);
int __real_fsync(int);
int __real_linkat(int, const char*, int, const char*, int);
int __real_link(const char*, const char*);
int __real_close(int);

int __wrap_open(const char* path, int flags, ...) {
  mode_t mode = 0;
  const bool anonymous = (flags & O_TMPFILE) == O_TMPFILE;
  if ((flags & O_CREAT) != 0 || anonymous) {
    va_list args;
    va_start(args, flags);
    mode = va_arg(args, mode_t);
    va_end(args);
  }
  if (enabled.load() && directory == path && anonymous) {
    ++anonymous_opens;
    if (boundary == "fallback" || boundary == "fallback-writing" ||
        boundary == "fail-open") {
      errno = boundary == "fail-open" ? EACCES : EOPNOTSUPP;
      return -1;
    }
  }
  const int result = __real_open(path, flags, mode);
  if (result >= 0 && staged(result)) {
    staged_descriptor.store(result);
    if (!anonymous) ++named_opens;
    pause_at(result, "staging-created");
  }
  return result;
}

ssize_t __wrap_write(int descriptor, const void* bytes, size_t count) {
  if (!staged(descriptor)) return __real_write(descriptor, bytes, count);
  if (boundary == "fail-write") {
    if (!fired.exchange(true))
      return __real_write(descriptor, bytes, std::min(count, size_t{13}));
    errno = EIO;
    return -1;
  }
  if ((boundary == "staging-writing" || boundary == "fallback-writing") &&
      !fired.load()) {
    const auto result =
        __real_write(descriptor, bytes, std::min(count, size_t{13}));
    if (result > 0) pause_at(descriptor, boundary);
    return result;
  }
  return __real_write(descriptor, bytes, count);
}

int __wrap_fsync(int descriptor) {
  if (staged(descriptor)) {
    if (boundary == "fail-sync") {
      errno = EIO;
      return -1;
    }
    const int result = __real_fsync(descriptor);
    if (result == 0) pause_at(descriptor, "staging-synced");
    return result;
  }
  if (enabled.load() && linked_output.load() && boundary == "fail-directory" &&
      descriptor_path(descriptor) == directory) {
    errno = EIO;
    return -1;
  }
  return __real_fsync(descriptor);
}

int __wrap_linkat(int from, const char* source, int to, const char* target,
                  int flags) {
  if (enabled.load() && directory + "/preview.png" == target) {
    ++link_calls;
    if (boundary == "fail-link") {
      errno = ENOSPC;
      return -1;
    }
    const int result = __real_linkat(from, source, to, target, flags);
    if (result == 0) {
      linked_output.store(true);
      // AT_EMPTY_PATH uses the live descriptor. The unprivileged /proc
      // fallback encodes that same descriptor in its source pathname.
      const int descriptor =
          source[0] == '\0' ? from : std::stoi(std::string(source).substr(14));
      pause_at(descriptor, "staging-linked");
    }
    return result;
  }
  return __real_linkat(from, source, to, target, flags);
}

int __wrap_link(const char* source, const char* target) {
  if (enabled.load() && directory + "/preview.png" == target) {
    ++link_calls;
    if (boundary == "fail-link") {
      errno = ENOSPC;
      return -1;
    }
    const int result = __real_link(source, target);
    if (result == 0) linked_output.store(true);
    if (result == 0 && boundary == "staging-linked") {
      const int descriptor = __real_open(source, O_RDONLY);
      if (descriptor < 0)
        throw std::runtime_error("could not observe named linked file");
      pause_at(descriptor, "staging-linked");
      __real_close(descriptor);
    }
    return result;
  }
  return __real_link(source, target);
}

int __wrap_close(int descriptor) {
  const bool fail = staged(descriptor) && boundary == "fail-close";
  int expected = descriptor;
  staged_descriptor.compare_exchange_strong(expected, -1);
  const int result = __real_close(descriptor);
  if (fail && result == 0) {
    errno = EIO;
    return -1;
  }
  return result;
}
}
