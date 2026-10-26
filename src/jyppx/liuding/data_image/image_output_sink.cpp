// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data_image/image_output_sink.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <condition_variable>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <span>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace jyppx::liuding::data::image {
namespace {

constexpr std::string_view kComponent = "data.image.output_sink";
std::atomic<std::uint64_t> next_temporary_id{0};

[[noreturn]] void Invalid(std::string_view operation,
                          std::string_view expected) {
  throw InvalidArgument("Invalid image output sink request",
                        ErrorDetails{.component = std::string(kComponent),
                                     .operation = std::string(operation),
                                     .expected = std::string(expected)});
}

[[noreturn]] void Io(std::string_view operation,
                     const std::filesystem::path& path,
                     const std::error_code& error) {
  throw IoError("Image output publication failed",
                ErrorDetails{.component = std::string(kComponent),
                             .operation = std::string(operation),
                             .actual = path.string() + ": " + error.message()});
}

#ifdef _WIN32
std::error_code NativeError() {
  return {static_cast<int>(GetLastError()), std::system_category()};
}
std::uint64_t ProcessId() { return GetCurrentProcessId(); }
#else
std::error_code NativeError() { return {errno, std::generic_category()}; }
std::uint64_t ProcessId() { return static_cast<std::uint64_t>(getpid()); }
#endif

std::filesystem::path Parent(const std::filesystem::path& path) {
  return path.has_parent_path() ? path.parent_path()
                                : std::filesystem::path{"."};
}

class StagedFile final {
 public:
  explicit StagedFile(const std::filesystem::path& destination)
      : destination_(destination) {
    for (int attempt = 0; attempt < 32; ++attempt) {
      path_ = destination;
      path_ += ".liuding-tmp-" + std::to_string(ProcessId()) + "-" +
               std::to_string(next_temporary_id.fetch_add(1));
#ifdef _WIN32
      handle_ = CreateFileW(path_.c_str(), GENERIC_WRITE, 0, nullptr,
                            CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
      if (handle_ != INVALID_HANDLE_VALUE) return;
      const auto error = NativeError();
      if (error.value() != ERROR_FILE_EXISTS &&
          error.value() != ERROR_ALREADY_EXISTS)
        Io("temporary.create", path_, error);
#else
      descriptor_ = ::open(path_.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
      if (descriptor_ >= 0) return;
      const auto error = NativeError();
      if (error.value() != EEXIST) Io("temporary.create", path_, error);
#endif
    }
    Io("temporary.create", path_, std::make_error_code(std::errc::file_exists));
  }

  StagedFile(const StagedFile&) = delete;
  StagedFile& operator=(const StagedFile&) = delete;

  ~StagedFile() {
    CloseNoThrow();
    if (!published_) {
      std::error_code ignored;
      std::filesystem::remove(path_, ignored);
    }
  }

  void Write(std::span<const std::byte> bytes) {
    const auto* data = reinterpret_cast<const unsigned char*>(bytes.data());
    std::size_t offset = 0;
    while (offset < bytes.size()) {
      const auto count = static_cast<unsigned int>(
          std::min<std::size_t>(bytes.size() - offset, 1U << 20U));
#ifdef _WIN32
      DWORD written = 0;
      if (!WriteFile(handle_, data + offset, count, &written, nullptr))
        Io("temporary.write", path_, NativeError());
#else
      const auto written = ::write(descriptor_, data + offset, count);
      if (written < 0 && errno == EINTR) continue;
      if (written < 0) Io("temporary.write", path_, NativeError());
#endif
      if (written == 0)
        Io("temporary.write", path_,
           std::make_error_code(std::errc::no_space_on_device));
      offset += static_cast<std::size_t>(written);
    }
#ifdef _WIN32
    if (!FlushFileBuffers(handle_)) Io("temporary.flush", path_, NativeError());
#else
    if (::fsync(descriptor_) != 0) Io("temporary.flush", path_, NativeError());
#endif
    Close();
  }

  void Publish(bool replace_existing) {
#ifdef _WIN32
    DWORD flags = MOVEFILE_WRITE_THROUGH;
    if (replace_existing) flags |= MOVEFILE_REPLACE_EXISTING;
    if (!MoveFileExW(path_.c_str(), destination_.c_str(), flags))
      Io("publish", destination_, NativeError());
#else
    if (replace_existing) {
      if (::rename(path_.c_str(), destination_.c_str()) != 0)
        Io("publish", destination_, NativeError());
    } else {
      if (::link(path_.c_str(), destination_.c_str()) != 0)
        Io("publish", destination_, NativeError());
      // The published hard link remains valid if removing the staging name
      // fails.
      if (::unlink(path_.c_str()) != 0)
        Io("temporary.remove", path_, NativeError());
    }
#endif
    published_ = true;
#ifndef _WIN32
    const auto parent = Parent(destination_);
    const int directory = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY);
    if (directory < 0) Io("directory.open", parent, NativeError());
    const int result = ::fsync(directory);
    const auto flush_error = NativeError();
    const int closed = ::close(directory);
    if (result != 0) Io("directory.flush", parent, flush_error);
    if (closed != 0) Io("directory.close", parent, NativeError());
#endif
  }

 private:
  void Close() {
#ifdef _WIN32
    const auto handle = std::exchange(handle_, INVALID_HANDLE_VALUE);
    if (!CloseHandle(handle)) Io("temporary.close", path_, NativeError());
#else
    const auto descriptor = std::exchange(descriptor_, -1);
    if (::close(descriptor) != 0) Io("temporary.close", path_, NativeError());
#endif
  }

  void CloseNoThrow() noexcept {
#ifdef _WIN32
    if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);
    handle_ = INVALID_HANDLE_VALUE;
#else
    if (descriptor_ >= 0) ::close(descriptor_);
    descriptor_ = -1;
#endif
  }

  std::filesystem::path destination_;
  std::filesystem::path path_;
  bool published_ = false;
#ifdef _WIN32
  HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
  int descriptor_ = -1;
#endif
};

void ValidateImage(const std::filesystem::path& path,
                   const EncodedImage& image) {
  if (path.empty() || path.filename().empty())
    Invalid("submit.path", "a nonempty file path");
  if (image.width == 0 || image.height == 0 || image.channels == 0 ||
      image.bytes.empty())
    Invalid("submit.image", "nonempty encoded image and positive dimensions");
  if (image.format != ImageFormat::png && image.format != ImageFormat::jpeg)
    Invalid("submit.format", "PNG or JPEG");
  if (detect_image_format(image.bytes) != image.format)
    Invalid("submit.bytes", "encoded bytes matching declared format");
  auto extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char value) {
                   return static_cast<char>(std::tolower(value));
                 });
  if ((image.format == ImageFormat::png && extension != ".png") ||
      (image.format == ImageFormat::jpeg && extension != ".jpg" &&
       extension != ".jpeg"))
    Invalid("submit.extension", "extension matching encoded image format");
}

bool SelectSample(std::string_view key, std::uint32_t every) {
  if (every == 1) return true;
  if (key.empty() || key.size() > 4096)
    Invalid("submit.sample_key", "1..4096 stable key bytes");
  Sha256 hash;
  constexpr std::string_view domain = "liuding.image-output-sample.v1";
  hash.update(std::as_bytes(std::span(domain.data(), domain.size())));
  hash.update(std::as_bytes(std::span(key.data(), key.size())));
  std::uint64_t value = 0;
  const auto digest = hash.finish();
  for (const auto byte : std::span(digest).first<8>())
    value = (value << 8U) | std::to_integer<std::uint8_t>(byte);
  return value % every == 0;
}

}  // namespace

class ImageOutputSink::Impl final {
 public:
  explicit Impl(ImageOutputSinkOptions options) : options_(std::move(options)) {
    if (options_.max_pending_items == 0 || options_.max_pending_bytes == 0 ||
        options_.max_item_bytes == 0 || options_.sample_every == 0)
      Invalid("options", "positive item, byte and sampling limits");
    if (options_.overflow != ImageSinkOverflowPolicy::block &&
        options_.overflow != ImageSinkOverflowPolicy::drop_newest)
      Invalid("options.overflow", "block or drop_newest");
    worker_ = std::jthread([this] { Run(); });
  }

  ImageSinkSubmission Submit(std::filesystem::path path, EncodedImage image,
                             std::string_view sample_key,
                             std::stop_token stop) {
    ValidateImage(path, image);
    const auto bytes = image.bytes.size();
    if (bytes > options_.max_item_bytes || bytes > options_.max_pending_bytes)
      throw ResourceExhausted("Encoded image exceeds output sink capacity",
                              ErrorDetails{.component = std::string(kComponent),
                                           .operation = "submit.bytes",
                                           .expected = std::to_string(std::min(
                                               options_.max_item_bytes,
                                               options_.max_pending_bytes)),
                                           .actual = std::to_string(bytes)});
    const bool selected = SelectSample(sample_key, options_.sample_every);
    std::unique_lock lock(mutex_);
    if (closed_) Invalid("submit.state", "open sink");
    if (first_error_ && !options_.continue_after_error)
      std::rethrow_exception(first_error_);
    ++stats_.submitted;
    if (!selected) {
      ++stats_.sampled_out;
      return {.disposition = ImageSinkDisposition::sampled_out,
              .completion = {}};
    }
    const auto capacity = [&] {
      return stats_.pending_items < options_.max_pending_items &&
             bytes <= options_.max_pending_bytes - stats_.pending_bytes;
    };
    if (!capacity() &&
        options_.overflow == ImageSinkOverflowPolicy::drop_newest) {
      ++stats_.dropped;
      return {.disposition = ImageSinkDisposition::dropped, .completion = {}};
    }
    if (!capacity()) {
      const bool ready = changed_.wait(lock, stop, [&] {
        return capacity() || closed_ ||
               (first_error_ && !options_.continue_after_error);
      });
      if (!ready || stop.stop_requested())
        throw Cancelled("Image output admission cancelled",
                        ErrorDetails{.component = std::string(kComponent),
                                     .operation = "submit.wait"});
      if (closed_) Invalid("submit.state", "open sink");
      if (first_error_ && !options_.continue_after_error)
        std::rethrow_exception(first_error_);
    }
    if (stop.stop_requested())
      throw Cancelled("Image output admission cancelled",
                      ErrorDetails{.component = std::string(kComponent),
                                   .operation = "submit.wait"});
    ImageResourceLease queue_lease;
    if (options_.resource_budget)
      queue_lease = options_.resource_budget->reserve(
          ImageResourceKind::encoded_queue_bytes, bytes);
    Work work;
    work.sequence = stats_.queued + 1;
    work.path = std::move(path);
    work.image = std::move(image);
    work.queue_lease = std::move(queue_lease);
    auto completion = work.promise.get_future();
    queue_.push_back(std::move(work));
    ++stats_.queued;
    ++stats_.pending_items;
    stats_.pending_bytes += bytes;
    changed_.notify_all();
    return {.disposition = ImageSinkDisposition::queued,
            .completion = std::move(completion)};
  }

  void Flush(std::stop_token stop) {
    std::unique_lock lock(mutex_);
    const auto target = stats_.queued;
    const bool settled =
        changed_.wait(lock, stop, [&] { return settled_ >= target; });
    if (!settled || stop.stop_requested())
      throw Cancelled("Image output flush cancelled",
                      ErrorDetails{.component = std::string(kComponent),
                                   .operation = "flush.wait"});
    if (first_failed_sequence_ != 0 && first_failed_sequence_ <= target)
      std::rethrow_exception(first_error_);
  }

  void Close() {
    {
      std::scoped_lock lock(mutex_);
      closed_ = true;
      changed_.notify_all();
    }
    std::call_once(join_once_, [this] { worker_.join(); });
    std::scoped_lock lock(mutex_);
    if (first_error_) std::rethrow_exception(first_error_);
  }

  ImageOutputSinkStats Stats() const {
    std::scoped_lock lock(mutex_);
    return stats_;
  }

  std::vector<ImageSinkFailure> Failures() const {
    std::scoped_lock lock(mutex_);
    return failures_;
  }

 private:
  struct Work {
    std::uint64_t sequence = 0;
    std::filesystem::path path;
    EncodedImage image;
    ImageResourceLease queue_lease;
    std::promise<ImageSinkWriteResult> promise;
  };

  ImageSinkWriteResult Publish(const Work& work) {
    const auto bytes = work.image.bytes.size();
    ImageResourceLease spool_lease;
    if (options_.resource_budget)
      spool_lease = options_.resource_budget->reserve(
          ImageResourceKind::disk_spool_bytes, bytes);
    const auto digest = sha256_hex(work.image.bytes);
    StagedFile staged(work.path);
    staged.Write(work.image.bytes);
    staged.Publish(options_.replace_existing);
    return {.path = work.path, .bytes = bytes, .sha256 = digest};
  }

  void Run() {
    for (;;) {
      Work work;
      std::exception_ptr skipped_error;
      {
        std::unique_lock lock(mutex_);
        changed_.wait(lock, [&] { return closed_ || !queue_.empty(); });
        if (queue_.empty()) return;
        work = std::move(queue_.front());
        queue_.pop_front();
        if (first_error_ && !options_.continue_after_error)
          skipped_error = first_error_;
      }
      std::exception_ptr error = skipped_error;
      if (!error) {
        try {
          work.promise.set_value(Publish(work));
        } catch (...) {
          error = std::current_exception();
        }
      }
      if (error) work.promise.set_exception(error);
      const auto bytes = work.image.bytes.size();
      work.queue_lease.release();
      work.image.output_lease.release();
      std::vector<std::byte>().swap(work.image.bytes);
      {
        std::scoped_lock lock(mutex_);
        if (error) {
          ++stats_.failed;
          if (!first_error_) {
            first_error_ = error;
            first_failed_sequence_ = work.sequence;
          }
          if (failures_.size() < options_.max_reported_failures) {
            try {
              try {
                std::rethrow_exception(error);
              } catch (const std::exception& cause) {
                failures_.push_back({work.path, cause.what()});
              } catch (...) {
                failures_.push_back({work.path, "unknown write failure"});
              }
            } catch (...) {
              // Reporting must not terminate the writer or strand its queue.
            }
          }
        } else {
          ++stats_.completed;
        }
        ++settled_;
        --stats_.pending_items;
        stats_.pending_bytes -= bytes;
        changed_.notify_all();
      }
    }
  }

  ImageOutputSinkOptions options_;
  mutable std::mutex mutex_;
  std::condition_variable_any changed_;
  std::deque<Work> queue_;
  std::jthread worker_;
  std::once_flag join_once_;
  ImageOutputSinkStats stats_;
  std::vector<ImageSinkFailure> failures_;
  std::exception_ptr first_error_;
  std::uint64_t first_failed_sequence_ = 0;
  std::uint64_t settled_ = 0;
  bool closed_ = false;
};

ImageOutputSink::ImageOutputSink(ImageOutputSinkOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

ImageOutputSink::~ImageOutputSink() {
  try {
    impl_->Close();
  } catch (...) {
  }
}

ImageSinkSubmission ImageOutputSink::submit(std::filesystem::path path,
                                            EncodedImage image,
                                            std::string_view sample_key,
                                            std::stop_token stop) {
  return impl_->Submit(std::move(path), std::move(image), sample_key, stop);
}

void ImageOutputSink::flush(std::stop_token stop) { impl_->Flush(stop); }
void ImageOutputSink::close() { impl_->Close(); }
ImageOutputSinkStats ImageOutputSink::stats() const { return impl_->Stats(); }
std::vector<ImageSinkFailure> ImageOutputSink::failures() const {
  return impl_->Failures();
}

}  // namespace jyppx::liuding::data::image
