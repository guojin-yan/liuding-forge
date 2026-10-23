// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

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
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/syscall.h>
#endif
extern char** environ;
#endif

#include "sidecar_process.hpp"

#include <jyppx/liuding/foundation/error.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstring>
#include <cwchar>
#include <limits>
#include <map>
#include <random>
#include <utility>

namespace jyppx::liuding::cli {
namespace {

using Clock = std::chrono::steady_clock;
using Milliseconds = std::chrono::milliseconds;
constexpr auto kPollInterval = Milliseconds(5);
constexpr std::size_t kChunkBytes = 4096U;

ErrorDetails details(std::string operation) {
  return {.component = "cli.sidecar_process",
          .operation = std::move(operation),
          .expected = "one bounded reply from the fixed supervised child",
          .actual = "child and launch data redacted",
          .hint =
              "Create a new transport after failure and replay the original "
              "durable observations; external delivery remains at least once."};
}

[[noreturn]] void io_failure(const char* operation) {
  throw IoError("Sidecar process I/O failed", details(operation));
}

[[noreturn]] void protocol_failure(const char* operation) {
  throw SerializationError("Sidecar process framing failed",
                           details(operation));
}

[[noreturn]] void launch_failure(const char* operation) {
  throw BackendError("Sidecar process launch or execution failed",
                     details(operation));
}

[[noreturn]] void capacity_failure(const char* operation) {
  throw ResourceExhausted("Sidecar process byte quota exceeded",
                          details(operation));
}

bool valid_utf8(std::string_view value) noexcept {
  std::size_t offset = 0U;
  while (offset < value.size()) {
    const auto first = static_cast<unsigned char>(value[offset++]);
    if (first < 0x80U) {
      if (first == 0U) return false;
      continue;
    }
    unsigned count = 0U;
    std::uint32_t scalar = 0U;
    std::uint32_t minimum = 0U;
    if (first >= 0xc2U && first <= 0xdfU) {
      count = 1U;
      scalar = first & 0x1fU;
      minimum = 0x80U;
    } else if (first >= 0xe0U && first <= 0xefU) {
      count = 2U;
      scalar = first & 0x0fU;
      minimum = 0x800U;
    } else if (first >= 0xf0U && first <= 0xf4U) {
      count = 3U;
      scalar = first & 0x07U;
      minimum = 0x10000U;
    } else {
      return false;
    }
    if (count > value.size() - offset) return false;
    for (unsigned index = 0U; index < count; ++index) {
      const auto next = static_cast<unsigned char>(value[offset++]);
      if ((next & 0xc0U) != 0x80U) return false;
      scalar = (scalar << 6U) | (next & 0x3fU);
    }
    if (scalar < minimum || scalar > 0x10ffffU ||
        (scalar >= 0xd800U && scalar <= 0xdfffU))
      return false;
  }
  return true;
}

void launch_text(std::string_view value) {
  if (!valid_utf8(value)) {
    throw InvalidArgument("Sidecar launch text must be NUL-free UTF-8",
                          details("launch.utf8"));
  }
}

void charge(std::size_t& current, std::size_t count, std::size_t maximum) {
  if (count > maximum - std::min(current, maximum))
    capacity_failure("launch.bytes");
  current += count;
}

Clock::time_point bounded_deadline(TelemetryDeadline supplied,
                                   Milliseconds maximum) {
  const auto configured = Clock::now() + maximum;
  return supplied ? std::min(configured, *supplied) : configured;
}

int poll_milliseconds(Clock::time_point deadline) {
  const auto remaining = deadline - Clock::now();
  if (remaining <= Clock::duration::zero()) return 0;
  const auto rounded = std::chrono::ceil<Milliseconds>(remaining);
  return static_cast<int>(std::min(rounded, kPollInterval).count());
}

void validate_options(const SidecarProcessOptions& options) {
  const auto bounded_time = [](Milliseconds value, Milliseconds maximum) {
    return value.count() > 0 && value <= maximum;
  };
  constexpr auto kMaximumBytes = std::size_t{1U} << 30U;
  if (options.executable.empty() || !options.executable.is_absolute() ||
      options.working_directory.empty() ||
      !options.working_directory.is_absolute() ||
      options.max_request_bytes == 0U ||
      options.max_request_bytes > kMaximumBytes ||
      options.max_reply_bytes == 0U ||
      options.max_reply_bytes > kMaximumBytes ||
      options.max_stderr_bytes == 0U ||
      options.max_stderr_bytes > kMaximumBytes ||
      options.max_launch_bytes == 0U ||
      options.max_launch_bytes > kMaximumBytes ||
      !bounded_time(options.operation_timeout, Milliseconds(600000)) ||
      !bounded_time(options.startup_timeout, Milliseconds(600000)) ||
      !bounded_time(options.cleanup_timeout, Milliseconds(10000)) ||
      !bounded_time(options.reply_quiet_period, Milliseconds(100))) {
    throw InvalidArgument(
        "Invalid sidecar process paths, quotas or time bounds",
        details("launch.options"));
  }
  std::error_code error;
  if (!std::filesystem::is_regular_file(options.executable, error) || error) {
    throw InvalidArgument("Sidecar executable must be an existing regular file",
                          details("launch.executable"));
  }
  if (!std::filesystem::is_directory(options.working_directory, error) ||
      error) {
    throw InvalidArgument("Sidecar cwd must be an existing isolated directory",
                          details("launch.cwd"));
  }
  std::size_t bytes = 0U;
  for (const auto& argument : options.arguments) {
    launch_text(argument);
    charge(bytes, argument.size() + 1U, options.max_launch_bytes);
  }
  for (const auto& [name, value] : options.environment) {
    launch_text(name);
    launch_text(value);
    if (name.empty() || name.find('=') != std::string::npos) {
      throw InvalidArgument("Invalid sidecar child environment key",
                            details("launch.environment"));
    }
    charge(bytes, name.size() + value.size() + 2U, options.max_launch_bytes);
  }
}

struct PumpResult {
  std::array<char, kChunkBytes> output{};
  std::array<char, kChunkBytes> diagnostic{};
  std::size_t output_size = 0U;
  std::size_t diagnostic_size = 0U;
  bool stdout_eof = false;
  bool process_exited = false;
  bool progressed = false;
};

#ifdef _WIN32

class Handle {
 public:
  explicit Handle(HANDLE value = nullptr) noexcept : value_(value) {}
  ~Handle() { reset(); }
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  HANDLE get() const noexcept { return value_; }
  explicit operator bool() const noexcept {
    return value_ != nullptr && value_ != INVALID_HANDLE_VALUE;
  }
  void reset(HANDLE value = nullptr) noexcept {
    if (*this) CloseHandle(value_);
    value_ = value;
  }

 private:
  HANDLE value_ = nullptr;
};

std::wstring wide(std::string_view value) {
  launch_text(value);
  if (value.empty()) return {};
  if (value.size() > static_cast<std::size_t>(INT_MAX))
    capacity_failure("launch.utf16");
  const auto count =
      MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                          static_cast<int>(value.size()), nullptr, 0);
  if (count == 0) launch_failure("launch.utf16");
  std::wstring result(static_cast<std::size_t>(count), L'\0');
  if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                          static_cast<int>(value.size()), result.data(),
                          count) != count) {
    launch_failure("launch.utf16");
  }
  return result;
}

void validate_native(std::wstring_view value) {
  if (value.empty() || value.size() > 32766U ||
      value.find(L'\0') != std::wstring::npos ||
      WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
                          static_cast<int>(value.size()), nullptr, 0, nullptr,
                          nullptr) == 0) {
    throw InvalidArgument("Invalid sidecar native Unicode path",
                          details("launch.path"));
  }
}

std::wstring quote_argument(std::wstring_view value) {
  // Microsoft CRT argv rules: double trailing backslashes and escape quotes
  // with 2*n+1 backslashes. Shell metacharacters have no special treatment.
  std::wstring result(1U, L'"');
  std::size_t slashes = 0U;
  for (const auto character : value) {
    if (character == L'\\') {
      ++slashes;
    } else if (character == L'"') {
      result.append(slashes * 2U + 1U, L'\\');
      result.push_back(L'"');
      slashes = 0U;
    } else {
      result.append(slashes, L'\\');
      result.push_back(character);
      slashes = 0U;
    }
  }
  result.append(slashes * 2U, L'\\');
  result.push_back(L'"');
  return result;
}

struct EnvironmentOrder {
  bool operator()(const std::wstring& left,
                  const std::wstring& right) const noexcept {
    return CompareStringOrdinal(left.data(), static_cast<int>(left.size()),
                                right.data(), static_cast<int>(right.size()),
                                TRUE) == CSTR_LESS_THAN;
  }
};

std::vector<wchar_t> child_environment(const SidecarProcessOptions& options) {
  std::map<std::wstring, std::wstring, EnvironmentOrder> values;
  if (options.inherit_environment) {
    auto* block = GetEnvironmentStringsW();
    if (!block) launch_failure("launch.environment.snapshot");
    try {
      for (const auto* position = block; *position;
           position += std::wcslen(position) + 1U) {
        std::wstring entry(position);
        const auto separator =
            entry.find(L'=', entry.front() == L'=' ? 1U : 0U);
        if (separator != std::wstring::npos) {
          values[entry.substr(0U, separator)] = entry.substr(separator + 1U);
        }
      }
    } catch (...) {
      FreeEnvironmentStringsW(block);
      throw;
    }
    FreeEnvironmentStringsW(block);
  }
  for (const auto& [name, value] : options.environment)
    values[wide(name)] = wide(value);
  std::vector<wchar_t> result;
  std::size_t bytes = 0U;
  for (const auto& [name, value] : values) {
    charge(bytes, (name.size() + value.size() + 2U) * sizeof(wchar_t),
           options.max_launch_bytes);
    result.insert(result.end(), name.begin(), name.end());
    result.push_back(L'=');
    result.insert(result.end(), value.begin(), value.end());
    result.push_back(L'\0');
  }
  result.push_back(L'\0');
  if (result.size() == 1U) result.push_back(L'\0');
  return result;
}

struct WinPipe {
  Handle server;
  Handle child;
  Handle event;
  OVERLAPPED operation{};
  std::array<char, kChunkBytes> buffer{};
  bool pending = false;
  bool eof = false;

  void create(bool parent_reads) {
    static std::atomic<unsigned long long> serial{0U};
    std::random_device random;
    const auto name = L"\\\\.\\pipe\\jyppx.telemetry." +
                      std::to_wstring(GetCurrentProcessId()) + L"." +
                      std::to_wstring(serial.fetch_add(1U)) + L"." +
                      std::to_wstring(random()) + L"." +
                      std::to_wstring(random());
    const auto access =
        (parent_reads ? PIPE_ACCESS_INBOUND : PIPE_ACCESS_OUTBOUND) |
        FILE_FLAG_OVERLAPPED | FILE_FLAG_FIRST_PIPE_INSTANCE;
    server.reset(CreateNamedPipeW(name.c_str(), access,
                                  PIPE_TYPE_BYTE | PIPE_READMODE_BYTE |
                                      PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
                                  1U, 65536U, 65536U, 0U, nullptr));
    if (!server) launch_failure("launch.pipe.server");
    event.reset(CreateEventW(nullptr, TRUE, FALSE, nullptr));
    if (!event) launch_failure("launch.pipe.event");
    operation.hEvent = event.get();
    if (!ConnectNamedPipe(server.get(), &operation)) {
      const auto error = GetLastError();
      if (error == ERROR_IO_PENDING)
        pending = true;
      else if (error != ERROR_PIPE_CONNECTED)
        launch_failure("launch.pipe.connect");
    }
    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    child.reset(CreateFileW(
        name.c_str(), parent_reads ? GENERIC_WRITE : GENERIC_READ, 0U,
        &security, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr));
    if (!child) launch_failure("launch.pipe.client");
    if (pending) {
      DWORD ignored = 0U;
      if (!GetOverlappedResult(server.get(), &operation, &ignored, TRUE)) {
        launch_failure("launch.pipe.connect.complete");
      }
      pending = false;
    }
    ResetEvent(event.get());
  }

  void begin_read() {
    if (pending || eof) return;
    operation = {};
    operation.hEvent = event.get();
    ResetEvent(event.get());
    DWORD count = 0U;
    const auto success =
        ReadFile(server.get(), buffer.data(), static_cast<DWORD>(buffer.size()),
                 &count, &operation);
    if (!success) {
      const auto error = GetLastError();
      if (error == ERROR_BROKEN_PIPE || error == ERROR_PIPE_NOT_CONNECTED) {
        eof = true;
        return;
      }
      if (error != ERROR_IO_PENDING) io_failure("pipe.read.begin");
    }
    pending = true;
  }

  std::size_t collect_read(std::array<char, kChunkBytes>& destination) {
    if (!pending) return 0U;
    DWORD count = 0U;
    if (!GetOverlappedResult(server.get(), &operation, &count, FALSE)) {
      const auto error = GetLastError();
      if (error == ERROR_IO_INCOMPLETE) return 0U;
      pending = false;
      if (error == ERROR_BROKEN_PIPE || error == ERROR_PIPE_NOT_CONNECTED) {
        eof = true;
        return 0U;
      }
      io_failure("pipe.read.complete");
    }
    pending = false;
    if (count == 0U) {
      eof = true;
      return 0U;
    }
    std::copy_n(buffer.begin(), count, destination.begin());
    return static_cast<std::size_t>(count);
  }

  void cancel() noexcept {
    if (pending && server) {
      CancelIoEx(server.get(), &operation);
      DWORD ignored = 0U;
      GetOverlappedResult(server.get(), &operation, &ignored, TRUE);
      pending = false;
    }
    child.reset();
    server.reset();
    event.reset();
  }
};

class AttributeList {
 public:
  AttributeList() {
    SIZE_T size = 0U;
    InitializeProcThreadAttributeList(nullptr, 1U, 0U, &size);
    if (size == 0U) launch_failure("launch.handle_list.size");
    bytes_.resize(size);
    value_ = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(bytes_.data());
    if (!InitializeProcThreadAttributeList(value_, 1U, 0U, &size)) {
      value_ = nullptr;
      launch_failure("launch.handle_list.initialize");
    }
  }
  ~AttributeList() {
    if (value_) DeleteProcThreadAttributeList(value_);
  }
  LPPROC_THREAD_ATTRIBUTE_LIST get() const noexcept { return value_; }

 private:
  std::vector<std::byte> bytes_;
  LPPROC_THREAD_ATTRIBUTE_LIST value_ = nullptr;
};

class NativeProcess {
 public:
  NativeProcess(const SidecarProcessOptions& options, std::stop_token stop,
                Clock::time_point deadline)
      : cleanup_timeout_(options.cleanup_timeout) {
    try {
      check_telemetry_operation(stop, deadline);
      const auto executable = options.executable.native();
      const auto cwd = options.working_directory.native();
      validate_native(executable);
      validate_native(cwd);
      auto command = quote_argument(executable);
      for (const auto& argument : options.arguments) {
        command.push_back(L' ');
        command += quote_argument(wide(argument));
        if (command.size() > 32766U) capacity_failure("launch.command_line");
      }
      if (command.size() > 32766U) capacity_failure("launch.command_line");
      if (command.size() * sizeof(wchar_t) > options.max_launch_bytes)
        capacity_failure("launch.command_line");
      auto environment = child_environment(options);
      const auto command_bytes = (command.size() + 1U) * sizeof(wchar_t);
      std::size_t launch_bytes = 0U;
      charge(launch_bytes, command_bytes, options.max_launch_bytes);
      charge(launch_bytes, environment.size() * sizeof(wchar_t),
             options.max_launch_bytes);
      input_.create(false);
      output_.create(true);
      diagnostic_.create(true);
      job_.reset(CreateJobObjectW(nullptr, nullptr));
      if (!job_) launch_failure("launch.job.create");
      JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
      limits.BasicLimitInformation.LimitFlags =
          JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
      if (!SetInformationJobObject(job_.get(),
                                   JobObjectExtendedLimitInformation, &limits,
                                   sizeof(limits))) {
        launch_failure("launch.job.limits");
      }
      AttributeList attributes;
      std::array<HANDLE, 3U> inherited{input_.child.get(), output_.child.get(),
                                       diagnostic_.child.get()};
      if (!UpdateProcThreadAttribute(
              attributes.get(), 0U, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
              inherited.data(), sizeof(inherited), nullptr, nullptr)) {
        launch_failure("launch.handle_list.set");
      }
      STARTUPINFOEXW startup{};
      startup.StartupInfo.cb = static_cast<DWORD>(sizeof(startup));
      startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
      startup.StartupInfo.wShowWindow = SW_HIDE;
      startup.StartupInfo.hStdInput = inherited[0U];
      startup.StartupInfo.hStdOutput = inherited[1U];
      startup.StartupInfo.hStdError = inherited[2U];
      startup.lpAttributeList = attributes.get();
      PROCESS_INFORMATION process{};
      check_telemetry_operation(stop, deadline);
      if (!CreateProcessW(
              executable.c_str(), command.data(), nullptr, nullptr, TRUE,
              CREATE_SUSPENDED | CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT |
                  EXTENDED_STARTUPINFO_PRESENT,
              environment.data(), cwd.c_str(), &startup.StartupInfo,
              &process)) {
        launch_failure("launch.create_process");
      }
      process_.reset(process.hProcess);
      Handle thread(process.hThread);
      if (!AssignProcessToJobObject(job_.get(), process_.get())) {
        TerminateProcess(process_.get(), 1U);
        WaitForSingleObject(process_.get(),
                            static_cast<DWORD>(cleanup_timeout_.count()));
        launch_failure("launch.job.assign");
      }
      assigned_ = true;
      check_telemetry_operation(stop, deadline);
      if (ResumeThread(thread.get()) == static_cast<DWORD>(-1))
        launch_failure("launch.resume");
      input_.child.reset();
      output_.child.reset();
      diagnostic_.child.reset();
      output_.begin_read();
      diagnostic_.begin_read();
      check_telemetry_operation(stop, deadline);
    } catch (...) {
      terminate();
      throw;
    }
  }

  ~NativeProcess() { terminate(); }

  PumpResult pump(std::string_view input, std::size_t& position,
                  int timeout_ms) {
    PumpResult result;
    output_.begin_read();
    diagnostic_.begin_read();
    if (!input_.pending && position < input.size()) {
      const auto chunk = std::min(input.size() - position, std::size_t{65536U});
      input_.operation = {};
      input_.operation.hEvent = input_.event.get();
      ResetEvent(input_.event.get());
      DWORD count = 0U;
      if (!WriteFile(input_.server.get(), input.data() + position,
                     static_cast<DWORD>(chunk), &count, &input_.operation) &&
          GetLastError() != ERROR_IO_PENDING) {
        io_failure("pipe.write.begin");
      }
      input_.pending = true;
    }
    const auto collect = [&] {
      result.output_size = output_.collect_read(result.output);
      result.diagnostic_size = diagnostic_.collect_read(result.diagnostic);
      result.stdout_eof = output_.eof;
      if (input_.pending) {
        DWORD count = 0U;
        if (GetOverlappedResult(input_.server.get(), &input_.operation, &count,
                                FALSE)) {
          input_.pending = false;
          if (count == 0U) io_failure("pipe.write.zero");
          position += static_cast<std::size_t>(count);
          result.progressed = true;
        } else if (GetLastError() != ERROR_IO_INCOMPLETE) {
          input_.pending = false;
          io_failure("pipe.write.complete");
        }
      }
      const auto process_state = WaitForSingleObject(process_.get(), 0U);
      if (process_state == WAIT_FAILED) io_failure("process.poll");
      result.process_exited = process_state == WAIT_OBJECT_0;
      result.progressed = result.progressed || result.output_size != 0U ||
                          result.diagnostic_size != 0U;
    };
    collect();
    if (!result.progressed && !result.stdout_eof && !result.process_exited &&
        timeout_ms > 0) {
      std::array<HANDLE, 4U> waiting{};
      DWORD count = 0U;
      if (output_.pending) waiting[count++] = output_.event.get();
      if (diagnostic_.pending) waiting[count++] = diagnostic_.event.get();
      if (input_.pending) waiting[count++] = input_.event.get();
      waiting[count++] = process_.get();
      if (WaitForMultipleObjects(count, waiting.data(), FALSE,
                                 static_cast<DWORD>(timeout_ms)) ==
          WAIT_FAILED) {
        io_failure("pipe.wait");
      }
      collect();
    }
    return result;
  }

  void terminate() noexcept {
    const auto deadline = Clock::now() + cleanup_timeout_;
    if (job_ && assigned_)
      TerminateJobObject(job_.get(), 1U);
    else if (process_)
      TerminateProcess(process_.get(), 1U);
    // Cancel/drain pending kernel I/O before freeing its OVERLAPPED buffers.
    // Process termination ensures no writer is waiting for the child to read.
    if (input_.pending && input_.server)
      CancelIoEx(input_.server.get(), &input_.operation);
    if (output_.pending && output_.server)
      CancelIoEx(output_.server.get(), &output_.operation);
    if (diagnostic_.pending && diagnostic_.server)
      CancelIoEx(diagnostic_.server.get(), &diagnostic_.operation);
    input_.cancel();
    output_.cancel();
    diagnostic_.cancel();
    if (process_) {
      const auto remaining = std::max(
          Milliseconds::zero(),
          std::chrono::duration_cast<Milliseconds>(deadline - Clock::now()));
      WaitForSingleObject(process_.get(),
                          static_cast<DWORD>(remaining.count()));
    }
    if (job_ && assigned_) {
      JOBOBJECT_BASIC_ACCOUNTING_INFORMATION accounting{};
      while (Clock::now() < deadline &&
             QueryInformationJobObject(
                 job_.get(), JobObjectBasicAccountingInformation, &accounting,
                 sizeof(accounting), nullptr) &&
             accounting.ActiveProcesses != 0U) {
        Sleep(1U);
      }
    }
    job_.reset();
    process_.reset();
    assigned_ = false;
  }

 private:
  Milliseconds cleanup_timeout_;
  Handle job_;
  Handle process_;
  WinPipe input_;
  WinPipe output_;
  WinPipe diagnostic_;
  bool assigned_ = false;
};

#else

class Descriptor {
 public:
  explicit Descriptor(int value = -1) noexcept : value_(value) {}
  ~Descriptor() { reset(); }
  Descriptor(const Descriptor&) = delete;
  Descriptor& operator=(const Descriptor&) = delete;
  int get() const noexcept { return value_; }
  explicit operator bool() const noexcept { return value_ >= 0; }
  void reset(int value = -1) noexcept {
    if (value_ >= 0) ::close(value_);
    value_ = value;
  }

 private:
  int value_ = -1;
};

void nonblocking(int descriptor) {
  const auto flags = fcntl(descriptor, F_GETFL);
  if (flags < 0 || fcntl(descriptor, F_SETFL, flags | O_NONBLOCK) < 0)
    launch_failure("launch.nonblocking");
}

void make_pipe(Descriptor& reader, Descriptor& writer) {
  std::array<int, 2U> descriptors{-1, -1};
#if defined(__linux__)
  if (pipe2(descriptors.data(), O_CLOEXEC) != 0) launch_failure("launch.pipe");
#else
  if (pipe(descriptors.data()) != 0) launch_failure("launch.pipe");
#endif
  reader.reset(descriptors[0U]);
  writer.reset(descriptors[1U]);
  for (auto* handle : {&reader, &writer}) {
    if (handle->get() < 3) {
      const auto relocated = fcntl(handle->get(), F_DUPFD_CLOEXEC, 3);
      if (relocated < 0) launch_failure("launch.pipe.relocate");
      handle->reset(relocated);
    }
    if (fcntl(handle->get(), F_SETFD, FD_CLOEXEC) < 0)
      launch_failure("launch.pipe.cloexec");
  }
}

// Block SIGPIPE only in the current calling thread. Do not change process-wide
// signal disposition, consume an older pending signal, or expose a fatal EPIPE.
ssize_t write_without_sigpipe(int descriptor, const char* bytes,
                              std::size_t count) {
  sigset_t blocked{};
  sigset_t original{};
  sigemptyset(&blocked);
  sigaddset(&blocked, SIGPIPE);
  if (pthread_sigmask(SIG_BLOCK, &blocked, &original) != 0)
    io_failure("pipe.signal_mask");
  sigset_t pending{};
  const bool had_pending =
      sigpending(&pending) == 0 && sigismember(&pending, SIGPIPE) == 1;
  const auto result = ::write(descriptor, bytes, count);
  const auto saved_error = errno;
  if (result < 0 && saved_error == EPIPE && !had_pending) {
    timespec immediate{};
    while (sigtimedwait(&blocked, nullptr, &immediate) < 0 && errno == EINTR) {
    }
  }
  const auto restore = pthread_sigmask(SIG_SETMASK, &original, nullptr);
  errno = saved_error;
  if (restore != 0) io_failure("pipe.signal_restore");
  return result;
}

class NativeProcess {
 public:
  NativeProcess(const SidecarProcessOptions& options, std::stop_token stop,
                Clock::time_point deadline)
      : cleanup_timeout_(options.cleanup_timeout) {
    try {
      check_telemetry_operation(stop, deadline);
      const auto executable = options.executable.native();
      const auto cwd = options.working_directory.native();
      launch_text(executable);
      launch_text(cwd);
      std::vector<std::string> arguments{executable};
      arguments.insert(arguments.end(), options.arguments.begin(),
                       options.arguments.end());
      std::map<std::string, std::string> environment;
      if (options.inherit_environment) {
        for (auto** position = environ; position && *position; ++position) {
          const std::string entry(*position);
          const auto separator = entry.find('=');
          if (separator != std::string::npos && separator != 0U) {
            environment[entry.substr(0U, separator)] =
                entry.substr(separator + 1U);
          }
        }
      }
      for (const auto& [name, value] : options.environment)
        environment[name] = value;
      std::vector<std::string> environment_storage;
      std::size_t launch_bytes = 0U;
      for (const auto& argument : arguments)
        charge(launch_bytes, argument.size() + 1U, options.max_launch_bytes);
      for (const auto& [name, value] : environment) {
        charge(launch_bytes, name.size() + value.size() + 2U,
               options.max_launch_bytes);
        environment_storage.push_back(name + "=" + value);
      }
      std::vector<char*> argv;
      for (auto& argument : arguments) argv.push_back(argument.data());
      argv.push_back(nullptr);
      std::vector<char*> envp;
      for (auto& entry : environment_storage) envp.push_back(entry.data());
      envp.push_back(nullptr);
      Descriptor child_input, child_output, child_diagnostic, status_reader,
          status_writer;
      make_pipe(child_input, input_);
      make_pipe(output_, child_output);
      make_pipe(diagnostic_, child_diagnostic);
      make_pipe(status_reader, status_writer);
      nonblocking(input_.get());
      nonblocking(output_.get());
      nonblocking(diagnostic_.get());
      nonblocking(status_reader.get());
      rlimit descriptor_limit{};
      if (getrlimit(RLIMIT_NOFILE, &descriptor_limit) != 0)
        launch_failure("launch.fd_limit");
      const auto fallback_limit = sysconf(_SC_OPEN_MAX);
      const auto limit =
          descriptor_limit.rlim_cur == RLIM_INFINITY
              ? (fallback_limit > 0 ? static_cast<rlim_t>(fallback_limit)
                                    : static_cast<rlim_t>(INT_MAX))
              : descriptor_limit.rlim_cur;
      const auto max_descriptor = std::min(limit, static_cast<rlim_t>(INT_MAX));
      const auto in = child_input.get(), out = child_output.get(),
                 err = child_diagnostic.get();
      const auto status = status_writer.get();
      const auto status_reader_descriptor = status_reader.get();
      const auto input_parent = input_.get(), output_parent = output_.get(),
                 diagnostic_parent = diagnostic_.get();
      struct sigaction child_policy {};
      if (sigaction(SIGCHLD, nullptr, &child_policy) != 0) {
        launch_failure("launch.sigchld_policy");
      }
      if (child_policy.sa_handler == SIG_IGN ||
          (child_policy.sa_flags & SA_NOCLDWAIT) != 0) {
        throw CompatibilityError(
            "Sidecar ownership requires a reapable child process",
            details("launch.sigchld_policy"));
      }
      check_telemetry_operation(stop, deadline);
      pid_ = fork();
      if (pid_ < 0) launch_failure("launch.fork");
      if (pid_ == 0) {
        // Only async-signal-safe system calls after fork. All strings, argv,
        // environment storage and fd bounds were materialized by the parent.
        // Parent-facing pipe ends must not survive in the fork child even
        // briefly: a stale stdin writer could otherwise hide a launch error.
        ::close(status_reader_descriptor);
        ::close(input_parent);
        ::close(output_parent);
        ::close(diagnostic_parent);
        int failure = 0;
        if (setpgid(0, 0) != 0 || dup2(in, STDIN_FILENO) < 0 ||
            dup2(out, STDOUT_FILENO) < 0 || dup2(err, STDERR_FILENO) < 0 ||
            chdir(cwd.c_str()) != 0) {
          failure = errno;
        }
        if (failure == 0) {
          bool descriptors_closed = false;
#if defined(__linux__) && defined(SYS_close_range)
          // CLOSE_RANGE_CLOEXEC keeps the error pipe available on exec failure
          // while preventing inheritance of arbitrary parent descriptors.
          descriptors_closed = syscall(SYS_close_range, 3U, ~0U, 4U) == 0;
#endif
          if (!descriptors_closed) {
            for (rlim_t descriptor = 3U; descriptor < max_descriptor;
                 ++descriptor) {
              if (static_cast<int>(descriptor) != status)
                ::close(static_cast<int>(descriptor));
            }
          }
          execve(executable.c_str(), argv.data(), envp.data());
          failure = errno;
        }
        const auto* bytes = reinterpret_cast<const char*>(&failure);
        std::size_t written = 0U;
        while (written < sizeof(failure)) {
          const auto count =
              ::write(status, bytes + written, sizeof(failure) - written);
          if (count > 0)
            written += static_cast<std::size_t>(count);
          else if (count < 0 && errno == EINTR)
            continue;
          else
            break;
        }
        _exit(127);
      }
      // The child also performs setpgid before exec; parent placement closes
      // the cancellation window before the child's first instruction runs.
      if (setpgid(pid_, pid_) != 0 && errno != EACCES && errno != ESRCH)
        launch_failure("launch.process_group");
      child_input.reset();
      child_output.reset();
      child_diagnostic.reset();
      status_writer.reset();
      std::array<char, sizeof(int)> failure{};
      std::size_t received = 0U;
      while (true) {
        check_telemetry_operation(stop, deadline);
        const auto count =
            ::read(status_reader.get(), failure.data() + received,
                   failure.size() - received);
        if (count == 0) {
          if (received != 0U) launch_failure("launch.exec_status.partial");
          break;
        }
        if (count > 0) {
          received += static_cast<std::size_t>(count);
          if (received == failure.size()) launch_failure("launch.execve");
          continue;
        }
        if (errno == EINTR) continue;
        if (errno != EAGAIN && errno != EWOULDBLOCK)
          launch_failure("launch.exec_status.read");
        pollfd waiting{status_reader.get(), POLLIN, 0};
        if (::poll(&waiting, 1U, poll_milliseconds(deadline)) < 0 &&
            errno != EINTR) {
          launch_failure("launch.exec_status.wait");
        }
      }
      check_telemetry_operation(stop, deadline);
    } catch (...) {
      terminate();
      throw;
    }
  }

  ~NativeProcess() { terminate(); }

  PumpResult pump(std::string_view input, std::size_t& position,
                  int timeout_ms) {
    PumpResult result;
    std::array<pollfd, 3U> waiting{
        {{output_.get(), POLLIN, 0},
         {diagnostic_.get(), POLLIN, 0},
         {position < input.size() ? input_.get() : -1, POLLOUT, 0}}};
    const auto ready =
        ::poll(waiting.data(), static_cast<nfds_t>(waiting.size()), timeout_ms);
    if (ready < 0 && errno != EINTR) io_failure("pipe.poll");
    const auto read_pipe = [&](Descriptor& descriptor, short events,
                               std::array<char, kChunkBytes>& bytes,
                               std::size_t& count) {
      if (!descriptor || (events & (POLLIN | POLLHUP | POLLERR)) == 0) return;
      const auto received =
          ::read(descriptor.get(), bytes.data(), bytes.size());
      if (received > 0) {
        count = static_cast<std::size_t>(received);
        result.progressed = true;
      } else if (received == 0)
        descriptor.reset();
      else if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK)
        io_failure("pipe.read");
    };
    if (ready >= 0) {
      for (const auto& descriptor : waiting) {
        if ((descriptor.revents & POLLNVAL) != 0)
          io_failure("pipe.invalid_descriptor");
      }
      read_pipe(output_, waiting[0U].revents, result.output,
                result.output_size);
      read_pipe(diagnostic_, waiting[1U].revents, result.diagnostic,
                result.diagnostic_size);
      if ((waiting[2U].revents & (POLLOUT | POLLHUP | POLLERR)) != 0) {
        const auto count = write_without_sigpipe(
            input_.get(), input.data() + position,
            std::min(input.size() - position, std::size_t{65536U}));
        if (count > 0) {
          position += static_cast<std::size_t>(count);
          result.progressed = true;
        } else if (count == 0 || (errno != EINTR && errno != EAGAIN &&
                                  errno != EWOULDBLOCK)) {
          io_failure("pipe.write");
        }
      }
    }
    // Observe exit without releasing the group leader's PID. Keeping the
    // zombie until termination prevents PID reuse before group-wide cleanup.
    siginfo_t status{};
    const auto observed = waitid(P_PID, static_cast<id_t>(pid_), &status,
                                 WEXITED | WNOHANG | WNOWAIT);
    if (observed < 0 && errno != EINTR) io_failure("process.poll");
    result.process_exited = observed == 0 && status.si_pid == pid_;
    result.stdout_eof = !output_;
    return result;
  }

  void terminate() noexcept {
    if (pid_ > 0) {
      // Kill the whole original group even if its leader has already exited.
      ::kill(-pid_, SIGKILL);
      if (!reaped_) ::kill(pid_, SIGKILL);
      const auto deadline = Clock::now() + cleanup_timeout_;
      while (!reaped_ && Clock::now() < deadline) {
        int status = 0;
        const auto result = waitpid(pid_, &status, WNOHANG);
        if (result == pid_ || (result < 0 && errno == ECHILD))
          reaped_ = true;
        else if (result < 0 && errno != EINTR)
          break;
        else
          ::poll(nullptr, 0U, 1);
      }
      // POSIX descendants are reaped by their actual parent or the system
      // reaper. Do not change the application's global subreaper/SIGCHLD
      // policy.
      pid_ = -1;
    }
    input_.reset();
    output_.reset();
    diagnostic_.reset();
  }

 private:
  Milliseconds cleanup_timeout_;
  pid_t pid_ = -1;
  bool reaped_ = false;
  Descriptor input_;
  Descriptor output_;
  Descriptor diagnostic_;
};

#endif

}  // namespace

class SidecarProcessTransport::Impl {
 public:
  Impl(SidecarProcessOptions configured, std::stop_token stop,
       TelemetryDeadline deadline)
      : options(std::move(configured)) {
    check_telemetry_operation(stop, deadline);
    const auto end = bounded_deadline(
        deadline, std::clamp(options.startup_timeout, Milliseconds(1),
                             Milliseconds(600000)));
    validate_options(options);
    check_telemetry_operation(stop, end);
    native = std::make_unique<NativeProcess>(options, stop, end);
  }

  void terminate() noexcept {
    poisoned = true;
    native.reset();
  }

  PumpResult pump(std::string_view input, std::size_t& position,
                  int timeout_ms) {
    auto result = native->pump(input, position, timeout_ms);
    const auto add = [](std::uint64_t& destination, std::size_t count) {
      if (count > std::numeric_limits<std::uint64_t>::max() - destination)
        capacity_failure("counter.overflow");
      destination += static_cast<std::uint64_t>(count);
    };
    add(counters.stdout_bytes, result.output_size);
    add(counters.stderr_bytes, result.diagnostic_size);
    if (counters.stderr_bytes > options.max_stderr_bytes)
      capacity_failure("stderr.bytes");
    if (result.process_exited) launch_failure("process.exited");
    if (result.stdout_eof) protocol_failure("stdout.eof");
    return result;
  }

  SidecarProcessOptions options;
  // Windows overlapped writes reference this storage until completion/cancel.
  // It outlives native handles even when an exception unwinds transact locals.
  std::string input_frame;
  std::unique_ptr<NativeProcess> native;
  SidecarProcessStats counters;
  bool poisoned = false;
};

SidecarProcessTransport::SidecarProcessTransport(SidecarProcessOptions options,
                                                 std::stop_token stop,
                                                 TelemetryDeadline deadline)
    : impl_(std::make_unique<Impl>(std::move(options), stop, deadline)) {}

SidecarProcessTransport::~SidecarProcessTransport() noexcept = default;

std::string SidecarProcessTransport::transact(std::string_view request_json,
                                              std::stop_token stop,
                                              TelemetryDeadline deadline) {
  if (impl_->poisoned)
    throw NotReady("Sidecar process requires a new transport",
                   details("process.poisoned"));
  try {
    const auto end =
        bounded_deadline(deadline, impl_->options.operation_timeout);
    check_telemetry_operation(stop, end);
    if (request_json.size() > impl_->options.max_request_bytes)
      capacity_failure("request.bytes");
    if (request_json.empty() || !valid_utf8(request_json) ||
        request_json.find_first_of("\r\n") != std::string_view::npos) {
      throw InvalidArgument("Sidecar request must be one NUL-free UTF-8 line",
                            details("request.frame"));
    }
    std::size_t drained = 0U;
    while (true) {
      check_telemetry_operation(stop, end);
      const auto idle = impl_->pump({}, drained, 0);
      if (idle.output_size != 0U) protocol_failure("stdout.unsolicited");
      if (!idle.progressed) break;
    }
    impl_->input_frame.assign(request_json);
    impl_->input_frame.push_back('\n');
    const auto& framed = impl_->input_frame;
    std::size_t written = 0U;
    std::string reply;
    bool complete = false;
    std::optional<Clock::time_point> quiet_until;
    while (true) {
      check_telemetry_operation(stop, end);
      const auto wait_until = quiet_until ? std::min(end, *quiet_until) : end;
      const auto activity =
          impl_->pump(complete ? std::string_view{} : std::string_view(framed),
                      written, poll_milliseconds(wait_until));
      if (activity.output_size != 0U) {
        if (complete || written != framed.size())
          protocol_failure("stdout.extra_or_early");
        if (activity.output_size >
            impl_->options.max_reply_bytes + 1U - reply.size()) {
          capacity_failure("reply.bytes");
        }
        reply.append(activity.output.data(), activity.output_size);
        const auto newline = reply.find('\n');
        if (newline != std::string::npos) {
          if (newline + 1U != reply.size()) protocol_failure("stdout.extra");
          reply.pop_back();
          if (reply.empty() || !valid_utf8(reply) ||
              reply.find('\r') != std::string::npos) {
            protocol_failure("reply.utf8_or_frame");
          }
          complete = true;
          quiet_until = Clock::now() + impl_->options.reply_quiet_period;
        } else if (reply.size() > impl_->options.max_reply_bytes) {
          capacity_failure("reply.bytes");
        }
      }
      check_telemetry_operation(stop, end);
      if (complete && Clock::now() >= *quiet_until) {
        if (impl_->counters.completed_requests ==
            std::numeric_limits<std::uint64_t>::max()) {
          capacity_failure("counter.requests");
        }
        ++impl_->counters.completed_requests;
        return reply;
      }
    }
  } catch (...) {
    impl_->terminate();
    throw;
  }
}

void SidecarProcessTransport::terminate() noexcept { impl_->terminate(); }
bool SidecarProcessTransport::poisoned() const noexcept {
  return impl_->poisoned;
}
SidecarProcessStats SidecarProcessTransport::stats() const noexcept {
  return impl_->counters;
}

}  // namespace jyppx::liuding::cli
