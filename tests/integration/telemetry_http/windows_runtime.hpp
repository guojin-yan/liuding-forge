// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_TEST_HTTP_WINDOWS_RUNTIME_HPP
#define JYPPX_LIUDING_TEST_HTTP_WINDOWS_RUNTIME_HPP

#ifdef _WIN32
#include "../../cli/windows_handle_diagnostics.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <rpc.h>
#include <stdexcept>
#include <string>
#include <windows.h>
#include <winsock2.h>

namespace jyppx::liuding::test {

// Test-process isolation only. RPC and Winsock retain an event and a duplicated
// thread handle per initialized worker. A newly scheduled OS worker otherwise
// changes the total process count even after all request resources are freed.
// Pin and initialize every default-pool worker before measuring; the measured
// raw handle counts, cancellation budgets and peer-close assertions stay exact.
class WindowsHttpRuntime final {
  static constexpr DWORD kWorkers = 4;

  struct Winsock final {
    Winsock() {
      WSADATA data{};
      if (WSAStartup(MAKEWORD(2, 2), &data) != 0)
        throw std::runtime_error("Cannot initialize test Winsock runtime");
    }
    ~Winsock() { WSACleanup(); }
  };

  struct Pool final {
    PTP_POOL value = nullptr;
    DWORD minimum = 0;
    DWORD maximum = 0;

    Pool() {
      // Materialize the process default pool without using network services.
      auto* work = CreateThreadpoolWork(
          [](PTP_CALLBACK_INSTANCE, void*, PTP_WORK) {}, nullptr, nullptr);
      if (!work) throw std::runtime_error("Cannot initialize test OS pool");
      SubmitThreadpoolWork(work);
      WaitForThreadpoolWorkCallbacks(work, FALSE);
      CloseThreadpoolWork(work);

      using Query = LONG(NTAPI*)(HANDLE, ULONG, void*, ULONG, ULONG*);
      const auto query = reinterpret_cast<Query>(GetProcAddress(
          GetModuleHandleW(L"ntdll.dll"), "NtQueryInformationWorkerFactory"));
      if (!query)
        throw std::runtime_error("Cannot inspect test worker factory");
      struct Information {
        LARGE_INTEGER timeout, retry_timeout, idle_timeout;
        BOOLEAN paused, timer_set, queued, may_create, creating, inserted,
            shutdown;
        ULONG binding_count, minimum, maximum, pending, waiting, total,
            release_count;
        LONGLONG infinite_wait_goal;
        void* start_routine;
        void* start_parameter;
        HANDLE process;
        SIZE_T stack_reserve, stack_commit;
        LONG last_creation_status;
      };
      static_assert(offsetof(Information, start_parameter) == 0x50);
      for (const auto& [handle, metadata] :
           CaptureWindowsSelfHandles().handles) {
        if (metadata.type != "TpWorkerFactory") continue;
        Information info{};
        ULONG bytes = 0;
        const auto status = query(reinterpret_cast<HANDLE>(handle), 7U, &info,
                                  sizeof(info), &bytes);
        if (status < 0 || bytes < offsetof(Information, stack_reserve) ||
            info.process != reinterpret_cast<HANDLE>(static_cast<ULONG_PTR>(
                                GetCurrentProcessId())) ||
            !info.start_parameter)
          throw std::runtime_error(
              "Invalid test worker-factory metadata; status=" +
              std::to_string(status) + "; bytes=" + std::to_string(bytes) +
              "; pid=" +
              std::to_string(reinterpret_cast<ULONG_PTR>(info.process)) +
              "; minimum=" + std::to_string(info.minimum) +
              "; maximum=" + std::to_string(info.maximum));
        // Ignore pre-existing private factories below the required capacity.
        // The default pool materialized above has its initial 500-worker cap.
        if (info.maximum < kWorkers) continue;
        // Before the first HTTP/RPC operation this isolated executable must
        // have exactly one eligible pool. Never guess among private pools.
        if (value)
          throw std::runtime_error("Ambiguous test process default pool");
        value = static_cast<PTP_POOL>(info.start_parameter);
        minimum = info.minimum;
        maximum = info.maximum;
      }
      if (!value) throw std::runtime_error("Missing test process default pool");
      SetThreadpoolThreadMaximum(value, kWorkers);
      if (!SetThreadpoolThreadMinimum(value, kWorkers)) {
        SetThreadpoolThreadMaximum(value, maximum);
        throw std::runtime_error("Cannot pin test process default pool");
      }
    }
    ~Pool() {
      SetThreadpoolThreadMinimum(value, minimum);
      SetThreadpoolThreadMaximum(value, maximum);
    }
  };

  struct Warmup {
    std::atomic<unsigned> entered{0};
    std::atomic<bool> release{false};
    std::atomic<bool> failed{false};
  };

 public:
  WindowsHttpRuntime() {
    Warmup state;
    auto* work = CreateThreadpoolWork(
        [](PTP_CALLBACK_INSTANCE, void* context, PTP_WORK) noexcept {
          auto& state = *static_cast<Warmup*>(context);
          RPC_BINDING_HANDLE binding = nullptr;
          auto* address = reinterpret_cast<RPC_WSTR>(
              const_cast<wchar_t*>(L"ncalrpc:[liuding.http.test]"));
          if (RpcBindingFromStringBindingW(address, &binding) != RPC_S_OK)
            state.failed = true;
          if (binding && RpcBindingFree(&binding) != RPC_S_OK)
            state.failed = true;
          const auto socket = WSASocketW(AF_INET, SOCK_STREAM, IPPROTO_TCP,
                                         nullptr, 0, WSA_FLAG_OVERLAPPED);
          if (socket == INVALID_SOCKET)
            state.failed = true;
          else if (closesocket(socket) == SOCKET_ERROR)
            state.failed = true;
          ++state.entered;
          // Occupy each worker until all four have initialized their caches.
          const auto until =
              std::chrono::steady_clock::now() + std::chrono::seconds(5);
          while (!state.release && std::chrono::steady_clock::now() < until)
            Sleep(1);
        },
        &state, nullptr);
    if (!work) throw std::runtime_error("Cannot warm test OS workers");
    for (unsigned i = 0; i < kWorkers; ++i) SubmitThreadpoolWork(work);
    const auto until =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (state.entered != kWorkers &&
           std::chrono::steady_clock::now() < until)
      Sleep(1);
    state.release = true;
    WaitForThreadpoolWorkCallbacks(work, TRUE);
    CloseThreadpoolWork(work);
    if (state.entered != kWorkers || state.failed)
      throw std::runtime_error("Test OS worker caches did not initialize");
  }
  WindowsHttpRuntime(const WindowsHttpRuntime&) = delete;
  WindowsHttpRuntime& operator=(const WindowsHttpRuntime&) = delete;

 private:
  Winsock winsock_;
  Pool pool_;
};

}  // namespace jyppx::liuding::test
#endif  // _WIN32
#endif  // JYPPX_LIUDING_TEST_HTTP_WINDOWS_RUNTIME_HPP
