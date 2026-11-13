// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

int main(int argc, char** argv) {
#if defined(_WIN32)
  if (_setmode(_fileno(stdout), _O_BINARY) == -1 ||
      _setmode(_fileno(stderr), _O_BINARY) == -1)
    return 8;
#endif
  if (argc != 2) return 9;
  const std::string mode(argv[1]);
  std::cout << "native stdout\n" << std::flush;
  std::cerr << "native stderr\n" << std::flush;
  if (mode == "success") return 0;
  if (mode == "retry") {
    const auto* index = std::getenv("JYPPX_LIUDING_LOCAL_ATTEMPT_INDEX");
    return index && std::string(index) == "1" ? 75 : 0;
  }
  if (mode == "flood") {
    std::cout << std::string(8192, 'x') << std::flush;
    return 0;
  }
  if (mode == "wait") {
    std::this_thread::sleep_for(std::chrono::seconds(30));
    return 0;
  }
  return 9;
}
