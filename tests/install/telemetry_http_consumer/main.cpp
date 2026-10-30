// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/telemetry_http/otlp_http.hpp>

#include <iostream>
#include <utility>

int main() {
  try {
    jyppx::liuding::experiment::OtlpHttpOptions options;
    options.endpoint = {.host = "127.0.0.1", .port = 1U, .tls = false};
    options.mapping.resource_instance_id = "installed.consumer";
    jyppx::liuding::experiment::OtlpHttpTelemetrySink sink(std::move(options));
    if (!sink.emit({}).dispositions.empty() ||
        sink.close().exported_events != 0U)
      return 2;
    // Construction, empty batches and cleanup never connect. This proves the
    // installed linkage/ABI boundary; real network semantics use the wire gate.
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
