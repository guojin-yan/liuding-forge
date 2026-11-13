// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "telemetry.hpp"

#include <jyppx/liuding/telemetry_native/tensorboard.hpp>

#include <array>
#include <chrono>
namespace jyppx::liuding::vision::tasks::detail {
MetricObserver tensorboard_observer(const std::filesystem::path& output,
                                    bool enabled) {
  if (!enabled) return {};
  auto sink = std::make_shared<experiment::TensorBoardTelemetrySink>(
      experiment::TensorBoardTelemetryOptions{.directory =
                                                  output / "tensorboard"});
  return [sink, sequence = std::uint64_t{0}](
             const training::CallbackEvent& event,
             const std::map<std::string, double>& metrics) mutable {
    TelemetryMetricBatch batch;
    for (const auto& [key, value] : metrics) {
      TelemetryMetricPoint p;
      p.definition.metric_id = key;
      p.step.value = event.global_step;
      p.observation = make_telemetry_metric_value(value);
      batch.points.push_back(std::move(p));
    }
    TelemetryEnvelope envelope;
    const auto id = std::to_string(++sequence);
    envelope.identity = {.stream_id = event.run_id,
                         .event_id = "epoch." + id,
                         .idempotency_key = "epoch." + id,
                         .sequence = sequence};
    envelope.time.wall_time_unix_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
    envelope.payload = std::move(batch);
    const std::array events{std::move(envelope)};
    auto receipt = sink->emit(events);
    if (receipt.dispositions.size() != 1 ||
        receipt.dispositions[0] != TelemetryDisposition::accepted)
      throw IoError("TensorBoard did not accept vision metrics");
    sink->flush();
  };
}
}  // namespace jyppx::liuding::vision::tasks::detail
