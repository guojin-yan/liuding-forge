// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "fuzz_support.hpp"
#include "telemetry_seed.hpp"

#include <jyppx/liuding/experiment/telemetry_exporters.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
namespace ex = jyppx::liuding::experiment;
namespace fuzz = jyppx::liuding::fuzz;
using Json = nlohmann::json;

enum class Operation { emit, flush, close };

class ReplyTransport final : public ex::TelemetrySidecarTransport {
 public:
  explicit ReplyTransport(std::vector<std::string> responses)
      : responses_(std::move(responses)) {}

  std::string transact(std::string_view request, std::stop_token stop,
                       jyppx::liuding::TelemetryDeadline deadline) override {
    jyppx::liuding::check_telemetry_operation(stop, deadline);
    fuzz::require(request.size() <= 256U * 1024U,
                  "sidecar request exceeded configured budget");
    const auto parsed = Json::parse(request);
    fuzz::require(parsed.at("protocol_version") == 1U &&
                      (parsed.at("backend") == "tensorboard" ||
                       parsed.at("backend") == "opentelemetry"),
                  "official sink generated invalid protocol identity");
    fuzz::require(!responses_.empty(), "protocol fuzzer has no reply");
    const auto index = (std::min)(calls_, responses_.size() - 1U);
    ++calls_;
    return responses_[index];
  }

  std::size_t calls() const noexcept { return calls_; }

 private:
  std::vector<std::string> responses_;
  std::size_t calls_ = 0U;
};

ex::OfficialTelemetryOptions options(std::size_t reply_bytes,
                                     ex::OfficialTelemetryBackend backend) {
  return {.backend = backend,
          .sink_id = "fuzz.sidecar",
          .destination_id = "fuzz.destination",
          .limits = fuzz::telemetry_limits(),
          .max_request_bytes = 256U * 1024U,
          .max_reply_bytes = reply_bytes};
}

std::vector<std::uint64_t> call(ex::OfficialTelemetrySink& sink,
                                Operation operation) {
  if (operation == Operation::emit) {
    const auto batch = std::array{fuzz::seed_envelope()};
    const auto result = sink.emit(batch);
    jyppx::liuding::validate_telemetry_emit_result(
        sink.descriptor(), batch, result, fuzz::telemetry_limits());
    std::vector<std::uint64_t> encoded;
    for (auto disposition : result.dispositions)
      encoded.push_back(static_cast<std::uint64_t>(disposition));
    return encoded;
  }
  const auto result =
      operation == Operation::flush ? sink.flush() : sink.close();
  return {result.exported_events, result.sampled_out_events,
          result.unsupported_events};
}

void exercise_operation(std::string_view input, Operation operation,
                        ex::OfficialTelemetryBackend backend) {
  auto transport = std::make_shared<ReplyTransport>(
      std::vector<std::string>{std::string(input)});
  ex::OfficialTelemetrySink sink(transport,
                                 options(fuzz::kMaxInputBytes, backend));
  const auto accepted =
      fuzz::parse_input([&] { return call(sink, operation); }, true);
  if (!accepted) return;
  fuzz::require(transport->calls() == 1U,
                "protocol operation exchanged extra frames");
  // Repeating accepted emit/flush replies must preserve the exact receipts and
  // monotonic counters. A second close uses cached counters without transport.
  fuzz::require(call(sink, operation) == *accepted,
                "replaying identical protocol reply changed receipts/counters");
  fuzz::require(transport->calls() == (operation == Operation::close ? 1U : 2U),
                "closed sidecar unexpectedly transacted another request");
  auto exact_transport = std::make_shared<ReplyTransport>(
      std::vector<std::string>{std::string(input)});
  ex::OfficialTelemetrySink exact(exact_transport,
                                  options(input.size(), backend));
  fuzz::require(call(exact, operation) == *accepted,
                "exact reply byte budget changed protocol acceptance");
  fuzz::require(input.size() > 1U,
                "accepted impossible one-byte protocol reply");
  auto small_transport = std::make_shared<ReplyTransport>(
      std::vector<std::string>{std::string(input)});
  ex::OfficialTelemetrySink small(small_transport,
                                  options(input.size() - 1U, backend));
  fuzz::require_resource_rejection(
      [&] { static_cast<void>(call(small, operation)); });
}

void exercise_counter_sequence(std::string_view input) {
  const auto newline = input.find('\n');
  if (newline == std::string_view::npos || newline == 0U ||
      newline + 1U == input.size())
    return;
  const std::string first(input.substr(0U, newline));
  const std::string second(input.substr(newline + 1U));
  auto transport = std::make_shared<ReplyTransport>(
      std::vector<std::string>{first, second, first});
  ex::OfficialTelemetrySink sink(
      transport,
      options(fuzz::kMaxInputBytes, ex::OfficialTelemetryBackend::tensorboard));
  const auto baseline =
      fuzz::parse_input([&] { return call(sink, Operation::flush); }, true);
  if (!baseline) return;
  const auto next =
      fuzz::parse_input([&] { return call(sink, Operation::flush); }, true);
  if (next) {
    fuzz::require(next->size() == baseline->size(),
                  "protocol counter arity changed");
    for (std::size_t index = 0U; index < next->size(); ++index)
      fuzz::require((*next)[index] >= (*baseline)[index],
                    "sidecar accepted regressing cumulative counters");
  } else {
    // A rejected frame may not advance committed sink counters.
    fuzz::require(call(sink, Operation::flush) == *baseline,
                  "rejected protocol reply mutated committed counters");
  }
}

void exercise(std::span<const std::byte> input) {
  const auto value = fuzz::text(input);
  for (const auto backend : {ex::OfficialTelemetryBackend::tensorboard,
                             ex::OfficialTelemetryBackend::opentelemetry}) {
    for (const auto operation :
         {Operation::emit, Operation::flush, Operation::close})
      exercise_operation(value, operation, backend);
  }
  exercise_counter_sequence(value);
}

std::string response(std::string request_id, std::uint64_t exported = 1U) {
  return Json{{"protocol_version", 1U},
              {"request_id", std::move(request_id)},
              {"counters",
               {{"exported_events", exported},
                {"sampled_out_events", 0U},
                {"unsupported_events", 0U}}}}
      .dump();
}

const std::vector<std::vector<std::byte>>& seeds() {
  static const auto value = [] {
    const auto content = jyppx::liuding::telemetry_content_digest(
        fuzz::seed_envelope(), fuzz::telemetry_limits());
    const auto identity = "jyppx.telemetry.sidecar-batch/v1" + content.hex;
    auto receipt = Json::parse(
        response(jyppx::liuding::sha256_hex(fuzz::bytes(identity))));
    receipt["receipts"] = Json::array({1U});
    std::vector<std::vector<std::byte>> result{
        fuzz::copy_bytes(response("flush")),
        fuzz::copy_bytes(response("close")),
        fuzz::copy_bytes(receipt.dump()),
        fuzz::copy_bytes(response("flush", 9U) + "\n" + response("flush", 8U)),
        fuzz::copy_bytes(response("flush", 8U) + "\n" + response("flush", 9U)),
        fuzz::copy_bytes("{\"protocol_version\":1,\"request_id\":\"flush\","
                         "\"error\":\"backend failure\"}")};
    for (std::size_t index = 0U; index < 3U; ++index) {
      auto channel = std::make_shared<ReplyTransport>(
          std::vector<std::string>{std::string(fuzz::text(result[index]))});
      ex::OfficialTelemetrySink sink(
          channel, options(fuzz::kMaxInputBytes,
                           ex::OfficialTelemetryBackend::tensorboard));
      const auto operation = index == 0U   ? Operation::flush
                             : index == 1U ? Operation::close
                                           : Operation::emit;
      static_cast<void>(call(sink, operation));
    }
    for (const auto& seed : result) exercise(seed);
    return result;
  }();
  return value;
}
}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data,
                                      std::size_t size) {
  return fuzz::exercise_raw_and_seed(data, size, seeds(), exercise);
}
