// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
#include "detail.hpp"

#include <jyppx/serving/service.hpp>

#include <chrono>
#include <utility>
namespace jyppx::serving {
namespace {
ServiceStatus Status(ErrorCode code) {
  switch (code) {
    case ErrorCode::invalid_argument:
    case ErrorCode::serialization:
    case ErrorCode::numerical:
      return ServiceStatus::invalid_argument;
    case ErrorCode::resource_exhausted:
      return ServiceStatus::resource_exhausted;
    case ErrorCode::cancelled:
      return ServiceStatus::cancelled;
    case ErrorCode::deadline_exceeded:
      return ServiceStatus::deadline_exceeded;
    case ErrorCode::not_ready:
    case ErrorCode::device:
    case ErrorCode::compatibility:
    case ErrorCode::not_implemented:
      return ServiceStatus::unavailable;
    default:
      return ServiceStatus::internal;
  }
}
ServiceReply Reply(ServiceStatus status, std::string id) {
  return {.status = status,
          .request_id = std::move(id),
          .message = std::string(service_status_name(status)),
          .body = {},
          .timing = {}};
}
}  // namespace
std::string_view service_status_name(ServiceStatus status) noexcept {
  switch (status) {
    case ServiceStatus::ok:
      return "ok";
    case ServiceStatus::invalid_argument:
      return "invalid_argument";
    case ServiceStatus::unauthenticated:
      return "unauthenticated";
    case ServiceStatus::permission_denied:
      return "permission_denied";
    case ServiceStatus::resource_exhausted:
      return "resource_exhausted";
    case ServiceStatus::cancelled:
      return "cancelled";
    case ServiceStatus::deadline_exceeded:
      return "deadline_exceeded";
    case ServiceStatus::unavailable:
      return "unavailable";
    default:
      return "internal";
  }
}
InferenceServiceAdapter::InferenceServiceAdapter(
    std::shared_ptr<Predictor> predictor, ProtocolAdapter protocol,
    ServiceOptions options, std::shared_ptr<DynamicBatcher> batcher)
    : predictor_(std::move(predictor)),
      protocol_(std::move(protocol)),
      options_(options),
      batcher_(std::move(batcher)) {
  detail::Text(protocol_.protocol_id, true);
  if (!predictor_ || !protocol_.schema_version || !protocol_.decode ||
      !protocol_.encode || !protocol_.authorize ||
      !options_.max_request_bytes || !options_.max_response_bytes)
    detail::Invalid("service.adapter",
                    "owning predictor, explicit versioned codecs/authorization "
                    "and positive sizes");
  if (batcher_ && batcher_->predictor() != predictor_)
    detail::Invalid("service.scheduler", "scheduler owning the same predictor");
}
ServiceReply InferenceServiceAdapter::handle(std::span<const std::byte> body,
                                             const TransportContext& context,
                                             std::stop_token stop) const {
  std::string id;
  const auto start = std::chrono::steady_clock::now();
  TimingBreakdown timing;
  const auto reply = [&](ServiceStatus status) {
    auto response = Reply(status, id);
    response.timing = timing;
    response.timing.total =
        detail::Seconds(std::chrono::steady_clock::now() - start);
    return response;
  };
  try {
    detail::Text(context.correlation.request_id, true);
    id = context.correlation.request_id;
    detail::Context(context.correlation, predictor_->info().model_version);
    if (!context.authenticated) return reply(ServiceStatus::unauthenticated);
    if (context.correlation.principal_id.empty() ||
        context.correlation.project_id.empty())
      return reply(ServiceStatus::unauthenticated);
    if (!protocol_.authorize(context.correlation))
      return reply(ServiceStatus::permission_denied);
    detail::Boundary(stop, context.deadline);
    if (body.size() > options_.max_request_bytes)
      return reply(ServiceStatus::resource_exhausted);
    if (!health().ready) return reply(ServiceStatus::unavailable);
    auto stage = std::chrono::steady_clock::now();
    PredictRequest request{.input = protocol_.decode(body),
                           .context = context.correlation,
                           .deadline = context.deadline};
    timing.decode = detail::Seconds(std::chrono::steady_clock::now() - stage);
    detail::Boundary(stop, context.deadline);
    PredictResponse result;
    if (batcher_) {
      auto future = batcher_->submit(request, stop);
      while (future.wait_for(std::chrono::milliseconds(1)) !=
             std::future_status::ready)
        detail::Boundary(stop, context.deadline);
      result = future.get();
    } else
      result = predictor_->predict(request, stop);
    const auto decode = timing.decode;
    timing = result.timing;
    timing.decode = decode;
    detail::Boundary(stop, context.deadline);
    stage = std::chrono::steady_clock::now();
    auto encoded = protocol_.encode(result.output);
    timing.encode = detail::Seconds(std::chrono::steady_clock::now() - stage);
    if (encoded.size() > options_.max_response_bytes)
      return reply(ServiceStatus::resource_exhausted);
    detail::Boundary(stop, context.deadline);
    auto response = reply(ServiceStatus::ok);
    response.body = std::move(encoded);
    return response;
  } catch (const Error& error) {
    return reply(Status(error.code()));
  } catch (const std::bad_alloc&) {
    return reply(ServiceStatus::resource_exhausted);
  } catch (...) {
    return reply(ServiceStatus::internal);
  }
}
ServiceHealth InferenceServiceAdapter::health() const {
  auto model = predictor_->info();
  const auto ready = model.health == HealthStatus::ready &&
                     (!batcher_ || batcher_->statistics().accepting);
  return {.live = true,
          .ready = ready,
          .streaming = false,
          .model = std::move(model)};
}
bool InferenceServiceAdapter::close(Deadline deadline, std::stop_token stop) {
  if (batcher_ && !batcher_->close(deadline, stop)) return false;
  return predictor_->close(deadline, stop);
}
}  // namespace jyppx::serving
