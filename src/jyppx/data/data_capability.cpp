// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "jyppx/data/data_pipeline.hpp"
#include "jyppx/foundation/error.hpp"

#include <ATen/core/ScalarType.h>
#include <algorithm>
#include <new>
#include <set>
#include <string_view>
#include <utility>

namespace jyppx::data {
namespace {

constexpr char kComponent[] = "data.capability";

[[noreturn]] void throw_invalid(std::string operation, std::string expected,
                                std::string actual) {
  throw InvalidArgument(
      "Invalid data capability",
      ErrorDetails{.component = kComponent,
                   .operation = std::move(operation),
                   .expected = std::move(expected),
                   .actual = std::move(actual),
                   .hint = "Use the descriptor returned by a validated "
                           "DataPipeline or IterableDataPipeline instance."});
}

[[noreturn]] void throw_resource(std::string operation, std::string expected,
                                 std::string actual, std::string hint) {
  throw ResourceExhausted("Unable to describe data capability",
                          ErrorDetails{.component = kComponent,
                                       .operation = std::move(operation),
                                       .expected = std::move(expected),
                                       .actual = std::move(actual),
                                       .hint = std::move(hint)});
}

std::string worker_operation(DataWorkerModel model) {
  switch (model) {
    case DataWorkerModel::inline_execution:
      return "worker.inline";
    case DataWorkerModel::thread_pool:
      return "worker.thread-pool";
    case DataWorkerModel::producer_thread:
      return "worker.producer-thread";
  }
  throw_invalid("worker_model", "a known data worker model",
                std::to_string(static_cast<std::uint8_t>(model)));
}

std::string shard_operation(DataWorkerShardPolicy policy) {
  switch (policy) {
    case DataWorkerShardPolicy::single_shard:
      return "worker.shard.single";
    case DataWorkerShardPolicy::round_robin_batch:
      return "worker.shard.round-robin-batch";
  }
  throw_invalid("worker_shard_policy", "a known logical shard policy",
                std::to_string(static_cast<std::uint8_t>(policy)));
}

std::string restore_operation(RestoreLevel level) {
  switch (level) {
    case RestoreLevel::exact:
      return "restore.exact";
    case RestoreLevel::bounded:
      return "restore.bounded";
    case RestoreLevel::best_effort:
      return "restore.best-effort";
  }
  throw_invalid("restore_level", "a known restore level",
                std::to_string(static_cast<std::uint8_t>(level)));
}

void collect_tensor_constraints(const ValueSchema& schema,
                                std::set<std::string>& dtypes,
                                std::set<std::string>& devices) {
  if (schema.kind == ValueKind::tensor && schema.tensor_constraints) {
    if (schema.tensor_constraints->dtype) {
      dtypes.emplace(c10::toString(*schema.tensor_constraints->dtype));
    }
    if (schema.tensor_constraints->device) {
      devices.emplace(schema.tensor_constraints->device->str());
    }
  }
  for (const auto& element : schema.elements) {
    collect_tensor_constraints(element, dtypes, devices);
  }
  for (const auto& [name, field] : schema.fields) {
    static_cast<void>(name);
    collect_tensor_constraints(field, dtypes, devices);
  }
}

void validate_data_capability(const DataCapability& capability) {
  if (capability.map_style == capability.iterable_style) {
    throw_invalid("source_style", "exactly one of map or iterable", "invalid");
  }
  validate_batch_schema(capability.batch_schema, capability.schema_limits);
  const auto expected_fingerprint = batch_schema_fingerprint(
      capability.batch_schema, capability.schema_limits);
  if (capability.batch_schema_fingerprint != expected_fingerprint) {
    throw_invalid("batch_schema_fingerprint", expected_fingerprint,
                  capability.batch_schema_fingerprint);
  }
  if (capability.worker_model == DataWorkerModel::inline_execution &&
      capability.worker_count != 0U) {
    throw_invalid("worker_count", "zero for inline execution",
                  std::to_string(capability.worker_count));
  }
  if (capability.worker_model == DataWorkerModel::thread_pool &&
      capability.worker_count == 0U) {
    throw_invalid("worker_count", "positive for a thread pool", "0");
  }
  if (capability.worker_model == DataWorkerModel::producer_thread &&
      capability.worker_count != 1U) {
    throw_invalid("worker_count", "one for an iterable producer thread",
                  std::to_string(capability.worker_count));
  }
  const auto expected_shard_policy =
      capability.worker_model == DataWorkerModel::thread_pool
          ? DataWorkerShardPolicy::round_robin_batch
          : DataWorkerShardPolicy::single_shard;
  if (capability.worker_shard_policy != expected_shard_policy) {
    throw_invalid("worker_shard_policy", shard_operation(expected_shard_policy),
                  shard_operation(capability.worker_shard_policy));
  }
  if (capability.degradation_reasons.size() > kMaxCapabilityValues) {
    throw_resource(
        "degradation_reasons",
        "at most " + std::to_string(kMaxCapabilityValues) + " reasons",
        std::to_string(capability.degradation_reasons.size()) + " reasons",
        "Reduce degradation metadata before capability publication.");
  }
  if (capability.restore_level == RestoreLevel::exact &&
      !capability.degradation_reasons.empty()) {
    throw_invalid("degradation_reasons", "empty for exact restore",
                  std::to_string(capability.degradation_reasons.size()));
  }
  if (capability.restore_level != RestoreLevel::exact &&
      capability.degradation_reasons.empty()) {
    throw_invalid("degradation_reasons",
                  "at least one reason for degraded restore", "empty");
  }
  if (capability.non_blocking_transfer && !capability.target_device) {
    throw_invalid("target_device", "a destination for non-blocking transfer",
                  "unset");
  }
  if ((capability.pin_memory || capability.target_device ||
       capability.non_blocking_transfer) &&
      !capability.memory_stage) {
    throw_invalid("memory_stage", "enabled for pinning or transfer",
                  "disabled");
  }
  if (capability.memory_stage && capability.memory_stage_capacity == 0U) {
    throw_invalid("memory_stage_capacity", "positive when staging is enabled",
                  "0");
  }
  if (!capability.memory_stage && capability.memory_stage_capacity != 0U) {
    throw_invalid("memory_stage_capacity", "zero when staging is disabled",
                  std::to_string(capability.memory_stage_capacity));
  }
  if (capability.transfer_stream_policy !=
      BatchTransferStreamPolicy::caller_current) {
    throw_invalid("transfer_stream_policy", "caller-current",
                  std::to_string(static_cast<std::uint8_t>(
                      capability.transfer_stream_policy)));
  }
  static_cast<void>(worker_operation(capability.worker_model));
  static_cast<void>(shard_operation(capability.worker_shard_policy));
  static_cast<void>(restore_operation(capability.restore_level));
}

void add_limit(std::vector<CapabilityLimit>& limits, std::string name,
               std::uint64_t value, std::string unit) {
  limits.push_back(
      {.name = std::move(name), .value = value, .unit = std::move(unit)});
}

}  // namespace

CapabilityDescriptor describe_data_capability(
    const DataCapability& capability) {
  try {
    validate_data_capability(capability);
    CapabilityDescriptor descriptor{
        .feature_id = capability.map_style ? "jyppx.data.pipeline.map"
                                           : "jyppx.data.pipeline.iterable",
        .scope_id = capability.pipeline_id,
        .feature_version = 1U,
        .status = CapabilityStatus::available,
        .implementation = CapabilityImplementation::native,
        .source = CapabilitySource::configured_instance,
        .backend = "libtorch",
        .devices = {},
        .dtypes = {},
        .schemas = {},
        .operations = {},
        .limits = {},
        .limitations = {},
    };

    std::set<std::string> dtypes;
    std::set<std::string> devices;
    collect_tensor_constraints(capability.batch_schema.value, dtypes, devices);
    if (capability.target_device) {
      devices.clear();
      devices.emplace(capability.target_device->str());
    } else if (devices.empty()) {
      devices.emplace("producer-defined");
    }
    descriptor.devices.assign(devices.begin(), devices.end());
    descriptor.dtypes.assign(dtypes.begin(), dtypes.end());
    descriptor.schemas = {
        capability.batch_schema.id + "/v" +
            std::to_string(capability.batch_schema.version),
        "fnv1a64/" + capability.batch_schema_fingerprint,
    };
    descriptor.operations = {
        capability.map_style ? "source.map" : "source.iterable",
        "epoch.explicit",
        "schema.validate",
        worker_operation(capability.worker_model),
        shard_operation(capability.worker_shard_policy),
        restore_operation(capability.restore_level),
    };
    if (capability.finite_epoch) {
      descriptor.operations.emplace_back("epoch.finite");
    }
    if (capability.deterministic_epoch) {
      descriptor.operations.emplace_back("epoch.deterministic");
    }
    if (capability.in_epoch_cursor) {
      descriptor.operations.emplace_back("checkpoint.in-epoch-cursor");
    }
    if (capability.process_workers) {
      descriptor.operations.emplace_back("worker.process");
    } else {
      descriptor.limitations.emplace_back("process workers are unsupported");
    }
    if (capability.worker_prefetch) {
      descriptor.operations.emplace_back("worker.prefetch");
    }
    if (capability.persistent_workers) {
      descriptor.operations.emplace_back("worker.persistent");
    }
    if (capability.batch_timeout) {
      descriptor.operations.emplace_back("batch.timeout");
    }
    if (capability.memory_stage) {
      descriptor.operations.emplace_back("memory.stage");
    }
    if (capability.pin_memory) {
      descriptor.operations.emplace_back("memory.pin");
    }
    if (capability.target_device) {
      descriptor.operations.emplace_back("memory.transfer");
      if (capability.non_blocking_transfer) {
        descriptor.operations.emplace_back("memory.transfer.non-blocking");
      }
      descriptor.operations.emplace_back("memory.stream.caller-current");
    }

    add_limit(descriptor.limits, "worker_count", capability.worker_count,
              "workers");
    add_limit(descriptor.limits, "prefetch_batches",
              capability.prefetch_batches, "batches");
    add_limit(descriptor.limits, "memory_stage_capacity",
              capability.memory_stage_capacity, "batches");
    add_limit(descriptor.limits, "max_batch_tensors",
              capability.max_batch_tensors, "tensors");
    add_limit(descriptor.limits, "max_batch_value_nodes",
              capability.max_batch_value_nodes, "nodes");
    add_limit(descriptor.limits, "max_batch_bytes", capability.max_batch_bytes,
              "bytes");
    add_limit(descriptor.limits, "max_schema_depth",
              capability.schema_limits.max_depth, "count");
    add_limit(descriptor.limits, "max_schema_nodes",
              capability.schema_limits.max_schema_nodes, "nodes");
    add_limit(descriptor.limits, "max_schema_metadata_bytes",
              capability.schema_limits.max_metadata_bytes, "bytes");
    add_limit(descriptor.limits, "max_runtime_value_nodes",
              capability.schema_limits.max_runtime_nodes, "nodes");
    add_limit(descriptor.limits, "max_replayed_batches",
              capability.max_replayed_batches, "batches");
    if (capability.max_replayed_samples) {
      add_limit(descriptor.limits, "max_replayed_samples",
                *capability.max_replayed_samples, "samples");
    }
    if (capability.max_omitted_samples) {
      add_limit(descriptor.limits, "max_omitted_samples",
                *capability.max_omitted_samples, "samples");
    }
    descriptor.limitations.insert(descriptor.limitations.end(),
                                  capability.degradation_reasons.begin(),
                                  capability.degradation_reasons.end());
    validate_capability_descriptor(descriptor);
    return descriptor;
  } catch (const Error&) {
    throw;
  } catch (const std::bad_alloc&) {
    throw_resource("describe", "bounded capability metadata allocation",
                   "allocation failed",
                   "Reduce schema or degradation metadata.");
  }
}

}  // namespace jyppx::data
