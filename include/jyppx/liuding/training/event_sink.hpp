// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_TRAINING_EVENT_SINK_HPP
#define JYPPX_LIUDING_TRAINING_EVENT_SINK_HPP
/**
 * @file event_sink.hpp
 * @brief Defines bounded post-commit event delivery and built-in local sinks.
 */

#include <jyppx/liuding/foundation/checkpoint.hpp>
#include <jyppx/liuding/foundation/telemetry.hpp>
#include <jyppx/liuding/training/callback_dispatcher.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace jyppx::liuding::training {

struct TrainingTelemetryAdapterOptions;

/**
 * @brief Declares the retry guarantee provided by an event sink.
 *
 * Numeric values are persisted in outbox topology and must remain stable.
 */
enum class EventDeliveryGuarantee : std::uint8_t {
  at_least_once = 1,  ///< A failed or interrupted publish may be replayed.
};

/**
 * @brief Returns the stable lowercase event-delivery guarantee name.
 * @param guarantee Guarantee to render.
 * @return A process-lifetime name, or `"unknown"` for an invalid value.
 */
std::string_view event_delivery_guarantee_name(
    EventDeliveryGuarantee guarantee) noexcept;

/** @brief Freezes one event sink's logical destination and retry semantics. */
struct EventSinkDescriptor {
  std::uint32_t schema_version = 1;  ///< Current-only descriptor schema.
  std::string type_id;  ///< Stable implementation and event schema identity.
  std::string sink_id;  ///< Stable identity unique within one outbox.
  std::string destination_id;  ///< Logical destination without credentials.
  EventDeliveryGuarantee delivery_guarantee =
      EventDeliveryGuarantee::at_least_once;  ///< Crash/retry behavior.
};

/**
 * @brief Contains one committed lifecycle event and its callback decisions.
 *
 * `event.run_id` plus `event.event_index` is the stable idempotency key.
 * Consumers must deduplicate that key when external exactly-once effects are
 * required because a process can stop after a sink accepts an event but before
 * the outbox cursor is checkpointed.
 */
struct EventEnvelope {
  std::uint32_t schema_version = 1;  ///< Current event-envelope schema.
  CallbackEvent event;  ///< Immutable lifecycle snapshot accepted by callbacks.
  CallbackDispatchResult callback_result;  ///< Auditable callback decisions.
};

/**
 * @brief Receives post-commit events without writable Trainer access.
 *
 * `publish()` must either accept the complete envelope or throw. Once it
 * returns, the sink owns any internal retry required for its destination.
 * `close()` must be idempotent so a failed multi-sink shutdown can be retried.
 * Implementations are invoked serially and need not synchronize themselves.
 */
class EventSink {
 public:
  /** @brief Enables destruction through the sink interface. */
  virtual ~EventSink() = default;

  /**
   * @brief Returns immutable sink identity and delivery semantics.
   * @return A descriptor reference valid for the sink lifetime.
   */
  virtual const EventSinkDescriptor& descriptor() const noexcept = 0;

  /**
   * @brief Accepts one complete committed event.
   * @param envelope Validated event and callback result.
   * @param stop Cooperative cancellation checked before blocking work.
   * @throws Cancelled If cancellation is already requested.
   * @throws Error If the complete event cannot be accepted.
   */
  virtual void publish(const EventEnvelope& envelope,
                       std::stop_token stop = {}) = 0;

  /**
   * @brief Flushes accepted output to the sink's durable boundary.
   * @param stop Cooperative cancellation checked before blocking work.
   * @throws Cancelled If cancellation is already requested.
   * @throws Error If buffered output cannot be flushed.
   */
  virtual void flush(std::stop_token stop = {}) = 0;

  /**
   * @brief Idempotently flushes and releases the destination.
   * @param stop Cooperative cancellation checked before blocking work.
   * @throws Cancelled If cancellation is already requested.
   * @throws Error If finalization cannot complete.
   */
  virtual void close(std::stop_token stop = {}) = 0;
};

/** @brief Configures a bounded checkpointable event outbox. */
struct EventOutboxOptions {
  std::string outbox_id =
      "events.outbox";         ///< Stable checkpoint participant identity.
  std::size_t max_sinks = 64;  ///< Positive maximum retained sink count.
  std::size_t max_pending_events =
      4096;  ///< Positive maximum queued committed events.
  std::size_t max_metrics_per_event =
      4096;  ///< Maximum finite metric observations per envelope.
  std::size_t max_commands_per_event =
      1024;  ///< Maximum callback commands retained per envelope.
  std::size_t max_string_bytes =
      4096;  ///< Maximum bytes in one identity, reason, or metric name.
  std::uint64_t max_event_bytes =
      1024ULL * 1024ULL;  ///< Maximum encoded bytes for one envelope.
  std::uint64_t max_pending_bytes =
      64ULL * 1024ULL * 1024ULL;  ///< Maximum encoded queued state bytes.
  std::uint64_t max_topology_bytes =
      1024ULL * 1024ULL;  ///< Maximum canonical sink topology bytes.
  std::uint64_t max_control_bytes =
      65ULL * 1024ULL * 1024ULL;  ///< Maximum checkpoint control blob bytes.
  std::vector<std::pair<std::string, std::string>>
      v1_sink_type_migrations;  ///< Explicit old/new sink implementation pairs.
};

/**
 * @brief Buffers committed callback events before deterministic sink delivery.
 *
 * Enqueue is bounded and has a strong guarantee. `CallbackDispatcher` uses the
 * outbox overload of `dispatch()` to enqueue after all callbacks succeed but
 * before its event cursor commits; an enqueue failure therefore rolls back
 * every attempted stateful callback. Sink publication happens only through
 * `drain_one()`, `drain()`, `flush()`, or `close()` after that transaction.
 *
 * Delivery is ordered by event index and then sink ID. A sink failure retains
 * the front event and its next-sink cursor for retry. State schema 2 persists
 * canonical neutral envelopes, callback snapshots, frozen metric bindings,
 * the producer stream, and per-envelope delivery cursors. The strict N-1 reader
 * upgrades real schema-1 checkpoint blobs atomically. Cross-process delivery
 * remains at-least-once.
 *
 * Instances are single-owner and not internally synchronized. Destruction does
 * not perform I/O; callers must invoke `close()` and handle any failure.
 */
class EventOutbox final : public CheckpointParticipant {
 public:
  /**
   * @brief Creates an outbox with deterministic sink order.
   * @param sinks Nonempty sinks retained by shared ownership.
   * @param options Stable identity and positive resource limits.
   * @throws InvalidArgument If options or descriptors are invalid or duplicate.
   * @throws ResourceExhausted If sink topology exceeds configured limits.
   */
  explicit EventOutbox(std::vector<std::shared_ptr<EventSink>> sinks,
                       EventOutboxOptions options = {});

  /**
   * @brief Creates the same bounded outbox for neutral telemetry destinations.
   * @param sinks Nonempty foundation sinks, ordered by immutable sink ID.
   * @param telemetry Frozen typed metric bindings and optional producer stream.
   * An empty stream ID generates a new stream once; restore adopts its saved
   * ID. An explicit stream ID must match the checkpoint being restored.
   * @param options Stable participant identity, limits, and N-1 type mappings.
   * @note enqueue captures wall/monotonic time once inside callback staging;
   * no sink I/O occurs until post-commit drain. Required evidence receipts are
   * validated before advancing the one shared delivery cursor.
   */
  EventOutbox(std::vector<std::shared_ptr<TelemetrySink>> sinks,
              const TrainingTelemetryAdapterOptions& telemetry,
              EventOutboxOptions options = {});

  /** @brief Prevents copying queued delivery state. */
  EventOutbox(const EventOutbox&) = delete;

  /** @brief Prevents assignment over queued delivery state. */
  EventOutbox& operator=(const EventOutbox&) = delete;

  /** @brief Prevents moving an object retained by a participant registry. */
  EventOutbox(EventOutbox&&) = delete;

  /** @brief Prevents move assignment of checkpointed delivery state. */
  EventOutbox& operator=(EventOutbox&&) = delete;

  /** @brief Leaves external I/O to explicit `close()`. */
  ~EventOutbox() override;

  /**
   * @brief Atomically appends one callback-committed event.
   * @param event Validated lifecycle event.
   * @param callback_result Validated commands produced for `event`.
   * @throws InvalidArgument If event order, identity, or data is invalid.
   * @throws CompatibilityError If callback results are internally inconsistent.
   * @throws ResourceExhausted If queue or byte limits would be exceeded.
   * @throws NotReady If the outbox has been closed.
   */
  void enqueue(const CallbackEvent& event,
               const CallbackDispatchResult& callback_result);

  /**
   * @brief Attempts delivery of the oldest event to all remaining sinks.
   * @param stop Cooperative cancellation checked before each sink call.
   * @param deadline Optional absolute steady-clock bound for this operation.
   * @return `true` when one complete event was removed, or `false` if empty.
   * @throws Cancelled If cancellation is requested before completion.
   * @throws DeadlineExceeded If the operation deadline is reached.
   * @throws Error If the next sink rejects the event; queued state is retained.
   */
  bool drain_one(std::stop_token stop = {},
                 TelemetryDeadline deadline = std::nullopt);

  /**
   * @brief Delivers every currently pending event in order.
   * @param stop Cooperative cancellation checked between sink calls.
   * @param deadline Optional absolute steady-clock bound for this operation.
   * @throws Cancelled If cancellation is requested before completion.
   * @throws DeadlineExceeded If the operation deadline is reached.
   * @throws Error If a sink rejects an event; remaining state is retained.
   */
  void drain(std::stop_token stop = {},
             TelemetryDeadline deadline = std::nullopt);

  /**
   * @brief Drains the outbox and flushes every sink in deterministic order.
   * @param stop Cooperative cancellation checked between sink calls.
   * @param deadline Optional absolute steady-clock bound for this operation.
   * @throws Cancelled If cancellation is requested before completion.
   * @throws DeadlineExceeded If the operation deadline is reached.
   * @throws Error If delivery or sink flushing fails.
   */
  void flush(std::stop_token stop = {},
             TelemetryDeadline deadline = std::nullopt);

  /**
   * @brief Idempotently drains, flushes, and closes every sink.
   * @param stop Cooperative cancellation checked between sink calls.
   * @param deadline Optional absolute steady-clock bound for this operation.
   * @throws Cancelled If cancellation is requested before completion.
   * @throws DeadlineExceeded If the operation deadline is reached.
   * @throws Error If delivery or sink close fails; the operation may be
   * retried.
   */
  void close(std::stop_token stop = {},
             TelemetryDeadline deadline = std::nullopt);

  /** @brief Returns the producer stream reused by retries and restore.
   * @return Immutable producer stream identity for this outbox.
   */
  std::string_view telemetry_stream_id() const noexcept;

  /**
   * @brief Reports whether finalization completed.
   * @return `true` only after every sink closed successfully.
   */
  bool closed() const noexcept;

  /**
   * @brief Returns the number of pending envelopes.
   * @return Queue length including a partially delivered front event.
   */
  std::size_t pending_events() const noexcept;

  /**
   * @brief Returns encoded bytes retained by pending envelopes.
   * @return Total canonical event bytes charged to the queue budget.
   */
  std::uint64_t pending_bytes() const noexcept;

  /**
   * @brief Returns the last accepted event index.
   * @return Zero before the first event, otherwise the committed enqueue
   * cursor.
   */
  std::uint64_t last_enqueued_event_index() const noexcept;

  /**
   * @brief Returns the last event delivered to every sink.
   * @return Zero before the first complete delivery.
   */
  std::uint64_t last_delivered_event_index() const noexcept;

  /**
   * @brief Returns immutable outbox limits and participant identity.
   * @return A reference valid for the outbox lifetime.
   */
  const EventOutboxOptions& options() const noexcept;

  /** @copydoc CheckpointParticipant::participant_id */
  std::string participant_id() const override;

  /** @copydoc CheckpointParticipant::describe_state */
  StateManifest describe_state() const override;

  /** @copydoc CheckpointParticipant::save_state */
  void save_state(StateWriter& writer) const override;

  /** @copydoc CheckpointParticipant::load_state */
  void load_state(const StateReader& reader, RestorePolicy policy) override;

 private:
  /** @brief Hides queue nodes and checkpoint codec from the public ABI. */
  struct Impl;

  std::unique_ptr<Impl> impl_;  ///< Owned bounded queue and delivery state.
};

/** @brief Configures one machine-readable append-only local JSONL sink. */
struct JsonlEventSinkOptions {
  std::string sink_id = "sink.jsonl";  ///< Stable identity within an outbox.
  std::string destination_id =
      "local.events";          ///< Stable logical destination used in topology.
  std::filesystem::path path;  ///< Nonempty output file path opened in append.
};

/**
 * @brief Appends one canonical JSON object per committed event.
 *
 * Each successful `publish()` flushes the line before returning. Existing files
 * are never truncated. The sink does not create missing parent directories and
 * does not attempt cross-process deduplication.
 */
class JsonlEventSink final : public EventSink {
 public:
  /**
   * @brief Opens an append-only JSONL destination.
   * @param options Stable sink identity and existing-parent output path.
   * @throws InvalidArgument If identity or path is invalid.
   * @throws IoError If the file cannot be opened.
   */
  explicit JsonlEventSink(JsonlEventSinkOptions options);

  /** @brief Closes the owned stream without throwing. */
  ~JsonlEventSink() override;

  /** @copydoc EventSink::descriptor */
  const EventSinkDescriptor& descriptor() const noexcept override;

  /** @copydoc EventSink::publish */
  void publish(const EventEnvelope& envelope,
               std::stop_token stop = {}) override;

  /** @copydoc EventSink::flush */
  void flush(std::stop_token stop = {}) override;

  /** @copydoc EventSink::close */
  void close(std::stop_token stop = {}) override;

 private:
  /** @brief Hides standard stream state from the public ABI. */
  struct Impl;

  std::unique_ptr<Impl> impl_;  ///< Owned descriptor, path, and output stream.
};

/** @brief Configures one caller-owned console or text-stream sink. */
struct ConsoleEventSinkOptions {
  std::string sink_id = "sink.console";  ///< Stable identity within an outbox.
  std::string destination_id =
      "console.stdout";  ///< Stable logical destination used in topology.
};

/**
 * @brief Writes canonical event JSON lines to a caller-owned output stream.
 *
 * The stream must outlive the sink. `close()` flushes but never destroys or
 * closes the borrowed stream.
 */
class ConsoleEventSink final : public EventSink {
 public:
  /**
   * @brief Binds a console sink to a caller-owned output stream.
   * @param output Stream that outlives this sink.
   * @param options Stable sink and logical destination identities.
   * @throws InvalidArgument If identity fields are invalid.
   */
  explicit ConsoleEventSink(std::ostream& output,
                            ConsoleEventSinkOptions options = {});

  /** @copydoc EventSink::descriptor */
  const EventSinkDescriptor& descriptor() const noexcept override;

  /** @copydoc EventSink::publish */
  void publish(const EventEnvelope& envelope,
               std::stop_token stop = {}) override;

  /** @copydoc EventSink::flush */
  void flush(std::stop_token stop = {}) override;

  /** @copydoc EventSink::close */
  void close(std::stop_token stop = {}) override;

 private:
  EventSinkDescriptor descriptor_;  ///< Immutable sink topology descriptor.
  std::ostream* output_;  ///< Non-null borrowed stream owned by the caller.
  bool closed_ = false;   ///< Whether explicit finalization completed.
};

}  // namespace jyppx::liuding::training

#endif  // JYPPX_LIUDING_TRAINING_EVENT_SINK_HPP