# Tutorial 4: Callbacks, Checkpoints and Events

Current callbacks observe immutable events. They never receive a writable
Trainer pointer, so callback ordering cannot create hidden model, optimizer or
loop mutations.

## Callback descriptors

Implement `CallbackHandler` with a frozen `CallbackDescriptor`. It declares a
stable ID, subscribed event kinds, execution priority, permitted commands and
whether the callback owns checkpoint state.

```cpp
class StopCallback final : public jyppx::liuding::training::CallbackHandler {
 public:
  StopCallback()
      : descriptor_{
            .type_id = "example.stop.v1",
            .callback_id = "callback.stop",
            .subscriptions = {
                jyppx::liuding::training::CallbackEventKind::epoch_finished},
            .allowed_commands = {
                jyppx::liuding::training::CallbackCommandKind::stop}} {}

  const jyppx::liuding::training::CallbackDescriptor& descriptor()
      const noexcept override {
    return descriptor_;
  }

  jyppx::liuding::training::CallbackCommandRequest handle(
      const jyppx::liuding::training::CallbackEvent& event) override {
    if (event.epoch >= 9) {
      return {.kind = jyppx::liuding::training::CallbackCommandKind::stop,
              .reason = "ten epochs completed"};
    }
    return {};
  }

 private:
  jyppx::liuding::training::CallbackDescriptor descriptor_;
};
```

The dispatcher executes higher priority first and then callback ID order. The
effective command uses fixed semantic priority:
`stop > request_checkpoint > skip_validation`.

## Early stopping

`EarlyStoppingCallback` is a stateful callback and a separate checkpoint
participant. Its default monitored key is `validation.loss`, matching Trainer's
epoch metric identity.

```cpp
auto early = std::make_shared<
    jyppx::liuding::training::callbacks::EarlyStoppingCallback>(
        jyppx::liuding::training::callbacks::EarlyStoppingOptions{
            .callback_id = "callback.early-stop",
            .monitor = "validation.loss",
            .patience = 5,
            .min_delta = 1e-4});
auto callbacks = std::make_shared<jyppx::liuding::training::CallbackDispatcher>(
    std::vector<std::shared_ptr<jyppx::liuding::training::CallbackHandler>>{early});
request.callbacks = callbacks;
```

The Trainer registers both the dispatcher cursor and stateful handlers in its
checkpoint coordinator.

## Checkpoint commands

A callback requests checkpointing; it does not choose a file format. Trainer
waits for an accumulation-safe boundary and invokes the request's
`checkpoint_action` with the complete coordinator:

```cpp
request.checkpoint_action = [&artifact](
    const jyppx::liuding::training::CheckpointCoordinator& coordinator) {
  coordinator.save(artifact);
};
```

The artifact object implements `CheckpointStateSink` and owns persistence,
atomic publication and physical integrity. Restore uses a matching immutable
`CheckpointStateSource` in a fresh, topology-identical prepared domain.

## Event delivery

Attach an `EventOutbox` when lifecycle records must reach console, telemetry or
another sink. Dispatcher state and the outbox record commit before sink I/O.
The bounded outbox retries only unacknowledged sinks and is itself a checkpoint
participant, preventing duplicate delivery after exact restore.

Task metrics normally belong in `StepOutput::metrics`. Stateful metric objects
that retain observations across events must also be added to
`FitRequest::additional_participants`.
