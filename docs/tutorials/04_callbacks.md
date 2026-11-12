# 教程 4：回调、检查点与事件

回调观察不可变事件，不持有可写 Trainer 指针；回调顺序不会隐式修改模型、优化器或训练循环。

## 回调描述符

实现 `CallbackHandler` 并提供冻结的 `CallbackDescriptor`，声明稳定 ID、订阅事件、执行优先级、允许的命令及是否拥有检查点状态。

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

调度器先按优先级从高到低，再按回调 ID 排序执行。命令采用固定语义优先级：`stop > request_checkpoint > skip_validation`。

## 提前停止

`EarlyStoppingCallback` 是有状态回调，也是独立检查点参与者；默认监测 `validation.loss`，与 Trainer 的 epoch 指标身份一致。

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

Trainer 将调度游标及有状态处理器登记到检查点协调器中。

## 请求检查点

回调请求保存检查点，由制品对象决定文件格式。Trainer 等待梯度累积安全边界，再将完整协调器交给请求中的 `checkpoint_action`：

```cpp
request.checkpoint_action = [&artifact](
    const jyppx::liuding::training::CheckpointCoordinator& coordinator) {
  coordinator.save(artifact);
};
```

制品对象实现 `CheckpointStateSink`，负责持久化、原子发布及物理完整性；恢复使用匹配的不可变 `CheckpointStateSource`，在全新且拓扑相同的已准备执行域中进行。完整接口见[检查点协调器](../api/checkpoint_coordinator.md)。

## 事件投递

生命周期记录需要写入控制台、遥测或其他接收端时，可绑定 `EventOutbox`。调度状态和出箱记录先于接收端 I/O 提交；有界出箱重试尚未确认的接收端，自身也是检查点参与者。

跨进程失败或恢复旧检查点时，投递语义为至少一次：接收端已持久提交、出箱 ACK 尚未保存的记录可能重放相同字节。接收端应按操作身份与内容实现幂等去重；精确恢复不保证外部接收端绝不重复。具体边界见[训练遥测接口](../api/training_telemetry.md)。

任务指标通常写入 `StepOutput::metrics`。跨事件保留观测的有状态指标对象还需加入 `FitRequest::additional_participants`。
