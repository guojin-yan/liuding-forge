# 检查点参与者

公共头为 `<jyppx/liuding/foundation/checkpoint.hpp>`，链接 `jyppx::liuding::foundation`。接口类型位于 `jyppx::liuding`；foundation组件提供与存储无关的状态协议，training负责多参与者协调，interop负责容器、摘要与文件发布。

## 状态与恢复等级

每个 `CheckpointParticipant` 提供稳定且唯一的 `participant_id()`、正整数 `schema_version` 和完整的 `StateManifest`。字段名是逻辑标识，不是文件路径；每个字段声明媒体类型、正数 `max_bytes` 和是否必需。实际长度、摘要与构建身份由[制品清单](checkpoint_artifact_manifest.md)记录。

| 等级 | 含义 | 可接受策略 |
| --- | --- | --- |
| `RestoreLevel::exact` | 在声明兼容环境下等价续行 | 全部策略 |
| `RestoreLevel::bounded` | 在明确的数值或数据顺序边界内等价 | `allow_bounded`、`allow_best_effort` |
| `RestoreLevel::best_effort` | 仅恢复列出的状态 | `allow_best_effort` |

非exact清单须提供至少一个降级原因。`exact`描述该参与者的承诺，不自动证明完整训练域、跨设备或跨后端逐位等价；调用方须登记所有实际状态所有者，并在相应环境验证恢复。

## 保存与载入

`save_state(StateWriter&)`写入全部必需字段。writer必须在返回前消费或复制传入字节，拒绝未声明、重复和超限字段；它不接管参与者数据的所有权。

`load_state(const StateReader&, RestorePolicy)`应先验证reader中保存的清单、恢复策略、版本和全部字段，再一次性修改实时状态。发生异常时该参与者必须保持原状态；跨参与者回滚由[训练协调器](checkpoint_coordinator.md)负责。不能用当前对象的清单代替存储清单判断迁移和恢复等级。

`read_blob(name, max_bytes)`在返回视图前检查上限，返回内容由reader持有。文件适配器的reader只在对应存储对象下一次成功提交或销毁前有效；需要长期保存内容时由调用方复制。参与者负责定义可移植编码和版本迁移，不能直接持久化包含指针、padding或本机字节序的C++对象内存。

可运行的[自定义状态示例](../../examples/training/checkpoint_state.cpp)演示8字节小端计数器、两个参与者的新进程恢复、应用前置条件拒绝时的回滚和预算不足时的预检拒绝；运行方式见[持久化检查点](durable_checkpoint.md#自定义状态示例)。
