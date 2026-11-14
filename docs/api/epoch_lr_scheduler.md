# 按 epoch 推进的学习率调度器

包含 `<jyppx/liuding/optim/epoch_lr_scheduler.hpp>`，链接 `jyppx::liuding::optim`。四个算法均继承 `LrScheduler`，使用共享的 `OptimizerRateControl`，遵循相同的事务与检查点接口。

| 算法 | 配置和行为 |
| --- | --- |
| `ExponentialLrScheduler` | `gamma` 有限非负，每个 epoch 乘以当前学习率 |
| `MultiStepLrScheduler` | `milestones` 非空、为正且严格递增，位于 int64 范围；指定 epoch 应用 `gamma` |
| `CosineAnnealingLrScheduler` | `t_max` 为正且在 int64 范围；`eta_min` 有限非负且不超过任一基础学习率；使用可组合的递归余弦轨迹 |
| `CosineAnnealingWarmRestartsLrScheduler` | `t_0`、`t_mult` 为正且在 int64 范围；由基础学习率计算余弦，每个边界重启并扩展周期 |

options 包含稳定 `scheduler_id`。全部选项、构造时基础学习率、optimizer ID 和有序 group ID 写入不可变描述符；恢复时任何不一致都会在修改前拒绝。

`step()` 先读取并校验全部组学习率，预览完整候选学习率和计数，成功后一次提交全部组并更新调度器状态。指数与 MultiStep 使用当前学习率，可用于 `ChainedLrScheduler`；重启余弦保存周期索引、长度与位置，并检查其与 `completed_steps` 一致。周期长度或索引溢出在修改前拒绝。

`LrSchedulerCheckpointParticipant` 使用 participant schema3：`topology` 保存完整描述符，`control` 保存状态树及当前组学习率，仅接受当前格式与 exact 恢复策略。Trainer 接入和其他算法见[调度器适配](scheduler_adapter.md)与[优化器总览](optim.md)。
