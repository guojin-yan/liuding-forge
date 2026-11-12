# 检查点协调器

公共头为 `<jyppx/liuding/training/checkpoint_coordinator.hpp>`，链接 `jyppx::liuding::training`。`CheckpointCoordinator`将[参与者协议](checkpoint_participant.md)组合成事务保存与恢复；[文件存储](durable_checkpoint.md)提供实际容器和原子发布。

## 登记与所有权

`CheckpointParticipantRegistry`通过shared pointer持有参与者。单个或整组登记会先验证ID、清单和唯一性；`register_participants()`任一项失败时整个登记组都不生效。遍历始终按ID升序，与登记顺序无关。

协调器借用registry，registry必须存活更久。登记、保存和恢复不具备线程安全性，调用方须外部串行化，并暂停模型、优化器及其他参与者的状态更新；存储事务不能冻结正在运行的训练。

`CheckpointDescriptor`保存规范顺序的参与者清单，其 `restore_level`必须等于所有参与者中最弱的等级；不包含物理长度、路径和摘要。`validate_checkpoint_descriptor()`拒绝空集合、重复、顺序错误和不一致的等级。

## 保存事务

`save(sink)`先验证并固定完整描述符，再调用一次 `begin()`，逐个请求 `writer_for(id)`并保存状态，最后调用一次 `commit()`。从begin到commit任一阶段失败，协调器调用不抛异常的 `rollback()`后传播原异常。

sink须复制描述符、限制字段并检查必需字段完整性，成功提交时原子发布。具体文件格式、摘要和持久化保证由存储适配器承担。

## 恢复事务

`restore(source, policy)`要求来源提供稳定描述符和已验证的reader。协调器在修改前检查参与者ID集合完全相同、reader清单与来源描述符一致、存储恢复等级被调用方策略接受，并为所有当前参与者捕获有界回滚快照。每个参与者仍须检查自身版本、字段和应用约束。

载入按ID升序进行；任一参与者失败时，所有已尝试参与者（包含失败者）按逆序恢复到操作前快照。回滚成功后重新抛出原失败；回滚自身失败则抛出包含原异常的 `InternalError`，此时调用方须将实时训练状态视为不确定，不能继续训练或将其发布为成功结果。

`CheckpointCoordinatorOptions`默认限制1024个参与者和1GiB回滚预算。预算按当前参与者所有字段的声明最大值合计，包含可选字段；不足时在载入参与者前拒绝，不会静默关闭回滚，也不等于进程RSS上限。

## 与Trainer配合

当前Trainer已接入模型、优化器、调度器、数据、RNG、精度缩放器、回调及其他登记状态，通过 `save_checkpoint()`和 `restore_checkpoint()`在已提交安全边界保存与恢复；具体要求见[训练API](training.md)。协调器本身无法发现漏登的状态，也不将旧容器、任意pickle或未提交梯度转换为可恢复训练域。自定义循环可直接组合registry、coordinator和FileCheckpointStore，参见[可运行示例](durable_checkpoint.md#自定义状态示例)。
