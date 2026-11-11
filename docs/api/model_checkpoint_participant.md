# 模型检查点参与者

头文件：`#include <jyppx/liuding/nn/model_checkpoint_participant.hpp>`；链接目标：`jyppx::liuding::nn`。

`ModelCheckpointParticipant` 将 `ModelStateView` 中的参数和 buffer 暴露为 exact 检查点状态，实现与存储无关的 foundation 参与者合同，可直接登记到训练检查点协调器。

## 状态合同

Schema 1 写入一个 `topology` 字段及按确定顺序编号的张量字段。拓扑包含稳定状态键、参数或 buffer 角色、dtype、shape、stride、存储别名组、存储偏移和 `requires_grad` 标志；恢复要求与当前已构造模型的完整拓扑一致。

张量按逻辑连续顺序编码为小端原始字节，不使用 `torch::save`、pickle、可执行代码、文件路径或 LibTorch 模块归档。支持常见训练 dtype 的 dense、strided、非重叠稠密张量；稀疏、量化、meta、重叠、未支持 dtype 和无状态模块会显式拒绝。

## 恢复事务

加载先校验 manifest、策略、必需字段、拓扑编码、状态键、角色、dtype、shape、stride、别名、偏移、字节数和 bool 字节，再将全部候选值及当前张量的独立回滚副本暂存在 CPU 内存。CPU 模型也使用独立副本，避免提交时覆盖回滚值。

提交在 no-grad 下逐项拷贝到原张量的当前设备，保留原张量、存储、别名和步幅。恢复不再将两份完整模型暂存到 CUDA；非连续布局的后端拷贝仍可能需要设备临时工作缓冲，设备分配或拷贝错误按事务失败处理。

任一拷贝失败后，按逆序恢复所有已尝试张量，包括可能已被部分修改的失败项。回滚不完整时抛出保留原异常的 `InternalError`；此时模型状态不确定，调用方必须停止训练。

## 资源限制

`ModelCheckpointParticipantOptions` 限制参与者 ID、张量数量、状态键长度、拓扑字节、单张量字节和总声明字节；描述、保存、元数据解析和加载分配均检查这些限制。

恢复期间的主机内存还须容纳 reader、完整候选值、回滚张量，以及协调器另行持有的全部参与者回滚状态。声明字节限制与协调器的 aggregate rollback budget 均不是 RSS 或 CUDA 工作缓冲的硬上限，大模型需要显式预算并实测系统资源。

## 使用边界

本参与者恢复张量值并保留模型现有拓扑，不恢复 train/eval 模式、优化器、梯度、hook、任意模块属性、设备策略或调用 schema；这些状态由其他参与者或 Trainer 管理。Schema 迁移、状态键重映射、宽松部分加载、互操作摘要、压缩和制品发布不属于此 schema。

描述、保存和加载期间，调用方必须暂停模型修改。本类共享持有模型，但不提供内部同步。
