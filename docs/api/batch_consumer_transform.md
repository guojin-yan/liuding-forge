# 数据批次的提交前变换

`DataPipeline` 和 `IterableDataPipeline` 都可以通过 `options.consumer_transform` 配置一个 `BatchConsumerTransform`。它在调用 `next()` 的线程上运行：factory 组批 → CPU prepare/pin → consumer transfer → 输入检查 → consumer transform → 输出资源与 schema 检查 → 提交 cursor。未配置时保持直接传输、校验和提交的路径。

## 输入、输出和身份

包含 `<jyppx/liuding/data/batch_transform.hpp>`。`input_schema` 描述传输完成后的输入，pipeline 的 `options.schema` 描述最终返回值，因此变换可以改变 dtype、形状或容器结构。配置 CUDA 传输时，factory 仍须返回 CPU Tensor，而输入 schema 应声明传输后的设备。

`id` 必须是非空稳定身份，包含实现版本和全部影响输出的配置；修改缩放、颜色、mean/std、随机策略或实现时也必须更新它。框架保存完整输入和输出 schema，不能从任意 lambda 推断配置。

```cpp
using namespace jyppx::liuding::data;
using namespace jyppx::liuding::data::image;

// prototype 是 factory 将产生的 ImageBatch；numeric 是明确的数值策略。
const auto output = transform_image_batch(prototype, numeric);
options.schema = image_batch_schema(output, device);
options.consumer_transform = BatchConsumerTransform{
    .id = "my-source-stage/v1/" + image_batch_identity(collate, numeric),
    .input_schema = image_batch_schema(prototype, device),
    .deterministic = true,
    .apply = [numeric](const c10::IValue& value,
                       const BatchConsumerContext& context,
                       std::stop_token stop) {
      return image_batch_value(transform_image_batch(
          image_batch_from_value(value), numeric, stop));
    }};
```

在创建 pipeline 前完成配置；选定逻辑 epoch 后调用 `next()`。CPU 和 CUDA 共用接口，CUDA 使用调用方当前 stream。框架不创建隐藏 stream，也不为回调建立额外 event。自定义回调必须保持该流上的依赖；跨 stream 使用数据时，由调用方按 LibTorch 规则建立依赖和生命周期。

## 上下文与确定性

`BatchConsumerContext` 提供 `dataset_version`、`epoch`、未提交的 `sequence`、有序 `sample_ids` 和候选 `source_watermark`，map 的 watermark 为空。所有视图仅在回调期间有效。随机性应由来源/变换身份、epoch、稳定样本身份及显式 seed 派生；不能把物理线程 ID、全局可变 RNG 或未保存的回调计数器当作 exact replay 状态。

回调只接收当前批次，应将输入 Tensor 与容器视为只读，并返回结果；需要修改照片时分配自己的输出。回调须响应 stop token，不得在同一 pipeline 上再次调用 `next()`、生命周期操作或 checkpoint 操作。框架不能回滚回调写文件、修改外部状态或原地修改共享输入造成的副作用，也不保存闭包中的可变状态。

默认 `deterministic=true` 是调用方声明。不能保证同一上下文和配置重放相同值时，设置 `false`。map 的 exact 还要求 sampler 有稳定身份/状态且 factory 确定；iterable 的 exact 还要求 source 声明 exact 且 factory 确定。iterable 新增 `deterministic_batch_factory`；任一计算阶段不确定时，exact 降为 bounded，原本的 best-effort 不会升级。来源 replay/omission 数量仍单独描述，不等于输出数值一致。

## 失败、取消与恢复

factory、传输、consumer transform、资源检查或 schema 检查在成功提交前的同步失败，都不推进 map cursor/sequence 或 iterable 的 committed_samples、source_state、watermark。当前 epoch 停止并释放其队列和运行时；再次 `next()` 会重新报告终止错误。处理问题后，显式 `load_state()` 恢复已提交位置，或用 `set_epoch()` 重新选择 epoch；不自动重试失败的回调。

通过 `next(stop_token)` 请求取消时返回空值，当前 epoch 同样停止，后续未带取消的 `next()` 报终止状态。恢复时会重新组装失败批次；训练步骤若已在 pipeline 外产生副作用，仍须使用训练协调器的 checkpoint 边界管理。`next()` 返回后才调用变换，错误不能回滚已提交 cursor。CUDA 提交后才被观察到的异步错误也不属于同步回滚保证。

## 资源和统计

启用变换后，`memory.max_tensors`、`max_value_nodes` 和 `max_bytes` 即使未启用 pin/transfer，也会检查输入和输出。检查只访问 Tensor 的布局、dtype、shape、device 等主机元数据，不读取 GPU 像素或主动同步。回调输出是在返回后才接受上限检查；这些限制不能阻止用户回调内部的大分配，也不代表 scratch、总 in-flight bytes、GPU allocator 占用或 RSS 门槛。

`BatchEnvelope.memory` 记录变换前 memory stage 的实际输入与传输工作；例如 uint8 照片传输时按 uint8 计数。`consumer_transform_id` 标识实际应用的阶段，`transformed_memory` 单独报告最终输出的 Tensor 次数、逻辑字节和容器节点；别名 Tensor 重复出现时重复计数。未配置变换时它为空。`inspect_batch_memory()` 可单独执行同样的无拷贝检查，不会 pin 或传输数据。ImageBatch 自身的旧 runtime memory 信息仍由数值变换清空，以免误当成输出统计。

## Checkpoint 格式

map 当前写 v5，iterable 当前写 v4，分别保存 consumer 身份、完整输入 schema 和确定性声明；iterable 还保存 factory 确定性。相邻的无 consumer 格式 map v4 / iterable v3 仍可读取完整结构 schema，但不能被静默恢复到新启用 consumer 的配置。

保留 map v3 / iterable v2 的显式 label 迁移：只允许原先可证明等价的 `BatchSchema::any(id, version)`，不能从旧标签推断新的结构约束。所有迁移后重写为当前格式；身份、确定性、schema、来源或 worker/memory topology 不匹配时拒绝恢复，live epoch 保持原状。仓库中的历史 writer fixture 原样保留，新增相邻格式 fixture 由独立二进制格式构造，并已与旧提交 `22e88a8`、Ubuntu LibTorch 2.14.1+cu126 的真实 C++ writer 输出逐字节核对一致；两种来源在 metadata 中分别说明。
