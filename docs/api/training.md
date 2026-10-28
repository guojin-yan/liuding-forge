# API：训练与评估

训练接口位于 `jyppx::liuding::training`。模型调用、目标、损失、检查点内容与调度器时点均由调用方显式配置。

## Trainer 与 FitRequest

使用 `<jyppx/liuding/training/trainer.hpp>` 创建 Trainer，然后调用 `prepare(request)`、`fit()`，最后显式 `close()`。

```cpp
jyppx::liuding::training::Trainer trainer({
    .trainer_id = "trainer.main",
    .run_id = "run.training",
    .trace_id = "trace.training",
    .max_epochs = 20,
    .accumulate_steps = 4,
    .gradient_clip_norm = 1.0,
});
```

`max_steps == 0` 表示按 epoch 限制。梯度累积尾批由 `accumulation_tail` 选择校正后更新或丢弃；恢复时必须保持这些训练策略一致。

`FitRequest` 绑定模型、优化器、数据管线、任务函数与执行策略。模型、每个优化器及 RNG 的 checkpoint participant 为必需项；scheduler、callbacks、outbox、诊断与其他任务 participant 可选，启用后一起保存恢复。

- `ModelAdapter` 校验模型输入输出 schema；`BatchMapper`、可选 `TargetMapper` 与 `StepFunction` 定义任务语义。
- `BatchPipeline` 接受 map-style `DataPipeline` 或流式 `IterableDataPipeline`；训练与验证使用独立游标。
- `TrainingStrategy` 负责设备、反向传播、清梯度、更新与 barrier。准备前优化器应无历史更新和原生状态；准备后不得改变参数 dtype/device。
- 多个 `OptimizerExecutionBinding` 的参数必须互不重叠，并完整覆盖可训练参数；按稳定 ID 顺序共同更新成功后才推进一次 `global_step`。任意后端更新或调度失败会终止训练，不承诺撤销后端副作用。

训练函数返回可微标量 loss 和与 batch sample ID 一致的 `sample_count`。验证可不返回 loss，但 metric scheduler 仍需取得配置的 `scheduler_metric`。

```cpp
auto step = [](const jyppx::liuding::training::StepInput& input) {
    const auto prediction = input.model.invoke(input.invocation).value.toTensor();
    return jyppx::liuding::training::StepOutput{
        .loss = torch::mse_loss(prediction, input.target->toTensor()),
        .sample_count = input.batch.sample_ids.size(),
    };
};
```

`fit()` 从准备或恢复状态执行一次，返回 `FitResult`。非有限 loss/梯度、空训练 epoch、外部优化/调度更新或缺失恢复状态会被拒绝。`StepOutput::metrics` 必须有限，epoch 结果按 `sample_count` 加权。

## 检查点、回调与调度

`save_checkpoint(sink)` 保存全部已注册 participant；新建同拓扑执行域后调用 `restore_checkpoint(source)`，再 `fit()` 续训。协调器预检描述符、保留有界回滚快照并事务恢复。Trainer 当前为 schema 4，只接受当前格式；诊断执行角色也属于拓扑，不能用相同 ID 的闲置额外 participant 替代。文件存储与资源限制见[持久检查点](durable_checkpoint.md)。

`CallbackHandler` 接收不可变事件，声明订阅、顺序、状态性与允许命令。`stop > request_checkpoint > skip_validation` 为固定优先级；有状态 callback 单独保存。内置 `EarlyStoppingCallback` 默认监测 `validation.loss`。`EventOutbox` 在 callback 状态及分发游标提交后投递事件。`SchedulerAdapter` 只接受声明的 batch、epoch 或 metric 事件，拒绝重复和缺口；Trainer 是唯一事件推进者。

## 混合精度与任务指标

`TrainerOptions::precision` 选择 full、CPU bfloat16、CUDA float16 或 CUDA bfloat16。CUDA float16 必须绑定启用的 `GradScaler` 及其 participant；Trainer 反缩放所有优化器后统一决定更新，一处非有限梯度便跳过整组、更新一次 scale，不推进 `global_step` 或 batch scheduler。

指标头位于 `<jyppx/liuding/training/metrics/...>`；需要恢复的指标通过 `additional_participants` 注册。loss 头位于 `<jyppx/liuding/nn/loss/...>`，提供 Focal、contrastive、Dice 与 label-smoothing cross entropy 的函数/模块接口；蒸馏的软硬损失、温度和 teacher 梯度策略由任务配方确定。

## 训练诊断

`<jyppx/liuding/training/diagnostics.hpp>` 提供可选 `TrainingDiagnostics`。绑定后复用既有有限值检查记录训练/验证 loss 的 NaN/Inf 次数、非有限梯度尝试和 AMP 溢出跳步；默认不额外测范数或读取时钟。

```cpp
auto diagnostics = std::make_shared<jyppx::liuding::training::TrainingDiagnostics>(
    jyppx::liuding::training::TrainingDiagnosticsOptions{
        .gradient_norm_every = 16,
        .measure_data_wait = true,
    });
request.diagnostics = diagnostics;
// trainer.prepare(request); trainer.fit();
const auto statistics = diagnostics->snapshot();
```

`gradient_norm_every=N` 在第 N、2N…次优化尝试采样，0 关闭；AMP 跳步也推进尝试游标，但不产生有限范数样本。范数覆盖所有优化器的梯度，在尾批校正与 AMP 反缩放后、裁剪前计算。分块 FP64 缩放归约支持 CPU/CUDA 与非连续 Tensor，不 flatten 或修改原梯度；`gradient_norm` 限制 Tensor 数、总元素、tile 和已知临时 payload，超限在范数 Tensor 分配及优化器更新前失败。独立调用可使用 `measure_gradient_l2(gradients, limits)`。

数据等待计时只统计成功返回非空 batch 的 `next()` 主机耗时，包含消费端变换/传输，排除 EOF 和抛异常的读取；它不是纯 I/O 等待或同步 GPU 执行时间。`snapshot()` 保留精确 uint64 计数、采样位置、范数与累计耗时；participant 自动接入 Trainer 检查点，恢复已记录值，不承诺未来墙钟耗时一致。

采样值随 optimizer event、等待值随 batch event、运行累计问题的 0/1 标志随 terminal event 发布。任务指标不得使用保留前缀 `diagnostics.`。中立遥测目的地需将 `training_diagnostic_metric_bindings()` 合并进 `TrainingTelemetryAdapterOptions::metrics`，再配置既有 outbox/run sink；这些事件可由现有 run/报告链消费，尚无独立诊断报告导出器。已知 Tensor payload 限额不包括后端私有 workspace、allocator 驻留或完整 RSS/显存；采样尤其在 CUDA 上会增加同步与扫描成本，应按工作负载测量。

## 分类数据集评估

`<jyppx/liuding/training/classification_evaluation.hpp>` 提供 `training::evaluate_classification(model, source, options)`。`source` 每次返回一批模型参数和一维整数标签，完整读取后返回 `std::nullopt`；模型返回形状为 `N × C` 的有限浮点 logits。数据读取、预处理和设备放置由调用方明确完成，评估器不联网。

结果包含整体和显式来源切片的准确率、宏平均 precision/recall/F1、逐类 support/precision/recall/F1，以及行是真实类别、列是预测类别的混淆矩阵。宏平均包含所有固定类别；logits 并列时取最小类别索引，`ignore_index` 和零分母策略显式配置。空数据或全被忽略的切片返回缺失标量，不记为零分或通过。拆批后汇总使用整数计数，不平均每批指标。

```cpp
training::ClassificationEvaluationOptions options;
options.num_classes = 10;
options.max_samples = 10000;
auto result = training::evaluate_classification(model, next_batch, options);
// next_batch(std::stop_token) -> optional<ClassificationEvaluationBatch>
// 使用 result.observations 作为 experiment::run_evaluation 的实际观测值。
```

配置 `sources` 后，每行必须在 `source_indices` 中指定唯一来源；结果顺序与配置一致。`observations` 自带指标定义、切片上下文和有效样本数，可交给现有 `experiment::run_evaluation` 产生带阈值的质量报告；调用方必须绑定真实的模型、数据、配置和构建身份。

评估关闭梯度并临时切换为 eval，结束或异常后恢复各子模块原来的模式。取消、超时、非法输出和超限均抛异常，不返回部分数据的完整报告。评估期间应独占模型，并保持模块拓扑稳定；自定义 `train()` 必须支持恢复。批次数、样本数、类别、来源和已知混淆矩阵 payload 有显式上限，不保存全量 logits；这些限制不等于模型工作空间、allocator 或 RSS 的硬上限。

## 语义分割 IoU/Dice 与数据集评估

`<jyppx/liuding/training/metrics/segmentation.hpp>` 的 `metrics::SemanticSegmentationMetric` 接受 `N,C,H,W` 有限浮点 logits 或显式指定的 `N,H,W` 整数类别 mask，以及同设备的 `N,H,W` 整数标签。整数类型为 uint8 或 int8/int16/int32/int64。默认按数据集计算 macro IoU；`score=dice` 选择 Dice，`average` 支持 micro、macro、真实像素数加权和仅逐类结果。类别必须映射为 `[0,C)`；不执行阈值、resize、标签重映射或实例匹配。

```cpp
metrics::SemanticSegmentationMetric iou({
    .num_classes = 3,
    .ignore_index = 255,
    .include_background = false});
iou.update(logits, labels);
auto dataset_iou = iou.compute();
auto class_iou = iou.per_class_scores();
```

`ignore_index` 去除对应像素；`include_background=false` 仅从汇总排除 `background_index`，背景与前景之间的错误仍计入前景指标。真实和预测均不存在的类别按 `absent` 选择排除、零、一或报错。空集、全忽略或没有可汇总分母时抛 `NotReady`；逐类被排除的值使用 `std::nullopt`。统计使用每类三个 int64 计数，支持相同身份的 `merge()` 和当前格式的 exact 检查点；拆批、合并或恢复均从总计数计算，不能平均每批 IoU/Dice。每次更新按 `chunk_pixels` 分块，不创建整幅 one-hot；逻辑限额不等于 RSS/显存硬上限。

`<jyppx/liuding/training/segmentation_evaluation.hpp>` 的 `training::evaluate_semantic_segmentation(model, source, options)` 实际逐批推理，返回整体、逐类和显式来源统计；其 `average` 选择 micro、macro 或 weighted。`source` 返回 `SegmentationEvaluationBatch`，以 `std::nullopt` 结束；来源按图像指定，数据准备及设备放置由调用方负责。评估关闭梯度、恢复原模块模式，并在错误、取消、超时或超限时抛异常。`observations` 可接入 `experiment::run_evaluation`；选择与 suite 规则匹配的整体/类别/来源观测值，并绑定真实数据、模型、配置和构建身份。评估器不持久化中途结果，不提供崩溃恢复；检测 AP 与实例 mask AP 不属于此接口。

## 轴对齐框 AP 与数据集评估

`<jyppx/liuding/training/metrics/detection.hpp>` 的 `metrics::DetectionAveragePrecisionMetric` 接受完整图像的 `DetectionImage`：稳定且唯一的 uint64 图像 ID、显式来源、真实框和预测框。框为同一坐标系的有限连续 `xyxy`，宽高严格为正；类别映射到 `[0,C)`，分数为任意有限实数。必须指定不可变的 `dataset_id` 和有序类别映射的 `class_vocabulary_id`，调用方负责将其绑定真实数据、split 和筛选条件。

默认按 IoU 0.50–0.95（间隔 0.05）计算 101 点插值 AP（召回率网格为 COCO 的 FP64 `k*0.01`，与 FP64 recall 比较，保留浮点边界行为），再对有有效 GT 的类别和阈值取宏平均。`iou_thresholds`、闭区间 `min_area/max_area`、每图每类的 `max_detections_per_image_class` 和无 GT 类的排除/零/报错策略均显式进入身份；GT 面积可指定标注面积，否则用框面积。maxDet 在匹配前按分数截断，类别相互独立。每图每类贪心匹配优先非忽略 GT；等 IoU 取标注顺序中的最后一个。crowd 不计入有效 GT，用交集除以预测框面积，可吸收多个预测；普通 ignore GT 仅匹配一次。未匹配且面积越界的预测不计 FP。分数并列按图像 ID、预测行号排序，不做隐式 NMS、阈值、裁剪或坐标还原。

```cpp
metrics::DetectionAveragePrecisionMetric ap({
    .dataset_id = dataset_fingerprint,
    .class_vocabulary_id = vocabulary_fingerprint,
    .num_classes = 10});
ap.update(complete_images);
auto dataset_ap = ap.compute();
auto details = ap.summarize(); // 逐类、逐 IoU AP 和最大 recall
```

整图拆批、无重复图像的 `merge()` 和当前格式 exact 状态恢复保持同一数据集结果；不能平均 batch AP。重复图像（包括空图）会被拒绝，合并要求完整语义和数据身份相同。状态只保留各阈值匹配位，不保留完整框或 IoU 矩阵，来源切片共享记录；图像数、输入数、匹配操作、逻辑状态和工作量均有显式限额。这些限制不等于 allocator/RSS 或模型显存上限。

`<jyppx/liuding/training/detection_evaluation.hpp>` 的 `training::evaluate_detection(model, source, options)` 实际逐批调用模型，要求每图返回一个 `(boxes[D,4], scores[D], labels[D])` tuple，所有图构成 list；框和分数是浮点数，类别为 uint8/int8/int16/int32/int64。同图三个 Tensor 在同一 CPU/CUDA 设备，支持非连续 strided 输出，经有限额的 CPU FP64/int64 暂存后匹配。`source` 提供 `DetectionEvaluationBatch`，其中 `images` 只包含元数据和 GT，`predictions` 必须为空；EOF 返回 `std::nullopt`。可用 `expected_images` 拒绝提前结束。评估关闭梯度并恢复各子模块原模式，错误、取消、超时、超限不返回部分报告。整体/类别/显式来源的 `observations` 按 suite 选取后可接现有质量报告；不提供评估游标持久化、mask AP、旋转框或姿态指标。合成模型测试只验证执行契约，真实检测任务精度需另行验收。
