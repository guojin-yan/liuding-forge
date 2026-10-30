# 使用预训练 ResNet50 与 DINOv2 迁移学习

本教程将本地预训练权重接入 ImageFolder 分类，演示冻结骨干、替换分类头、完整微调和独立进程恢复。入口复用框架的 Trainer、图像管线、实际分类评估和模型包；当前选定架构为 `resnet50` 与 `vit_small_patch14_dinov2`，不自动下载权重或数据。

## 1. 准备环境与数据

构建时启用 `JYPPX_LIUDING_BUILD_TOOLS`、`JYPPX_LIUDING_BUILD_MODELS` 和 `JYPPX_LIUDING_BUILD_DATA_IMAGE`，然后构建 `jyppx_cli`。安装和 LibTorch 环境见[项目主页](../../README.md)。训练与验证各使用一个 ImageFolder 目录，两端按目录名排序的类别表必须完全一致，至少两个类别。

```text
dataset/
  train/
    class-a/
    class-b/
  validation/
    class-a/
    class-b/
```

使用自己的图片时声明其许可。Fashion-MNIST 的固定官方数据、离线准备和类别顺序见[图像分类教程](image_classification_recipe.md#7-准备-fashion-mnist)；已准备的灰度图片以精确重复通道的 RGB PNG 接入。数据准备成功不代表已经完成迁移学习或达到任务准确率。

## 2. 准备本地权重

仓库提供两份显式描述文件，固定来源版本、完整权重摘要、原始 head 形状、许可和预处理。它们描述外部输入，不分发权重，也不改变内置候选的可用状态。调用者须核查来源和许可；摘要验证只证明字节身份。

| 架构 | 固定来源 | 描述文件 | 本地输入 |
| --- | --- | --- | --- |
| ResNet50 | [timm/resnet50.a1_in1k](https://huggingface.co/timm/resnet50.a1_in1k/tree/767268603ca0cb0bfe326fa87277f19c419566ef) | [resnet50-a1.weights.json](../../recipes/transfer-learning/resnet50-a1.weights.json) | 原始 `model.safetensors`，102,469,840 字节 |
| DINOv2-S | [官方 facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056) | [dinov2-small.weights.json](../../recipes/transfer-learning/dinov2-small.weights.json) | 下述离线转换结果，88,240,512 字节 |

从固定来源自行取得文件，保存在仓库外。DINOv2 的原始 HF 文件需要使用已有转换工具显式合并 Q/K/V、映射 state 键并省略无掩码图像编码器不使用的 mask token；原始文件不能直接冒充转换结果。

```sh
python scripts/models/convert_dinov2_hf.py --source /weights/dinov2-small.facebook.safetensors --destination /weights/dinov2-small.liuding.safetensors --report /weights/dinov2-small.conversion.json
```

工具只接受对应固定来源，离线运行，拒绝覆盖既有输出。描述文件中的 `sha256` 对应最终载入的 safetensors，`source_revision` 和 `state_mapping_version` 则保留来源与映射身份。`source_num_classes` 用于先按原始拓扑严格加载；随后删除原 head，按当前数据集类别数创建新的带偏置 Linear。迁移后的列顺序由数据集决定，不使用原始 ImageNet 类别表。

## 3. 配置与预处理

复制 [ResNet50 模板](../../recipes/transfer-learning/resnet50.json)或 [DINOv2-S 模板](../../recipes/transfer-learning/dinov2-small.json)，并修改 `train_root`、`validation_root`、`data_license`、`weights_file` 和 `weights_metadata`。所有相对路径均相对于配置文件所在目录；复制模板后应同时复制对应的权重描述文件，或为它指定绝对路径。

```sh
liuding-forge recipe schema --name transfer --format json
liuding-forge recipe check --file my-transfer.json --format json
liuding-forge recipe run --file my-transfer.json --root runs/transfer --format json
```

`check` 检查闭合配置、类别和完整图片身份、权重描述及文件大小/SHA-256；`run` 还要严格验证并导入实际模型的全部 state 键。字段拼写错误、错误类型、重复 JSON 键、摘要不符和不匹配的源 head 都会报错，不会回退到随机权重。

当前输入配方是 RGB float32 `[0,1]`，按短边做 antialiased bicubic 缩放，再使用 nearest-even 中心偏移裁剪到 224×224，最后执行 `(pixel-mean)/std`。ResNet50 短边 235，DINOv2-S 短边 256；两者使用描述文件中的 ImageNet mean/std。该配方保留原图到当前画布的 GeometryTrace，不声称与 PIL 插值逐位相同。图像抗锯齿与 DINO 的位置编码插值是两项独立设置；后者由 `position_antialias` 显式指定。

## 4. 冻结与完整微调

| 配置 | 骨干梯度 | BN 与 dropout | 分类头 |
| --- | --- | --- | --- |
| `backbone_mode: "frozen"`、`batch_norm: "eval"` | 关闭，前向不记录 autograd 图 | 骨干始终 eval，包含 running mean/variance 等 buffer | 新 Linear 训练 |
| `backbone_mode: "finetune"`、`batch_norm: "eval"` | 全部开启 | BN 统计保持 eval，BN affine 参数仍有梯度；dropout 随 train/eval 切换 | 新 Linear 训练 |
| `backbone_mode: "finetune"`、`batch_norm: "train"` | 全部开启 | BN 和 dropout 随 train/eval 切换 | 新 Linear 训练 |

冻结模式不允许 `batch_norm: "train"`。Trainer 每轮切换训练与验证模式时仍会执行上述策略；仅关闭 `requires_grad` 不足以冻结 BN 统计。使用训练态 BN 时，训练批次及最后一个尾批次至少需要两张图片。DINOv2 没有 BatchNorm，相关策略不会将其 LayerNorm 切换为另一种计算规则。

先把冻结骨干作为基线：数据量较小、类别与预训练特征接近、显存有限时，可以先只训练新分类头，观察数据、类别顺序和预处理是否适合当前任务。若冻结特征无法满足任务，再考虑完整微调，让骨干适应新的图像分布。完整微调增加内存和训练成本，也可能破坏已有特征；它不保证优于冻结。冻结任务的准确率不能证明完整微调已收敛。

优化器使用带 `momentum` 和 `weight_decay` 的 SGD，每轮更新完成后把各组当前学习率乘以 `lr_gamma`。`learning_rate` 保持必填；新增可选的 `backbone_learning_rate` 与 `head_learning_rate`，只适用于 transfer 配方：

| 配置 | 实际 SGD 分组及初始学习率 |
| --- | --- |
| 两个可选字段均省略 | 保持原来的 `main` 单组，全部可训练参数使用 `learning_rate` |
| 完整微调，提供任一可选字段 | 固定顺序为 `backbone`、`head`；已提供的值覆盖该组，省略的组回退到 `learning_rate` |
| 冻结骨干，只提供 `head_learning_rate` | 只有 `head` 组；骨干不加入优化器 |

可选值必须是有限的数且位于 `(0,1]`；冻结模式不接受 `backbone_learning_rate`，避免配置写了却不起作用。例子中的数值只解释字段对应关系，不是通用训练建议，也没有对应的准确率保证：

```json
{
  "backbone_mode": "finetune",
  "batch_norm": "eval",
  "learning_rate": 0.001,
  "backbone_learning_rate": 0.0001,
  "head_learning_rate": 0.01,
  "lr_gamma": 0.9
}
```

这是需要合并到完整模板的配置片段。首轮两组分别使用 `0.0001` 和 `0.01`；轮末变成 `0.00009` 和 `0.009`，比例保持不变。骨干较小的学习率用于减少已有表示的改动，新头的学习率可以独立选择。学习率、epochs、batch 和质量阈值仍需针对任务确定；不要因为分组能够执行，就把未达标的微调写成合格任务。

冻结前向可以减少 autograd 内存，但仍逐批执行骨干，不隐式缓存整个数据集的特征。单图、prefetch、数据总量及图像逻辑字节受配置限制；256 MiB 的单制品上限覆盖本教程的两种骨干。逻辑预算不等同于进程 RSS 或显存 allocator 的严格上界。

## 5. 恢复和交付

```sh
liuding-forge recipe run --file my-transfer.json --root runs/resumable-transfer --pause-after 10 --format json
liuding-forge recipe resume --file my-transfer.json --root runs/resumable-transfer --format json
```

暂停点是绝对优化器更新次数，必须小于总次数；完整模型、head、optimizer、scheduler、RNG、数据游标和 Trainer 状态一起保存。恢复要求权重字节、完整描述、冻结/BN 策略、配置、图片内容、类别和构建身份一致；骨干/head 初始学习率、分组参数身份和顺序也绑定到同一运行。改动这些内容时创建新运行。恢复从检查点读取已衰减的当前学习率、SGD momentum 和调度器计数，不用配置的初始学习率覆盖它们，也不多调用一次调度器。已完成运行的同配置 `resume` 返回已验证结果，不重新训练。

暂停点不必与 `checkpoint_every` 对齐。CLI 在 Trainer 安全边界显式保存，不额外推进周期检查点或回调计数；新进程恢复后继续原调度。

输出布局与[分类配方](image_classification_recipe.md#4-训练和评估)相同。`result.json` 额外记录初始/最终骨干与 head 摘要、可训练/冻结参数数目以及权重输入身份；显式启用分组学习率时还记录 `optimization.groups` 的参数名、初始/最终学习率和调度器计数，并在模型包 provenance 中保留相同信息；冻结配方在发布模型前核对全部骨干参数和 buffer 未变化。模型包保留实际缩放/裁剪/mean/std、权重来源、模型选项、冻结策略及当前数据集类别，state 名称使用 `parameter.backbone.*`、`parameter.head.*` 和 `buffer.backbone.*`。

CPU 与 CUDA 需要分别验收。显式使用 `device: "cuda"` 时还需可用的 LibTorch CUDA 环境，并设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；没有设备时明确拒绝，不自动转 CPU。本配方为 FP32，TF32 关闭，不隐式启用 AMP。模型包当前无密码学签名，传递 SBOM、完整任务矩阵和发布资格仍需独立验证。


## 6. 用 C++ 公共 API 理解骨干与分类头

[完整 C++ 示例](../../examples/transfer_learning/group_rates.cpp)展示同一组公共 API 的使用顺序：创建无分类头的 DINOv2-S → 严格加载本地预训练权重 → 注册新的二分类头 → 设置冻结策略 → 收集命名参数并创建 SGD 分组 → backward/step → 轮末 StepLR。它不依赖 CLI 的私有实现，也不调用 Python。

启用 `JYPPX_LIUDING_BUILD_EXAMPLES`、`JYPPX_LIUDING_BUILD_MODELS` 和 `JYPPX_LIUDING_BUILD_OPTIM` 后，显式构建示例目标（Windows 多配置树加 `--config Release`）：

```sh
cmake --build out/build/<profile> --target jyppx_transfer_group_rates --parallel 2
liuding-forge-example-transfer-group-rates /weights/dinov2-small.liuding.safetensors frozen
liuding-forge-example-transfer-group-rates /weights/dinov2-small.liuding.safetensors finetune
```

这两个命令各创建一次新模型。示例使用两张生成的 RGB float32 图片与 `[0,1]` 类别标签做一次更新，用于看清参数与学习率关系；它不是数据集准确率测试。文件大小、SHA-256 和 state 拓扑仍严格检查，没有权重时会报错。真实图片的解码、短边缩放、裁剪和类别映射由前面的 ImageFolder 配方承担；在自己的 C++ 程序中则需显式接入 `data_image` 管线，不能用随机图片替代任务验收。

示例的 `TransferClassifier` 把骨干和新头注册为 `backbone`、`head` 两棵子树。冻结模式既关闭骨干 `requires_grad`，也在每次 `train()` 后让骨干保持 eval，并用 `NoGradGuard` 关闭其前向图；分类头继续训练。完整微调则把所有骨干参数放进 `backbone` 组，把新头放进 `head` 组。DINOv2 没有 BN；迁移到 ResNet 时，完整微调还需像 CLI 一样明确选择 BN 的 eval/train 策略。

```cpp
auto optimizer = std::make_shared<optim::SgdOptimizerAdapter>(
    "example.transfer.sgd",
    std::vector<optim::SgdParameterGroup>{
        {.group_id = "backbone", .parameters = backbone_bindings,
         .options = {.learning_rate = 0.0001, .momentum = 0.9}},
        {.group_id = "head", .parameters = head_bindings,
         .options = {.learning_rate = 0.01, .momentum = 0.9}}});
optim::StepLrScheduler scheduler(
    optimizer, {.scheduler_id = "example.transfer.lr", .step_size = 1,
                .gamma = 0.9});
```

其中每个 `OptimizerParameterBinding` 包含稳定参数名及模型拥有的 tensor handle。骨干导入、head 替换、device/dtype 放置及冻结策略都应在创建 optimizer 前完成；更换 head 或可训练参数集合后，需要重新建立 optimizer 和 checkpoint 拓扑，不能把旧 momentum 对到一组新参数上。`learning_rates()` 的顺序与 `parameter_group_ids()` 一致，示例会打印更新前后各组学习率、绑定张量数量、patch/head 的实际变化量。

示例刻意不保存一个不完整的“仅模型”恢复文件。真实可恢复训练应把同一个 `SgdOptimizerAdapter` 交给 `SgdOptimizerCheckpointParticipant`，把同一个 scheduler 包装进 `training::SchedulerAdapter`，与模型、RNG、数据游标、Trainer 一起交给 `Trainer::prepare()`；创建同一拓扑后调用 `restore_checkpoint()`，再继续 `fit_until()`。完整的运行和进程恢复可直接采用第 5 节 CLI；不能在保存权重后重新创建一个初始 SGD，就声称接续了原训练。
