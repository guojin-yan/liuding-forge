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

当前优化器是带 `momentum` 和 `weight_decay` 的 SGD，所有可训练参数共用 `learning_rate`，每轮按 `lr_gamma` 衰减；骨干和新分类头没有独立学习率。完整微调需要单独选择学习率、epochs、batch 和质量阈值，降低共同学习率也会减慢新分类头的学习；模板不是所有数据集通用的超参数或准确率保证，冻结任务的准确率不能证明完整微调已收敛。

冻结前向可以减少 autograd 内存，但仍逐批执行骨干，不隐式缓存整个数据集的特征。单图、prefetch、数据总量及图像逻辑字节受配置限制；256 MiB 的单制品上限覆盖本教程的两种骨干。逻辑预算不等同于进程 RSS 或显存 allocator 的严格上界。

## 5. 恢复和交付

```sh
liuding-forge recipe run --file my-transfer.json --root runs/resumable-transfer --pause-after 10 --format json
liuding-forge recipe resume --file my-transfer.json --root runs/resumable-transfer --format json
```

暂停点是绝对优化器更新次数，必须小于总次数；完整模型、head、optimizer、scheduler、RNG、数据游标和 Trainer 状态一起保存。恢复要求权重字节、完整描述、冻结/BN 策略、配置、图片内容、类别和构建身份一致；改动这些内容时创建新运行。已完成运行的同配置 `resume` 返回已验证结果，不重新训练。

输出布局与[分类配方](image_classification_recipe.md#4-训练和评估)相同。`result.json` 额外记录初始/最终骨干与 head 摘要、可训练/冻结参数数目以及权重输入身份；冻结配方在发布模型前核对全部骨干参数和 buffer 未变化。模型包保留实际缩放/裁剪/mean/std、权重来源、模型选项、冻结策略及当前数据集类别，state 名称使用 `parameter.backbone.*`、`parameter.head.*` 和 `buffer.backbone.*`。

CPU 与 CUDA 需要分别验收。显式使用 `device: "cuda"` 时还需可用的 LibTorch CUDA 环境，并设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；没有设备时明确拒绝，不自动转 CPU。本配方为 FP32，TF32 关闭，不隐式启用 AMP。模型包当前无密码学签名，传递 SBOM、完整任务矩阵和发布资格仍需独立验证。
