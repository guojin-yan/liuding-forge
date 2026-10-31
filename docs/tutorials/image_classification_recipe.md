# 用自己的图像训练 ResNet18

本教程使用 ImageFolder 目录、JSON 配置和 `liuding-forge recipe` 命令完成 ResNet18 分类训练、验证集评估、保存检查点及独立进程续训。模型默认从随机参数开始，也可导出并严格复用完整初始状态，支持单设备 CPU 或显式 CUDA、FP32、SGD momentum 或 AdamW、可选梯度裁剪，以及逐 epoch StepLR 或逐成功更新的线性衰减。预训练骨干换头见[迁移学习教程](transfer_learning_recipe.md)。

## 1. 构建工具

使用[支持矩阵](../../compatibility.json)中的编译器和 LibTorch 组合，同时启用 `models`、`data_image` 和 CLI。构建不会自动下载 LibTorch。

```sh
cmake -S . -B out/build/release -DCMAKE_PREFIX_PATH=/path/to/libtorch -DJYPPX_LIUDING_BUILD_MODELS=ON -DJYPPX_LIUDING_BUILD_DATA_IMAGE=ON -DJYPPX_LIUDING_BUILD_TOOLS=ON
cmake --build out/build/release --config Release --target jyppx_cli --parallel 2
```

以下命令使用可在 PATH 中找到的 `liuding-forge`。Windows 构建产物位于 `out/build/release/tools/cli/Release/liuding-forge.exe`，单配置 Linux 构建位于 `out/build/release/tools/cli/liuding-forge`。省略任一可选组件时，该配方会返回明确的不支持错误。

### 1.1 启用离线数据准备

Fashion-MNIST 与 CIFAR-10 共用 zlib 1.3.2 的静态 codec SDK，需要匹配构建平台与编译器；Windows 使用 x64、Release、`/MD`。当前 SDK 由维护者提供，不会自动下载。它只包含准备器需要的 codec 子集，不能用系统 zlib、zlib-ng 或任意完整 zlib 库替代。

SDK 目录必须包含以下文件；Linux 的库名为 `lib/libz.a`，Windows 为 `lib/zlibstatic.lib`。

```text
dataset-codec-sdk/
├── include/zlib.h
├── include/zconf.h
├── lib/libz.a
├── source/zlib-1.3.2.tar.gz
├── share/licenses/zlib/LICENSE
└── provenance.json
```

在第 1 节已配置好的构建目录上启用准备器。将 SDK 根路径和 `REPLACE_WITH_REVIEWED_LIBRARY_SHA256` 替换为所用平台 SDK 的绝对路径及经审查的静态库摘要；两个开关可以分别开启。源码包与许可证摘要如下固定，静态库摘要随平台和编译器变化。

```sh
cmake -S . -B out/build/release -DJYPPX_LIUDING_PREPARE_FASHION_MNIST=ON -DJYPPX_LIUDING_PREPARE_CIFAR10=ON -DJYPPX_LIUDING_DATASET_ZLIB_ROOT=/absolute/path/to/dataset-codec-sdk -DJYPPX_LIUDING_DATASET_ZLIB_SOURCE_SHA256=bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16 -DJYPPX_LIUDING_DATASET_ZLIB_LIBRARY_SHA256=REPLACE_WITH_REVIEWED_LIBRARY_SHA256 -DJYPPX_LIUDING_DATASET_ZLIB_LICENSE_SHA256=e32ff4e00d9d94930537635291da39e7e612703334bf6fde8c7f1686fe8a45a2
cmake --build out/build/release --config Release --target jyppx_cli --parallel 2
```

配置会核对两个头文件、实际源码包、静态库和原始许可证，缺失或摘要不符立即失败。安装后的数据准备命令由 C++ 执行，不需要 Python。

## 2. 准备数据

训练集和验证集分别放在两个目录中。每个子目录是一类，子目录名称和类别集合必须完全一致。支持 `.jpg`、`.jpeg`、`.png`，每张图只有一个分类标签。目录必须保持稳定，不得在训练或续训期间改写图片。

```text
images/
├── train/
│   ├── cat/
│   │   ├── 001.jpg
│   │   └── 002.jpg
│   └── dog/
│       ├── 001.jpg
│       └── 002.jpg
└── validation/
    ├── cat/
    │   └── 101.jpg
    └── dog/
        └── 101.jpg
```

类别 ID 按目录名称的稳定顺序生成。每张图转为 RGB、float32、CHW 和 `[0,1]`，再以 bilinear 缩放到 `image_size × image_size`；不隐式使用 ImageNet 的 mean/std。训练集至少两张图，训练 batch 至少两个样本；若尾 batch 只有一个样本，检查阶段会拒绝，需调整 `batch_size`。验证集可以有单样本 batch，不丢弃样本。

## 3. 编写并检查配置

复制[配置模板](../../recipes/image-classification/resnet18.json)后修改。`train_root` 和 `validation_root` 相对于配置文件目录解析，也可以使用绝对路径；Windows JSON 路径建议写成 `F:/datasets/images/train`。填写实际数据许可或使用条款，并按任务冻结训练预算和 `minimum_accuracy`。模板中的轮数、学习率和阈值不代表你的任务已通过质量验收。

```sh
liuding-forge recipe schema --format json
liuding-forge recipe check --file my-resnet18.json --format json
```

`schema` 的 JSON 响应中，`data` 是可交给 `config lint` 使用的闭合配置 schema；保存时提取这个字段。模板中的必填字段必须提供，可选学习率调度字段按下一节选择；未知字段、重复键、错误类型和越界值会被拒绝。`check` 校验两个类别表，流式读取全部图片字节，输出内容 SHA256、样本数、有效配置和总更新数；它不训练模型。配置上限包括单图字节、两个 split 的总字节、每个 split 的样本数和最多一百万次优化器更新。

`image_budget_bytes` 限制合作图像阶段的已知 decoded、scratch、pinned、GPU payload 和在途 batch 逻辑字节；worker、prefetch 和单图限制共同约束数据准备。它不是整个进程 RSS 或 CUDA allocator 的严格上界，也不包含模型、优化器及检查点缓冲。图像逐样本解码，不全量缓存到内存。

### 3.1 选择学习率调度

省略 `lr_schedule` 或设为 `"epoch_step"` 时，沿用原有行为：每轮成功完成训练后，将当前学习率乘以 `lr_gamma`。此模式不接受 `warmup_updates`、`warmup_start_factor` 或 `lr_end_factor`。

设为 `"linear"` 时，要求 `lr_gamma = 1`，且不能提供上述三个预热字段。对初始学习率 `B`、总更新数 `T` 和已完成成功更新数 `k`，当前学习率为 `B × (1 - k / T)`；第一次更新使用 `B`，最后一次使用 `B / T`，完成后为零。该模式不执行预热，可与 SGD 或 AdamW 组合。

若需按优化器更新预热和衰减，将下列片段合并到完整配置，再运行 `recipe check`：

```json
{
  "learning_rate": 0.001,
  "lr_schedule": "warmup_linear",
  "lr_gamma": 1.0,
  "warmup_updates": 20,
  "warmup_start_factor": 0.1,
  "lr_end_factor": 0.01
}
```

设总优化器更新数为 `T = ceil(训练样本数 / batch_size) × epochs`，`warmup_updates` 为整数 `W`，必须满足 `1 ≤ W < T`；两个 factor 都必须是 `(0,1]` 内的有限数。此模式要求 `lr_gamma = 1`，避免同时配置不会执行的轮末衰减。模式、字段组合和范围错误会在创建 run 目录前拒绝。示例要求 `T > 20`；数值只说明调度关系，需根据任务自行确定。

对初始学习率 `B`，调度在每次成功的优化器更新后推进，当前学习率为 `B × f(k)`，其中 `k` 是已完成的成功更新数：

| 阶段 | 倍率 `f(k)` |
| --- | --- |
| 更新前及预热阶段，`0 ≤ k ≤ W` | `warmup_start_factor + (1 - warmup_start_factor) × k / W` |
| 线性衰减阶段，`W ≤ k ≤ T` | `1 + (lr_end_factor - 1) × (k - W) / (T - W)` |

首次实际更新使用 `B × warmup_start_factor`，完成第 `W` 次更新后达到 `B`；第 `T` 次更新使用 `B × f(T-1)`，完成后达到 `B × lr_end_factor`。这里的逐 batch 调度以成功的优化器更新计数，样本数和微批次数不推进计数。`warmup_linear` 的暂停输出和最终 `result.json` 中，`optimization.groups` 给出各组初始及当前学习率，`optimization.scheduler.completed_steps` 给出已成功更新数；选择调度方式本身不保证任务准确率或训练加速。

### 3.2 选择优化器与梯度裁剪

省略 `optimizer` 或设为 `"sgd"` 时，必须提供 `momentum`，且不能提供 AdamW 专用字段。选择 AdamW 时，从完整配置中删除 `momentum`，再合并下列片段；三个 AdamW 参数必须显式提供，不能只添加片段而保留原来的 `momentum`：

```json
{
  "optimizer": "adamw",
  "learning_rate": 0.001,
  "adam_beta1": 0.9,
  "adam_beta2": 0.999,
  "adam_epsilon": 1e-8,
  "weight_decay": 0.0,
  "gradient_clip_norm": 1.0,
  "lr_schedule": "linear",
  "lr_gamma": 1.0
}
```

`adam_beta1` 和 `adam_beta2` 必须是 `[0,1)` 内的有限数，`adam_epsilon` 必须是有限正数；AdamW 使用解耦权重衰减且不启用 AMSGrad。`gradient_clip_norm` 对两种优化器均可选，必须是有限正数，按全体可训练参数的 L2 范数在更新前裁剪；省略时不裁剪。优化器种类、参数、裁剪阈值和调度方式参与恢复身份校验，修改后应创建新运行。显式选择优化器或裁剪时，暂停结果、`result.json` 和 bundle provenance 的 `optimization` 记录相同配置与实际学习率。示例数值仅说明配置方法，不保证任务质量或性能。

### 3.3 固定完整初始模型

需要让不同运行使用完全相同的参数和缓冲区时，先从完整配置导出零步初始模型。输出目录的父目录必须存在，目标目录必须尚未存在：

```sh
liuding-forge recipe initialize --file my-resnet18.json --root initial-model --format json
```

命令在 C++ 中构造 CPU 模型，输出 `initial-model/initial-state.safetensors` 与 `initial-model/initialization.json`，不创建优化器、训练检查点或 run-store，不执行前向、反向和训练更新。报告包含实际完整模型身份、类别顺序、构建身份、文件大小与 SHA-256；`global_step` 和 `optimizer_steps` 均为零。即使配置目标为 CUDA，初始文件仍是 CPU 上的 FP32 参数和 FP32/int64 缓冲区。

从命令响应的 `data.artifact.sha256` 或输出报告的 `artifact.sha256` 取得完整摘要，将以下字段一起加入原配置；路径相对于配置文件解析，摘要必须是 64 位小写十六进制：

```json
{
  "initial_state_file": "initial-model/initial-state.safetensors",
  "initial_state_sha256": "REPLACE_WITH_INITIALIZATION_REPORT_SHA256"
}
```

`recipe check` 预检文件字节和格式、配方、架构及有序类别元数据；`initialize`、`run` 和 `resume` 还会严格匹配模型的全部参数与缓冲区，不接受缺失键、多余键、形状变化、类型转换或部分导入。导入发生在优化器创建之前，完整输入缓冲随后释放。单文件仍受配方的 256 MiB 制品上限约束；这不是高容量模型的放宽入口。

加入上述字段后，再执行 `recipe initialize` 到另一个新目录；即使改变 `seed`，重新导出的模型文件仍与输入逐字节一致。seed 仍控制采样及训练期 RNG，固定初始参数不保证不同设备的浮点运算或训练结果逐位相同。正常 `recipe run` 会将初始化来源和实际模型身份写入运行证据及 bundle；`resume` 会拒绝更换文件内容、摘要、配置或类别。该文件只提供零步模型状态，完整续训仍使用训练检查点。

### 3.4 训练后重估 BatchNorm 统计

小数据或短训练后，可显式选择用训练集重新估计 BatchNorm 的运行统计。当前配方只在 CPU FP32 ResNet18 分类中支持此功能，默认关闭；迁移学习和 CUDA 配方会拒绝启用。将下列片段合并到完整配置，并重新运行 `recipe check`：

```json
{
  "device": "cpu",
  "batch_norm_calibration": "train_cumulative_once",
  "calibration_max_snapshot_bytes": 134217728
}
```

`calibration_max_snapshot_bytes` 限制完整编码模型快照，允许 1 MiB 至 1 GiB；示例为 128 MiB。只有启用校准时才提供该字段，关闭时应一并删除。实际快照不足以容纳时明确失败，回滚和最终模型编码还会使用额外内存；它不是整个进程的内存限额。

训练完成后，配方以独立单 worker 管线遍历训练集一遍，固定 epoch 0 的可移植顺序和原预处理，不推进原训练游标、不更新权重或优化器，也不使用验证数据选择统计。仅 BatchNorm2d 临时处于 train 模式，关闭梯度并以 `momentum=None` 累计每批统计；尾批不等长时不承诺精确逐样本加权。最终仍使用正常 eval 模式评估完整验证集，结果及质量报告采用这次评估；启用本功能不保证提升准确率。

校准开始前和完成后均保存协调检查点。中途退出时，`resume` 从校准前的完整状态重放整遍，不重复训练；已完成校准则直接继续最终评估与交付。不能在原运行上改配置后续训，启用或关闭应建立新运行。

## 4. 训练和评估

输出目录必须是新目录，其父目录需预先创建。使用 CUDA 时，先将配置中的 `device` 改为 `cuda`，并在启动每个进程前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；PowerShell 可执行 `$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'`，Linux 可执行 `export CUBLAS_WORKSPACE_CONFIG=:4096:8`。CUDA 不可用时会拒绝运行，不自动回退到 CPU。

```sh
liuding-forge recipe run --file my-resnet18.json --root runs/my-run --format json
```

配方固定单个 intra-op/inter-op 线程，启用确定性算法，关闭 cuDNN benchmark 及 cuDNN/cuBLAS 的 TF32，并将 float32 matmul precision 固定为 `highest`；该策略纳入模型与恢复契约。训练每个 epoch 后遍历整个验证集，完成后再次运行分类 evaluator，输出 accuracy 和逐类混淆矩阵，并将 accuracy 与配置中预先声明的阈值比较。退出码 `0` 表示配方通过；`4` 表示训练和交付已完成，但质量阈值未通过；其他错误使用 CLI 的通用错误码。

主要输出如下：

| 路径 | 内容 |
| --- | --- |
| `config.json`、`dataset.json`、`lineage.json`、`build.json` | 有效配置、完整图片内容身份、模型契约和实际构建身份 |
| `latest.jyck` | 按 `checkpoint_every` 或显式暂停边界覆盖的最新运行中检查点 |
| `final.jyck` | 训练完成态的完整检查点 |
| `model.safetensors` | 所有参数和 BN 等 buffer；名称前缀为 `parameter.`、`buffer.` |
| `evaluation-suite.json`、`evaluation.json` | 冻结质量门槛和实际评估报告 |
| `diagnostics.json` | 与运行、配置、数据、模型及最终检查点关联的诊断报告 |
| `run-store/` | SQLite 运行记录与 CAS 制品关系 |
| `bundle/` | 经过完整性校验的 data-only 模型包、输入输出、预后处理、类别和来源说明 |
| `result.json` | 更新次数、评估结果、内容身份和图像逻辑资源峰值 |

模型包当前不签名，SBOM 只包含直接运行时声明，完整传递依赖 SBOM 和签名交付尚未提供。包的完整性校验不能证明模型在其他设备或运行时上的可执行性。导出的 safetensors 是显式的模型状态数据，不含可执行 Python 对象。

## 5. 暂停并在新进程恢复

`--pause-after` 指定绝对优化器更新次数，必须大于已恢复的次数且小于最终次数。以下第一条命令在第 19 次更新后保存完整检查点，再有序结束进程；下一条在新进程中恢复同一目录。

暂停点不必与 `checkpoint_every` 对齐。CLI 在 Trainer 的安全边界显式保存状态，不额外增加训练的周期检查点或回调计数。

```sh
liuding-forge recipe run --file my-resnet18.json --root runs/resumable --pause-after 19 --format json
liuding-forge recipe resume --file my-resnet18.json --root runs/resumable --format json
```

恢复覆盖模型参数和 buffer、所选优化器的状态、完整学习率调度器树、Trainer、训练与验证数据游标、采样器、CPU/选定 CUDA RNG、回调游标和诊断统计。当前学习率、SGD momentum 或 AdamW 一阶/二阶矩与步数、预热或衰减阶段及调度计数一起恢复，不会重新预热或多推进一步。新的 `pause-after` 可以继续分段运行。异常终止后，已有 `latest.jyck` 时也可从该检查点重做尚未持久化的工作；尚未发布首个检查点时不能恢复。

配方使用可移植、带版本的 `RandomSampler`，并在结果的 `sampling` 和模型包来源中保存算法、seed 与训练/验证 sampler 身份。同一数据顺序、seed 和 epoch 在 Windows/Linux 上使用相同排列；跨平台比较还需冻结完整初始参数、预处理和数值策略，不能仅凭相同 seed 推断训练结果相同。旧 `std::shuffle` v1 采样状态不接受为当前算法续训，已有运行应使用原版本恢复。算法及状态契约见[数据 API](../api/data.md#samplers)。

启用 [BatchNorm 校准](#34-训练后重估-batchnorm-统计)时，可用 `--pause-calibration-after N` 在处理 N 批后有序暂停；N 必须大于零且小于校准总批数，不能与 `--pause-after` 同时提供。该暂停会丢弃本遍局部统计并保留校准前检查点，新进程会重放完整校准，不增加训练更新。例如训练集至少有三批时：

```sh
liuding-forge recipe run --file my-resnet18.json --root runs/calibrated --pause-calibration-after 2 --format json
liuding-forge recipe resume --file my-resnet18.json --root runs/calibrated --format json
```

恢复要求有效配置、类别表、全部图片内容和构建身份一致，包括 seed、设备、线程策略、训练预算及全部学习率调度字段。若需要改配置，创建新的输出目录重新训练。暂停的 CLI 结果为 `paused`，run-store 中仍保留同一条 `running` 的未完成逻辑运行；不要在计划续训时将它标为终态。已经通过 `run recover` 等命令标为 `interrupted` 的终态不接受原运行续训。完成后的同配置 `resume` 校验结果与 bundle 后返回既有结果，不重复训练；`run` 始终拒绝覆盖既有目录。同一输出目录同时只允许一个配方进程，操作系统会在进程退出时释放锁。

```sh
liuding-forge checkpoint verify --path runs/my-run/final.jyck --format json
liuding-forge model bundle verify --path runs/my-run/bundle --format json
liuding-forge run verify --root runs/my-run/run-store --format json
```

图片摘要在运行前和交付前检查，数据目录仍须由调用者保持稳定；这不是文件系统快照或对并发替换的隔离保证。文件、模型包和运行记录的发布有各自的持久化边界，不能视为一次跨目录原子事务。

## 6. 使用 CIFAR-10

从 [CIFAR-10 官方页面](https://www.cs.toronto.edu/~kriz/cifar.html)自行下载 **binary version** 的 `cifar-10-binary.tar.gz`，保存到仓库外。数据集适用原始条款，本项目的 Apache-2.0 许可证不覆盖这些图片；[数据身份清单](../../recipes/image-classification/cifar10.dataset.json)固定压缩包大小、SHA-256、官方 MD5、类别和完整 split 数量。

```sh
liuding-forge data prepare --name cifar10 --archive /datasets/cifar-10-binary.tar.gz --data-root /datasets
```

也可以设置 `JYPPX_LIUDING_DATA_ROOT` 并省略 `--data-root`。

按[构建步骤](#11-启用离线数据准备)启用 `JYPPX_LIUDING_PREPARE_CIFAR10=ON`；未启用时命令返回不支持。准备由 C++ 实现，离线核对完整 SHA-256 与官方 MD5，受限读取 gzip/tar 后逐条转换 3,073 字节的 binary record，生成相同像素和格式的无损 PNG；不会加载 pickle、联网或将整套图片缓存到内存。输出 `/datasets/cifar10-imagefolder-v1/` 包含完整 50,000 张训练图、10,000 张官方 test 图及 `preparation.json`。官方 test 在输出中名为 `validation`，每类 train 5,000 张、validation 1,000 张；它会参与每个 epoch 的验证，不是额外保留且从未观察的测试集。准备失败不发布本次目标并尝试清理本次暂存，已有目标目录始终拒绝覆盖。解压内容上限沿用数据清单的 185,000,000 字节，tar 结构另有 64 KiB 上限；PNG 单文件最多 8 KiB，实际 RSS 和耗时需要在本机测量。进程被终止可能留下 `.cifar10-preparing-*` 暂存目录，确认归属后再清理；准备过程中应保持源和输出父目录稳定。原生调用与报告读取示例见 [cifar10.cpp](../../examples/data/cifar10.cpp)。

复制 [CIFAR-10 ResNet18 模板](../../recipes/image-classification/cifar10-resnet18.json)，把 `train_root`、`validation_root` 改为准备结果的实际路径，再执行 `recipe check` 和 `recipe run`。模板使用完整 split、32×32 RGB `[0,1]`、随机初始化 ResNet18、50 epochs、batch 128、seed `20261028`、SGD momentum 和 StepLR，共 19,550 次更新，预先声明 accuracy ≥0.65。它是限定训练配方，不表示与采用数据增强、不同 stem 或更长预算的其他 CIFAR-10 实现达到相同准确率。CPU 与 CUDA 的使用和恢复方式沿用前文，CUDA 需显式修改 `device` 并设置工作空间环境变量。

## 7. 准备 Fashion-MNIST

按[构建步骤](#11-启用离线数据准备)启用 `JYPPX_LIUDING_PREPARE_FASHION_MNIST=ON`，并提供匹配平台的 codec SDK。数据准备由原生 C++ 完成，安装后的工具不需要 Python。

从 [Fashion-MNIST 固定版本的数据目录](https://github.com/zalandoresearch/fashion-mnist/tree/b2617bb6d3ffa2e429640350f613e3291e10b141/data/fashion)下载 `train-images-idx3-ubyte.gz`、`train-labels-idx1-ubyte.gz`、`t10k-images-idx3-ubyte.gz`、`t10k-labels-idx1-ubyte.gz`，原样放入仓库外的同一个目录。它与普通 MNIST 是不同的数据集；[数据身份清单](../../recipes/image-classification/fashion-mnist.dataset.json)固定四个文件的大小、SHA-256、官方 MD5 和类别映射，数据适用原始 MIT 许可证。

```sh
liuding-forge data prepare --name fashion-mnist --source-dir /datasets/fashion-mnist-source --data-root /datasets
```

默认输出 `/datasets/fashion-mnist-first100-per-class-imagefolder-v1/`，在原始 train/test 中各按源顺序选取每类最早 100 张，共 1,000 张训练图和 1,000 张验证图。需要完整数据时增加 `--selection full`，输出独立的 `fashion-mnist-full-imagefolder-v1/`，包含 60,000 张训练图和 10,000 张验证图。两种选择均检查全部源记录、类别数量、IDX 长度和 gzip CRC，不会只检查被选中的部分。也可以设置 `JYPPX_LIUDING_DATA_ROOT` 并省略 `--data-root`。

每张原始 28×28 uint8 灰度图逐像素复制为 RGB 无损 PNG，不缩放、不归一化，也不将整套数据加载到内存。`train/` 和 `validation/` 使用带数字前缀的类别目录，例如 `00-t-shirt-top`，按目录名称排序得到的类别 ID 与原始标签 0–9 一致。官方 test 在此命名为 `validation`，不是额外保留的最终测试集。

`index/train.jsonl` 和 `index/validation.jsonl` 记录原始索引、类别 ID、相对图片路径、灰度像素 SHA-256 与 PNG 文件 SHA-256；`preparation.json` 保存来源、选择规则、类别计数和内容身份，`fashion-mnist.LICENSE` 保留原始许可文本。内容身份基于原始索引、标签和灰度字节，不依赖 PNG 压缩实现。准备命令离线运行，拒绝覆盖已有输出，失败时清理本次私有暂存目录。逐图读取与解码示例见[数据准备示例](../../examples/data/README.md#fashion-mnist-图像接入)。

上述工具提供数据接入，不会自动加载预训练权重。使用现有随机初始化 ResNet18 配方时，将数据路径指向对应的两个 split，并为 Fashion-MNIST 单独设置配置与质量阈值；模型所需的缩放、裁剪及 mean/std 由图像管线显式完成，不能沿用 CIFAR-10 的任务阈值来宣称该任务已经通过。预训练 ResNet50 和 DINOv2-S 的冻结骨干、完整微调与恢复步骤见[迁移学习教程](transfer_learning_recipe.md)。
