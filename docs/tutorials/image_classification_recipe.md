# 用自己的图像训练 ResNet18

本教程使用 ImageFolder 目录、JSON 配置和 `liuding-forge recipe` 命令完成 ResNet18 分类训练、验证集评估、保存检查点及独立进程续训。模型从随机参数开始，支持单设备 CPU 或显式 CUDA、FP32、SGD momentum 和逐 epoch StepLR；当前入口不加载预训练权重。

## 1. 构建工具

使用[支持矩阵](../../compatibility.json)中的编译器和 LibTorch 组合，同时启用 `models`、`data_image` 和 CLI。构建不会自动下载 LibTorch。

```sh
cmake -S . -B out/build/release -DCMAKE_PREFIX_PATH=/path/to/libtorch -DJYPPX_LIUDING_BUILD_MODELS=ON -DJYPPX_LIUDING_BUILD_DATA_IMAGE=ON -DJYPPX_LIUDING_BUILD_TOOLS=ON
cmake --build out/build/release --config Release --target jyppx_cli --parallel 2
```

以下命令使用可在 PATH 中找到的 `liuding-forge`。Windows 构建产物位于 `out/build/release/tools/cli/Release/liuding-forge.exe`，单配置 Linux 构建位于 `out/build/release/tools/cli/liuding-forge`。省略任一可选组件时，该配方会返回明确的不支持错误。

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

`schema` 的 JSON 响应中，`data` 是可交给 `config lint` 使用的闭合配置 schema；保存时提取这个字段。所有字段都必须提供，未知字段、重复键、错误类型和越界值会被拒绝。`check` 校验两个类别表，流式读取全部图片字节，输出内容 SHA256、样本数、有效配置和总更新数；它不训练模型。配置上限包括单图字节、两个 split 的总字节、每个 split 的样本数和最多一百万次优化器更新。

`image_budget_bytes` 限制合作图像阶段的已知 decoded、scratch、pinned、GPU payload 和在途 batch 逻辑字节；worker、prefetch 和单图限制共同约束数据准备。它不是整个进程 RSS 或 CUDA allocator 的严格上界，也不包含模型、优化器及检查点缓冲。图像逐样本解码，不全量缓存到内存。

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

```sh
liuding-forge recipe run --file my-resnet18.json --root runs/resumable --pause-after 19 --format json
liuding-forge recipe resume --file my-resnet18.json --root runs/resumable --format json
```

恢复覆盖模型参数和 buffer、SGD 状态、StepLR、Trainer、训练与验证数据游标、采样器、CPU/选定 CUDA RNG、回调游标和诊断统计。新的 `pause-after` 可以继续分段运行。异常终止后，已有 `latest.jyck` 时也可从该检查点重做尚未持久化的工作；尚未发布首个检查点时不能恢复。

恢复要求有效配置、类别表、全部图片内容和构建身份一致，包括 seed、设备、线程策略和训练预算。若需要改配置，创建新的输出目录重新训练。暂停的 CLI 结果为 `paused`，run-store 中仍保留同一条 `running` 的未完成逻辑运行；不要在计划续训时将它标为终态。已经通过 `run recover` 等命令标为 `interrupted` 的终态不接受原运行续训。完成后的同配置 `resume` 校验结果与 bundle 后返回既有结果，不重复训练；`run` 始终拒绝覆盖既有目录。同一输出目录同时只允许一个配方进程，操作系统会在进程退出时释放锁。

```sh
liuding-forge checkpoint verify --path runs/my-run/final.jyck --format json
liuding-forge model bundle verify --path runs/my-run/bundle --format json
liuding-forge run verify --root runs/my-run/run-store --format json
```

图片摘要在运行前和交付前检查，数据目录仍须由调用者保持稳定；这不是文件系统快照或对并发替换的隔离保证。文件、模型包和运行记录的发布有各自的持久化边界，不能视为一次跨目录原子事务。

## 6. 使用 CIFAR-10

从 [CIFAR-10 官方页面](https://www.cs.toronto.edu/~kriz/cifar.html)自行下载 **binary version** 的 `cifar-10-binary.tar.gz`，保存到仓库外。数据集适用原始条款，本项目的 Apache-2.0 许可证不覆盖这些图片；[数据身份清单](../../recipes/image-classification/cifar10.dataset.json)固定压缩包大小、SHA-256、官方 MD5、类别和完整 split 数量。

```sh
python scripts/datasets/prepare_cifar10.py --archive /datasets/cifar-10-binary.tar.gz --data-root /datasets
```

也可以设置 `JYPPX_LIUDING_DATA_ROOT` 并省略 `--data-root`。

准备脚本仅用 Python 标准库，离线核对压缩包、逐条读取 binary record，再按精确的 RGB uint8 像素生成无损 PNG；不会加载 pickle、联网或将整套图片缓存到内存。输出 `/datasets/cifar10-imagefolder-v1/` 包含完整 50,000 张训练图、10,000 张官方 test 图及 `preparation.json`。官方 test 在输出中名为 `validation`，每类 train 5,000 张、validation 1,000 张；它会参与每个 epoch 的验证，不是额外保留且从未观察的测试集。准备失败不发布目标目录，已有目标目录始终拒绝覆盖。

复制 [CIFAR-10 ResNet18 模板](../../recipes/image-classification/cifar10-resnet18.json)，把 `train_root`、`validation_root` 改为准备结果的实际路径，再执行 `recipe check` 和 `recipe run`。模板使用完整 split、32×32 RGB `[0,1]`、随机初始化 ResNet18、50 epochs、batch 128、seed `20261028`、SGD momentum 和 StepLR，共 19,550 次更新，预先声明 accuracy ≥0.65。它是限定训练配方，不表示与采用数据增强、不同 stem 或更长预算的其他 CIFAR-10 实现达到相同准确率。CPU 与 CUDA 的使用和恢复方式沿用前文，CUDA 需显式修改 `device` 并设置工作空间环境变量。

## 7. 准备 Fashion-MNIST

从 [Fashion-MNIST 固定版本的数据目录](https://github.com/zalandoresearch/fashion-mnist/tree/b2617bb6d3ffa2e429640350f613e3291e10b141/data/fashion)下载 `train-images-idx3-ubyte.gz`、`train-labels-idx1-ubyte.gz`、`t10k-images-idx3-ubyte.gz`、`t10k-labels-idx1-ubyte.gz`，原样放入仓库外的同一个目录。它与普通 MNIST 是不同的数据集；[数据身份清单](../../recipes/image-classification/fashion-mnist.dataset.json)固定四个文件的大小、SHA-256、官方 MD5 和类别映射，数据适用原始 MIT 许可证。

```sh
python scripts/datasets/prepare_fashion_mnist.py --source-dir /datasets/fashion-mnist-source --data-root /datasets
```

默认输出 `/datasets/fashion-mnist-first100-per-class-imagefolder-v1/`，在原始 train/test 中各按源顺序选取每类最早 100 张，共 1,000 张训练图和 1,000 张验证图。需要完整数据时增加 `--selection full`，输出独立的 `fashion-mnist-full-imagefolder-v1/`，包含 60,000 张训练图和 10,000 张验证图。两种选择均检查全部源记录、类别数量、IDX 长度和 gzip CRC，不会只检查被选中的部分。也可以设置 `JYPPX_LIUDING_DATA_ROOT` 并省略 `--data-root`。

每张原始 28×28 uint8 灰度图逐像素复制为 RGB 无损 PNG，不缩放、不归一化，也不将整套数据加载到内存。`train/` 和 `validation/` 使用带数字前缀的类别目录，例如 `00-t-shirt-top`，按目录名称排序得到的类别 ID 与原始标签 0–9 一致。官方 test 在此命名为 `validation`，不是额外保留的最终测试集。

`index/train.jsonl` 和 `index/validation.jsonl` 记录原始索引、类别 ID、相对图片路径、灰度像素 SHA-256 与 PNG 文件 SHA-256；`preparation.json` 保存来源、选择规则、类别计数和内容身份，`fashion-mnist.LICENSE` 保留原始许可文本。内容身份基于原始索引、标签和灰度字节，不依赖 PNG 压缩实现。脚本仅使用 Python 标准库，离线运行，拒绝覆盖已有输出，失败时清理本次私有暂存目录。

上述工具提供数据接入，不会自动加载预训练权重。使用现有随机初始化 ResNet18 配方时，将数据路径指向对应的两个 split，并为 Fashion-MNIST 单独设置配置与质量阈值；模型所需的缩放、裁剪及 mean/std 由后续图像管线显式完成，不能沿用 CIFAR-10 的任务阈值来宣称该任务已经通过。
