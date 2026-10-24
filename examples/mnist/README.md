# MNIST 分类示例

这个示例通过 `jyppx::liuding::sdk` 使用卷积分类器、可恢复数据管线、SGD、训练与验证循环。它读取本地 MNIST IDX 文件，不在训练时联网，不将数据写入代码仓库。运行需要已构建的示例程序；数据准备脚本只需要 Python 3 标准库。

先选择仓库之外的数据根目录，例如在 PowerShell 中执行：

```powershell
$env:JYPPX_DATA_ROOT = 'F:\Datasets\liuding-forge'
python scripts/datasets/prepare_mnist.py
out/build/config-windows/examples/Release/liuding-forge-example-mnist.exe --epochs 3 --device cpu
```

Linux 上可使用：

```sh
export JYPPX_DATA_ROOT=/path/outside/repository/datasets
python3 scripts/datasets/prepare_mnist.py
out/build/cuda-release/examples/liuding-forge-example-mnist --epochs 3 --device cpu
```

准备脚本从 Google 托管的 MNIST 镜像下载四个 gzip 文件，依据 [数据清单](dataset-manifest.json) 校验文件大小与 SHA-256，再执行有界解压和原子写入。缓存中的压缩包每次使用都会验证；摘要不符会失败并指出文件，用户检查来源后可移走损坏文件并重新准备。数据来源信息见清单的 `source` 与 `checksum_reference`，项目不随源码分发数据集。

训练默认读取 `$JYPPX_DATA_ROOT/mnist`。也可直接传入包含四个已解压 IDX 文件的目录：

```sh
liuding-forge-example-mnist /external/datasets/mnist 3 --device cpu
```

只有缓存压缩包且需要离线解压时，运行 `python scripts/datasets/prepare_mnist.py --offline`；完整缓存可重复验证，缺少压缩包、摘要不符或 gzip 损坏时明确失败，始终不会联网。训练始终使用本地 IDX；`--offline` 可显式标注这一行为。未设置数据根且未传目录、未知参数、非正整数 epoch、缺失文件、损坏头部或图像与标签数量不一致都会报错。

`--device cpu` 选择 CPU，`--device cuda` 等同于 `--device cuda:0`，也可显式指定其他 CUDA 索引；不传时优先使用 CUDA 设备 0，不可用时选择 CPU。训练策略记录具体设备索引，以保持 checkpoint 拓扑稳定。正常输出包括设备、完成 epoch 数、优化步数、有限的训练 loss 与验证 accuracy。随机初始化的示例不承诺固定准确率；CTest 使用生成的小型 IDX 数据验证真实训练、参数和离线路径，不能代表完整 MNIST 数据集的模型质量。
