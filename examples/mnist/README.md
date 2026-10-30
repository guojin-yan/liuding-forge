# MNIST 分类示例

这个示例通过 `jyppx::liuding::sdk` 使用卷积分类器、可恢复数据管线、SGD 和训练/验证循环。训练只读取本地 IDX；四个本地 gzip 可通过 C++ 命令离线准备，也可在准备阶段显式下载。

## 准备官方本地数据

从[数据清单](dataset-manifest.json)的 source 镜像自行取得四个未改动的官方 gzip，放在仓库外同一目录。需要 train-images、train-labels、t10k-images、t10k-labels 四个文件，原始文件名见清单；数据适用原始 MNIST 条款，本项目不随源码分发数据集。

构建 CLI 时启用 `JYPPX_LIUDING_PREPARE_MNIST=ON`，并提供构建配置要求的私有静态 codec SDK。选择新的外部数据根，然后运行：

~~~powershell
$env:JYPPX_DATA_ROOT = 'F:\Datasets\liuding-forge-native'
liuding-forge data prepare --name mnist --source-dir F:\Datasets\mnist-source
out/build/config-windows/examples/Release/liuding-forge-example-mnist.exe --epochs 3 --device cpu
~~~

Linux 对应命令：

~~~sh
export JYPPX_DATA_ROOT=/external/datasets/liuding-forge-native
liuding-forge data prepare --name mnist --source-dir /external/mnist-source
out/build/cuda-release/examples/liuding-forge-example-mnist --epochs 3 --device cpu
~~~

准备核对四个 gzip 的完整大小与 SHA-256，再分块解压并检查 IDX magic、60,000/10,000 样本数、28×28 形状、全部标签、原始长度与摘要、gzip CRC 和尾部。只发布新的 `$JYPPX_DATA_ROOT/mnist`，包含四份原字节 IDX 与 preparation.json，不复制 gzip，不覆盖已有目录。重复体验请选择新根；准备失败尝试回滚本次暂存，进程终止仍可能留下 .mnist-preparing-*，应在确认归属后清理。准备时保持源与父目录稳定。

## 读取与训练

训练默认读取 `$JYPPX_DATA_ROOT/mnist`；也可显式传入包含四个已解压 IDX 的目录：

~~~sh
liuding-forge-example-mnist /external/datasets/mnist 3 --device cpu
~~~

独立的[数据准备教学示例](../data/README.md#mnist-idx-接入)展示相同原生接口和第一条灰度样本。准备阶段保留固定块缓冲，不加载整套图片；训练示例仍将完整 IDX 转为内存 Tensor，二者内存开销不同。

`--device cpu` 选择 CPU，`--device cuda` 等同 `cuda:0`，也可指定其他 CUDA 索引；未指定时优先 CUDA 0，不可用时使用 CPU。`--offline` 仅标注训练只读本地数据。缺少数据根、文件损坏、未知参数或非正整数 epoch 均明确失败。正常输出含设备、完成 epoch、优化步数、有限 loss 和 validation accuracy；随机初始化的演示不承诺固定准确率。CTest 中的小型合成 IDX 用例验证训练和参数，不代表官方完整数据集的模型质量。


需要联网获取数据时，使用[数据准备示例的显式下载模式](../data/README.md#显式下载-mnist)；训练程序本身不发起下载。已有本地文件时无需启用网络组件。
