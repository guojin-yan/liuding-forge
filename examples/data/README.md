# 数据准备示例

## 双输入训练数据

[paired_affine.cpp](paired_affine.cpp) 展示如何直接调用原生 C++ 数据准备实现，再读取一条 JSONL 样本，区分两个 positional 模型输入和分类、回归监督目标。生成、核验和读取均不需要 Python 运行环境；完整训练流程见[双输入分类与回归教程](../../docs/tutorials/multi_io_recipe.md)。

启用示例后构建目标，将运行参数替换为仓库外的数据根目录：

```sh
cmake -S . -B out/build/release -DCMAKE_PREFIX_PATH=/path/to/libtorch -DJYPPX_LIUDING_BUILD_EXAMPLES=ON
cmake --build out/build/release --config Release --target jyppx_paired_affine_data --parallel 2
liuding-forge-example-paired-affine-data /path/to/datasets
```

Windows 可执行文件位于 `out/build/release/examples/Release/`，单配置 Linux 位于 `out/build/release/examples/`。程序在新建的 `paired-affine-v1/` 中逐条生成全部 1,280 条样本，核验冻结摘要后发布目录，然后输出第一条训练样本的 ID、两组特征和两个监督目标。已有目录会被拒绝；再次体验时选择另一个数据根目录。

`left`、`right` 各有两个特征；训练组批后为两个 `[batch,2]` Tensor。`class_id` 组批为 `[batch]`，`regression` 组批为 `[batch,1]`，不能拼入模型输入。数据数量、数学公式和许可只在[数据说明](../../recipes/multi-io/README.md)维护。日常准备同一数据可直接运行 `liuding-forge data prepare --name paired-affine --data-root /path/to/datasets`。

## Fashion-MNIST 图像接入

[fashion_mnist.cpp](fashion_mnist.cpp) 通过 C++ 离线准备数据，再用库的有界图像解码器读取第一张 PNG，展示原始样本索引、类别 ID 与 `[28,28,3]` CPU uint8 RGB Tensor。它保留原始灰度值和标签；模型输入需要的缩放、归一化及组批由训练管线显式完成。

此例需要 `JYPPX_LIUDING_BUILD_EXAMPLES=ON`、`JYPPX_LIUDING_BUILD_DATA_IMAGE=ON`、`JYPPX_LIUDING_PREPARE_FASHION_MNIST=ON` 和构建时指定的 zlib 1.3.2 静态 SDK。先按[图像分类教程](../../docs/tutorials/image_classification_recipe.md#7-准备-fashion-mnist)准备四个未改动的官方压缩包，再运行：

```sh
cmake --build out/build/release --config Release --target jyppx_fashion_mnist_data --parallel 2
liuding-forge-example-fashion-mnist-data /path/to/fashion-mnist-source /path/to/new-dataset-root
```

默认选取每类最早 100 张原始样本，输出独立的训练和验证目录；日常准备完整数据时，在教程的准备命令中加入 `--selection full`。示例会创建数据，因此第二次运行应选择新的输出根目录。数据准备和读取均不要求 Python；此例不训练模型，也不推导任务准确率。

## CIFAR-10 完整图像数据

[cifar10.cpp](cifar10.cpp) 调用同一原生准备接口，离线核对官方 binary archive，逐条将平面 RGB 转为无损 PNG，再读取准备报告，展示类别顺序、两个 split 的计数和像素契约。它始终准备完整 60,000 张图，不选择子集、不改变训练 seed 或准确率门槛。

启用 `JYPPX_LIUDING_BUILD_EXAMPLES=ON`、`JYPPX_LIUDING_PREPARE_CIFAR10=ON`，并提供构建配置要求的静态 codec SDK。按[图像分类教程](../../docs/tutorials/image_classification_recipe.md#6-使用-cifar-10)准备未改动的官方 binary archive，再运行：

```sh
cmake --build out/build/release --config Release --target jyppx_cifar10_data --parallel 2
liuding-forge-example-cifar10-data /datasets/cifar-10-binary.tar.gz /datasets/new-root
```

输出 `cifar10-imagefolder-v1/` 包含 50,000 张训练图、10,000 张验证图和 `preparation.json`；验证集就是原官方 test split，会参与每个 epoch 的评估。每次只保留一个 record、RGB/PNG 和有界 codec 缓冲，完整输出含 60,000 个文件，请预留对应磁盘空间。既有目录拒绝覆盖，重复体验需选择新根目录；此例展示数据接入，不训练模型。
