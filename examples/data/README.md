# 准备双输入训练数据

[paired_affine.cpp](paired_affine.cpp) 展示如何直接调用原生 C++ 数据准备实现，再读取一条 JSONL 样本，区分两个 positional 模型输入和分类、回归监督目标。生成、核验和读取均不需要 Python 运行环境；完整训练流程见[双输入分类与回归教程](../../docs/tutorials/multi_io_recipe.md)。

启用示例后构建目标，将运行参数替换为仓库外的数据根目录：

```sh
cmake -S . -B out/build/release -DCMAKE_PREFIX_PATH=/path/to/libtorch -DJYPPX_LIUDING_BUILD_EXAMPLES=ON
cmake --build out/build/release --config Release --target jyppx_paired_affine_data --parallel 2
liuding-forge-example-paired-affine-data /path/to/datasets
```

Windows 可执行文件位于 `out/build/release/examples/Release/`，单配置 Linux 位于 `out/build/release/examples/`。程序在新建的 `paired-affine-v1/` 中逐条生成全部 1,280 条样本，核验冻结摘要后发布目录，然后输出第一条训练样本的 ID、两组特征和两个监督目标。已有目录会被拒绝；再次体验时选择另一个数据根目录。

`left`、`right` 各有两个特征；训练组批后为两个 `[batch,2]` Tensor。`class_id` 组批为 `[batch]`，`regression` 组批为 `[batch,1]`，不能拼入模型输入。数据数量、数学公式和许可只在[数据说明](../../recipes/multi-io/README.md)维护。日常准备同一数据可直接运行 `liuding-forge data prepare --name paired-affine --data-root /path/to/datasets`。
