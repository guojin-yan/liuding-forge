# 训练完成一批后发布预览

`image_training_output.cpp` 演示 ImageFolder 读取、32×32 联合变换、组批、Trainer 前后向与 SGD、PNG 输出以及 CAS/SQLite 证据登记。模型是随机初始化的两层分类器，每批8张，运行一轮；输入至少包含两个非空类别和9张图。它用于学习数据与输出生命周期，不提供预训练权重或任务准确率承诺。

启用 `data_image`、八 Core 与 examples 后，构建 `jyppx_image_training_output`。复用现有构建目录；Windows 多配置生成器添加 `--config Release`。

```sh
cmake --build out/build/release --target jyppx_image_training_output --parallel 1
liuding-forge-example-image-training-output ./images ./new-output cpu current
```

输入目录按 `images/类别/图像.png` 组织。输出目录必须不存在。CUDA 构建可以选择当前流传输或显式拷贝流；CPU-only 构建拒绝 CUDA 请求。

```sh
liuding-forge-example-image-training-output ./images ./new-current cuda:0 current
liuding-forge-example-image-training-output ./images ./new-copy cuda:0 copy
```

需要确定性 CUDA 运算时，在启动进程前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。两条路径使用相同 seed、数据顺序、模型、精度和 SGD 参数；可比较 `report.json` 中逐批输入、logits、更新后参数摘要及预览文件摘要。报告的耗时包含训练与输出，不是独立 DMA 时间或性能资格。

示例只准入一个源批次和1MiB保守组批额度，两个生产线程可等待准入。模型 mapper 显式保留完整源 envelope；`IValue` 不携带运行期租约。显式拷贝 ticket 与设备批次保留至更新后的回调，预览 D2H 和摘要读取确认消费流工作完成，然后释放输入。异步证据回调继续持有源准入，直到文件引用与事件原子登记结束。

首批证据等待 Trainer 发起下一次读取，再检查第二批尚未解码、源准入仍持有、CPU/pinned/GPU输入租约已释放。这个检查有10秒截止。后续每批保存一个 `step-N.png`，`run-store/` 中的SQLite与CAS 保存文件证据，`report.json` 保存更新、输入身份、资源计数和传输流信息。调用方应检查进程退出码，成功退出才表示全部文件及证据已确认。

文件与数据库独立发布；失败可能留下已发布的PNG。本例不实现训练检查点、自动重试或联合可见恢复。源存储准入和已知图像预算不覆盖模型、梯度、allocator缓存、任意私有workspace或完整RSS；逐批摘要和D2H会同步消费流，不能用本例声明copy/compute重叠或端到端提速。详细接口见[图像数据](../../docs/api/data_image.md)与[本地run存储](../../docs/api/local_run_storage.md)。
