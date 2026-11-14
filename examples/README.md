# C++ 示例

这里保存可运行的教学程序。先准备匹配的 LibTorch 与依赖，再按各目录说明构建；运行输出使用 `out/` 或自己的工作目录。安装 SDK 消费者与主工程示例的配置入口不同，不能在同一个 CMake 缓存中互换。

| 目标 | 示例与说明 |
| --- | --- |
| 基础训练、优化器调度、检查点、GradScaler | [training](training/README.md)，支持消费已安装 SDK 的独立构建 |
| Tensor 数据管线与数据准备 | [data](data/README.md)，基础管线支持独立构建；数据准备程序按主工程组件开启 |
| MNIST 训练 | [mnist](mnist/README.md)，读取本地 IDX，数据准备另行执行 |
| 迁移学习、线性探测与特征提取 | [transfer_learning](transfer_learning/README.md)及[中间特征](transfer_learning/intermediate_features.md) |
| 蒸馏训练循环 | [distillation.cpp](distillation.cpp)，自包含教师/学生合成数据演示；完整视觉配方见[扩展任务](../docs/api/vision/extensions.md) |
| 图像训练与文件输出 | [training/IMAGE_OUTPUT.md](training/IMAGE_OUTPUT.md) |
| CIFAR-10 模型重载与图像输出 | [分类教程](../docs/tutorials/image_classification_recipe.md#61-在新进程中重载模型并评估)及[inference 源码](inference/) |
| 分类、检测与语义分割评估 | [evaluation](evaluation/README.md)，分类评估支持独立 SDK 构建 |
| 本地任务、搜索、遥测与制品 | [experiment](experiment/README.md)及[本地执行](experiment/local_execution/README.md) |
| 检查点到 ONNX、图像部署与签名 | [deployment](deployment/README.md) |
| 本地 Serving 与版本切换 | [serving](serving/README.md) |
| 模型目录查询与 YOLOv8 训练 | [vision](vision/README.md) |

主工程示例使用 `JYPPX_LIUDING_BUILD_EXAMPLES=ON`，每个程序还要求所链接的可选组件。具体逻辑 target 和依赖条件见 [CMakeLists.txt](CMakeLists.txt)；`EXCLUDE_FROM_ALL` 的程序需显式指定 target 构建。只用于验证、故障注入或参考对照的程序保存在 `tests/`，开发维护工具见 [scripts](../scripts/README.md)。
