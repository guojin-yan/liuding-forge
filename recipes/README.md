# 训练与搜索配方

这里保存原生 C++ CLI 使用的 JSON 参数模板、数据集身份和权重来源描述。它们用于接入和复现配置，不包含图片、数据集、预训练权重、检查点或训练结果；执行前复制到自己的工作目录，修改数据路径、类别、设备和训练预算。

| 目录 | 内容与入口 |
| --- | --- |
| `image-classification/` | ImageFolder、CIFAR-10、Fashion-MNIST 分类配置及数据描述，见[图像分类教程](../docs/tutorials/image_classification_recipe.md) |
| `multi-io/` | 双输入分类/回归和确定性合成数据合同，见[多输入教程](../docs/tutorials/multi_io_recipe.md) |
| `transfer-learning/` | ResNet50、DINOv2 迁移学习配置和权重摘要；实际权重由使用者另行准备，见[迁移学习教程](../docs/tutorials/transfer_learning_recipe.md) |
| `vision/suite/` | 统一视觉 CLI 的 schema1 训练与搜索起点，覆盖 YOLO、Faster R-CNN 和分类，见[视觉训练套件](../docs/api/vision/index.md) |
| `vision/extensions/` | schema2 扩展任务与模型配置，包括语义分割、深度、开放词汇、蒸馏和自监督，见[扩展任务](../docs/api/vision/extensions.md) |
| `vision/yolov8/` | [独立 YOLOv8 C++ 示例](../examples/vision/yolo_training.cpp)使用的旧示例格式；它与统一视觉 CLI 的配方格式不同 |

训练配方、搜索空间和 `*.dataset.json`、`*.weights.json` 各有独立格式，应使用对应教程中的命令；不能把数据或权重描述直接传给训练入口。模板中的轮数、学习率和质量阈值是待调整的起点，不表示已经取得对应数据集精度或全设备资格。

数据、检查点、运行目录和导出模型保存在仓库外，或被忽略的 `out/` 中。开发维护与验证工具另见 [scripts](../scripts/README.md)，可复制运行的 C++ 程序见 [examples](../examples/)。
