# 视觉 C++ 示例

本目录包含模型注册表查询与独立 YOLOv8 检测教学程序。日常多模型训练、调参及 ONNX 工作流使用[视觉套件](../../docs/api/vision/index.md)和 `liuding-forge-vision`。

## 查询已安装模型

`jyppx_liuding_vision_catalog` 生成 `liuding-forge-example-vision-catalog`。[catalog.cpp](catalog.cpp) 调用 `vision_catalog()`，打印模型、任务、变体、组件、输入约束与权重/数值/质量字段，不分配模型参数或运行训练。

## YOLOv8 检测训练

`jyppx_liuding_yolo_training` 生成 `liuding-forge-example-yolo-training`。[yolo_training.cpp](yolo_training.cpp) 演示统一标注导入、LetterBox、任务 loss、Trainer、调度、EMA、检查点和检测评估。它使用[独立示例配置](../../recipes/vision/yolov8/detect.example.json)，与统一 CLI 的配方格式不同。

```text
liuding-forge-example-yolo-training CONFIG.json NEW_OUTPUT_DIRECTORY
liuding-forge-example-yolo-training CONFIG.json NEW_OUTPUT_DIRECTORY --pause-at STEP
liuding-forge-example-yolo-training CONFIG.json NEW_OUTPUT_DIRECTORY --resume CHECKPOINT
```

配置中的数据路径相对配置文件解析，输出目录必须全新。`--pause-at` 是绝对优化器更新次数；续训保留配置与数据身份，使用新的输出目录。`--smoke NEW_OUTPUT_DIRECTORY` 生成微型合成数据供检查接入，不代表实际检测精度。

在主工程开启 `JYPPX_LIUDING_BUILD_EXAMPLES=ON`、`JYPPX_LIUDING_BUILD_VISION_TASKS=ON` 与 `JYPPX_LIUDING_BUILD_MODELS_YOLO=ON`，并按视觉套件说明开启它们所需的基础组件。显式构建上述 target；运行时使用与构建匹配的 LibTorch。
