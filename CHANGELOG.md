# 变更记录

项目尚未发布。本页记录当前未发布功能及用户可见修复；源码版本以根目录 `CMakeLists.txt` 为准。

## [Unreleased]

### 训练与数据

- 提供显式任务 step、模型适配、多优化器、梯度累积、裁剪、AMP、回调及可恢复调度器。
- 提供 map/iterable 数据管线、确定性采样、线程 worker、结构化批次和游标恢复。
- 协调保存模型、优化器、scheduler、RNG、数据、GradScaler、回调和训练控制状态，支持文件检查点及完整恢复。
- 提供图像解码、COCO/VOC/YOLO/LabelMe 导入、联合变换、数据增强、缓存、组批、设备传输、绘制及有界图像输出。

### 模型与视觉套件

- 提供分类骨干、ViT/Swin、DINOv2/v3、统一特征提取、分类头替换与显式 safetensors 权重导入。
- 提供原始 YOLOv5 P5/P6、YOLOv8/11/12/26、Faster/Fast R-CNN、FCOS、RetinaNet、SSD、RT-DETR 与开放词汇模型的原生组件和统一 CLI。
- 提供分类、多标签、检测、实例分割、姿态、旋转框、语义分割、深度、SimMIM、ReID、蒸馏及序列跟踪工作流。
- 提供严格配方、数据预览、参数搜索、EMA、早停、best/last 保存、模型初始化及固定尺寸 ONNX bundle。
- 修复 VOC 图片扩展名与子目录导入、YOLOv5/SSD buffer 视图权重导入、完成态重复训练、扩展模型预测对训练标注的依赖及 attention 布局反向差异。
- 修复视觉任务 CUDA 输入与辅助张量设备不一致、结束多图混合后错误拒绝配对目标，以及大型检查点的元数据预算限制。

### 实验、评估与交付

- 提供 SQLite/CAS 运行与制品存储、事务化发布、本地原生进程执行、有界搜索、原生 TensorBoard 与 OTLP 遥测。
- 提供分类、检测和分割评估、检查点续评、质量规则及 JSON/HTML 报告。
- 提供通用 C++ ONNX 导出、独立 ONNX Runtime、模型 bundle、Ed25519 签名、本地 Serving 与受控版本切换。
- 修复 ONNX 归约类型提升、混合类型 cat/stack、可选仿射 BatchNorm 及 Double LayerNorm 方差计算。
- 增加 `initialize_runtime()`，规避适用 Windows LibTorch OpenMP 配置的辅助线程池退出等待；原生 HTTP 完成回调采用无等待事件通知。

### 文档与仓库维护

- 文档按当前公共接口、组件和可运行示例组织，移除开发阶段流水、测试成绩及无代码支持的功能说明。
- 视觉模型验证探针归入测试目录；示例目录保留教学程序，并提供集中索引。
- 移除一次性环境报告、未接入的预研脚本及依赖元数据中的本机路径和执行记录；增加嵌套缓存、生成物及可移植配置的卫生检查。

具体接口及限制见[文档索引](docs/index.md)。权重、任务精度和设备支持按当前构建与兼容矩阵分别判断。
