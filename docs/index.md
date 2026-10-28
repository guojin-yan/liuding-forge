# 六丁炼算（Liuding Forge）文档

六丁炼算是基于 LibTorch 的 C++20 深度学习训练框架，采用 [Apache License 2.0](../LICENSE)。安装、构建和最小接入方式见[项目主页](../README.md)，当前支持边界见 [Beta 限制](user-guide/beta_limits.md)。

## 教程

1. [基础训练](tutorials/01_basic_training.md)
2. [数据加载](tutorials/02_data_loading.md)
3. [优化器与调度器](tutorials/03_optimizers.md)
4. [回调、检查点与指标](tutorials/04_callbacks.md)
5. [混合精度训练](tutorials/05_mixed_precision.md)
6. [用自己的图像训练 ResNet18](tutorials/image_classification_recipe.md)
7. [训练双输入、分类与回归模型](tutorials/multi_io_recipe.md)

## 用户指南

- [命令行工具](user-guide/cli.md)
- [部署](user-guide/deployment.md)
- [性能调优](user-guide/performance_tuning.md)
- [故障排查](user-guide/troubleshooting.md)
- [离线工作流示例](examples/offline_workflow.md)

## API

- 训练基础：[数据](api/data.md)、[图像数据](api/data_image.md)、[优化器与调度器](api/optim.md)、[训练器](api/training.md)、[模型适配](api/model_adapter.md)
- 恢复与实验：[检查点参与者](api/checkpoint_participant.md)、[持久化检查点](api/durable_checkpoint.md)、[本地运行存储](api/local_run_storage.md)、[训练遥测](api/training_telemetry.md)、[评估](api/evaluation.md)
- 模型交付：[模型](api/models.md)、[权重导入](api/model_weight_import.md)、[模型包](api/model_bundle.md)、[互操作后端](api/interop_backends.md)
- 可选预览：[量化](api/quantization.md)、[Serving](api/serving.md)
- 图像输出：[绘制](api/vision_render.md)

## 开发维护

- [贡献指南](../CONTRIBUTING.md)
- [组件依赖](development/component_dependencies.md)
- [仓库卫生](development/repository_hygiene.md)
- [构建元数据](development/build_metadata.md)
- [发布流程](development/releasing.md)
- [Benchmark 操作](benchmarks/README.md)

规划、审计、历史性能结果和验收证据保存在代码仓库之外，不属于用户文档。
