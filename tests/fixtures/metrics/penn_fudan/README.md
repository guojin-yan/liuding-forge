# Penn-Fudan 真实图像任务验证

此可选集成测试验证本地 PNG → 联合缩放 → C++ Trainer/Adam → 小型 U-Net → IoU/Dice、候选框 AP → 持久检查点恢复与质量报告。固定划分、训练配置和首次执行前确定的质量门槛见 [task-contract.json](task-contract.json)，170 对图像和实例掩码的文件摘要见 [manifest.json](manifest.json)。测试默认跳过，不联网或下载数据。

数据来自 [Penn-Fudan 官方页面](https://www.cis.upenn.edu/~jshi/ped_html/)。自行取得官方 `PennFudanPed.zip` 后，先核对 manifest 中的归档 SHA-256，再解压到仓库外；保留官方 readme 的版权声明。原图、掩码和训练权重均不随仓库分发，也不适用本仓库的代码许可证。此划分不是官方划分，没有场景独立性保证。

配置已验证的 LibTorch 环境，启用 `JYPPX_LIUDING_BUILD_TESTS=ON`、`JYPPX_LIUDING_BUILD_DATA_IMAGE=ON` 和默认的 training/optim/interop/experiment 组件，然后构建 `jyppx_test_training_real_spatial_task`。测试源码位于 [test_real_spatial_task.cpp](../../../unit/training/test_real_spatial_task.cpp)。

在解压得到的 `PennFudanPed` 目录中放入本目录的 `manifest.json`，设置 `JYPPX_LIUDING_SPATIAL_TASK_DIR` 指向该目录，并设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。每次运行将 `JYPPX_LIUDING_SPATIAL_OUTPUT_DIR` 指向一个尚不存在的仓库外目录，避免覆盖结果。

1. 显式设置 `JYPPX_LIUDING_TRAIN_SPATIAL_TASK=1`，运行 `RealSpatialTask.TrainCUDA`，得到 safetensors 权重。此入口固定执行最终轮训练，不使用验证集或测试集选择权重。
2. 将生成的 `trained.safetensors` 放入数据目录；测试仅接受源码中固定摘要的权重。其他 LibTorch/GPU 组合训练出的字节需要作为新的参考候选单独核验，不直接替换现有参考。
3. 清除训练开关，分别运行 `RealSpatialTask.HeldOutInferenceAndDurableEvaluationCPU` 和 `RealSpatialTask.HeldOutInferenceAndDurableEvaluationCUDA`，每次使用新的输出目录。可通过 `--gtest_filter=测试名` 单独执行，或用 CTest 的精确 `-R` 选择，并保持 `-j 1`。

输出包含实际前处理图像、实例 ID、模型 logits、候选框、统计值、质量报告、物理检查点与资源观测，便于独立参考核对。照片采用 FP32 RGB/255 和带抗锯齿的双线性缩放，离散实例 ID 用 nearest-exact；评估按单张图像执行前向，拆批不改变模型运算形状。IoU/Dice 和 AP 均核对 batch 1/4 等价，并销毁重建模型和会话后续评。

候选框只来自模型预测前景的连通分量，不读取真实标注。它用于检验实际预测到 AP 的数据链路，不代表独立检测模型、完整检测配方、官方 benchmark 或训练性能优势。
