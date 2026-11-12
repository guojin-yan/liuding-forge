# C++ 训练与评估示例

无需外部数据的分类推理、断点续评和JSON/HTML质量报告见[classification/](classification/README.md)；分割训练、完整检查点和IoU/Dice评估见[segmentation/](segmentation/README.md)。下文介绍行人检测训练与评估。

[pedestrian_detector.cpp](detection/pedestrian_detector.cpp) 从随机初始化训练直接预测行人框的小型卷积模型，使用 `Trainer`、完整检查点和实际 detection evaluator 生成质量报告。模型声明和实现在同模块的 [model/](detection/model/) 子目录，训练与评估入口在 `detection/` 内。预测只读取 RGB；实例掩码用于训练目标和评估真值，不参与预测或解码。这是小规模接线示例，不能据此推导生产检测精度或完整平台支持。

本例已验收的完整训练与新进程恢复组合为 Ubuntu 的 CPU/CUDA 和 Windows 的 CUDA；Windows CPU 尚未验收。各组合使用[兼容矩阵](../../compatibility.json)列出的工具链与 LibTorch，不保证跨设备权重或数值相同。

## 准备数据与构建

自行取得 [Penn-Fudan 官方数据](https://www.cis.upenn.edu/~jshi/ped_html/)，核对 `PennFudanPed.zip` 的 SHA-256 为 `9095a9613c95586f1c7f2a327d454833d16e0f5e17e5f83d35027ffd315b48e2`，解压到仓库外，并保留官方版权说明。将 [manifest.json](../../tests/fixtures/metrics/penn_fudan/manifest.json) 复制到解压后的 `PennFudanPed` 目录。示例逐一核对所用原图和掩码的摘要；图像、掩码与训练权重不随代码分发，不适用本仓库的代码许可证。

固定划分为120张训练、25张验证、25张测试；示例只在训练集更新，验证集不参与选择，使用第80轮最终权重。这不是官方划分，也不保证场景独立。模型、解码、匹配、阈值和恢复规则见[任务契约](../../tests/fixtures/metrics/penn_fudan/direct-detector-contract.json)。

按[构建指南](../../CONTRIBUTING.md)准备精确兼容的 LibTorch，启用 examples、training、data_image、interop、optim 和 experiment，复用已有兼容构建目录：

```sh
cmake --build out/build/YOUR_PROFILE --target jyppx_liuding_pedestrian_detector --parallel 1
```

下例假定程序位于 `out/build/YOUR_PROFILE/examples/`。Windows 使用对应 Release 程序路径，并通过 PowerShell 的 `$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'` 设置环境；CPU 使用不需要设置该 CUDA 环境。每次输出必须是尚不存在的仓库外目录；程序不下载数据。

## 训练、暂停与恢复

```sh
export CUBLAS_WORKSPACE_CONFIG=:4096:8
DETECTOR=out/build/YOUR_PROFILE/examples/liuding-forge-example-pedestrian-detector
"$DETECTOR" train /data/PennFudanPed /runs/detector-full cuda
"$DETECTOR" train /data/PennFudanPed /runs/detector-pause cuda - 137
"$DETECTOR" train /data/PennFudanPed /runs/detector-resumed cuda /runs/detector-pause/checkpoint.jyck
```

将 `cuda` 改为 `cpu` 可选择 CPU。默认固定 batch8、seed20261108、Adam学习率0.001、80轮／1200次更新、FP32和严格确定性，不在测试集调参。暂停发生在更新137的安全边界，恢复加载模型、优化器、采样器、流水线、Trainer 和 RNG 的完整状态；同设备、构建、数据和配置下，连续路线与新进程恢复路线应生成相同的最终模型和完整检查点。

每条训练路线生成 `model.safetensors`、`checkpoint.jyck` 和 `training.json`。JSON 保存初始／最终更新数、实际应用更新数、权重与检查点摘要、资源计数和构建身份；暂停输出中的模型只完成了137次更新，不作为最终质量候选。

长任务可用可选尾参数 `--checkpoint-every STEP` 定期保存同一 `checkpoint.jyck`，STEP 为1至1199。检查点按绝对更新数的倍数写入，最终或主动暂停边界也会保存；恢复后继续下一个绝对倍数。保存完成后输出并刷新一行包含 `event=checkpoint`、`global_step` 和 `checkpoint_sha256` 的 JSON；它表示已提交的检查点，不表示整条训练完成。没有该选项时只在原先的最终/暂停边界保存。

```sh
"$DETECTOR" train /data/PennFudanPed /runs/detector-periodic cpu --checkpoint-every 137
"$DETECTOR" train /data/PennFudanPed /runs/detector-restart cpu \
  /runs/detector-periodic/checkpoint.jyck --checkpoint-every 137
```

进程意外结束后，从最后完整提交的检查点恢复到全新输出目录；检查点之后尚未提交的更新会重新执行，不能算作已保存进度。中断目录可能没有最终模型或 `training.json`。写入采用 `FileCheckpointStore` 的原子替换语义；本例不保证断电或任意文件系统的持久性。

## 从预测生成 AP 与报告

从 `training.json` 读取最终 `weights_sha256`，显式传入期望摘要：

```sh
"$DETECTOR" evaluate /data/PennFudanPed /runs/detector-evaluation cuda \
  /runs/detector-full/model.safetensors EXPECTED_SHA256
```

每张图输出三组 anchor 的中心、宽高及置信度，固定解码、裁剪、稳定排序与 NMS 后交给 `DetectionEvaluationSession`。报告按完整数据集计算 COCO101点 AP，同时保留 Penn/Fudan 来源切片；不平均每批 AP。实际模型前向始终逐图执行，比较批次1／4只改变评估聚合分组，不证明不同模型前向 batch 形状的数值等价。

评估会在两批后保存状态，销毁并重建会话与模型，再恢复到 EOF；连续、拆批和恢复报告必须逐字节相同，完成态重开不再读取数据或推理。该评估恢复在同一进程内重建对象；训练恢复使用上面三次独立程序调用。

输出包含 `quality.json`、`results.json`、`predictions.json`、`raw.f32`、`instances.i64` 和评估检查点；原始预测与实例张量用于独立检查，可在核验后删除。固定测试集要求 AP50≥0.10；不通过时保留报告并返回非零。耗时、进程峰值 RSS 及 CUDA 构建可用的 allocated/reserved 峰值分别记录，不作为跨设备性能比较。

## 可选独立验收

安装后的训练和评估只需要 C++ 程序及其依赖。维护者可另外准备具有 NumPy 和 `pycocotools==2.0.11` 的 checker 环境，运行仓库的可选参考测试；Python只检查保存的原生输出和编排程序调用，不实现训练、预测或用户运行入口。

```sh
python tests/scripts/test_detector_example.py --binary "$DETECTOR" \
  --data /data/PennFudanPed --output /runs/detector-check --device cuda
```

测试只训练连续与恢复两条路线，合计2400次逻辑更新，核对完整检查点字节，独立重算解码／NMS，并用 COCO 官方工具核对整体和来源 AP。CUDA路线另对同一冻结权重执行CPU评估。CTest入口为 `RealDetectorExample`；须显式设置 `JYPPX_LIUDING_DETECTOR_DATA_DIR`、全新 `JYPPX_LIUDING_DETECTOR_OUTPUT_DIR` 和可选 `JYPPX_LIUDING_DETECTOR_DEVICE`，未选外部数据时报告跳过且不下载。

`RealDetectorCrashRecovery` 另需 `JYPPX_LIUDING_DETECTOR_BASELINE_DIR`，指向已完成同任务的 `model.safetensors` 和 `checkpoint.jyck`；用 `JYPPX_LIUDING_DETECTOR_BASELINE_MODEL_SHA256` 与 `JYPPX_LIUDING_DETECTOR_BASELINE_CHECKPOINT_SHA256` 显式固定两者摘要。它在第137步提交后强制结束原生进程，再由新进程定期保存并续训1063步；模型和全部检查点状态字节须与基线相同，并核对质量及独立参考，不重训连续基线。构建来源不同时，可预先用 `JYPPX_LIUDING_DETECTOR_EXPECTED_BUILD_COMMIT` 与 `JYPPX_LIUDING_DETECTOR_EXPECTED_DEPENDENCY_LOCK_DIGEST` 分别指定当前源码提交和依赖清单SHA；检查器仅替换容器内对应的两个构建来源值，从原基线独立重编码预期容器，要求其余全部元数据和状态字节相同，再核对整文件字节。未指定的字段要求原字节相同。被强杀进程在最后提交后实际应用了多少更新无法精确确定，结果分别保留计数下界和上界；不将这些更新隐记为零。缺少外部数据或基线时跳过，不下载。
