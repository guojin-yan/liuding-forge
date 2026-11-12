# 用 C++ 训练分割模型并恢复完整状态

[pedestrian_segmentation.cpp](pedestrian_segmentation.cpp) 使用原生 `Trainer`、Adam、随机初始化小型 U-Net 和完整检查点训练行人前景分割。模型声明与实现在 [model/](model/)；照片是唯一模型输入，实例掩码只提供训练目标和评估真值。该小规模任务不是生产分割模型或官方 benchmark。

先按[数据说明](../../../tests/fixtures/metrics/penn_fudan/README.md)取得并核验 Penn-Fudan 原图、掩码和 manifest，放在仓库外。固定120张训练、25张验证、25张测试；照片缩放至128×128，掩码使用 nearest-exact。训练为80轮、batch8、Adam学习率0.001、seed20261028、FP32，使用最终轮权重；不根据验证或测试选择权重。

在启用 `training`、`data_image`、`interop`、`optim`、`experiment` 和 examples 的兼容构建目录中编译：

```sh
cmake --build out/build/YOUR_PROFILE --target jyppx_liuding_pedestrian_segmentation --parallel 1
SEGMENT=out/build/YOUR_PROFILE/examples/liuding-forge-example-pedestrian-segmentation
"$SEGMENT" train /data/PennFudanPed /runs/segmentation cpu
```

Windows 使用 `.exe` 后缀及本机绝对路径；CUDA 使用匹配的 LibTorch 和 `cuda` 设备参数，并在启动前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。所有输出目录必须全新。示例支持的参数不表示所有平台和设备都已取得完整任务资格。

训练写出 `model.safetensors`、`checkpoint.jyck` 与 `training.json`；JSON包含初始/最终更新数、实际应用更新数、完整状态参与者数量、模型/检查点SHA和构建身份。主动暂停和恢复示例：

```sh
"$SEGMENT" train /data/PennFudanPed /runs/paused cpu - 137
"$SEGMENT" train /data/PennFudanPed /runs/resumed cpu /runs/paused/checkpoint.jyck
```

可选尾参数 `--checkpoint-every STEP` 在绝对更新数的倍数提交完整检查点，STEP为1至1199；最终或主动暂停边界也提交。保存后刷新一行包含 `event=checkpoint`、`global_step`、`checkpoint_sha256` 的JSON。意外中断后可从最后提交的检查点恢复到全新目录；之后尚未保存的更新会重做，不算作已保存进度。中断目录可能没有最终模型或训练摘要。原子替换遵循 `FileCheckpointStore` 合同，不保证断电或任意文件系统的持久性。

从完整训练的 `training.json` 读取 `weights_sha256`，显式指定期望模型身份后评估：

```sh
"$SEGMENT" evaluate /data/PennFudanPed /runs/evaluation cpu /runs/resumed/model.safetensors EXPECTED_SHA256
```

评估输出IoU、Dice、按Penn/Fudan来源的统计与质量报告，并验证batch1/4聚合、物理保存后重建模型/会话继续、完成态重开。前景IoU门槛为0.35，mean IoU为0.60。模型前景四连通分量面积至少16像素时生成tight exclusive xyxy候选框，分数为该分量的前景softmax均值，AP50门槛为0.10；候选框不读取真值，这是分割衍生评估。

可选维护检查器 `tests/scripts/test_segmentation_example.py` 使用独立进程核对连续、暂停恢复、真实强杀恢复的完整模型和检查点字节，并调用独立Pillow/NumPy/COCO参考检查前处理和计分。它需要全新输出目录及显式外部数据；未选数据时CTest跳过，不下载。`--recovery-only --model-sha256 SHA` 只执行两条恢复路线并固定既有最终模型SHA，不重新训练已合格连续基线；此时完整检查点只比较两条恢复路线，不声称与缺失的历史检查点相同。

已验证上述固定任务在 Ubuntu CUDA FP32 上的连续完整训练，以及同环境暂停、新进程强杀后的恢复等价；三条路线的最终模型与完整训练状态一致，构建来源单独核对。留出集 mean IoU 为0.6775、前景 IoU 为0.4917、分割衍生 AP50 为0.3507。Ubuntu CPU 也已完成连续训练与两条恢复路线；跨平台模型SHA不作为同环境等价承诺，设备参数支持与实际任务资格分别判断。
