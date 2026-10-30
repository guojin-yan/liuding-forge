# DINO 多层特征示例

[`intermediate_features.cpp`](intermediate_features.cpp) 演示从同一个 DINO 骨干提取第九和第三个 block 的 CLS 向量及 NCHW 特征网格，适用于学习多层表示、密集预测 head 的输入准备和特征可视化。默认使用随机初始化的 DINOv2-S、合成图像和 CPU；也可选择 DINOv3-S，无需数据、权重、Python 或联网。

## 构建和运行

准备项目支持的 LibTorch、编译器与本地依赖，在已有兼容构建目录启用 `JYPPX_LIUDING_BUILD_MODELS=ON` 和 `JYPPX_LIUDING_BUILD_EXAMPLES=ON`。Windows 示例：

```powershell
cmake --build out/build/config-windows --config Release --target jyppx_dino_intermediate_features --parallel 2
./out/build/config-windows/examples/Release/liuding-forge-example-dino-intermediate-features.exe
./out/build/config-windows/examples/Release/liuding-forge-example-dino-intermediate-features.exe dinov3_vits16
```

Linux 使用对应构建目录，运行 `examples/liuding-forge-example-dino-intermediate-features`，参数相同。程序只接受 `vit_small_patch14_dinov2` 或 `dinov3_vits16`，错误参数返回退出码 2。

## 输出与含义

输出一个 JSON 对象；`model` 随参数变化，四项输出顺序与形状相同：

```json
{
  "scope": "synthetic_api_demonstration",
  "weights": "random",
  "device": "cpu",
  "model": "vit_small_patch14_dinov2",
  "shared_state": true,
  "normalized": true,
  "outputs": [
    {"name": "block_8_cls", "shape": [1, 384]},
    {"name": "block_8_patch_grid", "shape": [1, 384, 2, 3]},
    {"name": "block_2_cls", "shape": [1, 384]},
    {"name": "block_2_patch_grid", "shape": [1, 384, 2, 3]}
  ]
}
```

`block_indices={8,2}` 是零基索引，返回顺序保持第九层、第三层；`out_indices={0,2}` 分别选择 CLS 和 patch grid。默认 `normalize=true` 对各选定层分别使用家族的最终 LayerNorm，不改变向后续 block 传递的原始 token。DINOv2 输入为 28×42；DINOv3 输入为 35×50，按其原有规则裁成 32×48，因此均产生 2×3 网格。

提取器与原 adapter 共享完整模型，前向到最大请求索引 8 后停止，后续 block 和分类头不参与本次计算。示例显式使用 `eval()` 与 `NoGradGuard`；训练时按需移除 no-grad，梯度将传入参与计算的层，不会自动冻结任何参数。同层视图共享存储且可能不连续，需要连续布局时显式调用 `contiguous()`。完整合同见[Models API](../../docs/api/models.md#dino-中间层特征)。

接入真实任务时，先严格加载已确认来源与身份的本地权重，按权重预处理图像，再将选定特征交给任务 head。本例只演示随机权重的 C++ API，不表示预训练准确率、完整微调质量、全部容量或 CUDA 已通过验证，也没有速度和内存优势的测量结论。
