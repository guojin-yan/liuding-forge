# 迁移学习示例

`linear_probe.cpp` 演示冻结骨干、按批提取特征到有界 CPU 缓存，再只训练分类头。每张输入只执行一次骨干前向，多次 head 更新复用同一缓存；提取结束后释放骨干，避免分类头训练阶段继续持有骨干参数和中间图。`group_rates.cpp` 另行展示骨干与分类头的独立学习率，见[迁移学习教程](../../docs/tutorials/transfer_learning_recipe.md)。

## 构建和运行

准备项目支持的 LibTorch、编译器和本地依赖，在已有兼容构建目录启用 `JYPPX_LIUDING_BUILD_MODELS=ON` 和 `JYPPX_LIUDING_BUILD_EXAMPLES=ON`，然后构建独立目标。以 Windows 开发构建目录为例：

```powershell
cmake --build out/build/config-windows --config Release --target jyppx_transfer_linear_probe --parallel 2
./out/build/config-windows/examples/Release/liuding-forge-example-transfer-linear-probe.exe
```

Linux 使用对应构建目录，示例通常位于 `examples/liuding-forge-example-transfer-linear-probe`。默认无需数据、权重、Python 或联网：固定 seed 创建随机初始化的 DINOv2-S 骨干，将三种确定性 RGB 小图模式各重复四次作为教学输入，提取 L2 归一化的特征，完成 12 次真实 SGD 分类头更新。

输出一个 JSON 对象，包含以下字段；`head_delta`、`loss_before` 和 `loss_after` 按实际浮点结果显示：

```json
{
  "scope": "synthetic_api_demonstration",
  "model": "vit_small_patch14_dinov2",
  "backbone_device": "cpu",
  "weights": "random",
  "samples": 12,
  "cache_bytes": 18528,
  "cache_device": "cpu",
  "features_requires_grad": false,
  "backbone_forward_calls": 4,
  "backbone_unchanged": true,
  "backbone_eval": true,
  "backbone_gradients": 0,
  "backbone_released_before_training": true,
  "head_device": "cpu",
  "optimizer_steps": 12
}
```

`head_delta` 必须大于零；示例会检查骨干全部参数与 buffer 的摘要保持不变、骨干没有梯度、特征不带 autograd 图及分类头确实更新。默认缓存上限为 1 MiB，包含 FP32 特征和 int64 标签的 payload；临时批次、模型、权重加载及 LibTorch allocator 不计入该缓存上限，因此它不是进程 RSS 或显存的硬上限。

```powershell
# 缓存不足时，在分配特征缓存和执行骨干前向前拒绝，退出码为 2。
./out/build/config-windows/examples/Release/liuding-forge-example-transfer-linear-probe.exe --cache-bytes 1
# 也可切换小型骨干；设备选择只影响提特征阶段，分类头在 CPU 上训练。
./out/build/config-windows/examples/Release/liuding-forge-example-transfer-linear-probe.exe --model resnet18
./out/build/config-windows/examples/Release/liuding-forge-example-transfer-linear-probe.exe --model dinov3_vits16 --device cuda
```

## 显式本地权重与真实图像

`--weights` 仅接受[迁移学习教程](../../docs/tutorials/transfer_learning_recipe.md)中固定官方 revision 转换的 DINOv2-S safetensors，文件须为 88,240,512 字节、SHA-256 为 `f3ef1ae01fd4af3cd15926a0d5e631fa0caea50ca68b29db8a68c8c84c9810e5`。公共 `models::load_weights()` 严格检查文件身份和完整 state，错误文件不回退到随机权重；权重不会自动下载。该参数仅适用于 `--model vit_small_patch14_dinov2`，输入仍然是教学张量。

```powershell
./out/build/config-windows/examples/Release/liuding-forge-example-transfer-linear-probe.exe --weights ./weights/dinov2-small.liuding.safetensors
```

迁移到真实图像时，用公共 `data::image` 的 ImageFolder 索引、decoder 和批次几何/数值变换替换 `teaching_images()`，按权重的预处理声明准备 NCHW Tensor，并把真实类别标签填入缓存。先校验样本数与 `samples × (features × 4 + 8)` 字节预算，再预分配缓存，逐批写入；大型数据应显式采用有界分片。缓存只适用于固定权重、固定预处理的冻结骨干，不能复用到骨干更新后或每轮随机增强不同的训练。若持久化缓存，须同时保存并核对模型/权重、样本及顺序、类别、预处理和 L2 归一化选择的身份；本示例不写持久缓存。

本例用于学习 API 和减少重复骨干计算，没有真实数据集的准确率、收敛或速度优势结论，也不是完整 DINO 微调任务的 375 步、65% 质量或 checkpoint 恢复验收。加载预训练权重不会改变这一范围。
