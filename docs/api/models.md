# 模型组件

`models` 是默认关闭的可选组件。构建时设置 `JYPPX_LIUDING_BUILD_MODELS=ON`，安装后使用 `find_package(jyppx CONFIG REQUIRED COMPONENTS models)`，并链接 `jyppx::liuding::models`。组件直接依赖 `nn` 和 `interop`；模型构造不联网，不自动下载权重，也不执行图像解码、归一化或训练循环。

## 创建模型

公共入口为 `<jyppx/liuding/models/models.hpp>`。`list_models()` 和 `model_info()` 查询本地注册表，`create_model()` 返回拥有模型状态的 `nn::ModelAdapter`。构造使用 LibTorch 的当前随机数状态；需要可复现初始化时，应由调用方在构造前设置种子并记录设备和版本。

```cpp
#include <jyppx/liuding/models/models.hpp>

namespace models = jyppx::liuding::models;

auto info = models::model_info("efficientnet_b0");
models::ModelOptions options;
options.efficientnet.num_classes = 10;
auto adapter = models::create_model(info.name, options);
adapter.state().eval();
torch::NoGradGuard no_grad;
auto result = adapter.invoke({.positional = {images}, .keyword = {}});
```

CNN 分类模型接收与模块设备、dtype 一致的稠密浮点 NCHW 图像，默认输入通道数为 3、类别数为 1000，返回未归一化的 `[N, num_classes]` logits。DINO 系列默认无分类头，返回 embedding；显式设置正类别数后才返回 logits。改变输入通道或分类头后，预训练 state 不再直接匹配默认结构。模块保持正常的 LibTorch autograd 与 train/eval 行为；调用方负责线程同步和推理时的 no-grad 边界。

## 当前架构

下表是公共注册表中的模型架构，并非可下载的预训练模型清单。表中的参考通过表示固定随机 state、输入和输出的独立差分；完整设备矩阵、权重和任务精度须分别验证。

| 家族 | 注册表名称 | 特征接口与验证边界 |
| --- | --- | --- |
| ResNet / ResNeXt | `resnet18`、`resnet34`、`resnet50`、`resnet101`、`resnext50_32x4d` | 四级残差 stage；34/101 与 ResNeXt50 已有固定 torchvision 随机参考，18/50 的精确 timm 映射另验。 |
| DenseNet | `densenet121` | 四级 dense/transition 特征；固定 torchvision 随机参考通过。 |
| MobileNetV2 | `mobilenetv2_100` | 五级特征；固定 timm 随机参考通过。 |
| MobileNetV3 | `mobilenet_v3_small`、`mobilenet_v3_large` | 最后一级卷积特征；与目标 timm 变体的精确映射尚未验收。 |
| EfficientNet | `efficientnet_b0`、`efficientnet_b2` | 步长 2/4/8/16/32 的五级特征；固定 timm 随机参考通过。 |
| EfficientNetV2 | `tf_efficientnetv2_s` | fused-MBConv、MBConv 与动态 TF SAME padding，返回五级特征；固定 timm 随机状态、logits 和特征差分通过。 |
| ConvNeXt | `convnext_tiny`、`convnext_small` | 四级 NCHW 特征与全局分类头；固定 timm 随机 state、logits 和四级特征差分通过。与 DINOv3 ConvNeXt 是独立架构及权重身份。 |
| ViT/16 | `vit_tiny_patch16_224`、`vit_small_patch16_224`、`vit_base_patch16_224` | 十二层预归一化注意力骨干，提供 CLS/patch token 和 NCHW patch grid；固定 timm 随机 state、logits、token 与 grid 差分通过。 |
| Swin Transformer | `swin_tiny_patch4_window7_224` | 四级 shifted-window 骨干，提供步长 4/8/16/32 的 NCHW 特征；固定 timm 随机 state、logits 和四级特征差分通过。 |
| DINOv2 ViT/14 | `vit_small_patch14_dinov2`、`vit_base_patch14_dinov2`、`vit_large_patch14_dinov2`、`vit_giant_patch14_dinov2` | 四容量的 token/CLS 固定 timm 随机参考通过；Giant 的 backward/资源及完整设备矩阵未关闭。 |
| DINOv3 ViT/16 | `dinov3_vits16`、`dinov3_vits16plus`、`dinov3_vitb16`、`dinov3_vitl16`、`dinov3_vitl16plus`、`dinov3_vith16plus`、`dinov3_vit7b16` | 七个官方 factory 已登记；S/S+/B、L/L+ 有分层固定 Meta 随机参考，H+/7B 尚无独立执行资源门禁。 |
| DINOv3 ConvNeXt | `dinov3_convnext_tiny`、`dinov3_convnext_small`、`dinov3_convnext_base`、`dinov3_convnext_large` | 四级空间特征；T/S/B/L 有固定 Meta 随机参考，B/L 的完整 dtype/设备矩阵未关闭。 |
| 特征 Transformer | `transformer_encoder` | 已嵌入序列的 batch-first encoder，不提供 tokenizer、图像 patch embedding 或分类头。 |

`ModelRegistry::add` 可注册调用方自己的显式工厂；同名和空工厂会被拒绝。注册表没有静态插件加载。`ModelInfo::default_parameters` 只统计默认参数，不包括 buffer；自定义配置会改变参数量。

## 特征与输入语义

ResNet/ResNeXt 与 ConvNeXt 返回四级 NCHW stage，DenseNet 返回四级 dense/transition 特征，MobileNetV2、EfficientNet 与 EfficientNetV2 返回五级特征，MobileNetV3 返回最后卷积特征图。普通 ConvNeXt Tiny/Small 的四级步长为 4/8/16/32，默认通道为 96/192/384/768，对不能整除步长的输入逐级取整，默认分类头在全局平均池化后执行通道归一化。`tf_efficientnetv2_s` 对奇数尺寸输入使用动态 TF SAME padding，五级特征对应步长 2/4/8/16/32。`forward_features` 保留 autograd；在提取特征前应按任务选择 train/eval 模式。

DINOv2 要求高宽为 14 的正整数倍，`forward_features` 返回 `[N, 1 + H/14 * W/14, D]`，CLS 在第一个位置，其后为行优先 patch token。非参考网格使用 bicubic 位置编码插值。`num_classes == 0` 时 `forward` 返回 CLS embedding；设置正类别数才返回 logits。`reset_classifier()` 后应重建 adapter，使输出 schema 与新 head 一致。

普通 ViT/16 默认要求 224×224 输入；设置 `ViTOptions::dynamic_image_size = true` 后可接受高宽均为 16 的正整数倍的图像，并对 14×14 参考位置编码网格作 bicubic 插值。`forward_features` 返回 `[N, 1 + H/16 * W/16, D]`，`forward_patch_grid` 返回归一化 patch token 组成的 `[N,D,H/16,W/16]`。默认分类头输出 1000 类 logits，设置 `num_classes = 0` 则返回 CLS embedding；替换分类头后须重建 adapter。随机架构差分不代表预训练权重可用，Tiny/Small/Base 的 CPU 单步 backward 已分别验收，完整设备和任务矩阵仍开放。

Swin-T 默认要求 224×224 输入；设置 `SwinOptions::dynamic_image_size = true` 后接受高宽均为 4 的正整数倍的图像。四级 `forward_features` 返回通道数 96/192/384/768 的 NCHW 特征，空间尺寸在 patch embedding 后按 patch merging 逐级向上取整；窗口大小为 7，交替使用局部窗口与位移窗口注意力。默认分类头先对最后一级归一化特征作全局平均池化，再输出 1000 类 logits；`num_classes = 0` 返回 768 维 embedding。更换 head 后应重建 adapter。当前随机架构参考与 CPU 单步训练不代表预训练权重、完整设备矩阵或任务质量。

DINOv3 ViT 的 token 顺序为 CLS、四个 extra/storage token、patch token；输入高宽会裁到不大于原尺寸的 16 倍数。DINOv3 ConvNeXt 输出四级 NCHW 特征，输入高宽会裁到 32 的倍数。调用方应在数据管线中明确记录这些裁剪及其几何语义，不能把输出当作原图全幅特征。高资源变体的构造、训练、显存和设备能力以各自实测门禁为准。

[图像训练基准](../benchmarks/README.md#图像读取到训练更新)可选择 DINOv2 四容量，以及 DINOv3 ViT-S/S+/B/L/L+ 和 ConvNeXt-T/S/B/L，检查文件读取、Trainer、反向传播与实际 SGD 更新。高容量模型须显式选择并串行测量；小尺寸随机初始化验证不代表预训练权重、任务准确率或常用分辨率的资源要求。DINOv3 H+/7B 尚未接入该训练基准。

`transformer_encoder` 接收已嵌入的 `[batch, sequence, model_dimension]` 浮点特征，返回相同形状。可选 `attention_mask` 是 `[S,S]` bool，`padding_mask` 是 `[N,S]` bool，`true` 分别表示屏蔽注意力边或忽略 key。调用方提供位置与因果语义；全屏蔽行保持 LibTorch 后端行为，可能产生 NaN。

## 统一特征提取

`model_info(name).features` 查询输出名称、布局、通道数、步幅和特殊 token 数，`feature_input` 说明输入尺寸规则。当前统一入口覆盖 DINOv2、DINOv3 ViT 与 DINOv3 ConvNeXt；其他骨干仍使用各自的 `forward_features`。元数据不代表每种容量都已通过设备和资源验收。

```cpp
auto extractor = models::create_feature_extractor("dinov3_vits16", {}, {0, 2});
extractor.state().eval();
auto outputs = extractor.invoke({.positional = {images}}).value.toTuple();
auto cls = outputs->elements().at(0).toTensor();
auto grid = outputs->elements().at(1).toTensor();
```

第三个参数是 `features` 的零起始索引，保持请求顺序；空列表选择全部输出，重复或越界索引在构造模型前报错。ViT 的四项依次为 `cls`（NC）、`patch_tokens`（NTC）、`patch_grid`（NCHW）和完整 `tokens`（NTC）；DINOv2 的完整序列含一个 CLS，DINOv3 含 CLS 加四个 extra token，patch 输出均移除这些前缀。这些是最终层的不同视图。ConvNeXt 的四项为 `stage1`–`stage4`，布局 NCHW，步幅为 4/8/16/32。

提取器执行一次完整特征前向，保留所选 Tensor 的 autograd、device 和 dtype；选择输出不减少主干计算。ViT 的 token/grid 输出共享存储，可能不连续；需要连续布局时由调用方显式转换。状态与模式操作仍通过 `ModelAdapter::state()`，不会自动冻结参数。DINOv2 要求输入高宽为 14 的倍数，DINOv3 ViT/ConvNeXt 分别将底部和右侧裁到 16/32 的倍数；调用方应将该裁剪纳入自己的几何记录。

## 训练状态恢复

启用 `models` 与 `training` 后，可将模型适配器交给 `Trainer`，通过 [`FileCheckpointStore`](durable_checkpoint.md) 保存和恢复完整训练状态。重建时必须保持模型、分类头、优化器、精度和数据管线拓扑一致，并登记模型、优化器、调度器、RNG、数据游标、回调及 CUDA FP16 所需的 GradScaler 状态；只保存模型权重不能恢复训练进度。

当前一致续训验证覆盖 DINOv2 ViT-S/B、DINOv3 ViT-S/S+/B/L/L+ 和 ConvNeXt-T/S/B/L 的 CPU FP32 与 CUDA FP16 autocast，CUDA 路径保留 FP32 参数；DINOv2 ViT-L 另有 CPU FP32 恢复验证，其 CUDA 完整恢复仍待验证。逐字节相同的续训结果要求固定软件、硬件、线程和确定性算法设置；这些验证使用单线程 CPU 计算、两个数据 worker 和严格确定性算法。DINOv2 的 CPU/CUDA 输入分别为 28×28/原生 518×518，DINOv3 为 32×32；DINOv2 位置插值的 CUDA 反向计算可能存在数值非确定性。其他容量、精度、分辨率和任务仍需分别验证。

大型模型需要为恢复预留足够的回滚空间，可通过 `TrainerOptions::checkpoint.max_rollback_bytes` 配置；DINOv2 ViT-L 和 DINOv3 ViT-L/L+ 的上述恢复验证显式使用 4 GiB 预算。该值限制回滚状态，不等于训练进程的全部内存需求。

## 权重与验证

`available_weights()` 返回的上游候选默认均为 `not_provisioned`，许可证为 `NOASSERTION`。候选名称或短哈希不构成可用权重证据。模型组件不会下载或反序列化 `.pth`/pickle，也不因架构参考通过而宣称预训练准确率。

调用方已独立确认完整 SHA-256、文件大小、来源和制品许可证时，可创建 `pinned_external` 的 `WeightRecord`。`verify_weights()` 验证字节身份及有界 safetensors 结构；`load_weights()` 通过 `interop::import_model_weights` 执行显式严格或部分映射，并返回实际映射报告。导入须在模型独占且静止时执行；之后应重建旧的存储视图和优化器状态。`cache_weights()` 仅在调用方提供显式传输回调时获取字节，缓存命中会重新验证；缓存目录必须由调用方独占管理。

当前固定随机参考覆盖选定架构的结构、state 导入和前向/特征差分，不覆盖预训练权重、任务精度或完整 Models profile。未提供参考目录时，相应可选测试会条件跳过；跳过不算验收通过。实际支持范围以具体变体、设备、dtype 和验证结果为准。
