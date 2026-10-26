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
| DINOv2 ViT/14 | `vit_small_patch14_dinov2`、`vit_base_patch14_dinov2`、`vit_large_patch14_dinov2`、`vit_giant_patch14_dinov2` | 四容量的 token/CLS 固定 timm 随机参考通过；Giant 的 backward/资源及完整设备矩阵未关闭。 |
| DINOv3 ViT/16 | `dinov3_vits16`、`dinov3_vits16plus`、`dinov3_vitb16`、`dinov3_vitl16`、`dinov3_vitl16plus`、`dinov3_vith16plus`、`dinov3_vit7b16` | 七个官方 factory 已登记；S/S+/B、L/L+ 有分层固定 Meta 随机参考，H+/7B 尚无独立执行资源门禁。 |
| DINOv3 ConvNeXt | `dinov3_convnext_tiny`、`dinov3_convnext_small`、`dinov3_convnext_base`、`dinov3_convnext_large` | 四级空间特征；T/S/B/L 有固定 Meta 随机参考，B/L 的完整 dtype/设备矩阵未关闭。 |
| 特征 Transformer | `transformer_encoder` | 已嵌入序列的 batch-first encoder，不提供 tokenizer、图像 patch embedding 或分类头。 |

`ModelRegistry::add` 可注册调用方自己的显式工厂；同名和空工厂会被拒绝。注册表没有静态插件加载。`ModelInfo::default_parameters` 只统计默认参数，不包括 buffer；自定义配置会改变参数量。

## 特征与输入语义

ResNet/ResNeXt 返回四级 NCHW stage，DenseNet 返回四级 dense/transition 特征，MobileNetV2、EfficientNet 与 EfficientNetV2 返回五级特征，MobileNetV3 返回最后卷积特征图。`tf_efficientnetv2_s` 对奇数尺寸输入使用动态 TF SAME padding，五级特征对应步长 2/4/8/16/32。`forward_features` 保留 autograd；在提取特征前应按任务选择 train/eval 模式。

DINOv2 要求高宽为 14 的正整数倍，`forward_features` 返回 `[N, 1 + H/14 * W/14, D]`，CLS 在第一个位置，其后为行优先 patch token。非参考网格使用 bicubic 位置编码插值。`num_classes == 0` 时 `forward` 返回 CLS embedding；设置正类别数才返回 logits。`reset_classifier()` 后应重建 adapter，使输出 schema 与新 head 一致。

DINOv3 ViT 的 token 顺序为 CLS、四个 extra/storage token、patch token；输入高宽会裁到不大于原尺寸的 16 倍数。DINOv3 ConvNeXt 输出四级 NCHW 特征，输入高宽会裁到 32 的倍数。调用方应在数据管线中明确记录这些裁剪及其几何语义，不能把输出当作原图全幅特征。高资源变体的构造、训练、显存和设备能力以各自实测门禁为准。

`transformer_encoder` 接收已嵌入的 `[batch, sequence, model_dimension]` 浮点特征，返回相同形状。可选 `attention_mask` 是 `[S,S]` bool，`padding_mask` 是 `[N,S]` bool，`true` 分别表示屏蔽注意力边或忽略 key。调用方提供位置与因果语义；全屏蔽行保持 LibTorch 后端行为，可能产生 NaN。

## 权重与验证

`available_weights()` 返回的上游候选默认均为 `not_provisioned`，许可证为 `NOASSERTION`。候选名称或短哈希不构成可用权重证据。模型组件不会下载或反序列化 `.pth`/pickle，也不因架构参考通过而宣称预训练准确率。

调用方已独立确认完整 SHA-256、文件大小、来源和制品许可证时，可创建 `pinned_external` 的 `WeightRecord`。`verify_weights()` 验证字节身份及有界 safetensors 结构；`load_weights()` 通过 `interop::import_model_weights` 执行显式严格或部分映射，并返回实际映射报告。导入须在模型独占且静止时执行；之后应重建旧的存储视图和优化器状态。`cache_weights()` 仅在调用方提供显式传输回调时获取字节，缓存命中会重新验证；缓存目录必须由调用方独占管理。

当前固定随机参考覆盖选定架构的结构、state 导入和前向/特征差分，不覆盖预训练权重、任务精度或完整 Models profile。未提供参考目录时，相应可选测试会条件跳过；跳过不算验收通过。实际支持范围以具体变体、设备、dtype 和验证结果为准。
