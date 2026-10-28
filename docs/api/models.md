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

CNN 分类模型接收与模块设备、dtype 一致的稠密浮点 NCHW 图像，默认输入通道数为 3、类别数为 1000，返回未归一化的 `[N, num_classes]` logits。全部视觉模型均支持 `num_classes=0` 返回 embedding，DINO 系列默认使用该模式。改变输入通道或分类头后，预训练 state 不再直接匹配默认结构。模块保持正常的 LibTorch autograd 与 train/eval 行为；调用方负责线程同步和推理时的 no-grad 边界。

DINOv2 在 CUDA 严格确定性模式下使用可求导的分离矩阵计算缩放位置编码，支持非原生分辨率的完整微调；位置参数、激活与梯度保留在设备上，仅小型插值系数从 CPU 传入。该路径按 FP32 计算并恢复参数 dtype，不改变 state 键或形状；与原生插值的浮点舍入可能不同。CUDA 确定性矩阵运算还要求进程启动前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，完整训练恢复须另按实际配方验收。

## 查询与替换分类头

`classifier_info(adapter)` 查询当前类别数、embedding 宽度、池化或 CLS 语义和被替换的参数前缀。`reset_classifier(adapter, classes)` 替换 head 并刷新传入 adapter 的输出 schema；`classes` 为 0–1,000,000，正数创建全新可训练 head，零移除 head。它保留具体模块与骨干参数身份、冻结标志、device、dtype 和当前模式，不自动冻结骨干或加载权重。

```cpp
auto model = models::create_model("resnet18");
auto info = models::classifier_info(model);
models::reset_classifier(model, 10);
auto logits = model.invoke({.positional = {images}}).value.toTensor();
```

ViT/DINO 的 headless 输出为最终归一化 CLS；CNN/Swin 输出为各自定义的池化向量，保留相应归一化和激活。MobileNetV3 替换完整的两层分类 MLP；移除后返回最后卷积的池化向量，默认 Small/Large 为 576/960 维，并随宽度配置变化。MobileNetV2、EfficientNet B0/B2 和 EfficientNetV2-S 的池化向量仍经过末端卷积，宽度分别为 1280、1280/1408、1280，因此可能不同于最后一个 stage 的通道数。

替换前须停止 forward/backward 和并发状态访问；替换后重建优化器、参数组、checkpoint 拓扑及其他共享 adapter。只有传入的 adapter 会刷新 schema，旧 adapter 别名需要重新创建。特征 tuple adapter 可查询底层 head，替换时会明确拒绝；通用 `transformer_encoder` 没有分类头契约。具体模型的 `reset_classifier()` 也可直接使用，此时调用方负责重建 adapter。

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

`create_feature_extractor()` 覆盖注册表中的 33 个视觉变体。`model_info(name).features` 和 `feature_input` 描述默认配置；`model_feature_spec(name, options)` 无需构造模型即可查询当前配置的输出名称、布局、通道数、步幅和输入规则。MobileNetV3 宽度与 ViT/Swin 动态尺寸选项会反映在查询结果和提取器的 schema 中。元数据不代表每种容量都已通过设备和资源验收；通用 `transformer_encoder` 不提供图像特征入口。

```cpp
auto extractor = models::create_feature_extractor("dinov3_vits16", {}, {0, 2});
extractor.state().eval();
auto outputs = extractor.invoke({.positional = {images}}).value.toTuple();
auto cls = outputs->elements().at(0).toTensor();
auto grid = outputs->elements().at(1).toTensor();
```

第三个参数是输出的零起始索引，保持请求顺序；空列表选择全部输出，重复或越界索引在构造模型前报错。ViT/DINO 的四项依次为 `cls`（NC）、`patch_tokens`（NTC）、`patch_grid`（NCHW）和完整 `tokens`（NTC）；普通 ViT 与 DINOv2 含一个 CLS，DINOv3 含 CLS 加四个 extra token，patch 输出均移除前缀。这四项是最终层的不同视图。

| 家族 | 输出 | 输入规则 |
| --- | --- | --- |
| ResNet/ResNeXt、DenseNet121 | `stage1`–`stage4`，NCHW，名义步幅 4/8/16/32；DenseNet 在 transition 前取各 block 输出 | 接受奇数尺寸；DenseNet 高宽至少 29 |
| MobileNetV2、EfficientNet B0/B2、TF EfficientNetV2-S | `stage1`–`stage5`，NCHW，名义步幅 2/4/8/16/32 | 接受奇数尺寸，保留各网络的 padding 规则 |
| MobileNetV3 Small/Large | 一个 `spatial`，NCHW，名义步幅 32，默认通道 576/960 | 接受奇数尺寸；宽度配置影响输出通道 |
| ConvNeXt Tiny/Small | `stage1`–`stage4`，NCHW，名义步幅 4/8/16/32 | 高宽至少 32，逐级向下取整，保留原输入 |
| Swin-T | `stage1`–`stage4`，NCHW，名义步幅 4/8/16/32 | 默认 224×224；动态模式接受 4 倍数，patch merging 逐级向上取整 |
| 普通 ViT/16 | CLS、patch token、patch grid、完整 token | 默认 224×224；动态模式接受 16 倍数 |
| DINOv2、DINOv3 ViT/ConvNeXt | ViT 的四种最终层视图或 ConvNeXt 的四级 stage | DINOv2 要求 14 倍数；DINOv3 ViT/ConvNeXt 分别裁到 16/32 倍数 |

`FeatureInputSpec::minimum_spatial` 指定高宽各自的最小值，`spatial_multiple` 指定对齐倍数，`fixed_size` 非零时指定精确高宽。`FeatureInfo::reduction` 是名义步幅；遇到奇数尺寸时不能直接用输入尺寸除以它推断输出尺寸。自定义注册表可通过 `FeatureSpecFactory` 解析配置，输出名称、顺序、布局与步幅须保持稳定；解析器只查询元数据，不构造模型或消耗随机数。

提取器执行一次完整特征前向，保留所选 Tensor 的 autograd、device 和 dtype；选择输出不减少主干计算。ViT 的 token/grid 输出共享存储，可能不连续；需要连续布局时由调用方显式转换。状态与模式操作仍通过 `ModelAdapter::state()`，不会自动冻结参数。DINOv2 要求输入高宽为 14 的倍数，DINOv3 ViT/ConvNeXt 分别将底部和右侧裁到 16/32 的倍数；调用方应将该裁剪纳入自己的几何记录。

## 训练状态恢复

启用 `models` 与 `training` 后，可将模型适配器交给 `Trainer`，通过 [`FileCheckpointStore`](durable_checkpoint.md) 保存和恢复完整训练状态。重建时必须保持模型、分类头、优化器、精度和数据管线拓扑一致，并登记模型、优化器、调度器、RNG、数据游标、回调及 CUDA FP16 所需的 GradScaler 状态；只保存模型权重不能恢复训练进度。

当前一致续训验证覆盖 DINOv2 ViT-S/B、DINOv3 ViT-S/S+/B/L/L+ 和 ConvNeXt-T/S/B/L 的 CPU FP32 与 CUDA FP16 autocast，CUDA 路径保留 FP32 参数；DINOv2 ViT-L 另有 CPU FP32 恢复验证，其 CUDA 完整恢复仍待验证。逐字节相同的续训结果要求固定软件、硬件、线程和确定性算法设置；这些验证使用单线程 CPU 计算、两个数据 worker 和严格确定性算法。DINOv2 的 CPU/CUDA 输入分别为 28×28/原生 518×518，DINOv3 为 32×32；DINOv2 位置插值的 CUDA 反向计算可能存在数值非确定性。其他容量、精度、分辨率和任务仍需分别验证。

大型模型需要为恢复预留足够的回滚空间，可通过 `TrainerOptions::checkpoint.max_rollback_bytes` 配置；DINOv2 ViT-L 和 DINOv3 ViT-L/L+ 的上述恢复验证显式使用 4 GiB 预算。该值限制回滚状态，不等于训练进程的全部内存需求。

## 权重与验证

`available_weights()` 返回的上游候选默认均为 `not_provisioned`，许可证为 `NOASSERTION`。候选名称或短哈希不构成可用权重证据。模型组件不会下载或反序列化 `.pth`/pickle，也不因架构参考通过而宣称预训练准确率。

调用方已独立确认完整 SHA-256、文件大小、来源和制品许可证时，可创建 `pinned_external` 的 `WeightRecord`。`verify_weights()` 验证字节身份及有界 safetensors 结构；`load_weights()` 通过 `interop::import_model_weights` 执行显式严格或部分映射，并返回实际映射报告。导入须在模型独占且静止时执行；之后应重建旧的存储视图和优化器状态。`cache_weights()` 仅在调用方提供显式传输回调时获取字节，缓存命中会重新验证；缓存目录必须由调用方独占管理。

需要复现权重输入时，在 `WeightRecord::input_spec` 中登记 `WeightInputSpec`：权重来源 revision、state 映射版本、预处理声明的来源与 revision、`ImagePreprocessSpec` 和按分类器输出列排序的类别 ID/名称。embedding 权重可省略类别表；缺少表不表示已知类别。`weight_input_spec(record)` 借用该记录中的元数据，缺失时报 `NotReady`，不会按架构名猜测；记录必须保持存活且不修改。当前内置候选没有经过核验的输入契约。

`ImagePreprocessSpec` 描述确定性评估输入：最终高宽、直接 resize 或保持比例的短边 resize 加中心裁剪、nearest-exact/bilinear/bicubic、gray/RGB/BGR 和逐通道 mean/std。处理顺序是几何变换、转单位强度、按输出通道顺序做 `(pixel-mean)/std`；整数照片在插值后先舍入饱和，浮点 bicubic 过冲保留。bilinear/bicubic 固定使用 `align_corners=false, antialias=true`，不承诺 PIL 逐位等价。训练随机增强、执行 dtype/device、EXIF/ICC、alpha 或灰度转换由调用方单独配置。

`resolve_image_preprocess_geometry(spec, {source_height, source_width})` 只计算坐标：短边 resize 的长边向下取整，中心偏移使用最近偶数舍入；尺寸溢出或裁剪超出画布时报错。将结果转换为 [`data_image` 的 resize/crop 步骤](data_image.md)，继续保留 `GeometryTrace`，再用 `transform_image_batch` 显式设置 `to_unit`、通道交换和同一组 mean/std。模型组件本身不依赖或执行图像解码，调用方要核对源 `PixelSpec`，避免重复归一化。

`weight_input_identity(record)` 以流式 SHA-256 绑定已固定权重的完整摘要/大小、来源和许可、输入配方与类别顺序，供配方、run 或缓存记录；没有固定权重或输入契约时报 `NotReady`。它只核验元数据格式，不能证明来源真实、实际执行了变换、分类头与类别表匹配或任务准确率，权重字节仍须单独 `verify_weights()`。类别表最多 100,000 项、文本合计 16 MiB，ID 唯一；类别名称可重复或为空。

当前固定随机参考覆盖选定架构的结构、state 导入和前向/特征差分，不覆盖预训练权重、任务精度或完整 Models profile。未提供参考目录时，相应可选测试会条件跳过；跳过不算验收通过。实际支持范围以具体变体、设备、dtype 和验证结果为准。

### 固定外部权重

以下两个来源可用于显式离线导入。下载应由使用者执行；库内候选仍保持 `not_provisioned`，完整数据集精度和微调资格尚待验证。ResNet50 的 1000 列输出暂未绑定经过独立核验的有序类别表，不能据此推断类别名称。

| 模型与固定来源 | 权重输入配方 | 导入设置 |
| --- | --- | --- |
| ResNet50：[timm/resnet50.a1_in1k，7672686](https://huggingface.co/timm/resnet50.a1_in1k/tree/767268603ca0cb0bfe326fa87277f19c419566ef)，Apache-2.0 | RGB 单位强度、bicubic 短边 235、中心裁剪 224×224、ImageNet mean/std | 原始 safetensors 可严格导入 `resnet50`，1000 类 |
| DINOv2 ViT-S/14：[facebook/dinov2-small，ed25f3a](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056)，Apache-2.0 | RGB 单位强度、bicubic 短边 256、中心裁剪 224×224、ImageNet mean/std | 先运行下述转换，`dinov2.num_classes=0`、`dinov2.position_antialias=false` |

两者 mean 为 `{0.485, 0.456, 0.406}`，std 为 `{0.229, 0.224, 0.225}`。上述图像配方可按前文组合 Tensor 图像变换；上游 PIL/图像处理器与 Tensor bicubic 不承诺逐位相同。DINOv2 的 `position_antialias` 控制模型内部位置编码插值，独立于图像 resize：默认 `true` 对应已有 timm 参考，官方 Transformers 权重路径使用 `false`；在原生 518×518 网格上不插值。

| 制品 | 字节数 | 完整 SHA-256 |
| --- | --- | --- |
| ResNet50 原始 `model.safetensors` | 102469840 | `773525d5821de224f8f30c33377b7a795d7863e08522698200d3217d3f2a41bb` |
| 官方 DINOv2-S 原始 `model.safetensors` | 88249960 | `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1` |
| DINOv2-S 转换结果 | 88240512 | `f3ef1ae01fd4af3cd15926a0d5e631fa0caea50ca68b29db8a68c8c84c9810e5` |

[`convert_dinov2_hf.py`](../../scripts/models/convert_dinov2_hf.py) 仅接受表中官方固定文件，使用 Python 标准库流式转换，不联网、不执行 pickle、不依赖 Torch。它保留参数位模式，按 Q/K/V 顺序合并 attention 参数，登记无掩码图像编码器不使用的 `mask_token`，并拒绝覆盖已有输出。

```powershell
python scripts/models/convert_dinov2_hf.py --source D:/weights/dinov2-small.facebook.safetensors --destination D:/weights/dinov2-small.liuding.safetensors --report D:/weights/dinov2-small.conversion.json
```

在 `WeightRecord` 中登记转换结果的完整摘要/大小，保留官方来源 revision、许可证链接和映射版本 `jyppx.liuding.dinov2-hf-small.unmasked.v1`，再调用 `load_weights()`。转换报告保留原始权重身份和全部参数映射；转换只确认数据布局，不证明任务准确率。该入口不支持掩码训练、其他容量或其他 revision 的自动转换。

迁移学习可以将 `num_classes=0` 的 embedding 接入新分类头，并显式冻结骨干参数、保持骨干 eval 模式以固定 BN 状态。分类数据集的逐批推理与质量报告接口见[训练 API](training.md#分类数据集评估)；输入配方、特征归一化和新类别顺序由任务明确指定。
