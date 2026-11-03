# 模型来源与许可

本说明记录模型组件已识别的上游参考和应保留的许可材料。本项目贡献者拥有并有权授权的贡献采用 Apache-2.0；该声明不将上游代码、模型权重或数据集重新许可为 Apache-2.0。提及上游项目或模型名称仅用于说明来源与兼容性，不表示关联或背书。

## 已识别的参考

下表记录独立参考脚本使用的上游输入，不是逐文件的源码衍生关系判定，也不是预训练权重分发清单。

| 模型或用途 | 已固定的参考 | 上游许可原文 |
| --- | --- | --- |
| ResNet34/101、ResNeXt50、DenseNet121 的随机参考 | torchvision 0.24.1 | [BSD-3-Clause](torchvision-0.24.1-LICENSE)，Copyright (c) Soumith Chintala 2016 |
| ResNet18/50、MobileNetV3 的随机参考，以及部分预处理和参考环境 | torchvision 0.29.1 | [BSD-3-Clause](torchvision-0.29.1-LICENSE)，Copyright (c) Soumith Chintala 2016 |
| MobileNetV2、EfficientNet/V2、ConvNeXt、ViT、Swin、DINOv2 及部分 ResNet 参考 | timm `af97f99efef62bd9bd7eaa908fffe5b7a2d98e2c` | [根 Apache-2.0 许可](timm-af97f99e-LICENSE)，Copyright 2019 Ross Wightman；各文件另保留原始署名 |
| DINOv3 ViT 和 DINOv3 ConvNeXt | Meta DINOv3 `6876159a11b4df116f30f667f8c9888617df0751` | [DINOv3 License](dinov3-6876159a-LICENSE.md)，Meta Platforms, Inc. and affiliates |

PyTorch 的 Tensor、autograd、Transformer 和其他运行接口另适用对应 LibTorch/PyTorch 许可。参考环境中的 Transformers、safetensors 等工具及实际加载的模型权重，也需要按各自版本和制品核对；上表不声称已经穷尽全部开发环境依赖。

## 文件声明与原始署名

除根许可证外，保留了固定 timm 源码中的 [ConvNeXt 声明](models-timm-convnext-SOURCE-NOTICE.txt)、[Swin Transformer 声明](models-timm-swin_transformer-SOURCE-NOTICE.txt)、[ViT 声明](models-timm-vision_transformer-SOURCE-NOTICE.txt)和 [EfficientNet 声明](models-timm-efficientnet-SOURCE-NOTICE.txt)。这些原文包含 Ross Wightman 的对应修改署名，以及 Meta、Microsoft 等原始来源；不能只用 timm 的根许可证替代。

原始 ConvNeXt 的 [Meta MIT 许可](convnext-upstream-MIT-LICENSE)及 Swin Transformer 的 [Microsoft MIT 许可](swin-transformer-upstream-MIT-LICENSE)一并留存，作为 timm 文件中继承声明的上下文。许可文档的准确抓取提交记录在来源索引中，不据此宣称这些提交就是 timm 或本项目实际派生代码的历史版本。ConvNeXt 文件还单独说明部分 ConvNeXt-V2 权重的 CC BY-NC 4.0 条款；这不表示当前框架已经包含这些权重。

固定 Meta 源码的 [DINOv3 ViT 文件头](models-dinov3-vision_transformer-SOURCE-NOTICE.txt)及 [DINOv3 ConvNeXt 文件头](models-dinov3-convnext-SOURCE-NOTICE.txt)均原样保留，明确指向 DINOv3 License。摘录仅供核对来源和署名，完整协议仍是适用条款的依据。

## DINOv3 的独立条款

此处 `LicenseRef-DINOv3` 指向保留的完整 [DINOv3 License](dinov3-6876159a-LICENSE.md)，并非 Apache-2.0。协议定义的 DINO Materials 包含模型代码、训练和推理代码、权重及其他材料，不仅限于预训练权重。

协议授予有条件的使用、复制、修改和分发权；分发 DINO Materials 或其衍生作品须按该协议进行，并向接收者提供协议副本。协议另有研究发表署名、使用与贸易管制限制、反向工程限制、终止及其他条款；使用者应阅读完整原文，不能将其理解为无条件宽松许可或一概禁止商业使用。

已确认独立参考脚本加载并执行固定 Meta 官方源码。当前 C++ ViT 的后续实现按官方源码修正二维 patch RoPE、`periods`、storage/mask token、`qkv.bias_mask` 和前馈网络状态布局等；DINOv3 ConvNeXt 也按官方状态、归一化公式和下采样顺序作过适配。这些事实确认了官方源码的使用和后续兼容适配，当前实现的独立开发来源尚未证实。参考测试使用官方材料本身应遵守该协议，不以是否加载预训练权重为前提。

数学公式、功能、状态键名或数值结果相同，本身不足以证明复制了受保护的源码表达。ConvNeXt 的通用 block 和归一化结构在更早的 [MIT 原版](convnext-upstream-MIT-LICENSE)中已经存在，需与 DINOv3 特有适配分别判断；这也不自动证明当前 C++ 实现只适用 MIT。现有证据尚不能最终认定 C++ 实现是否属于衍生作品，也不能证明其完全独立。

在具体复用范围厘清前，包含这些实现的源码或二进制不能被表述为已确认只受 Apache-2.0 覆盖。实际复制、翻译或实质改写 DINO Materials 的部分及相应衍生作品应遵守 DINOv3 协议的分发条件；只附协议或关闭编译选项均不替代该义务。

## 来源状态与分发

依赖清单中的 `models-reference-timm`、`models-reference-dinov3`、`models-reference-torchvision-0.24.1` 和 `models-reference-torchvision-0.29.1` 继续为 `pending`。已保留原文并核对参考归档身份，不等于已完成 C++ 源码来源或最终制品的再分发审查；发布前需要记录实际复用的上游文件、版本、修改和适用条款，再作对应处理。`models` 默认关闭不消除已经随源码分发的材料所产生的义务。

源码中的 `dependency-manifest.json` 和本目录 `manifest.json` 记录参考源归档及保留原文的 URL、SHA-256、字节数和归档成员；文件头摘录另外记录原始字节区间。上游原文不修改换行或措辞。启用 `models` 时，本说明与所列模型许可和声明一起安装到 `share/doc/jyppx/licenses/models/`，本说明中的原文链接在安装后仍可使用。

框架代码、具体权重和数据集分别适用各自授权。随机初始化参考、形状或数值测试只证明相应测试行为，不授予权重、数据集或源码的使用和分发权；微调、量化或格式转换也不会自动消除原始条款。
