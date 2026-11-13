# 扩展模型与训练任务

扩展模型使用 `JYPPX_LIUDING_BUILD_MODELS_VISION=ON`，并依赖 `models`、`vision_ops`、`nn`。训练、数据、检查点和 CLI 继续复用 `vision_tasks` 与既有 Trainer。当前为原生开发预览；官方权重映射、完整训练精度和设备资格须分别核对，模型名称不表示已经通过上游数值对照。

## 模型选择

| 任务 | architecture 起点 | 输入与输出 |
| --- | --- | --- |
| 多标签分类 | 既有分类骨干，如 `resnet50` | `classify_multilabel`；独立 logits，known mask 区分未知与负类 |
| FCOS | `fcos_resnet50_fpn` | 五级 FPN、中心采样、focal/GIoU/centerness |
| 有限特征组合 | `fcos_resnet50_bifpn`、`fcos_resnet50_bifpn_softmax`、`fcos_swin_tiny_fpn`、`fcos_swin_tiny_bifpn`、`fcos_convnext_tiny_bifpn`、`fcos_vit_tiny_token_pyramid` | 明确网格/stride；fast attention 和 softmax BiFPN 使用不同身份 |
| ViT-Adapter | `fcos_vit_adapter_tiny_fpn` | SPM、injector/extractor、四组中间层交互；区别于轻量 token pyramid |
| RetinaNet | `retinanet_resnet50_fpn` | v1 anchor/focal/L1 路线 |
| SSD | `ssd300_vgg16` | 固定 RGB 300×300、8732 个 default boxes |
| Fast R-CNN | `fast_rcnn_resnet50_fpn`、`fast_rcnn_vgg16` | 外部候选框，无 RPN；分别使用 ROIAlign、经典 ROIPool |
| 语义分割 | `fcn_resnet50`、`deeplabv3_resnet50` | `semantic_segment`；OS8 骨干、主/辅助 logits、ignore-aware CE |
| 深度 | `depth_resnet50_fpn` | `depth`；数值场和有效性，不以 RGB 热图代替深度 |
| 自监督预训练 | `simmim_swin_tiny`、`simmim_vit_tiny`、`simmim_vit_base` | `pretrain_simmim`；真实 mask token、像素重建与 masked L1 |
| RT-DETR v1 | `rtdetr_resnet18_vd`、`rtdetr_resnet50_vd` | vd 骨干、HybridEncoder、300 queries、6 层 decoder、DN/aux；至少 256×256 |
| YOLO-World | `yoloworld_v8s_text` | 原生文本条件 detect profile |
| YOLOE | `yoloe_v8s_text`、`yoloe_v8s_visual`、`yoloe_v8s_prompt_free` | detect；各名称追加 `_segment` 选择实例分割 |
| ReID | `reid_resnet50` | `reid`；CE/triplet、P×K 采样、跨 camera mAP/rank1、归一化检索输出 |

`catalog` 是实际已安装组合的完整入口；不接受任意骨干/neck/head 的笛卡尔积。VGG16 的 4096 维全连接层单模型参数超过 500 MiB，应预留相应内存。RT-DETR 的资源上限还约束输入 token 数与目标数量。

## 配方和统一数据

[扩展配方目录](../../../recipes/vision/extensions/)提供 schema2 起点，数据仍只进入 `VisionSample`、`ImageDataset` 和 `ImageBatch`。旧 schema1 及原五任务枚举保留；新增字段不能降级到 schema1。`template` 可生成指定模型与任务的起点。

```powershell
liuding-forge-vision template --model fcos_resnet50_fpn --classes 3 --output train.json
liuding-forge-vision check --recipe train.json
liuding-forge-vision inspect --recipe train.json --decode --output audit.json
liuding-forge-vision train --recipe train.json --output runs/extension-1
```

schema2 的 `extension.training_manifest`、`validation_manifest` 指向 `vision.dataset/v1` JSON，路径相对 manifest 解析。顶层至少包含 `image_root`、有序 `categories` 和 `images`；每个样本提供 `image`、`width`、`height`。目标文件必须留在 image_root 内。图片内容、标注和目标文件摘要绑定运行身份；训练/验证路径、内容和序列检查复用统一工作流。

| 任务数据 | 样本或顶层字段 | 语义 |
| --- | --- | --- |
| 多标签 | 样本 `labels` | 与类别数一致，`-1` 表示未知，`0/1` 表示负/正；训练损失还接受已知概率软标签，AP/F1 评估要求二值真值 |
| 语义 | 样本 `semantic_mask`；顶层 `semantic_ignore`、`semantic_encoding` | PNG class_index、palette_index 或 rgb_mapping；RGB 提供 `semantic_color_map: [{color:[r,g,b],id:n}]`；可用 `semantic_id_map: [[source,train],...]` 显式转训练 ID，未列出的源 ID 转 ignore |
| 深度 | 样本 `depth`；顶层 `depth_representation`、`depth_unit_scale` | safetensors：F32 `depth [1,H,W]` 和 BOOL `valid [1,H,W]`；当前 recipe 对应 `depth_meters` 或 `relative_inverse_depth`，两种指标口径严格区分 |
| 外部候选框 | 样本 `proposals: [[x1,y1,x2,y2],...]` | 顶层必须提供 `proposal_generator`、`proposal_inventory`；坐标来自当前输入图，训练采样不将验证 GT 当作候选框源 |
| 文本提示 | 顶层 `label_space_id`、`text_encoder_id`、`prompt_embeddings`、`prompt_dimension:512` | 完整有序词表、非零有限嵌入；或使用 literal `text_prompts` 和配方中的本地文本资产 |
| 视觉提示 | 样本 `visual_prompts: [[class,x1,y1,x2,y2],...]` | 与图像共同变换；每个类别都需要非空提示区域 |
| ReID/序列 | 样本 `identity`、`camera_id`、`sequence_id`、`frame_id` | identity 是本 split 的类别索引，类别对象的 `source_id` 是全局身份；训练/验证身份库存必须分离，检索验证需要跨 camera 正样本 |

实例分割在 `boxes` 的相同行序补充 `polygons: [object][ring][xy]` 或 `rles: [object][counts]`；RLE 采用 COCO 列主序，各行总数等于 width×height。

可选 `camera_intrinsics [3,3]`、对象对齐的 `track_ids` 和时间戳也保留在 typed payload 中。深度、候选框、提示、内参和 track IDs 随支持的几何变换同步；当前这些补充载荷拒绝透视、多源混合和未定义的投影组合。语义标签采用离散插值，深度采用 nearest 并独立传播 valid mask。manifest、张量、目标文件总量与中间画布均有显式预算。

## 训练策略

可用 `calibrate --recipe multilabel.json --weights student.safetensors --sha256 HASH --output thresholds.json` 在显式验证 split 生成阈值与验证身份，再将输出 `extension_patch` 写入配方的 `extension`；阈值随推理及 bundle 配方保存。多标签默认 BCE，可设置 `extension.asymmetric_multilabel=true` 使用 ASL。未知类别不产生监督梯度。验证报告 mAP、micro/macro F1；阈值校准 API 要求显式验证集身份，训练集不能自动充当校准集。

分类蒸馏在配方中指定 `teacher_architecture`、`teacher_weights` 和 `teacher_sha256`。教师冻结、eval、无梯度，与学生共享同一增强视图；单标签采用温度缩放 KL，多标签采用 Bernoulli KL。特征蒸馏首批限定 ResNet 全局特征，学生的显式 projection 在优化器和 EMA 之前注册。教师状态参加完整检查点，导出只保留学生推理状态。

ViT/Swin 可显式设置 `activation_recomputation`。原生 autograd 包装保留 RNG/autocast，支持输入本身无需梯度的参数训练；拒绝有 running stats 的训练 BN/IN 与状态副作用，不支持高阶梯度。它是激活重计算，与训练文件检查点不同。

SimMIM 的 `model_patch_size` 对应 Swin 的 4 或 ViT 的 16，`mask_patch_size` 默认 32，`mask_ratio` 默认 0.6；mask 身份绑定 epoch、sample、occurrence 和 view。`extract-encoder` 输出单独的编码器初始化制品，完整续训仍使用原训练检查点；下游 head 替换和 key mapping 必须显式配置。

## 提示资产和导出

原生 CLIP 文本入口使用本地 plain UTF-8 merge 表与 native safetensors 权重，配方分别提供 `tokenizer_merges/tokenizer_sha256`、`text_encoder_weights/text_encoder_sha256`。Tokenizer profile 为 `clip.utf8-lower-bpe/v1`，使用 Unicode lowercase 和 byte BPE，要求有效且已规范化的 UTF-8，不包含 ftfy 文本修复。这里的原生模型、tokenizer 和资产身份不授予官方 checkpoint 数值资格。

```powershell
liuding-forge-vision prepare-prompts --recipe text.json --prompts literal-prompts.json --output context.json
liuding-forge-vision predict --recipe text.json --weights student.safetensors --sha256 HASH --source image.jpg --sample-data context.json --output predictions
liuding-forge-vision export --recipe text.json --weights student.safetensors --sha256 HASH --output bundle
liuding-forge-vision bundle-predict --bundle bundle --source image.jpg --sample-data context.json --output result.json
```

`literal-prompts.json` 是按类别顺序排列的字符串数组。`vision.input_context/v1` 还可携带外部 proposals 或视觉提示，坐标指向原图。编码后的 context 可重复使用；摘要绑定词表顺序、原文、编码器/词表版本与归一化规则。

扩展导出使用 `vision.onnx_bundle/v3`：固定 batch/H/W、float32/int32/int64 I/O、标准 ONNX 算子与逐输入签名。Fast R-CNN 增加固定行数 `[R,5]` proposals 输入，必须匹配导出时的 `--proposals × batch`；RT-DETR 只交付最后层 logits/归一化 boxes；SimMIM 增加 int64 patch_mask；文本和视觉模式分别增加 embeddings 或 visual_masks；prompt-free 只输入图像。ReID 交付 embedding，语义/深度交付数值场。旧 v1/v2 bundle reader 保留。

导出器遇到未支持的算子会报错，只有实际通过导出及 ORT 差分的具体 profile 才能作为部署依据；完整组合矩阵没有因开发预览而自动获得资格。YOLOE 各模式使用独立原生身份，未宣称与任意官方权重可互换。

## 序列跟踪

ByteTrack 复用有界 Hungarian、XYAH Kalman、高低分二次关联与生命周期；BoT-SORT 原生模式增加 XYWH、ReID 和 affine GMC。当前 GMC 使用有界 patch matching/稳健拟合，与官方光流实现不同。视频解码由调用方负责，训练套件接收帧样本或图片序列。

```powershell
liuding-forge-vision track --recipe detector.json --weights detector.safetensors --sha256 HASH --sequence frames.json --tracker byte_track --output tracks
liuding-forge-vision track-eval --ground-truth gt.txt --predictions tracks/tracks.txt --output metrics.json
```

`frames.json` 包含 `schema:"vision.sequence/v1"`、`sequence_id` 和 `frames:[{image,frame_id}]`，图片路径相对该文件，frame_id 严格递增。BoT-SORT 使用 `--tracker bot_sort --reid-recipe reid.json`，该配方须指定已验证摘要的 ReID 权重。工作流保留 detector 的低分框供二次关联，ReID 在原图框内裁剪后使用分类预处理。

公开 `Tracker` 支持 step/reset/snapshot/restore，失败的 step 和 restore 不提交半更新状态。快照包含 Kalman、外观、ID 分配与生命周期；跨序列先 reset。MOTA/IDF1/HOTA 接收已清除 ignored/crowd/distractor 的单序列标注，MOT CSV 只负责数值 I/O；完整官方评测协议仍需调用方匹配数据集过滤政策。
