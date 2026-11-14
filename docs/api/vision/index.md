# 视觉模型训练套件

视觉套件在同一个 C++ SDK 下提供模型选择、数据导入、训练与续训、验证、预测、参数搜索、权重转换及 ONNX 交付。产品尚未发布，源码版本为 `0.3.0`；架构实现、权重可用性、任务精度、恢复一致性与设备资格分别记录，`catalog` 中的资格字段不能用编译通过替代。仓库不附带预训练权重。

## 模型和组件

| 家族 | 已接入的模型选择 | 任务 |
| --- | --- | --- |
| 原始 YOLOv5 | n/s/m/l/x，P5/P6 | detect |
| Ultralytics YOLOv8、YOLO11、YOLO12、YOLO26 | n/s/m/l/x，参考实际存在的 P2/P6/Ghost 变体 | detect、classify、segment、pose、obb；以 catalog 列出的组合为准 |
| Faster R-CNN | ResNet50-FPN、RPN、ROIAlign | detect |
| 既有骨干 | 复用 models 注册表 | classify、classify_multilabel |
| 扩展视觉模型 | FCOS/BiFPN/ViT-Adapter、RetinaNet、SSD、Fast R-CNN、FCN/DeepLabv3、RT-DETR、YOLO-World/YOLOE、SimMIM、深度、ReID | 见[扩展任务](extensions.md) |

模型选择以 `catalog` 返回的 architecture/task/variant 为准。Fast R-CNN、开放词汇、跟踪与新增训练策略的具体原生 profile 见[扩展任务](extensions.md)；任意骨干/Neck/Head 自由组合不在支持范围。已有骨干各自的输入尺寸、预训练与资格约束继续适用，注册表可查询不代表任意尺寸都能训练。

代码按 `vision/models`、`vision/ops`、`vision/tasks/data`、`vision/tasks/training`、`vision/tasks/workflow` 集中组织。五个默认关闭的组件分别导出 `jyppx::liuding::vision_ops`、`models_yolo`、`models_rcnn`、`models_vision`、`vision_tasks`；RCNN 还依赖既有 `models`。统一工具使用独立开关 `JYPPX_LIUDING_BUILD_VISION_CLI`，不需要开启其他全部 CLI。

在已有依赖的兼容构建配置中启用所需组件：

```powershell
cmake -S . -B out/build/vision -DJYPPX_LIUDING_BUILD_DATA_IMAGE=ON -DJYPPX_LIUDING_BUILD_VISION_OPS=ON -DJYPPX_LIUDING_BUILD_MODELS_YOLO=ON -DJYPPX_LIUDING_BUILD_VISION_TASKS=ON -DJYPPX_LIUDING_BUILD_VISION_CLI=ON -DFETCHCONTENT_FULLY_DISCONNECTED=ON
cmake --build out/build/vision --config Release --target jyppx_vision_cli --parallel 2
```

扩展包增加 `JYPPX_LIUDING_BUILD_MODELS_VISION=ON` 和既有 `models`，其配方、数据及提示输入见[扩展任务](extensions.md)。RCNN 增加 `JYPPX_LIUDING_BUILD_MODELS=ON` 和 `JYPPX_LIUDING_BUILD_MODELS_RCNN=ON`。导出开启 `JYPPX_LIUDING_BUILD_ONNX_EXPORT`，离线推理开启 `JYPPX_LIUDING_BUILD_ONNX_RUNTIME`；按[部署文档](../onnx.md)准备对应依赖。图像叠绘需要 `JYPPX_LIUDING_BUILD_VISION_RENDER`。TensorBoard 需要 `JYPPX_LIUDING_BUILD_TELEMETRY_NATIVE` 及其明确选择的 nanopb 源码，并在配方设置 `tensorboard: true`。这些组件均不在配置时下载依赖。

## 配方与统一入口

[套件配方目录](../../../recipes/vision/suite/)包含各家族、五种任务和调参的起点。它们是参数模板，需修改类别数、数据路径、设备和预算，不是已取得 COCO/ImageNet 精度的训练方案。旧的 [YOLOv8 检测示例](../../../examples/vision/README.md)使用单独的示例格式，继续用于小数据演示；原任务使用 `schema_version: 1`，新增任务使用 schema2；[扩展配方](../../../recipes/vision/extensions/)集中存放新增模型起点。

```powershell
liuding-forge-vision catalog --output catalog.json
liuding-forge-vision template --model yolo26n --task detect --classes 2 --output train.json
liuding-forge-vision check --recipe train.json --output resolved.json
liuding-forge-vision inspect --recipe train.json --decode --output data.json
liuding-forge-vision preview --recipe train.json --samples 8 --output preview
liuding-forge-vision train --recipe train.json --output runs/train-1
liuding-forge-vision resume --recipe runs/train-1/recipe.json --checkpoint runs/train-1/last.checkpoint --output runs/train-2
```

安装后的模板位于 `share/jyppx/recipes/vision/`。CMake 包版本、`version()`、构建记录和打包版本使用项目当前配置的 `0.3.0`；该配置值不表示已经发布。

配方路径相对配方所在目录解析；未知字段、错误任务组合和模型步长不匹配会报错。输出目录必须新建或为空，续训使用新的目录保存本次日志。`check` 检查配方和已安装模型选择，不读取全部数据或执行模型；`inspect`、`preview` 和 `preflight` 分别执行对应检查。`preflight` 没有任何成功候选时保留报告并返回退出码 2。

公开入口为 `<jyppx/liuding/vision/tasks/workflow/suite.hpp>` 的 `load_vision_data()`、`train_vision()`、`evaluate_vision()`、`predict_vision()`。`TaskBinding` 保留同一个模型状态，step 只执行任务 forward/loss，backward、优化器、AMP、调度和检查点由既有 Trainer 负责。

## 数据与预处理

底层始终使用现有 `AnnotationIndex`、`VisionAnnotationRecord`、`VisionSample`、`ImageDataset` 和 `ImageBatch`。外部数据格式只在导入边界区分，训练算法不直接读取 COCO/VOC/YOLO 文件。

| 外部布局 | 配方 dataset | 任务约定 |
| --- | --- | --- |
| COCO | JSON、image_root | 框、polygon/RLE 实例、COCO 关键点；OBB 可使用单个四点 polygon 或 oriented_bbox 八数扩展 |
| VOC | XML 目录、image_root、完整 categories | 检测；默认一基闭区间转连续边界 |
| YOLO | dataset YAML、train/validation/test split | 对应任务的 detect/segment/pose/obb TXT；姿态 shape 与 flip_idx 来自 YAML |
| LabelMe | JSON 目录、image_root、完整 categories | rectangle 检测、polygon 实例、四角 OBB；未提供 point 分组到 pose 的自动推断 |
| ImageFolder | classification_root、classification_validation_root | 文件夹类别的单标签分类 |

YOLO split 支持图片目录或 UTF-8 图片列表；最后一个 `images` 路径分量映射到 `labels`。空 TXT 为真负样本，缺文件默认报错，`allow_missing_labels` 显式允许其作为负样本。VOC/LabelMe 使用冻结类别表，各 split 按类别名称及源 ID 核对并统一顺序。转换输出复用同一份对象：`convert-data --recipe train.json --output annotations.json` 生成 COCO JSON，不复制图片；OBB 的 oriented_bbox 是本套件扩展。

像素输入为 CPU uint8 RGB，训练准备输出 float32 RGB NCHW、先映射到 `[0,1]` 再按配方 mean/std 标准化。确定性 LetterBox、resize/crop 和逆变换使用共享几何类型。预测输出以完成方向处理的原始显示图为坐标系。

`exif_policy` 默认 `reject`，遇到非标准 JPEG 方向要求明确选择。`encoded` 同步旋转编码坐标的图像和标注；`already_oriented` 只转图像并核对显示尺寸；`ignore` 保持编码方向。YOLO 归一化标签应选择 encoded，already_oriented 需用有明确尺寸的 COCO 等格式。默认不推断标签所属坐标系，不进行 ICC 色彩管理。

`verify_image_contents` 默认 true：启动时将图片实际内容加入训练/验证身份，并拒绝跨 split 的相同内容。恢复还要求文件在整个运行期间保持不变。关闭该选项时只绑定标注/路径等元数据，不能据此保证图片变化后仍有相同训练语义。

## 增强与加载

检测与相容任务提供 HSV、灰度、Gaussian/Median Blur、CLAHE、翻转、缩放/平移、旋转/剪切/透视、Mosaic、MixUp、CutMix 和实例 CopyPaste。几何同时更新框、polygon、RLE、关键点和 OBB，并保留来源身份与目标过滤。pose/OBB 拒绝未实现的 CutMix/CopyPaste 组合，OBB 拒绝会改变矩形语义的 shear/perspective；关键点翻转必须提供对称排列。

分类使用 RandomResizedCrop、ColorJitter、固定操作集合的 RandAugment、RandomErasing，以及带软标签的 MixUp/CutMix；验证使用确定性 resize/center crop。未实现 AutoAugment/AugMix。操作、插值和量化采用本套件版本化语义，尚未宣称与 PIL/OpenCV/Albumentations 逐像素等价。

随机抽样绑定 dataset、sample、seed、epoch、occurrence 和 view，不依赖物理 worker。`augmentation.close_epochs` 在末期关闭多源增强。rect 分桶采用稳定长宽比排序和批块 shuffle，拒绝与多源增强混用；有界 multiscale 只用于训练，按步长对齐。更换 batch/分组或配方会改变恢复身份。

`preprocessing` 可固定 LetterBox 的放大、居中、填充值与照片插值（默认 bilinear），以及 box/region 过滤、crowd/difficult 和目标容量政策；标注 mask 保持其离散插值合同。`augmentation` 可固定 Gaussian/median kernel、sigma、CopyPaste 遮挡上限及 projective 目标过滤，解析后的具体值进入配方身份。

`cache` 保存有界的确定性解码像素，RAM 返回独立 Tensor，磁盘条目验证内容摘要并原子替换；随机增强在缓存后执行。`workers`、`prefetch_batches`、`pin_memory` 复用已有数据管线。缓存/预取 Tensor 字节限额不等于操作系统 RSS/显存硬配额。关闭渲染时可通过 `preview --no-annotations` 保存像素与标签 JSON。

## 训练、迁移和恢复

优化器支持 SGD、AdamW 和原生 MuSGD；`auto` 在训练前解析为 RCNN 的 SGD、YOLO26 的 MuSGD、其余的 AdamW，并保存解析后配方。参数按 decay/no-decay/bias 分组，可按名冻结模块及 BN/buffer，或配置差分学习率。冻结和权重初始化先于优化器与 EMA 绑定。

支持梯度累积、裁剪、成功更新驱动的预热/余弦调度、显式 nominal/effective batch 的 decay 缩放、EMA、best/last/周期保存和早停。固定 `class_weights` 仅用于训练，验证损失清除类权重；数据统计只应来自训练集。调度器和 EMA 不随 AMP 跳步推进。

`deterministic` 默认开启，在训练作用域要求确定性算法并关闭 cuDNN benchmark，结束后恢复调用方设置；缺少确定性实现的操作会明确失败，可显式关闭以运行此类模型。同一软硬件环境下的精确 CUDA 恢复还需设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。`initial_loss_scale` 控制 CUDA FP16 的初始动态损失缩放，默认 65536；较短训练可显式降低它，避免初始化阶段多次溢出而尚未完成有效更新。两项设置都进入配方与恢复身份。

`full_validation_metrics` 默认开启，验证损失之外再完整执行任务评估。自动选优使用分类 top1、检测/多标签 AP、语义 mIoU、深度 RMSE、ReID rank1 或重建 loss；关闭完整指标时使用验证损失。各任务分别使用框 AP、实例 mask AP、OKS AP 和精确四边形 AP，报告逐类与 PR 数据。诊断混淆矩阵另用明确的 box IoU/置信度口径，不将其当作 mask/pose/OBB AP。

检测及几何任务同时报告 AP50，可通过 `validation.ap50` 选优或调参；评估报告保存 score/candidate/max_det、空类排除和密集目标容量诊断。指标的逐图逐类容量取配方的 `evaluation.max_detections`。姿态的 `keypoint_sigmas` 同时进入训练损失和 OKS，17 点人体模板显式采用 COCO sigmas；留空采用均匀 `1/K`，自定义骨架应提供自己的数值。分类 top1/top5 使用全部类别，不受展示过滤影响。

`last.checkpoint` 包含模型、优化器、scheduler、RNG、数据游标、回调、GradScaler 和 EMA 等恢复状态；safetensors 只用于初始化/推理。每个输出段写 recipe/identity/initialization、metrics JSON/CSV、HTML 和训练结果，启用时写独立 TensorBoard 目录。完整设备、跨进程恢复和精度矩阵仍需分别验收。

`diagnostics` 默认关闭。设置 `enabled=true` 可复用已有 Trainer 诊断；`measure_data_wait` 记录成功取批的主机耗时，`gradient_norm_every` 按更新尝试采样裁剪前梯度 L2，并支持 max_tensors/max_elements/tile_elements/max_scratch_bytes 的扫描边界。诊断计数、抽样位置和已捕获耗时进入检查点，diagnostics.json 保存带输入/构建身份的结构化结果；performance.json 另外记录本次输出段的实际训练样本数、最大逻辑 batch Tensor 字节与 fit 吞吐。fit 墙钟包含验证和检查点操作，取批时间包含消费者变换/传输，二者都不是纯磁盘 I/O、进程 RSS 或 GPU 峰值测量。

## 权重导入与调参

```powershell
liuding-forge-vision convert-weights --source upstream.pt --sha256 INPUT_SHA256 --root model --output upstream.safetensors
liuding-forge-vision map-weights --recipe train.json --source upstream.safetensors --sha256 SAFE_SHA256 --output mapping.json
liuding-forge-vision preflight --recipe train.json --batches 2,4,8 --output preflight.json
liuding-forge-vision tune --recipe train.json --search recipes/vision/suite/search-detect.json --output runs/search-1
```

`.pt` 转换用 C++ 提取现代 PyTorch ZIP 中的 Tensor、Parameter、OrderedDict 和 inert Module 状态，不执行 pickle 中的 Python 类/函数；只支持明确列出的协议和重建形式，拒绝任意 reduction、量化/稀疏及旧式非 ZIP。root 显式选择 model/ema/state_dict/root，不自动回退。转换保持源键名，另输出摘要和来源元数据；转换成功不代表模型兼容。

`map-weights` 只按确切名称、已声明的别名及 shape/dtype 生成报告，不按相同 shape 猜层。审核后把报告的 `key_mapping` 放入配方 `initialization`，同时指定 weights 和 sha256。默认严格加载；显式 replace_head_prefixes 允许保留新 head，ignored_source_keys 允许确认的额外源键，retain_buffer_keys 只允许保留确切的派生 buffer。报告列出加载、忽略和保留项目。官方预训练制品的来源、映射与质量尚未全部核验，catalog 保持 pretrained_weights=false。

预检对实际样本和一次性模型执行 loss/backward/update，比较 AMP 与 FP32 输出，恢复全局 RNG；不会修改正式训练对象或游标。AutoBatch 是受候选数、Tensor 字节和 CUDA 空闲显存比例约束的建议，报告当前活跃分配差值，不承诺未来训练峰值必定适配；正式 batch 由用户在配方固定。

训练目录内的 `experiment/` 复用已有 SQLite run store 和 CAS，登记配方、结果、模型和 best/last 检查点；搜索目录的同名存储按 trial 登记评分检查点及观测。CAS 会增加磁盘使用，摘要相同的内容在各自存储内去重；中断时已保存的本地检查点仍可恢复。

搜索支持 uniform/log_uniform/integer/discrete、条件参数、固定 seed 的 random/evolution、trial 预算与训练早停。默认串行，CLI 通过已有 LocalProcessRunner 为每个 trial 启动本机原生子进程，保存有界 stdout/stderr 和退出/超时收据；`--trial-timeout` 设置每次 attempt 的秒数。`--execution in-process` 显式选择进程内执行。C++ 默认执行器在进程内运行，也可通过 `execute_vision_trial` 注入相同的原生 launcher。保留待完成候选、每个 trial 的配方/attempt/checkpoint/观测和 best.recipe.json；再次使用同一搜索目录恢复。启用原本关闭的增强需 allow_enable。自定义 TrialExecutor 需要自行实现其中断恢复。TPE/ASHA/PBT 与并发 trial 尚未接入。

## 预测和 ONNX

```powershell
liuding-forge-vision val --recipe train.json --weights runs/train-1/best.safetensors --sha256 WEIGHTS_SHA256 --output validation.json
liuding-forge-vision predict --recipe train.json --weights runs/train-1/best.safetensors --sha256 WEIGHTS_SHA256 --source image.jpg --output prediction --text --render --crops --warmup 2
liuding-forge-vision export --recipe train.json --weights runs/train-1/best.safetensors --sha256 WEIGHTS_SHA256 --role ema --fuse --output bundle
liuding-forge-vision bundle-predict --bundle bundle --source image.jpg --output prediction.json
```

val/predict/export 需要明确权重与摘要；开发时 `--random-init` 显式选择未训练模型。标准结果包含 boxes、scores/classes，以及任务对应的 mask、keypoints 或 oriented_corners，支持 JSON/TXT、叠绘和裁剪。阶段计时区分预处理、模型执行、后处理与预热，CUDA 使用同步后的墙钟时间；不以单张耗时宣称吞吐资格。

ONNX 导出用独立 CPU eval 副本、固定 batch/H/W、FP32、opset 17。YOLO 输出原始任务 Tensor，C++ 复用对应解码器；YOLO26 保留其无 NMS 语义。RCNN 在图内执行 RPN 选择/NMS/ROIAlign，以固定上限 proposal 和 int64 有效行标记交付，再在图外完成检测解码。动态 H/W 和任意 opset 尚未接入。

`export --batch N` 固定导出的批大小，默认 1；C++ bundle API 需要提供相同数量的图片，单图 CLI 适用于 batch=1。RCNN 的 `--proposals N` 固定每图保留的 RPN proposal 数，默认 300，范围 1..2048；有效标记区分真实行与补齐行。该上限会截断候选，应随模型和数据单独验证其对最终检测的影响。

`--fuse` 在支持的 YOLO 副本中折叠 Conv/BN，记录融合块数与输入/输出权重角色；RCNN/既有骨干没有通用融合承诺。fused safetensors 供推理定义使用，不能直接装入未融合训练模型或当作 exact resume。bundle v2 记录模型摘要、完整确定性前后处理配方和类型配置；写齐后才发布目录。

离线 bundle 使用原生 TypedSession CPU 配置档执行。独立 `<jyppx/liuding/deployment/onnx_runtime/typed_session.hpp>` 提供 FP32/int32/int64、命名输入、固定维及有明确上限的前导动态输出；旧的 FP32 Session 维持原合同。动态结果由 ORT 内部分配后校验实际尺寸，逻辑 I/O 上限不等于 ORT 中间分配硬上限。各模型仍需独立 checker、raw 数值与完整任务结果验收，当前导出实现不代表全矩阵已取得部署资格。

模型架构配置和来源条款见[模型来源与许可](../../../third_party/licenses/MODEL_PROVENANCE.md)。
