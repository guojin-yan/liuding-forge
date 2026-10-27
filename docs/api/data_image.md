# 图像数据、离散标签与联合变换

图像组件提供本地图片解码、数据集索引和标注读取。构建时开启 `JYPPX_LIUDING_BUILD_DATA_IMAGE`，安装后使用 `find_package(jyppx CONFIG REQUIRED COMPONENTS data_image)`，并链接 `jyppx::liuding::data_image`。

## 像素语义

`read_image_file` 和 `decode_image` 返回拥有像素存储的 `DecodedImage`，其中 `pixels` 明确记录 dtype、CHW/HWC 布局、灰度/RGB/带 alpha 的通道含义以及数值编码范围。16 位 PNG 保留 uint16；选择 `float32_normalized` 时，输出 float32，范围语义为 `[0,1]`。解码器不执行 EXIF 定向或 ICC 色彩管理，RGB 只表示通道顺序。Apple CgBI PNG 尚未支持，会明确报错。

```cpp
#include <jyppx/liuding/data_image/image_decoder.hpp>

namespace image = jyppx::liuding::data::image;

image::ImageDecodeOptions options;
options.channels = image::ImageChannels::rgb;
options.dtype = image::ImageDtype::float32_normalized;
auto decoded = image::read_image_file("sample.png", options);
image::validate_pixels(decoded.tensor, decoded.pixels);
```

`validate_pixels` 检查形状、dtype、通道和 alpha/range 契约，支持有步幅的 CPU/CUDA 单图 Tensor。该函数不扫描像素数值、不传输设备数据、不执行颜色转换；范围表示编码语义，并非观测到的最小值和最大值。标准化后的浮点图像应显式改用 `PixelValueRange::unbounded`。单图 API 不接受 NCHW batch。

`ImageDataset::get` 返回的 `VisionSample` 携带 `pixels` 和初始恒等 `geometry`；配对图像另有 `paired_target_pixels`，不会继承输入图像的颜色或归一化策略。语义、实例或全景标签应选择下面的离散 mask 入口。

## 离散 mask

`decode_mask` 和 `read_mask_file` 专门读取 PNG 标签，返回拥有存储的 `DecodedMask`。`ids` 始终是 CPU 上连续的二维 int64 张量 `[H,W]`，不会做灰度强度缩放、颜色转换或浮点归一化。支持普通 PNG 和 Adam7 交错 PNG；源约定必须显式选择。

| `MaskEncoding` | 接受的 PNG | 输出 ID |
| --- | --- | --- |
| `class_index` | 1/2/4/8/16 位灰度 | 原始样本值，例如 2 位标签保留 0–3，16 位标签保留 0–65535 |
| `palette_index` | 1/2/4/8 位调色板 | 原始调色板索引；重复显示颜色不会合并类别 |
| `rgb_mapping` | 8 位 RGB | 按 `colors` 显式映射到 int64；默认拒绝未知颜色 |
| `panoptic_rgb` | 8 位 RGB | 按 COCO 约定计算 `R + 256*G + 65536*B`，保留完整 24 位 segment ID |

```cpp
#include <jyppx/liuding/data_image/mask_decoder.hpp>

image::MaskDecodeOptions mask_options;
mask_options.encoding = image::MaskEncoding::palette_index;
mask_options.ignore_id = 255;
auto mask = image::read_mask_file("labels.png", mask_options);
// mask.ids 是 [H,W] int64；mask.palette 保留 PLTE 颜色和 tRNS alpha。
```

`ignore_id` 和 `void_id` 是调用者声明的标签语义，不会自动改写像素。调色板的透明项保留原始索引，透明度仅存入 `palette` 元数据；不会自动变成背景或 ignore。RGB 映射允许多个颜色对应同一个 ID，拒绝重复颜色键；负 ID 必须声明为 ignore 或 void。需要把未知颜色设为 ignore 时，必须同时设置 `unknown_color_is_ignore=true` 和 `ignore_id`。

JPEG 等有损格式、带独立 alpha 通道的 PNG、灰度/RGB 的 tRNS、APNG 和 Apple CgBI PNG 当前会明确拒绝。读入全景 ID 图后，可使用下面的 `bind_panoptic_mask` 校验 segment 表与类别的关联。该入口检查所有 PNG chunk 的 CRC、zlib Adler-32、完整输出尺寸、调色板索引范围及文件结构。

`limits` 分别限制编码字节、尺寸、像素数和 int64 输出字节；`max_working_bytes` 限制压缩副本与滤波光栅的合计字节，不包含调用者输入、输出张量和映射表。chunk 数、RGB 映射条目数另有独立上限。取消在文件读取、chunk 校验和行处理边界检查，底层一次 DEFLATE 调用不支持中途取消。

### 从配对清单读取 mask

```cpp
#include <jyppx/liuding/data_image/image_dataset.hpp>

auto index = image::parse_manifest(
    "input,target\nimages/a.png,masks/a.png\n",
    image::ManifestFormat::csv, image::ManifestTask::paired);
image::ImageDatasetOptions dataset_options;
dataset_options.image.channels = image::ImageChannels::rgb;
dataset_options.image.dtype = image::ImageDtype::float32_normalized;
dataset_options.mask_target = mask_options;
image::ImageDataset dataset("dataset", std::move(index), dataset_options);
auto sample = dataset.get(0);
// sample.image 为归一化 RGB；sample.mask_target->ids 保留整数类别 ID。
```

配置 `mask_target` 后，配对清单中的 target 走标签 reader，结果存入 `VisionSample::mask_target`；`paired_target` 和 `paired_target_pixels` 为空。mask 必须与输入图像尺寸完全相同，`require_matching_target_size=false` 仅适用于普通配对图像。未配置 `mask_target` 时继续读取普通配对图像。数据集读取不自动增强；需要联合变换时显式调用下面的 `transform_sample`。

### 全景 ID 图与 segment 表绑定

`bind_panoptic_mask(record, mask, category_count)` 校验 COCO 全景 segment 表和 CPU int64 ID 图。必须声明 `panoptic_rgb`、`void_id` 和冻结类别数量；segment ID 必须是互不重复的正 24 位整数，不得与 void/ignore 冲突。每个 segment 都必须有像素，每个非保留像素都必须出现在表中。默认还校验源声明的 `area` 与 tight bbox；可通过 `PanopticBindOptions` 显式关闭原始元数据校验，输出仍会计算准确的当前统计。

绑定返回新的记录，设置 `mask_pixel_area`、当前 tight `box` 和 `panoptic_mask_materialized=true`，不修改源记录与 mask。源 `area`、ID、类别和标志保留，`panoptic_mask_path` 始终指向原始文件，不代表增强后的像素。对已经 materialized 的记录重新绑定时，当前像素计数和 bbox 必须匹配，不受关闭源元数据校验的选项影响。空 segment 表只允许包含 void/ignore 的图像。

`ImageDatasetOptions::panoptic_target` 可显式启用全景引用解码，要求分类清单、`coco_panoptic` 标注索引、每个样本都有标注引用，以及 `panoptic_rgb` 和显式 `void_id`。引用按数据集根目录的安全路径规则读取，再使用 `panoptic_validation` 绑定表。结果放入 `sample.mask_target`，`target_path` 保存来源路径；该选项不能与配对清单的 `mask_target` 同时使用。未启用时只保留标注引用，不主动打开 ID 图。

```cpp
image::ImageDatasetOptions panoptic_options;
panoptic_options.panoptic_target.emplace();
panoptic_options.panoptic_target->encoding = image::MaskEncoding::panoptic_rgb;
panoptic_options.panoptic_target->void_id = 0;
image::ImageDataset dataset("dataset", manifest, panoptic_annotations, panoptic_options);
auto panoptic_sample = dataset.get(0);
```

## 几何坐标追踪

`GeometryTrace` 记录连续像素边界坐标，原图范围为 `[0,W] × [0,H]`。resize、crop、pad 和水平/垂直 flip 返回新 trace，原对象不变。它只记录坐标关系，调用者仍需对像素和标注实际执行对应操作。

```cpp
#include <jyppx/liuding/data_image/geometry_trace.hpp>

auto trace = image::GeometryTrace({640, 480})
                 .resized({320, 240})
                 .padded(0, 40, 0, 40);
auto model_box = trace.forward_box({100, 80, 300, 280});
// model_box 为 [50,80,150,180]。
auto original_box = trace.inverse_box(*model_box);
// original_box 为 [100,80,300,280]。
auto padding_point = trace.inverse_point({10, 20});
// padding_point 为空：该位置属于补边，不对应原图内容。
```

点映射在内容区域外返回空值；框映射先与有效内容取交集，无正面积交集时返回空值。裁剪后 `source_region()` 表示仍保留的原图区域，`retains_full_source()` 表示原图内容是否完整保留，`output_region()` 排除人工补边。裁掉的区域不会因后续 resize 或 pad 而恢复。`matrix()` 提供数学映射矩阵，使用该矩阵时仍应自行遵守有效域；通常直接使用点和框 API 更合适。

trace 最多组合 128 次操作，保存固定大小的映射状态。空尺寸、越界 crop、uint32 补边溢出、非有限坐标或不可表示的映射均明确报错。当前实现仅覆盖上述轴对齐几何；旋转、透视、多图拼接和 EXIF 空间转换尚未提供。

## CPU 联合几何变换

`transform_sample` 将同一串 resize、crop、pad、水平/垂直 flip 同时应用到照片、对齐配对照片或离散 mask，以及水平框、关键点、polygon、OBB、RLE 与全景 segment 表。输入是带 `PixelSpec` 的 CPU CHW/HWC Tensor，允许非连续步幅；返回拥有独立存储、连续且未固定内存的 CPU Tensor，并组合已有 `geometry`。输入 Tensor 和共享标注不被修改。该数据处理接口拒绝带 autograd 的照片。

```cpp
#include <jyppx/liuding/data_image/joint_transform.hpp>

std::array<image::ImageGeometryStep, 3> steps{{
    {.kind = image::ImageGeometryKind::resize, .extent = {320, 240}},
    {.kind = image::ImageGeometryKind::pad, .top = 40, .bottom = 40},
    {.kind = image::ImageGeometryKind::horizontal_flip}}};
image::JointTransformOptions joint_options;
joint_options.interpolation = image::ImageInterpolation::bilinear;
joint_options.mask_padding_id = 255; // 显式选择填充标签，通常使用已声明的 ignore/void ID。
auto augmented = image::transform_sample(sample, steps, joint_options);
```

照片默认使用 `nearest_exact`；可显式选择 LibTorch `bilinear`，其策略为 `align_corners=false, antialias=true`。整数照片在双精度插值后按最近偶数舍入并限制到原位深范围，返回原 dtype；浮点照片保留原 dtype。通道、编码范围和 alpha 关联保持原有声明，插值逐通道执行，不隐式做颜色、线性色彩空间或 alpha 预乘转换。离散 mask 始终使用精确整数寻址：源索引为 `floor((i+0.5)*input_size/output_size)`，ID 不经过 float，包含大于 `2^53` 的 int64 ID 和显式负 ignore ID。补边时 `mask_padding_id` 必须显式给出，调色板和 ignore/void 元数据继续保留。

水平框先裁到仍有效的源区域，再映射到最终输出；无正面积交集的对象删除。`min_box_visibility` 是相对本次调用原始框面积的保留比例，`min_box_area` 是最终裁后输出框面积。关键点的 visibility 保留 0/1/2 语义，原本缺失或被裁掉的点变为 `(0,0,0)`；每个实际 flip 轴都要求显式的逐类别 output-to-input permutation，允许显式恒等排列，且必须为对合排列。`drop_empty_pose` 控制是否删除已无有效关键点的姿态对象。source ID、category 映射、group、crowd/ignore/difficult/truncated 及源声明的 `area` 原样保留；`area` 仍表示来源声明，不会改写成输出框面积。

polygon 使用连续像素边界坐标，每个元素是独立外环，不表示孔洞。未切割外环保留顶点顺序（去除冗余闭合点）；凸环直接裁剪，凹环部分裁剪时通过有界三角剖分保留多个无重叠内部的凸片，设置 `polygon_partitioned=true`，不会把断开的区域连接起来。`min_region_visibility` 按本次输入的每个外环计算保留几何面积比例，`min_region_area` 按该外环全部保留片的输出面积计算；多个外环分别筛选，不能将其重叠面积相加当作 union 面积。输出 bbox 来自保留区域，源 `area` 保持不变。

OBB 表示有序、严格凸的四角形；未裁掉内容时保留角点顺序，支持非等比 resize 与翻转。部分裁剪后默认 `OrientedBoxCrop::to_polygon`，清空四角字段并设置 `oriented_box_to_polygon=true`，保留真实裁后区域；也可显式选 `discard` 删除对象。不会把裁后多边形强制变成水平框或臆造角度约定。每个对象可带一种主要区域表示（polygon、OBB 或 RLE），混合主要区域表示会报错。

COCO RLE 按列主序交替 background/foreground 解码，各步骤使用与离散 mask 相同的 nearest-exact/crop/pad/flip，补边始终为无实例的 0。连续多次 resize 必须逐步执行，不能只对最终组合矩阵取一次最近邻。输出重新编码为 RLE，重算准确的 `mask_pixel_area` 和 tight bbox，空实例删除。全景变换要求绑定 ID 图和 segment 表；补边 ID 必须等于已声明的 void。被裁光或被面积/可见度/框策略删除的 segment 会从表中删除，其剩余像素改为 void，ignore 像素继续保留；输出表可再次绑定验证或继续变换。

对于 RLE 和全景 segment，`min_region_visibility` 是源 foreground 单元与有效源矩形交集的面积比例，允许分数边界；`min_region_area` 是重采样后的实际 foreground 像素数。输出像素可能因缩小而消失，即使连续几何可见度大于零也会删除。源声明的 `area` 与当前统计的 `mask_pixel_area` 是两个独立字段。

配对照片要求与输入等尺寸、同坐标空间，不能据此推导超分辨率或多相机配准。组批与最终 batch pinning 见下文；几何操作当前仍使用上述 CPU 单样本接口，CUDA 联合几何尚未提供。

`JointTransformLimits` 在分配像素前检查全部中间画布的边长、像素数、合计 Tensor 字节和保守工作 Tensor 估算，RLE 另预留单实例光栅工作空间。对象、关键点、输入/产生的 polygon 点与片段、输入/输出 RLE runs、验证/三角剖分候选比较另有独立预算；被筛除的候选几何也消耗已产生的预算。`max_mask_work_pixels` 汇总整个调用的离散光栅访问，包括所有实例的解码、复制、逐步重采样、统计、编码和全景关联扫描，避免仅限制单个实例却让总工作无界。表绑定也有独立 lookup 工作预算。Tensor 工作估算不包含标注容器、调用者输入、分配器开销或 LibTorch 内部 workspace，不能当作操作系统 RSS 上限。取消在步骤、复制行、几何比较和对象边界检查；单次 LibTorch 插值不能在调用内部中断，取消/失败不会发布半成品或修改输入。

## 图像组批与传输

`collate_images` 将非空的 CPU `VisionSample` 序列组成 `ImageBatch`。每个样本必须显式提供稳定 `ImageBatchKey`，包含字符串 ID、重复 occurrence 和 view；它们与照片、类别表、路径、标注和几何追踪保持原顺序。可混用非连续 CHW/HWC 照片，其他 PixelSpec 语义须一致；配对照片使用独立但全批一致的 PixelSpec。全批须统一为带配对照片、带离散 mask 或无配对目标，不能隐式丢弃部分样本的目标。

| 模式 | 照片存储 | 形状规则 |
| --- | --- | --- |
| `stack` | 连续 NCHW `images` | 所有样本同尺寸，否则报错 |
| `pad` | 连续 NCHW `images` | 右侧/底部补到最大尺寸或显式 `padded_extent`，可按 `size_divisor` 向上对齐 |
| `ragged` | CHW `image_list` | 每张照片保留自己的尺寸；配对目标和 valid 同样用列表 |

```cpp
#include <jyppx/liuding/data_image/batch_transform.hpp>

std::array samples{sample_a, sample_b};
std::array keys{image::ImageBatchKey{"a", 0, 0}, image::ImageBatchKey{"b", 0, 0}};
image::ImageCollateOptions collate;
collate.layout = image::ImageBatchLayout::pad;
collate.size_divisor = 32;
collate.mask_padding_id = 255; // 有离散目标且确实补边时必须显式指定。
auto batch = image::collate_images(samples, keys, collate);
// batch.images: [N,C,H,W]；mask 为 int64 [N,H,W]，不经过浮点转换。
```

`valid`/`valid_list` 是 bool，标记组批新加的 padding；`items[i].valid_extent` 保存该样本组批前的尺寸。此前增强已经产生的 padding 仍属于该 valid extent，是否对应原图内容应查询 `items[i].geometry`。组批会组合原 trace；缺失 trace 时以当前样本画布建立 identity。`GeometryTrace::state/from_state` 可完整保存和恢复映射、保留内容域和已消耗的 128 次操作预算，不会恢复裁掉的内容或重置预算。

分类标签打包为 int64 `labels`，由长度 N+1 的主机 `label_offsets` 定位每张图的区间；类别索引必须属于该样本的 `label_categories`。标注缺失与合法空对象列表保持区别，逐样本类别表不被重新映射。组批不调整框、polygon、OBB、关键点的位置，不改变源 ID、源 area、当前 mask 像素数或 flags；padding 会更新标注当前画布，并在 run 空间无损扩展 COCO RLE。全景表在 CPU 与 ID 图严格绑定后保留，新增 padding ID 必须等于该样本的 void，ignore 像素保持原值。

`image_batch_value/from_value` 提供版本 1 的完整结构化 IValue，照片/mask/valid/labels 是 Tensor，路径、标注、类别表、几何、offsets 和身份字段留在主机标量/容器中。uint64 occurrence/view 保留全部 64 位。解码检查结构、尺寸、dtype/device 和主机几何，不扫描设备上的像素；像素内容必须由可信生产者提供。该转换共享 Tensor，不复制其存储，也不序列化运行时 memory-stage 统计。

```cpp
jyppx::liuding::data::BatchMemoryOptions memory;
memory.pin_memory = true;
memory.target_device = jyppx::liuding::Device(torch::kCUDA, 0);
memory.non_blocking = true;
auto prepared = image::prepare_image_batch(batch, memory); // 在最终 batch 上 pin，仍在 CPU。
auto device_batch = image::transfer_image_batch(std::move(prepared), memory);
// device_batch.memory 给出 pinned/transferred Tensor 数量和逻辑字节数。
```

传输复用 Core 两阶段内存接口，在调用方当前 CUDA stream 上执行；没有独立 copy stream、隐式跨 stream event 或传输重叠保证。准备与传输必须使用相同的内存策略。输入 batch 和源样本保持原样；组批输出具有独立照片/mask/valid/label 存储。独立 pin/传输 API 以及原 pipeline memory stage 的限制各自生效。

## CPU/CUDA 批量数值变换

`transform_image_batch` 对 packed NCHW 或 ragged CHW 照片执行浮点类型转换、显式强度缩放、RGB/BGR 交换与逐通道标准化，在原 CPU/CUDA 设备及调用方当前 stream 上运行。packed 路径以整批执行，不逐张读取 GPU 标量。运算使用 float32，float64 输出使用 float64 运算，再转换到指定 float16/bfloat16/float32/float64。mask、valid、labels、标注和几何原样共享，不进行重采样或归一化。

```cpp
image::ImageBatchTransformOptions numeric;
numeric.image.dtype = torch::kFloat32;
numeric.image.scale = image::BatchPixelScale::to_unit;
numeric.image.mean = {0.485, 0.456, 0.406};
numeric.image.std = {0.229, 0.224, 0.225};
auto model_batch = image::transform_image_batch(device_batch, numeric);
```

默认 `to_unit` 将 uint8/uint16 分别除以 255/65535，已经声明为 unit interval 的浮点输入不会再次缩放；unbounded 输入会被拒绝，调用者应显式选择 `preserve`。`preserve` 保持数值，unsigned 编码转浮点后的范围标为 unbounded。mean/std 必须同时提供、长度等于通道数、数值有限且 std 为正，按交换后的通道顺序应用；归一化后范围标为 unbounded。alpha 的关联在缩放和通道交换中保持，带 alpha 的照片当前明确拒绝标准化，不会默默丢弃或反预乘 alpha。该接口信任 PixelSpec，不扫描像素来推断范围。

可通过 `numeric.paired` 显式配置配对照片的独立变换；默认保留其数值和 dtype。返回的变换照片有独立存储，其余 Tensor 和不可变元数据保持共享；由于 dtype/字节数可能变化，返回值清空先前 memory-stage 统计。padding 像素与照片接受相同数值变换。该接口尚未包含批量几何。可通过 map/iterable 的 `consumer_transform` 在 `next()` 提交前执行，分别声明传输后的输入与变换后的输出 schema；失败/取消不会提交 cursor，配置与恢复示例见[提交前批次变换](batch_consumer_transform.md)。在 `next()` 返回后单独调用时，错误不会回滚已经提交的 cursor。

`ImageBatchLimits` 限制单次调用的样本数、总输出画布像素、边长、全部 Tensor 字节、元数据节点/逻辑字节/单字符串、polygon 比较和保守工作 payload。组批在分配照片输出前预检；RLE 新增 runs 有全调用上限。数值变换另为转换/通道交换保守预留完整 compute buffer，输出过大或预算不足时明确报错，没有 CPU fallback。预算排除调用者输入、allocator/容器开销及 LibTorch 内部 workspace，是逻辑限制，不是 RSS 或全局在途内存上限。取消在样本、对象、RLE 列、字段边界和发布前检查，单次 LibTorch/CUDA 操作不能在内部中断。

接入 `DataPipeline` 时，可在 batch factory 返回 `image_batch_value(batch)`，并用 `image_batch_schema(exemplar, consumer_device)` 声明完整的 typed Tensor 与主机结构。schema descriptor 绑定布局、颜色/alpha/范围语义，N/H/W 为动态维度；具体 scalar 判别值和几何仍由 codec 校验。把 `image_batch_identity(collate, numeric)` 与逐样本几何/RNG 配置、身份分配规则共同绑定到 `batch_factory_id`，来源身份绑定到 `dataset_version`。摘要包含全部 collate/numeric 策略、可选字段和资源上限，不自动检查任意 lambda 的行为。CPU 数值变换可在 factory 内显式执行，此时 exemplar 应使用变换后的 PixelSpec；consumer CUDA 变换暂需调用方安排。已验证完整结构、尾 batch、worker 顺序、同拓扑恢复与 factory 失败不推进 cursor，跨 worker 拓扑 exact restore 仍会拒绝。

## 逐样本随机性与恢复绑定

`SampleRandom` 是无可变游标的 SHA-256 counter v1 采样器。键包含 dataset、稳定 sample ID、transform ID、seed、epoch、重复 occurrence 和 view；同一键与 draw 始终返回相同值，不使用物理线程、worker 数、系统时间或 torch 全局 RNG。`bits` 提供 uint64，`unit` 使用高 53 位得到 `[0,1)`，`bounded` 通过有界拒绝采样避免取模偏差。字符串按原始字节编码，不做路径、Unicode 或大小写规范化；内容身份必须由调用方冻结。

```cpp
image::RandomGeometryOptions random_options;
random_options.resize = image::ImageExtent{256, 256};
random_options.crop = image::ImageExtent{224, 224};
random_options.horizontal_flip_probability = 0.5;
const auto transform_id = image::joint_transform_identity(random_options, joint_options);
image::SampleRandom rng({.dataset_id = "dataset-v3",
                         .sample_id = sample.image_path,
                         .transform_id = transform_id,
                         .seed = 42, .epoch = 7, .occurrence = 0, .view = 0});
auto sampled = image::sample_geometry({sample.width, sample.height}, random_options, rng);
auto view = image::transform_sample(sample, sampled, joint_options);
```

`sample_geometry` 当前提供固定 resize、均匀整数原点 crop 和独立的水平/垂直 flip；x/y/horizontal/vertical 分别使用固定 draw slot 0/1/2/3。`joint_transform_identity` 摘要使用 `liuding.joint-transform.cpu.v2` 域，包含采样、插值、padding、框/区域筛选、OBB 裁剪策略、pose 排列和全部资源上限。升级后的指纹与旧版本不同，调用方须更新绑定后重新建立兼容的 checkpoint。示例中的 `dataset-v3` 应替换为实际冻结的数据/标注/reader 身份；重复采样时必须显式提供稳定 occurrence，不能用 worker 执行次数代替。

接入 `DataPipeline` 时，调用方须将来源身份绑定到 `dataset_version`，将上述配置摘要以及 RNG 版本、seed、view/occurrence 分配策略绑定到 `batch_factory_id`。直接提供手工 `ImageGeometryStep` 时，也要把完整 steps 纳入自己的 factory 身份；配置摘要不会自动读取任意 lambda 或推断其行为。当前已验证同一数据与配置在 0/1/3 workers 的像素结果一致、同拓扑 checkpoint 恢复重放，以及来源/变换身份变化被拒绝；pipeline 仍会拒绝不同 worker 拓扑的 exact restore。原 Core `data/transforms/vision.hpp` 的随机算子继续具有各自的旧 RNG 行为，不会自动变成此采样器。

## 异步图像输出与证据登记

`ImageOutputSink` 接受已编码的 CPU PNG/JPEG 字节，以单个写入线程和有界项数/字节容量发布本地文件；它不会隐式编码 Tensor 或执行 CUDA D2H。`submit` 的 `completion` 在文件成功发布后返回路径、字节数和 SHA-256。需要把每个成功文件关联到 run 时，使用 `submit_with_evidence(path, image, callback, sample_key, stop)`：调用方在回调中捕获稳定的 run、reference 和事件身份，并将发布结果传给 [`SQLiteRunStore::record_published_file_evidence`](local_run_storage.md)。`data_image` 组件不依赖 `experiment` 组件。

回调仅在文件成功发布后由现有写入线程执行；采样跳过、容量丢弃、文件写入失败均不会调用它。已排队的提交同时返回文件 `completion` 和 `evidence_completion`，前者即使证据登记失败仍保留成功发布的路径与摘要，后者单独报告登记结果。无回调的 `submit` 只有文件 future；未入队时两个 future 都无效。回调执行前已释放编码缓冲及其资源租约，但该项仍占用 sink 的项数和容量直到回调结束；调用方须保证捕获对象活到回调完成。回调捕获对象的内存不计入编码字节预算，应保持有界并及时返回，不得重新调用同一 sink。回调失败计入失败统计并由 `flush/close` 传播；默认策略停止后续已排队写入，`continue_after_error` 可允许继续。

文件发布、CAS 引用登记和事件追加不是一个原子事务。证据失败后保留 `completion` 的身份，按[本地 run 存储](local_run_storage.md)使用原有 reference、事件身份和描述符重试；已持久化的引用可在原文件移走后恢复事件。该回调只简化关联，不代表跨阶段严格内存上界或端到端训练提速。
