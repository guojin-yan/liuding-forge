# 图像数据、离散标签与坐标追踪

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

JPEG 等有损格式、带独立 alpha 通道的 PNG、灰度/RGB 的 tRNS、APNG 和 Apple CgBI PNG 当前会明确拒绝。读入全景 ID 图不等于已校验 segment 表与类别的关联，调用者仍需结合相应标注索引。该入口检查所有 PNG chunk 的 CRC、zlib Adler-32、完整输出尺寸、调色板索引范围及文件结构。

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

配置 `mask_target` 后，配对清单中的 target 走标签 reader，结果存入 `VisionSample::mask_target`；`paired_target` 和 `paired_target_pixels` 为空。mask 必须与输入图像尺寸完全相同，`require_matching_target_size=false` 仅适用于普通配对图像。未配置 `mask_target` 时继续读取普通配对图像。当前数据集读取不会自动重采样或联合变换图像与标签；后续变换必须保持离散 ID，不能对标签使用 bilinear。

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

trace 最多组合 128 次操作，保存固定大小的映射状态。空尺寸、越界 crop、uint32 补边溢出、非有限坐标或不可表示的映射均明确报错。当前实现仅覆盖上述轴对齐几何；旋转、透视、多图拼接、EXIF 空间转换和图像/标注自动联合变换尚未提供。
