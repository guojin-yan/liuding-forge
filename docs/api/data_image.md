# 图像数据与坐标追踪

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

`ImageDataset::get` 返回的 `VisionSample` 携带 `pixels` 和初始恒等 `geometry`；配对图像另有 `paired_target_pixels`，不会继承输入图像的颜色或归一化策略。配对图像解码本身不解释语义类别 ID，离散 mask 需要专门的读取和变换策略。

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
