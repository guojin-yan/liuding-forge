# 图像绘制

`jyppx::liuding::vision_render` 是独立的可选组件，依赖 `data_image`。它提供无窗口的 CPU raster 绘制，将带有 `PixelSpec` 的图片与当前图像坐标系中的结构化标注绘制到一份新建的 HWC RGB8 Tensor。

当前接口支持 RGB/BGR/灰度及显式 alpha 的 uint8 图片、xyxy 框、旋转框、polygon 外环、COCO keypoint 骨架、离散 int64 mask、RLE 实例 mask 和归一化 heatmap。通道顺序、CHW/HWC 布局和 alpha 关联均由 `PixelSpec` 指定；结果统一为新的 HWC RGB8 CPU Tensor。透明图层按给定背景合成，绘制顺序为热图、整图 mask、实例 RLE 和几何标注。mask ID 不经过浮点转换，颜色由 ID 稳定映射；输入 Tensor 与标注对象保持不变。

对象数、顶点/关键点/RLE run 数、像素数、输出字节和 raster 操作数均有单次调用上限。raster 预算覆盖基础像素转换、整图 overlay 扫描和可见绘制像素，支持 `stop_token` 取消。无效通道/尺寸、非有限热图、超限几何和超预算调用会在返回结果前失败。

该切片不包含文字和字体、mosaic、PNG/JPEG 编码与文件 sink、CUDA 绘制、模型后处理或 metric。调用方需要自行选择可视化颜色和结果保存策略；不得将绘制能力视为图像编码或完整可视化支持。

构建时显式启用 `JYPPX_LIUDING_BUILD_DATA_IMAGE=ON` 与 `JYPPX_LIUDING_BUILD_VISION_RENDER=ON`，安装后通过 `find_package(jyppx CONFIG REQUIRED COMPONENTS vision_render)` 发现，并链接 `jyppx::liuding::vision_render`。
