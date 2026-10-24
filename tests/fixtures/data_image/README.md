# 图像测试样本

本目录保存小尺寸自有测试图片，供解码、数据集和像素语义回归使用，不包含外部数据集或预训练权重。

`cgbi.png` 由同目录的 `generate_cgbi.py` 使用 Python 标准库生成，内容为两个像素的 Apple CgBI PNG，采用 raw DEFLATE、预乘 BGRA 和正确的 PNG chunk CRC。它专门验证解码器明确拒绝尚未支持的子格式，避免把输出误报为普通 RGB/straight-alpha。重新生成时运行 `python tests/fixtures/data_image/generate_cgbi.py`。

`masks/` 中的 PNG 由 `generate_masks.py` 根据 PNG 格式规范独立生成，仅依赖 Python 标准库 `struct` 和 `zlib`。运行 `python tests/fixtures/data_image/generate_masks.py` 可确定性重建。正例包含灰度 1/2/4/8/16 位、调色板 1/2/4/8 位、8 位 RGB、普通与 Adam7 交错光栅、五种行滤波、拆分和空 IDAT、小尺寸交错边界以及 16 位配对标签。调色板故意含有重复颜色与透明项，以检查索引不会被颜色展开或 alpha 策略改变。

`bad-filter`、`bad-adler`、`short-raster`、`bad-palette-index` 是刻意损坏的输入；`apng`、`critical`、`gray-trns` 和 `rgba` 用于验证当前 mask reader 明确拒绝对应子类型。损坏 zlib 的文件仍有正确 PNG chunk CRC，用于区分容器校验与数据流校验。
