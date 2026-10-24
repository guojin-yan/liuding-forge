# 图像测试样本

本目录保存小尺寸自有测试图片，供解码、数据集和像素语义回归使用，不包含外部数据集或预训练权重。

`cgbi.png` 由同目录的 `generate_cgbi.py` 使用 Python 标准库生成，内容为两个像素的 Apple CgBI PNG，采用 raw DEFLATE、预乘 BGRA 和正确的 PNG chunk CRC。它专门验证解码器明确拒绝尚未支持的子格式，避免把输出误报为普通 RGB/straight-alpha。重新生成时运行 `python tests/fixtures/data_image/generate_cgbi.py`。
