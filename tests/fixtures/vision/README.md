# 视觉训练固定参考

`yolo_opencv.json` 来自锁定的 Ultralytics `cb13d5f9cfbd6f299da3620c625f81d721dc2849` 原始图像加载、Mosaic、RandomPerspective、MixUp、HSV、翻转和 LetterBox 函数，记录源摘要、合成像素公式、固定随机取值、RGB 输出摘要及框；翻转后的上游 `cxcywh` 以 FP32 转为参考文件声明的 `xyxy`。用于独立验证 OpenCV 固定参数处理，不授予完整官方抽样、训练精度或性能资格。

`ema_original.json` 来自锁定的 RT-DETR `df2a96c8c3a098567a1149bf0248da6efe6d495d` 原始 `src/optim/ema.py`，文件内记录源摘要、Torch 版本、FP32 参数、FP64 缓冲区、整数及布尔初态和六次固定源状态更新后的完整 shadow。独立参考执行上游更新函数，不替代原生训练功能；用于验证乘法、加法的独立舍入及保留初始非浮点缓冲区。
