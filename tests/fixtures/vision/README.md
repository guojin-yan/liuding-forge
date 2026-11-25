# 视觉训练固定参考

`ema_original.json` 来自锁定的 RT-DETR `df2a96c8c3a098567a1149bf0248da6efe6d495d` 原始 `src/optim/ema.py`，文件内记录源摘要、Torch 版本、FP32 参数、FP64 缓冲区、整数及布尔初态和六次固定源状态更新后的完整 shadow。独立参考执行上游更新函数，不替代原生训练功能；用于验证乘法、加法的独立舍入及保留初始非浮点缓冲区。
