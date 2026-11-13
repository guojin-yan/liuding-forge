# 训练、优化器与状态示例

本目录集中原生 C++ 训练循环、优化器调度、梯度缩放和检查点示例。`CMakeLists.txt` 可以独立消费已安装 SDK 的 `training`、`interop` 组件；构建和运行库设置见[基础训练教程](../../docs/tutorials/01_basic_training.md)。按需构建目标，不必为体验一个流程构建全部示例。

| 源文件 | 目标 | 用途 |
| --- | --- | --- |
| [basic_trainer.cpp](basic_trainer.cpp) | `jyppx_basic_trainer` | 两轮线性回归、独立训练/验证游标、Trainer 管理的逐更新预热；保留原程序名和整体构建输出位置 |
| [optimizer_scheduling.cpp](optimizer_scheduling.cpp) | `jyppx_optimizer_scheduling` | 两参数组 SGD/动量、epoch/成功更新/指标调度，以及模型＋优化器＋调度器的新进程协调恢复 |
| [grad_scaler.cpp](grad_scaler.cpp) | `jyppx_grad_scaler` | 单/多优化器有限更新与溢出整组跳过，见[混合精度教程](../../docs/tutorials/05_mixed_precision.md) |
| [checkpoint_state.cpp](checkpoint_state.cpp) | `jyppx_checkpoint_state` | 自定义参与者、新进程恢复、失败回滚与预算预检，见[持久检查点](../../docs/api/durable_checkpoint.md) |
| [image_training_output.cpp](image_training_output.cpp) | 见[图像输出说明](IMAGE_OUTPUT.md) | 图像配方的持久输出和资源释放 |

优化器示例的五种运行方式、预期参数和状态边界见[优化器教程](../../docs/tutorials/03_optimizers.md)。这些微型程序用于学习和验证接口，真实任务质量、设备与资源支持按对应配方证据使用。
