# DINOv2-S 权重转换

`liuding-forge-convert-dinov2-hf` 离线转换固定官方 `facebook/dinov2-small` 权重，使用 C++ 实现。支持的 revision、完整 SHA-256、许可和模型选项见 [Models API](../../docs/api/models.md)；权重由使用者自行取得并保存在仓库外。当前仅支持无掩码 DINOv2-S 图像编码，不支持其他容量、revision 或 dtype。

启用 `JYPPX_LIUDING_BUILD_TOOLS=ON`，在已有构建配置中构建工具，再运行安装目录 `bin/` 下的命令。Windows 多配置构建添加 `--config Release`。

```sh
cmake --build out/build/<profile> --target jyppx_convert_dinov2_hf --parallel 2
liuding-forge-convert-dinov2-hf --source /weights/dinov2-small.facebook.safetensors --destination /weights/dinov2-small.liuding.safetensors --report /weights/dinov2-small.conversion.json
```

三个路径均必填；两个输出必须是已有可信目录中的不同新文件，工具拒绝覆盖普通文件、目录和符号链接。转换校验源与目标的完整大小和摘要，保持参数位模式、Q/K/V 顺序及既有目标格式，并记录全部映射与省略的 `mask_token`。转换同时保留约 168 MiB 的输入/输出字节缓冲，还需为校验和元数据预留内存。

文件独占暂存并 flush 后，以硬链接无覆盖发布；文件系统必须支持硬链接。两个输出不是原子事务：报告发布失败会尝试撤回本次模型链接，失败时保留暂存目录并在错误中给出路径。进程中断也可能留下仅模型或完整输出加暂存的状态。使用前同时核对完整报告与目标大小、摘要；报告缺失时先保留证据，确认残留属于本次运行后再清理，并使用两个新文件名重试。运行期间不要由其他程序改写相关目录。

成功返回 0 并打印摘要 JSON；参数错误返回 2，转换或文件错误返回 1。`--help` 显示参数。转换确认数据布局，任务准确率仍需单独验证；后续使用见[迁移学习教程](../../docs/tutorials/transfer_learning_recipe.md)。
