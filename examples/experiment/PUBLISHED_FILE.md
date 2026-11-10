# 图像文件的持久化发布与重启恢复

这个离线 C++ 示例生成一张 16×16 RGB 预览图，通过有界 `ImageOutputSink` 写入中间输入文件，在回调中调用 `SQLiteRunStore::prepare_file_publication`，以一次 SQLite 事务保留 CAS 内容和完整发布意图。随后删除输入文件、关闭原句柄，重新打开存储并发现意图，调用 `recover_file_publication` 从 CAS 发布最终 `preview.png`，共同提交结果引用和制品事件。

重启恢复只需要 run ID 和从 `file_publications` 读取的意图 ID，不需要生产者、原输入文件或内存中的事件请求。已准备的内容受 run 引用保护，参与现有 GC 和备份；准备阶段占两个引用，完整提交后共占四个引用及各自的字节配额。相同内容可重放，已有不同内容不会被覆盖。回调完成在本例中只代表准备完成，最终文件与证据以恢复方法的成功回执为准。

最终路径的父目录须已存在、由调用方控制，并位于 run 存储之外。恢复用同目录专用暂存及不覆盖发布，进程中断后的暂存按原意图恢复；文件先同步再提交 SQLite，仍存在原始文件可见但数据库尚未提交的窗口。成功回执确认完整结果，该协议不提供任意文件读取者与数据库的瞬时共同可见性或断电恢复资格。完整限制见[本地 run 存储](../../docs/api/local_run_storage.md#可恢复的最终文件发布)。

在匹配 LibTorch 与编译器的配置中开启 `experiment`、`data_image` 及 `JYPPX_LIUDING_BUILD_EXAMPLES`，构建并执行：

```sh
cmake --build out/build/your-profile --config Release --target jyppx_liuding_published_file_example --parallel 1
```

Windows 多配置构建的程序位于 `examples/Release/liuding-forge-example-published-file.exe`，Linux 单配置构建位于 `examples/liuding-forge-example-published-file`。不带参数运行时使用并清理独立临时目录；提供一个父目录已存在的新目录路径时，保留其中的 `preview.png` 与 `runs` 存储：

```sh
liuding-forge-example-published-file NEW_DIRECTORY
```

成功时输出 `Recorded preview and replayed one event:` 及文件 SHA-256。程序先等待中间文件与准备回调的两个 future，再关闭存储并删除中间文件；随后重新打开 SQLite，发现原意图、恢复并重放最终发布，检查仍为一个事件及四个引用，并读回 PNG 核对摘要。调用方须保留 run ID 和意图 ID，准备成功前重试还须保持完整请求身份；配额须容纳内容、意图、结果和事件四个引用。

已有最终文件只需补交证据时，原有 `record_published_file_evidence` 和 `record_published_file_evidence_atomic` 仍可使用；两者分别采用分阶段登记和 SQLite 引用/事件共同提交。提交期 I/O/同步错误可能导致结果未知，应重开存储并以全部原身份重试，确认引用前保留已发布文件。完整合同见[本地存储 API](../../docs/api/local_run_storage.md)。

原文件丢失而 CAS 完整字节仍在时，可清空原请求的 `path`，显式调用 `recover_published_file_evidence_atomic` 补齐同一事务中的引用和事件；每次都重新核验 CAS 摘要及精确大小。原描述符、时间戳和事件身份必须由调用方保留，CAS 缺失或已被 GC 清理时恢复失败，不会重建原文件。详见[输出证据与恢复](../../docs/api/data_image.md#异步图像输出与证据登记)。
