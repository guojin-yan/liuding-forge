# 图像文件、CAS 引用与事件联合登记

这个离线 C++ 示例生成一张 16×16 RGB 预览图，通过有界 `ImageOutputSink` 发布 PNG，然后在回调中调用 `SQLiteRunStore::record_published_file_evidence_atomic`，在一次 SQLite 写事务中登记文件引用、制品事件及 GC 引用。文件发布先于数据库事务；事务失败时保留原文件和请求身份重试，未引用的 CAS 对象可由 GC 清理。

在匹配 LibTorch 与编译器的配置中开启 `experiment`、`data_image` 及 `JYPPX_LIUDING_BUILD_EXAMPLES`，构建并执行：

```sh
cmake --build out/build/your-profile --config Release --target jyppx_liuding_published_file_example --parallel 1
```

Windows 多配置构建的程序位于 `examples/Release/liuding-forge-example-published-file.exe`，Linux 单配置构建位于 `examples/liuding-forge-example-published-file`。不带参数运行时使用并清理独立临时目录；提供一个父目录已存在的新目录路径时，保留其中的 `preview.png` 与 `runs` 存储：

```sh
liuding-forge-example-published-file NEW_DIRECTORY
```

成功时输出 `Recorded preview and replayed one event:` 及文件 SHA-256。程序先等待文件与证据两个 future，再关闭存储；随后重新打开 SQLite，省略原文件路径重放相同请求，检查仍为一个事件及两个引用。调用方拥有稳定的 run、reference、batch、stream、event 与 sequence 身份；重试须保留这些字段、文件大小、摘要及描述信息。两个引用分别属于显式文件关系和事件关系，配额须允许两者。

原有 `record_published_file_evidence` 保留文件引用先提交、事件另行提交的可恢复行为。需要一次提交的调用方应显式选择新接口；成功后可省略 path 重放，提交阶段 I/O/同步错误可能没有成功回执但事务已完整写入；此时重开存储并以全部原身份重试，确认引用前保留已发布文件。完整合同见[本地存储 API](../../docs/api/local_run_storage.md)。
