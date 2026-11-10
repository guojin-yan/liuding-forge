# 图像文件、CAS 引用与事件联合登记

这个离线 C++ 示例生成一张 16×16 RGB 预览图，通过有界 `ImageOutputSink` 发布 PNG，然后在回调中调用 `SQLiteRunStore::record_published_file_evidence_atomic`，在一次 SQLite 写事务中登记文件引用、制品事件及 GC 引用。文件发布先于数据库事务；事务失败时保留原文件和请求身份重试，未引用的 CAS 对象可由 GC 清理。

若进程在文件发布后退出，重建 sink 时可显式设置 `replay_identical_existing = true`，以同一路径、相同编码字节及原证据身份重新提交；sink 只接受普通文件的完整字节匹配，回调继续确认或重放联合提交。已有文件的不同内容不会被覆盖。输出目录须由调用方控制，文件与数据库仍有独立提交边界。

Linux 不覆盖路径在文件系统支持时使用无名暂存，发布前强杀不会留下具名临时文件；不支持的文件系统、其他平台及覆盖路径仍可能留下具名暂存，不会自动删除历史残留，详见[图像输出合同](../../docs/api/data_image.md#异步图像输出与证据登记)。

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

原文件丢失而 CAS 完整字节仍在时，可清空原请求的 `path`，显式调用 `recover_published_file_evidence_atomic` 补齐同一事务中的引用和事件；每次都重新核验 CAS 摘要及精确大小。原描述符、时间戳和事件身份必须由调用方保留，CAS 缺失或已被 GC 清理时恢复失败，不会重建原文件。详见[输出证据与恢复](../../docs/api/data_image.md#异步图像输出与证据登记)。
