# 原生遥测 worker 进程传输

`SidecarProcessTransport` 是 CLI 私有的 C++ 有界进程通道，实现 `experiment::TelemetrySidecarTransport`；类名和中立协议标识保持兼容，不表示安装或运行 Python sidecar。产品只启动同一 Liuding Forge 可执行文件中的原生 TensorBoard/OTLP worker，配置入口见[运行遥测 CLI](run_telemetry_cli.md)。

## 启动与数据边界

通道使用固定绝对可执行路径、UTF-8 argv、独立且已存在的工作目录和显式子进程环境；不搜索 PATH、不调用 shell、不修改父环境或 cwd。CLI 不接受任意 executable/arguments 配置，事件不能决定执行程序、目录、环境或目的地。环境默认不继承，调用方须显式选择继承或提供运行库所需环境。

请求与响应为单行 UTF-8 JSON；默认请求16 MiB、响应128 KiB，拒绝空帧、内嵌 NUL/CR/LF、EOF、不完整响应、多余输出和无效 UTF-8。写 stdin 时同时排空 stdout/stderr，避免有限管道缓冲死锁；默认2 ms静默观察无法预测更晚的 unsolicited输出，后续交换仍检查。`OfficialTelemetrySink` 另核 schema、请求身份、回执与累计计数。

stderr 按默认64 KiB累计配额丢弃，只返回排空字节数，不把内容、argv、环境、路径或系统错误文本带入协议错误。launch配额同时覆盖 argv和环境（默认256 KiB），Windows另受 UTF-16命令行限制。配额是容量上限，不是预分配或私有 workspace 保证。

## 截止、终止与确认

每次交换采用调用方绝对截止和 operation timeout（默认10秒）的较早者，startup timeout默认10秒。管道等待以最多5 ms片段检查 stop/deadline；失败后通道中毒并终止，后续调用报 `NotReady`。不自动重启或重放，因为失败前可能已有外部效果。

先显式 `close` 取得最终计数，再 `terminate()` 释放所有权；析构和 terminate不发送协议 close/flush。默认清理宽限2秒，可能延长已到达的操作截止；重复终止保留首次结果，不续借预算。原始交换错误优先，清理错误不泄露子进程诊断。

`process_tree_cleanup_confirmed()` 在清理前为 false，仅在原生层确认 leader已退出或回收、原 Job/进程组已空后为 true。CLI不能在 false时报告成功；错误为 `cli.run.export/process.cleanup`，原 SQLite证据保留。operation/startup最大10分钟，cleanup最大10秒，quiet period最大100 ms；操作系统创建、文件系统、DNS和调度仍不提供绝对墙钟保证。

## 平台所有权

Windows使用 Unicode `CreateProcessW`、CRT argv转义、显式句柄继承列表及 overlapped named pipes；进程先挂起，加入 kill-on-close Job后再恢复，不启用 breakaway。无法加入 Job时明确失败并终止挂起子进程，无非隔离回退；后代和持有继承管道的进程属于同一 Job。

POSIX使用独立进程组、固定绝对 `execve`、父端 nonblocking descriptors、`poll`和 exec状态管道；fork后的子进程只使用已准备数据与 async-signal-safe调用，关闭其他描述符，SIGPIPE仅在线程内屏蔽。清理向原组发送 SIGKILL并回收直接子进程；leader PID在组清理前保持所有权。后代 zombie由其真实父进程或系统 reaper回收，库不改变全局 subreaper/SIGCHLD策略；已有 `SIG_IGN/SA_NOCLDWAIT` 时拒绝启动。原生 worker按该所有权协议协作，进程组不是对恶意 `setsid/setpgid` 程序的沙箱。

进程通道测试中的 Python程序仅是故障夹具，官方 Python包仅用于独立读取/参考映射；二者不安装，也不参与产品导出。Collector 接收、TLS 信任、持久 ACK 与进程清理分别遵循[运行遥测](run_telemetry_cli.md)的契约。
