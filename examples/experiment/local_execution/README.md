# 在 C++ 中执行本地原生程序

[main.cpp](main.cpp) 直接链接 `jyppx::liuding::experiment`，通过安装后的公共头启动本机程序。示例复用自己的可执行文件作为工作进程，验证字面量参数、独立环境、环境摘要、CPU放置、非零退出码、双流输出、取消、超时回收和两次JSON行交互，不依赖Python、命令行工具或shell。

在兼容构建目录启用 `experiment` 和 examples 后构建、运行：

```sh
cmake --build out/build/YOUR_PROFILE --target jyppx_liuding_local_execution_example --parallel 1
out/build/YOUR_PROFILE/examples/liuding-forge-example-local-execution
```

Windows 使用相应配置子目录及 `.exe`。成功时输出 `passed: true` 的JSON；示例设置的150毫秒超时用于演示工作进程回收，不是性能承诺。若运行库需要搜索路径，应在启动示例前设置好。

用户程序使用 `<jyppx/liuding/experiment/local_execution/process.hpp>` 中的 `LocalProcessOptions` 和 `LocalProcessRunner`，无需解析CLI输出。程序路径和工作目录必须是绝对路径；标准输入为EOF，参数直接传给程序，stdout/stderr分别设置累计字节上限。普通非零退出和POSIX信号保持原生语义；取消、超时和输出超限为独立状态，返回结果不再拥有运行中的子进程。清理证明、直接进程RSS和可选OS内存计数分开报告。

`inherit_environment` 默认关闭；显式环境只影响子进程。`local_process_environment_sha256()` 可绑定本平台完整有效环境，实际启动使用同一份已核对快照；调用者需要的动态库搜索路径也须包含在环境中。`cpu_affinity` 使用调用进程允许的CPU编号，启动用户代码前设置并严格回读；它是放置约束，不是CPU时间配额。OS内存限制和进程树边界见[资源合同](../../../docs/development/local_process_runner.md#操作系统内存限制)。

需要持久JSON行交互时，使用独立头 `<jyppx/liuding/experiment/local_execution/json_lines.hpp>` 和 `JsonLineProcessTransport`。工作进程按每请求一行返回一行；传输检查UTF-8、帧长度、截止和额外输出，首次协议或I/O失败后对象不可复用。对象在同一串行所有者下使用，销毁或 `terminate()` 回收原生进程。

这个SDK接口负责单次进程执行和有界通道；持久attempt、run/CAS、有限重试及完整训练RunSpec仍须通过相应编排层组合，不能仅根据程序退出码认定训练成功。现有[本地执行和搜索CLI](../../../docs/user-guide/cli.md)保留原有持久协议。
