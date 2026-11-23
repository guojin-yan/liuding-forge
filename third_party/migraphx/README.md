# MIGraphX 指令名扩展

[`instruction-names-v1.patch`](instruction-names-v1.patch) 是 Liuding Forge 为外部 MIGraphX 2.18.0 提供的可选开发补丁，基于 AMDMIGraphX `rocm-10.1` 的 `src/program.cpp`、`src/include/migraphx/program.hpp` 和 `tools/api/{api.cpp,migraphx.h}`。上游归档身份见 [`dependency-manifest.json`](../../dependency-manifest.json) 的 `migraphx-native`；补丁沿用 [上游 MIT 许可](../licenses/migraphx-2.18.0-LICENSE.MIT)，保留原文件声明，不代表上游已接受或发行该扩展。

在独立 SDK 源码目录先执行 `git apply --check <补丁绝对路径>`，通过后再执行 `git apply <补丁绝对路径>`，按该 SDK 原有工具链重新构建并安装到独立目录。补丁同时修改 API 模板，正常构建重新生成 C API 即可；不要只改构建目录中的生成文件。Forge 不在配置或安装期间应用补丁，不下载或复制 SDK。Windows 平台构建本身所需的其他上游适配由 SDK 提供方维护，本补丁不替代这些适配。

扩展只增加 `program::eval_instruction_names` 和 C 入口 `migraphx_program_run_instruction_names_v1`，不改变 `program`、`execution_environment` 布局或现有 API。它使用同一 evaluator，在实际非空可检查指令执行后回调名称，省去逐指令同步和中间张量的主存拷贝，返回前完成所有 context。名称仅在回调期间有效；回调返回错误后继续完成本次执行，最终报告错误且不交付输出。执行错误仍由适配器同步清理借用输入。

Forge 从已加载的 C API 模块查询这一版本化入口；存在时逐请求使用它，不存在时使用原 `migraphx_program_run_trace()`。设置 `MIGRAPHX_TRACE_EVAL` 时保持原 trace 路径。两条路径均核对实际 GPU 算术、同步主存输出及 CPU/reference 指令拒绝，不以编译图清单代替执行记录，也不在失败后重放请求。

切换 SDK 会改变环境锁记录的实际 C API 模块摘要。部署时应重新观察环境、核对原模型与已知向量并生成对应可信包；旧包的锁不匹配时继续拒绝加载。具体验证组合以 [`compatibility.json`](../../compatibility.json) 为准，接口可用不表示持续负载、性能、其他设备或操作系统已通过验收。
