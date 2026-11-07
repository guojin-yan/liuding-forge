# 组件依赖约束

`component-dependencies.json` 记录组件的必要直接依赖、允许的可选依赖，以及公共头文件和生产源码的唯一归属。归属根相对于 `include/jyppx/liuding/` 或 `src/jyppx/liuding/`，支持 `deployment/onnx_export`、`deployment/onnx_runtime`、`deployment/onnx_serving` 这样的多级目录，分别归属同名组件。`onnx_serving` 是独立可选桥接组件，要求启用 `serving` 和 `onnx_runtime`；仅启用 `serving` 不引入 ORT。每段目录名使用小写字母开头的小写字母、数字或下划线，不允许绝对路径、路径别名或相互重叠的根。除清单声明的顶层聚合头文件外，每个公共头文件和生产代码文件都必须有唯一归属，每个归属根都必须包含代码文件。

CMake 在配置时生成 `component-target-graph.tsv`，`scripts/validation/component_graph.py` 同时检查实际 target、公共链接接口、文件归属及 `<jyppx/liuding/...>` 包含关系，并在构建目录写入 `component-dependency-report.json`。检查覆盖关闭测试的组件配置，也注册为 `ComponentDependencyGraph` CTest。

检查拒绝未声明或循环依赖、缺少直接链接、公共包含缺少公共链接接口、重复声明、非法或未归属路径以及归属断言漂移。新增组件或依赖时，同步更新架构决策与 JSON 清单；`jyppx::liuding::all`、`jyppx::liuding::sdk` 等聚合 target 或其他组件的传递链接不能替代必要的直接依赖。

`OptionalComponentMatrix` 和 `PreviewComponentMatrix` 的嵌套配置继承父构建的 `FETCHCONTENT_FULLY_DISCONNECTED`、`FETCHCONTENT_UPDATES_DISCONNECTED`（含逐依赖设置）及 `FETCHCONTENT_SOURCE_DIR_*`，两组矩阵各自复用一套构建树及其依赖构建目录。基础组件矩阵还复用一套消费者构建树及其 Windows DLL 暂存，并使用一个较短的缓存目录依次检查非法组合；每次重配显式选择完整组件集。矩阵暂存位于源码 `out/` 内，每种组合都先刷新安装目录，再核对启用组件、关闭组件及独立消费者。先准备锁定依赖并以 `-DFETCHCONTENT_FULLY_DISCONNECTED=ON` 配置父构建；矩阵覆盖组件关闭、安装及独立消费者，不能用完整 SDK 构建替代。

嵌套编译任务数优先使用配置项 `JYPPX_LIUDING_MATRIX_BUILD_JOBS`，为空时使用运行 CTest 时的 `CMAKE_BUILD_PARALLEL_LEVEL`，两者均未设置时为 2；必须是正整数。共享构建按平台串行，当前维护流程最多使用两个编译任务，可设置 `-DJYPPX_LIUDING_MATRIX_BUILD_JOBS=1` 降低资源占用。两组矩阵都包含重复构建和安装，CTest 总时间上限均为 10800 秒。
