# 组件依赖约束

`component-dependencies.json` 记录组件的必要直接依赖、允许的可选依赖，以及公共头文件和生产源码的唯一归属。归属根相对于 `include/jyppx/liuding/` 或 `src/jyppx/liuding/`，支持 `deployment/onnx_export`、`deployment/onnx_runtime`、`deployment/onnx_serving` 这样的多级目录，分别归属同名组件。`onnx_serving` 是独立可选桥接组件，要求启用 `serving` 和 `onnx_runtime`；仅启用 `serving` 不引入 ORT。每段目录名使用小写字母开头的小写字母、数字或下划线，不允许绝对路径、路径别名或相互重叠的根。除清单声明的顶层聚合头文件外，每个公共头文件和生产代码文件都必须有唯一归属，每个归属根都必须包含代码文件。

CMake 在配置时生成 `component-target-graph.tsv`，`scripts/validation/component_graph.py` 同时检查实际 target、公共链接接口、文件归属及 `<jyppx/liuding/...>` 包含关系，并在构建目录写入 `component-dependency-report.json`。检查覆盖关闭测试的组件配置，也注册为 `ComponentDependencyGraph` CTest。

检查拒绝未声明或循环依赖、缺少直接链接、公共包含缺少公共链接接口、重复声明、非法或未归属路径以及归属断言漂移。新增组件或依赖时，同步更新架构决策与 JSON 清单；`jyppx::liuding::all`、`jyppx::liuding::sdk` 等聚合 target 或其他组件的传递链接不能替代必要的直接依赖。
