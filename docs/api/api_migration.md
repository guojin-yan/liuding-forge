# 只读 API 迁移扫描

`liuding-forge api scan` 在应用源码中查找已退役的 include、namespace 和 CMake package 名称，返回需要人工审查的迁移方向。

```text
liuding-forge api scan --root application --format json
liuding-forge api scan --root application --max-files 2000 --max-bytes 8388608 --max-findings 512 --format json
```

扫描只读取文件，不改写源码、不执行 CMake 或构建应用。替换前须核对当前构造函数、优化器与 checkpoint 状态、worker 所有权及 callback 事务合同；消费者需用当前公开头与 target 重新编译。

这是有界词法扫描：注释、字符串、未启用的预处理分支和包含旧名称的较长标识符也可能匹配；宏、别名、生成拼写或未登记符号可能漏报。空结果只表示固定规则未发现匹配，不能证明 AST 完整性、行为等价或 ABI 兼容。

## 结果与迁移方向

`--format json|text` 返回 `scanner: bounded-lexical-v1`、`source_untouched`、扫描文件数/字节数、限制说明及 `findings`。

| 字段 | 含义 |
| --- | --- |
| `path` | 相对扫描根的 UTF-8 路径 |
| `line` | 从1开始的行号 |
| `feature_id` | 稳定的迁移规则标识 |
| `retired_surface` | 匹配到的固定旧名称 |
| `suggestion` | 当前所属合同或已移除能力的说明 |
| `automatic_change` | 始终为 false |

文件按路径排序，再按固定规则与文本顺序输出；同一行同一旧名称只返回一次，不返回源码片段或配置值。规则覆盖旧 package、工具/数据/损失与初始化头、优化器、scheduler 和训练 callback；表位于 `tools/cli/migration/api_migration.cpp`，不是完整符号数据库。

旧 `jyppx/optim/sgd.hpp` 指向 `jyppx/liuding/optim/sgd_adapter.hpp`，优化器状态仍需显式审查；旧 `jyppx::utils` 需选择实际的 foundation/data/training owner。旧 TensorBoard 头和 callback 指向 `run export --backend tensorboard-native`，配置与映射见[运行遥测](run_telemetry_cli.md)；历史 `feature_id` 保持兼容，不启用旧 Python 路由。量化与分布式占位符只报告已移除，不授予替代能力。

## 文件与资源边界

识别 `CMakeLists.txt`、`.cmake/.c/.cc/.cpp/.cxx/.h/.hpp/.hxx/.ipp/.inl`，区分大小写。跳过 `.git/.hg/.svn/out/build/third_party/node_modules/.venv` 目录；其他生成目录应通过选择合适扫描根避开。

| 选项或资源 | 默认及最大值 |
| --- | --- |
| `--max-files` | 10,000个识别文件 |
| `--max-bytes` | 累计67,108,864字节 |
| `--max-findings` | 4,096项 |
| 单文件 | 2 MiB |
| 遍历 | 50,000个目录项，深度32 |
| 相对路径 | 1,024个 UTF-8 字节 |

选项只能降低上限且必须为正；超限明确失败，不静默截断。扫描期间源树应保持静止并由调用方独占管理；[迁移文件系统规则](config_migration.md#filesystem-and-resource-bounds)拒绝路径穿越、已检查的链接/reparse point、特殊文件和硬链接输入。

退出码0表示无匹配，1表示有待审查结果，2表示无效输入，3表示资源、文件系统或其他运行失败。扫描不会写入源文件。
