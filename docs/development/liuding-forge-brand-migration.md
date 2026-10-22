# 六丁炼算（Liuding Forge）品牌迁移交付报告

工作开始日期：2026-10-22；验证延续至 2026-10-23；验证平台：Windows x64。

## 1. 执行摘要

项目对外展示为 **六丁炼算（Liuding Forge）**，生成物和推荐 CLI 使用
`liuding-forge`。历史名称 **JYPPX Training Framework** 继续用于兼容说明、
技术身份和原有版权署名。C++ API、ABI、目录、CMake 消费方式及模型协议保持不变。

本次在已有品牌迁移改动上续作，补齐嵌套安装消费工程的生成器传递、Unix
兼容 CLI 执行权限、跨平台库文件匹配、benchmark 库清单和安装命令回归验证。

## 2. 仓库与接手基线

| 项目 | 接手时状态 |
| --- | --- |
| 实际仓库根目录 | `F:/GitSpace/jyppx-training-framework/jyppx-training-framework` |
| 分支 | `improvement/DEV-001-extreme-condition-tests` |
| HEAD | `de2f3160a5bc4bf5a234a78361b0ec3b01bc9383` |
| 与上游关系 | 接手时已 ahead 51；本次不产生提交 |
| 已修改文件 | 42 个；完整保留并在其当前内容上增量修改 |
| 未跟踪文件 | 接手时 `git ls-files --others --exclude-standard` 为空 |
| 接手差异 | 245 行新增、111 行删除；这是续作基线，不冒充最初任务的干净状态 |

接手快照保存在忽略目录
`out/evidence/liuding-forge-rename-20261022-continuation/`，包括
`initial-status.txt`、`initial-diff-stat.txt`、`initial-diff.patch`
以及 `initial-remotes.txt`。初始状态中没有未跟踪文件。未执行 reset、clean、restore、
stash，也未删除来源不明的文件。

## 3. 初始命名引用清单

写入前执行完整 `rg -n -i --hidden -g '!.git/**' -g '!out/**'` 扫描，
检索 `jyppx|JYPPX|Liuding|liuding|六丁炼算|liuding-forge`。接手时共有
10,487 个匹配行，完整内容见 `initial-naming-scan.txt`。

| 类别 | 代表位置与标识 | 处理 |
| --- | --- | --- |
| C++ 命名空间、目录 | `namespace jyppx`、`include/jyppx`、`src/jyppx`、`<jyppx/...>` | 保留 |
| CMake project、option、macro、target | `project(jyppx VERSION 0.3.0)`、`JYPPX_*`、`jyppx_training`、`jyppx::training` | 保留技术标识，仅更新展示文案和 `OUTPUT_NAME` |
| 安装及导出包 | `jyppxConfig.cmake`、`jyppxTargets.cmake`、`lib/cmake/jyppx`、`share/doc/jyppx` | 保留 |
| 库和可执行文件 | 实体库、CLI、example、benchmark、fuzzer 的输出名 | 使用 `liuding-forge-*` |
| CLI 命令 | 逻辑 target `jyppx_cli`，原命令 `jyppx` | 推荐 `liuding-forge`，安装时保留旧入口 |
| JSON、checkpoint、bundle、telemetry | `jyppx_version`、`jyppx.model_bundle`、`jyppx.*`、`application/vnd.jyppx.*` | 保留已有字段、取值和 schema |
| README、docs、examples、CI | 首页标题、帮助示例、执行文件路径、发布说明 | 更新对外品牌及实际输出路径 |
| 测试断言及 fixture | checkpoint 元数据、fuzz corpus、CLI 合同、安装消费工程 | 保留协议 fixture，增加品牌和旧入口回归断言 |
| 第三方许可及外部名称 | LibTorch、GoogleTest、nlohmann-json、SQLite、stb、tinyxml2、yaml-cpp | 不改名、不改许可正文 |

扫描按目录分布：根目录 650、`.github` 19、`cmake` 508、`docs` 390、
`examples` 125、`include` 1,066、`scripts` 210、`src` 1,380、`tests` 5,917、
`tools` 222。匹配数是文本行计数，不等于需要替换的数量。

## 4. 修改文件清单

本次工作区最终差异包含 47 个已跟踪文件及 1 个新增报告文件；其中接手时的 42 个修改均保留。完整清单：

```text
.github/workflows/benchmark.yml
CHANGELOG.md
CMakeLists.txt
CONTRIBUTING.md
README.md
cmake/TestInstalledComponentMatrix.cmake
cmake/TestInstalledTargets.cmake
cmake/TestOptionalComponentMatrix.cmake
cmake/tests/TestPreviewComponentMatrix.cmake
docs/CMakeLists.txt
docs/api/api_migration.md
docs/api/config_migration.md
docs/api/evaluation.md
docs/api/run_telemetry_cli.md
docs/development/releasing.md
docs/examples/offline_workflow.md
docs/index.md
docs/user-guide/beta_limits.md
docs/user-guide/cli.md
docs/user-guide/deployment.md
docs/user-guide/performance_tuning.md
examples/CMakeLists.txt
examples/serving/README.md
examples/serving/jsonl_host.cpp
scripts/benchmarks/RESOURCE_BASELINE.md
scripts/benchmarks/Run-Baseline.ps1
scripts/benchmarks/run_resource_baseline.py
scripts/release/prepare_release.py
scripts/release/release_common.py
scripts/release/verify_release.py
scripts/validation/run-fuzz.sh
tests/CMakeLists.txt
tests/benchmarks/bench_cuda_optimizer.cpp
tests/benchmarks/bench_cuda_training.cpp
tests/benchmarks/bench_data_pipeline.cpp
tests/benchmarks/bench_memory_usage.cpp
tests/cli/sidecar_fixture.py
tests/cli/test_model_checkpoint.py
tests/fuzz/CMakeLists.txt
tests/fuzz/README.md
tests/install/cli/run_acceptance.py
tests/install/component_consumer/CMakeLists.txt
tests/integration/CMakeLists.txt
tests/scripts/test_release.py
tools/README.md
tools/cli/CMakeLists.txt
tools/cli/cli_main.cpp
docs/development/liuding-forge-brand-migration.md
```

主要改动涉及根 CMake、CLI、安装消费脚本、example/benchmark/fuzzer 输出配置、
发布工具、README、docs、benchmark CI 和对应验收测试。
`include/jyppx`、`src/jyppx`、`compatibility.json`、`component-dependencies.json`、
`dependency-manifest.json` 以及协议 fixture 均未因本次品牌调整改变。

## 5. 旧名称到新名称映射

| 旧名称或接口 | 新名称或处理结果 |
| --- | --- |
| JYPPX Training Framework 对外主标题 | 六丁炼算（Liuding Forge） |
| `project(jyppx)` | 保持 |
| 原 GitHub 仓库 `guojin-yan/jyppx-training-framework` | `guojin-yan/liuding-forge`；仅更新仓库展示与克隆入口 |
| `namespace jyppx`、`include/jyppx`、`src/jyppx` | 保持 |
| `jyppx_<component>`、`jyppx::<component>` | 保持 |
| `libjyppx_training.a` | `libliuding-forge-training.a` |
| `jyppx_training.lib` | `liuding-forge-training.lib` |
| 原 CLI `jyppx` | 推荐 `liuding-forge`，安装时同时提供 `jyppx` |
| `jyppx_bench_*` 逻辑 target | 保持；文件输出为 `liuding-forge-bench-*` |
| example 逻辑 target | 保持；文件输出为 `liuding-forge-example-*` |
| `jyppx_fuzz_*` 逻辑 target | 保持；文件输出为 `liuding-forge-fuzz-*` |
| 发布 ZIP `jyppx-<version>-<platform>.zip` | `liuding-forge-<version>-<platform>.zip` |
| `jyppxConfig.cmake`、`find_package(jyppx)` | 保持 |
| `jyppx_version`、`jyppx.model_bundle` | 保持 |

## 6. 品牌使用位置

README 与 docs 首页使用中文完整品牌；CMake 配置摘要、Doxygen、CPack、CLI
帮助/版本结果和发布文档使用中英文品牌。CLI 版本 JSON 增量提供
`display_name`、`display_name_en`、`brand`、`artifact_prefix`、`legacy_name`；
保留原有 `version` 和 `build`。SBOM 在已有 `jyppx` 技术身份之外添加品牌
properties。未将其他神话品牌引入命名空间、包名、target、CLI 或协议。

## 7. 保留的技术兼容接口

所有公共命名空间、头文件、源码根、include guard、宏、CMake 选项、逻辑 target、
`EXPORT_NAME` 和 `jyppx::` 导出名保持不变。继续支持：

```cmake
find_package(jyppx CONFIG REQUIRED COMPONENTS training)
target_link_libraries(app PRIVATE jyppx::training)
```

不增加新的 `find_package(liuding-forge)` 或 `liuding-forge::` 命名空间。
新库文件名由现有导出 target 自动传递；手写原始库文件名或 `-ljyppx_*` 的脚本
需要改用新文件名，或改为通过现有 CMake target 消费。

## 8. 历史名称保留位置

README、docs 首页、CMake 配置摘要、CLI `legacy_name`、Doxygen 副标题和
CHANGELOG 明确保留 JYPPX Training Framework。原有版权署名、NOTICE、历史
发布记录、技术协议及兼容性说明中的历史名称继续保留。

## 9. 库文件名称

12 个实体组件使用 `liuding-forge-` 前缀，依次为：`foundation`、`core`、
`interop`、`data`、`data-image`、`nn`、`optim`、`training`、`experiment`、
`models`、`quantization`、`serving`。Windows 静态库为 `<name>.lib`；Unix
静态库为 `lib<name>.a`。私有 CLI transport 输出
`liuding-forge-cli-transport`，不安装。

`all` 与 `sdk` 原本就是 INTERFACE 聚合 target，没有物理库文件，不为品牌
调整添加空的 `liuding-forge-sdk` 或 `liuding-forge-all` 库。当前框架实体库为
STATIC；不改变库类型、CRT、Debug/Release 后缀、前后缀或 LibTorch 的 DLL。
若以后切换共享库，平台 DLL/import library/so/dylib 命名仍由 CMake 属性负责。

## 10. 可执行文件名称

推荐 CLI 为 `liuding-forge`；兼容命令为安装后的 `jyppx`。
示例为 `liuding-forge-example-{basic-data-pipeline,basic-trainer,mnist,distillation,serving-jsonl}`。
所有 benchmark 与 fuzzer 的逻辑 target 保持不变，实际文件把旧前缀及下划线
转换为 `liuding-forge-bench-*` / `liuding-forge-fuzz-*`。Windows 自动附加 `.exe`。
单元测试及内部进程边界辅助程序保留原技术 target 和文件名。

## 11. CMake、安装与 CPack

通过 `OUTPUT_NAME` 调整生成物；不重命名 CMake project、target、别名或包目录。
安装导出继续使用 `jyppxConfig.cmake` 及 `jyppxTargets.cmake`。实体库路径由
CMake 导出生成，不维护另一套手工链接映射。

CPack 使用 `liuding-forge` 包名、中文/英文描述、`Liuding Forge` 安装目录及
NSIS 展示名；默认二进制归档名为 `liuding-forge-0.3.0-win64`，源码归档前缀为
`liuding-forge-0.3.0-Source`。发布工具自己的 ZIP 名称及验证器同时更新，
原有 release、provenance、SBOM 字段和技术身份继续保留。

本地 CPack ZIP `out/lf-pack/liuding-forge-0.3.0-win64.zip` 已生成并校验 CRC、
12 个新名称组件库、115 个公共/生成头文件、14 个原有 `jyppx::` 导出目标、
旧包配置、法律声明及双 CLI。两个 CLI 文件内容相同。该 ZIP 的 SHA-256 为
`5e1e531f6880cec1d8753ed6aa65457861652dd54bb4442ff0f4de8296bf2bcb`。

嵌套消费测试显式传递父工程的 generator/platform/toolset，再使用 `-G/-A/-T`
配置，覆盖稳定组件与 Preview 消费工程。Unix 库匹配包含 `lib` 前缀；
可选组件检查将技术组件名 `data_image` 映射到文件名中的 `data-image`。

## 12. CLI 处理

逻辑 target `jyppx_cli` 输出新品牌命令。旧命令从同一二进制以
`install(PROGRAMS ... RENAME jyppx)` 安装，保证 Unix 执行权限；两个入口使用
相同运行时依赖。安装验收实际执行两个入口的 `--version`、`--help` 和
`doctor --device cpu`，并核对品牌字段、旧版本字段与 schema-1 envelope。
旧命令承诺适用于安装目录，构建目录提供推荐的新命令。

## 13. 文档、示例和 CI

README、docs 首页、CLI/部署/迁移/离线工作流/性能文档、CONTRIBUTING、
tools 和 serving 示例说明、CHANGELOG、发布说明改用新品牌。
GitHub clone URL 已同步为 `https://github.com/guojin-yan/liuding-forge`，新克隆目录
推荐为 `liuding-forge`。本地当前 checkout 路径仍是用户指定的
`F:/GitSpace/jyppx-training-framework/jyppx-training-framework`，不会改动
`include/jyppx`、`src/jyppx` 或任何技术路径。README、贡献指南和 issue 链接
均指向新仓库。
benchmark CI、PowerShell 基线脚本和 Python resource runner 使用实际新输出名；
`--target jyppx_*` 命令继续引用原逻辑 target。

## 14. checkpoint、model bundle、telemetry 和 JSON

未改动 BuildFingerprint 的结构、布局、`jyppx_version` 字段或其 JSON 序列化。
checkpoint metadata、`jyppx.model_bundle`、manifest schema、MIME、telemetry
中的 `jyppx.*` 和 benchmark `kind` 取值全部保留。原 checkpoint 与 fuzz corpus
保持字节不变；新增品牌字段限于人类可读入口的增量元数据。

## 15. 名称冲突与法律风险初筛

实际检索时间：2026-10-22，约 23:27–23:42（Asia/Shanghai）；逐项 UTC 时间、
完整 URL、HTTP 状态和结果保存在 `name-screening.json` 与
`name-screening-exact.json`、`name-screening-search-engines.json`。关键词为 `六丁炼算`、`Liuding Forge`、
`liuding-forge`；包注册表的精确查询使用合法包名 `liuding-forge`。

| 平台 | 已执行的检索与结果 | 范围限制 |
| --- | --- | --- |
| GitHub | 三个完整名称的引号精确 repository search 均返回 0；额外 C++/CMake 语言筛选返回 0 | 不含私有库，repository search 不等于全部代码或全部商标检索 |
| GitLab | 三个名称的公开 projects API 均返回空数组 | 仅公开可检索项目 |
| Codeberg | 三个名称的 repository API 均返回空数组 | 仅该实例的检索结果 |
| npm | 三个名称的全文查询没有在前 20 条发现同名包；精确包 endpoint 返回 404 | 全文是分词搜索，不能把返回总数当作同名数 |
| PyPI | `liuding-forge` 精确项目 JSON 返回 404 | 不代表分散分发的软件不存在 |
| Conda / Anaconda | 三个名称的 search API 均返回空数组 | 不覆盖所有私有 channel |
| crates.io | 三个名称的 crates API 返回 0 | 仅该注册表 |
| vcpkg | 官方仓库 `ports/liuding-forge/vcpkg.json` 返回 404 | 不覆盖 overlay ports 或非官方 registry |
| Conan Center | 官方 index 的 `recipes/liuding-forge/config.yml` 返回 404 | 不覆盖私有 remote |
| C++ / CMake 包 | GitHub 语言筛选，加 vcpkg/Conan 官方路径查询未命中 | 不存在统一的全球 CMake 包注册表 |
| Bing | 三个引号关键词请求均返回页面，但结果与名称无关 | 结果质量不足，不据此得出无冲突结论 |
| Google | 三个引号关键词请求返回搜索入口页面 | 未取得可核实的结果清单 |
| DuckDuckGo | 三个关键词均返回 HTTP 202 验证页面 | 未解答或绕过验证，不视为完成有效检索 |
| WIPO | 官方查询入口可达 | 动态商标结果未完成核实，不能视为已检索通过 |
| 中国商标网 CNIPA | 官方查询入口可达 | 未取得有效商标结果清单 |
| USPTO | 连接握手超时 | 未完成检索 |
| EUIPO | 远端断开连接 | 未完成检索 |

本次没有确认到完整同名的公开深度学习、C++、LibTorch 或部署项目，因此没有
可进一步评估活跃度及产品重叠的同名候选。中文非精确搜索出现的分词噪声
不作为同名项目。单独 `Liuding` 的潜在既有使用不能由这次结果排除，持续使用
完整品牌。名称唯一性、近似商标、在先使用和目标司法辖区仍需进一步法务确认，
尤其软件及相关服务类别。此报告不构成商标审查或无风险结论。

## 16. 验证命令

验证证据目录简称 `E=out/evidence/liuding-forge-rename-20261022-continuation`。
全新构建目录简称 `B=out/build/lf-rename-20261022`，创建前确认不存在；没有使用
此前的 `out/build/config-windows` 或旧 rename build 作为本轮构建缓存。
Python 使用仓库已有 CPython 3.11.9，LibTorch 使用
`third_party/install/libtorch`（2.14.1 CUDA）。

```powershell
& out/envs/telemetry-py311/Scripts/python.exe scripts/validation/compatibility.py --file compatibility.json --validate
$env:TORCH_CUDA_ARCH_LIST = '7.5'
cmake -S . -B out/build/lf-rename-20261022 -G 'Visual Studio 18 2026' `
  -DCMAKE_BUILD_TYPE=Release `
  '-DCMAKE_PREFIX_PATH=F:/GitSpace/jyppx-training-framework/jyppx-training-framework/third_party/install/libtorch' `
  '-DPython3_EXECUTABLE=F:/GitSpace/jyppx-training-framework/jyppx-training-framework/out/envs/telemetry-py311/Scripts/python.exe' `
  '-DCUDAToolkit_ROOT=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9' `
  '-DCUDA_TOOLKIT_ROOT_DIR=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9' `
  -DCMAKE_CUDA_ARCHITECTURES=75 -DJYPPX_BUILD_BENCHMARKS=ON `
  -DJYPPX_BUILD_DATA_IMAGE=ON -DJYPPX_BUILD_MODELS=ON `
  -DJYPPX_BUILD_QUANTIZATION=ON -DJYPPX_BUILD_SERVING=ON
cmake --build out/build/lf-rename-20261022 --config Release --parallel 2
$env:JYPPX_RELEASE_TEST_OPENSSL = 'C:/Program Files/Git/usr/bin/openssl.exe'
ctest --test-dir out/build/lf-rename-20261022 -C Release --output-on-failure `
  --parallel 2 --output-junit out/evidence/liuding-forge-rename-20261022-continuation/ctest.xml
git diff --check
```

打包验证命令：

```powershell
cpack --config out/build/lf-rename-20261022/CPackConfig.cmake -G ZIP -C Release -B out/lf-pack
& out/envs/telemetry-py311/Scripts/python.exe out/evidence/liuding-forge-rename-20261022-continuation/verify_package.py out/lf-pack/liuding-forge-0.3.0-win64.zip
```

收尾验证命令：

```powershell
& out/envs/telemetry-py311/Scripts/python.exe out/evidence/liuding-forge-rename-20261022-continuation/verify_library_guards.py
& out/envs/telemetry-py311/Scripts/python.exe out/evidence/liuding-forge-rename-20261022-continuation/final_audit.py
rg -n -i --hidden -g '!.git/**' -g '!out/**' '(jyppx|JYPPX|Liuding|liuding|六丁炼算|liuding-forge)' .
git status --short --branch
git diff --stat
git diff --name-only
git ls-files --others --exclude-standard
git diff --check
```

`final_audit.py` 只读取 Git 状态和源码，将证据写入忽略的 `out/`；不暂存文件。
CTest 下各项脚本的精确命令还可从构建目录的 `CTestTestfile.cmake` 查阅。

## 17. 验证结果

Windows Release 完整 CTest 命令退出码为 **0**：发现 1,065 项，**1,052 项通过、
13 项跳过、0 项失败、0 项禁用**，墙钟耗时 2,333.91 秒。跳过项没有计为实际通过。
原始结果为 `E/ctest.log`、`E/ctest.xml`，逐项跳过原因归档在
`E/ctest-summary.json`。相对 JUnit 输出最初生成于 `B/out/evidence/.../ctest.xml`，
完成后复制到 E，内容不作改写。

| 命令或检查 | 结果与证据 |
| --- | --- |
| `compatibility.py --file compatibility.json --validate` | 通过；`compatibility.log` |
| 全新 B 的 `cmake -S . -B ...`，及最终重新配置 | 通过；`configure.log`、`configure-final.log` |
| `cmake --build ... --config Release --parallel 2` | 完整构建及最终增量构建通过；`build.log`、`build-final.log` |
| CMake 组件依赖图校验 | 12 个实体组件、27 条策略必需依赖边；`component-dependency-report.json` |
| 完整 `ctest --test-dir ... -C Release --output-on-failure --parallel 2` | 1,052 通过、13 跳过、0 失败；`ctest.log`、`ctest-summary.json` |
| `OptionalComponentMatrix` | 通过，444.67 秒；包含 data_image 配置及禁用组合 |
| `ComponentInstallMatrixConsumer` | 11 个组件安装消费通过，768.15 秒 |
| `PreviewComponentMatrix` | 单组件、三 Preview 同开、再次关闭、符号隔离及错误组合通过，1,847.23 秒 |
| SDK、interop、experiment、package、models、quantization、serving 安装消费 | 全部通过；继续使用 `find_package(jyppx)` 与 `jyppx::` |
| `CliInstalledConsumer` | 新旧 CLI 均执行 version/help/CPU doctor，通过，4.95 秒 |
| 6 项 `Cli_*` 工作流 | checkpoint、bundle、evaluation/data、offline、telemetry、migration 全部通过 |
| checkpoint、bundle、telemetry、manifest 和进程恢复测试 | 已执行项通过；精确 LibTorch 2.5.1 fixture 运行限制见下文 |
| `PublicHeaderGuards` | 114 个源码公共头文件通过 |
| 历史 checkpoint fixture 完整性校验 | 12 组通过；`checkpoint-fixtures.log`；不等同于旧版本迁移运行 |
| C++ clang-format、Python/PowerShell/Bash 脚本语法 | 变更文件检查通过；C++ 记录见 `format-check.log` |
| 生成物清单 | 12 个组件静态库、5 个示例、19 个 benchmark、CLI 及私有 transport 库正确；`build-artifact-inventory.json` |
| basic-data-pipeline 与 basic-trainer 实际运行 | 通过；后者输出 `status=1 global_step=4`；对应 `example-*.log` |
| data-pipeline benchmark 小规模运行 | 正常输出有效 CSV；`benchmark-smoke.csv` |
| CLI 文本版本输出 | 包含完整品牌及原有 fingerprint 字段；`cli-version-text.log` |
| `cpack ... -G ZIP -C Release -B out/lf-pack` | 通过；`cpack.log` |
| `verify_package.py .../liuding-forge-0.3.0-win64.zip` | CRC、库名、头文件、旧导出目标、旧包配置、法律声明、双 CLI 内容校验通过；`package-verification.json` |
| `verify_library_guards.py` | 最终安装守卫的 10 个定向场景通过；`library-guard-verification.json` |
| 最终完整 `rg` 命名扫描 | 已执行并分类；`final-naming-scan.txt`、`final-naming-summary.json` |
| `git diff --check` 与新增报告空白检查 | 通过；`final-diff-check.log`、`final-repository-audit.json` |
| 初始修改集合、HEAD、分支、远程、暂存区及受保护目录比较 | 通过；`final-repository-audit.json` |

13 项跳过分为两组：4 项 symlink 安全测试因本机无法创建符号链接而按原规则
跳过；8 个优化器历史 fixture 迁移测试与 1 个文件 checkpoint 历史迁移测试
要求精确 LibTorch 2.5.1，本机 2.14.1 因此跳过。12 组 fixture 完整性校验
和当前版本 checkpoint/bundle 回归已通过，但不能替代这 9 项旧版本迁移执行。

完整套件覆盖主体品牌改动。最终审查补齐的两处 `data_image` 文件名守卫另用
实际守卫代码和文件系统场景验证：禁用时拒绝残留的 Windows 新库、Unix 新库名
及旧库，启用时接受新库，完全缺失时正常通过。Unix `.a` 场景验证的是文件名
匹配，不代表实际 Unix 编译。未为这两处检查代码重新执行耗时的整个 Preview 矩阵。

独立执行发布工具测试时 18 项中 1 项按原规则跳过；随后完整 CTest 显式提供
Git 随附 OpenSSL 路径，`ReleaseToolUnitTests` 也已通过，包括临时测试密钥的
签名往返。此操作不为实际发布创建密钥、不签名或批准实际发布候选。

## 18. 剩余引用及原因

已按第 16 节命令重新执行全仓扫描，原始匹配及按目录的行数统计分别保存到
`E/final-naming-scan.txt`、`E/final-naming-summary.json`。扫描也包含这份报告，
因此新增匹配行不代表新增的技术接口或遗漏。

| 剩余引用类别 | 代表内容与保留原因 |
| --- | --- |
| C++ / API 兼容标识 | `namespace jyppx`、公共 include、源码目录、include guard 与宏；保持已有调用方兼容 |
| CMake / 安装兼容标识 | `project(jyppx)`、`JYPPX_*`、逻辑 target、`jyppx::`、`jyppxConfig.cmake`、`lib/cmake/jyppx`、`share/doc/jyppx` |
| 稳定协议字段 | `jyppx_version`、`jyppx.model_bundle`、`jyppx.*` telemetry、manifest、MIME 及 benchmark `kind`；不能按品牌替换 |
| 六丁炼算品牌 | 首页、帮助和版本元数据、配置摘要、打包描述及报告中的中文展示 |
| Liuding Forge 品牌 | 英文展示、文档、CPack、发布说明和 SBOM 增量 properties |
| `liuding-forge` 生成物标识 | OUTPUT_NAME、CLI、ZIP、文档命令、benchmark/fuzzer runner 和安装断言 |
| 历史名称 | JYPPX Training Framework 副标题、兼容说明、原版权署名、NOTICE、第三方许可汇总中的原技术身份及历史变更记录 |
| 测试与 fixture | 原协议 fixture 字节、兼容 CLI、拒绝旧格式的断言、同时检查新旧残留库的 glob、合成 `jyppx_fixture.a` |
| 第三方和外部项目 | 新 GitHub clone URL、当前本地 checkout 路径、LibTorch 等依赖名称与许可证原文；不虚构外部依赖改名 |
| 可能遗漏的旧品牌 | 对文档标题、CLI 调用、库/可执行文件扩展名和旧归档名作补充扫描；未发现需要进一步替换的当前品牌入口 |

源文件和内部 CMake 注释中的旧项目名作为技术/历史说明保留。手工链接旧库
文件名的外部脚本仍需迁移；仓库内的安装导出、发布验证和推荐执行路径已更新。

## 19. 限制与风险

验证环境为 Windows/MSVC；Linux/macOS 的实际编译和运行需要对应 CI。
MSVC 无 libFuzzer，因此本机不声称 fuzz runtime 通过。
本轮未单独运行覆盖率或 ASan/UBSan/TSan 运行时；相关脚本单测通过不代表
带这些编译插桩的构建已经验证。
13 项条件跳过的实际覆盖缺口如第 17 节所列，需在具备符号链接权限及精确
LibTorch 2.5.1 的环境补跑；没有改变跳过条件或降低断言。
工具链存在 LibTorch 原有 CUDA architecture/kineto 提示以及依赖头的编译警告，
品牌调整不修改这些依赖或平台支持承诺。商标动态检索及法律确认仍待完成。
原有 `dependency-manifest.json` 的 draft 状态未改变；本地 CPack ZIP 验证不等于
已完成依赖审计、签名或 GA 发布审批。

接手日志中的两个崩溃恢复测试在深路径下失败；相同既有二进制在浅目录下
完整通过 4 项/8 个 storage 场景和 3 项 CAS 场景，确认是 Windows 路径长度
约束。测试暂存目录改到构建根下的浅层目录，仍保留 Unicode、真实进程崩溃、
数据库和所有断言。

接手时失败的句柄继承测试另有 fixture 问题：`CompareObjectHandles` 由
`KernelBase` 导出，本机 `Kernel32` 不导出该函数。仅修正 Python fixture 的
DLL 解析后，相同既有测试二进制的该项测试通过。未改变侧车产品实现、
句柄隔离要求或断言。

## 20. Git 与外部状态

本次提交并推送到 `git@github.com:guojin-yan/liuding-forge.git` 的
`improvement/DEV-001-extreme-condition-tests` 分支；没有合并或重写历史，
也没有发布安装包。推送前 HEAD 为 `de2f3160a5bc4bf5a234a78361b0ec3b01bc9383`，
接手时的 42 个修改文件均保留，另加入品牌迁移报告和仓库链接更新。推送后的
提交号、远端跟踪关系和工作区状态记录在 E 下的 `post-push-status.txt`、
`post-push-remotes.txt` 和 `post-push-log.txt`。受保护的公共头文件、实现、
协议 fixture、兼容清单、组件依赖清单和依赖 manifest 没有相对推送前 HEAD 的
额外差异；构建、打包和验证产物仍只在本地忽略目录中。
