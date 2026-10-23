# Benchmark 操作说明

本文档只说明开发人员如何构建和运行 benchmark target。历史测量、审计结论和原始结果文件统一保存在代码仓库外的方案记录中；不要把带日期的结果文件或运行日志放入 docs/。

## 构建和运行

启用测试和 benchmark 后配置构建目录，再使用当前平台的主构建目录编译 benchmark target：

~~~powershell
cmake -S . -B out/build/benchmark-release -DCMAKE_BUILD_TYPE=Release `
  -DJYPPX_LIUDING_BUILD_TESTS=ON `
  -DJYPPX_LIUDING_BUILD_BENCHMARKS=ON `
  -DCMAKE_PREFIX_PATH=third_party/install/libtorch
cmake --build out/build/benchmark-release --config Release --parallel 2
~~~

Windows 基线协议使用 scripts/benchmarks/Run-Baseline.ps1。脚本执行一次预热和多次独立采样，记录构建与环境身份，并将可替换的运行输出写入被忽略的 out/baseline/：

~~~powershell
pwsh -NoProfile -File scripts/benchmarks/Run-Baseline.ps1 `
  -BuildDir out/build/baseline-release `
  -OutputDir out/baseline/release
~~~

只有在确实需要隔离的全新构建测量时才使用 -FreshBuild；-ReferenceFile 只能指定与当前工作负载和环境匹配的 reference。runner 会拒绝不稳定样本和环境不匹配，不会把本地测量结果变成跨机器阈值。

资源 profile、存储身份、CUDA 要求、来源校验和证据布局见 scripts/benchmarks/RESOURCE_BASELINE.md。CUDA 训练和优化器套件使用 scripts/benchmarks/run_cuda_baseline.py。

## 范围

benchmark 可执行文件位于 tests/benchmarks/。逻辑 CMake target 保留 jyppx_bench_* 技术身份，生成文件使用 liuding-forge 产品前缀。benchmark 输出属于本地证据，应写入被忽略的 out/ 或 CI artifact 存储。
