# 资源基准采样

`run_resource_baseline.py` 调用实际生产 API，默认每个 profile 启动一个预热进程和五个测量进程。runner 不配置、不构建、不运行 CTest，也不连接远程主机。先在现有 Release 构建中启用 `JYPPX_LIUDING_BUILD_BENCHMARKS=ON`；模型和 Serving 分别还需 `JYPPX_LIUDING_BUILD_MODELS=ON`、`JYPPX_LIUDING_BUILD_SERVING=ON`。可选组件由使用方显式链接，不改变 SDK 能力声明。

提交后用原有依赖和选项重新配置，使生成的 fingerprint 记录准确的干净提交，再以最多两个编译任务构建。runner 拒绝过期 fingerprint 和 `+dirty` 二进制。

```powershell
cmake --build out/build/config-windows --config Release --parallel 2 --target jyppx_bench_resource_stable jyppx_bench_resource_models jyppx_bench_resource_serving
python scripts/benchmarks/run_resource_baseline.py --build-dir out/build/config-windows --storage-environment-id windows-workspace-volume-defaults --output out/baselines/resources/windows-current
```

```sh
cmake --build out/build/cuda-release --config Release --parallel 2 --target jyppx_bench_resource_stable jyppx_bench_resource_models jyppx_bench_resource_serving
python3 scripts/benchmarks/run_resource_baseline.py --build-dir out/build/cuda-release --device cuda:0 --storage-environment-id ubuntu-workspace-mount-defaults --output out/baselines/resources/cuda-current
```

默认选择要求上述三个可执行程序，文件名为 `liuding-forge-bench-resource-{stable,models,serving}`，Windows 加 `.exe`。Core 构建可选 `--profiles storage checkpoint telemetry`，`--bin-dir` 可指定已知可执行文件目录。输出目录必须新建，使用被忽略的证据目录以保持源码干净。

| Profile | 固定负载与证据 |
| --- | --- |
| `storage` | 八个不同的 256 KiB CAS blob，SQLite FULL 发布、元数据重开、精确读取与完整性校验；发布/读取耗时、载荷和总磁盘字节 |
| `checkpoint` | 四个不同的 1 MiB 同步 checkpoint；包含序列化、hash、发布的平均保存暂停、恢复耗时、精确恢复字节 |
| `telemetry` | 256 个固定 envelope，16 轮编码，JSONL emit/stream flush，重开与重复重放；编码/写入/恢复耗时和输出字节 |
| `resnet18`、`resnet50`、`mobilenet_v3_small`、`mobilenet_v3_large` | 随机初始化 FP32 模型，七分类，`[2,3,32,32]`，eval，八次预热、256 次 forward，工作负载版本 `v2` |
| `transformer_encoder` | 两层、维度 32、四个 head、FFN 64、dropout 0、GELU/final norm；`[2,16,32]`，八次预热、256 次 forward，工作负载版本 `v2` |
| `serving` | 生成的 Linear bundle，NativeStateBackend 加载/预热，128 次 Predictor 调用、32 轮各八个 DynamicBatcher 请求；精确输出与实际合批计数 |

负载均为有界合成数据。模型 `load_seconds` 包含架构构建、随机初始化和设备传输，不代表预训练权重加载。输入和初始化种子固定，验证形状、dtype、设备、有限值与 eval 可重复性。forward 输出在循环期间保持存活并计入内存。Serving 测量进程内接口及调度，不含 HTTP。

模型 `v2` 延长固定预热和计量区间，以减少短负载的启动波动；描述符、摘要和环境键与原 `v1` 分离，原始测量须保留，两种版本不能共用 reference。Storage、checkpoint、telemetry 与 Serving 继续使用各自的 `v1`。

Storage、checkpoint 和 telemetry 必须提供 `--storage-environment-id`，标识实际卷、文件系统、挂载选项及持久化设置；同一存储配置的新目录沿用该 ID，改变配置则换 ID。ID 由 1..256 个字母、数字或 `_ . : / + -` 组成，首字符为字母或数字，无空格。runner 只比较操作者声明，不探测或验证底层设备；复用错误 ID 会破坏比较有效性。仅模型或 Serving 的采样不要求此 ID。

`--device cuda:0` 使用所选 LibTorch 的可见设备零运行模型和 Serving，CUDA 与 ROCm 均沿用这个设备名，各阶段前后同步。CPU 二进制拒绝 GPU。Storage、checkpoint 和 telemetry 始终在 CPU 执行。GPU 模型和 Serving 可执行文件提供独立 `--identity` 查询，runner 在各 profile 前后及每个样本核对原生设备、构建和线程身份。

CUDA 报告绑定 UUID、名称、compute capability、toolkit/runtime/driver API 版本和加载前开始统计的 `peak_cuda_allocated_bytes` / `peak_cuda_reserved_bytes`；用 `nvidia-smi` 验证 UUID 和显示驱动，不假定可见索引等于物理索引。

ROCm 报告使用真实 HIP UUID、PCI、gfx 架构、编译及运行时/API 版本、设备内存和 HIP 同步，显存指标为 `peak_rocm_allocated_bytes` / `peak_rocm_reserved_bytes`，不混入 CUDA 测量字段。runner 另记录 OS、CPU 和 Windows 显示适配器驱动列表；HIP driver API 不等于显示驱动版本，适配器列表不建立 HIP UUID 映射，也不调用 `nvidia-smi`。配置和运行环境按 [AMD 指南](../../docs/user-guide/amd_rocm.md)准备，例如：

```powershell
python scripts/benchmarks/run_resource_baseline.py --build-dir out/build/rocm `
  --device cuda:0 --profiles resnet18 resnet50 mobilenet_v3_small mobilenet_v3_large transformer_encoder serving `
  --output out/baselines/resources/rocm-current
```

CUDA 比较还要求可读的功率限额和 persistence mode，变化会拒绝历史比较或使当前采样失效。`N/A`/`Not Supported` 保留不可用原因，此时只能收集独立稳定性证据，不能发布 reference 或通过比较；查询失败与格式错误直接失败。ROCm 当前没有合格的功耗/频率控制观测，明确记录 `comparison_eligibility` 受阻，可以保留完整原始样本和稳定性结果，不能写成可比较的 reference；它不使用 NVIDIA persistence mode 作为 AMD 条件。runner 不修改 GPU 设置。

RSS 使用 Windows 进程生命周期峰值 working set 或 Linux `VmHWM`，包含运行库开销；设备峰值仅覆盖对应 LibTorch caching allocator。所有选定 profile 均要求 RSS，实际 GPU profile 另要求对应后端的两个 allocator 峰值。缺测保留原因并使完整性检查失败；CPU 的设备指标不适用不视为失败。JSONL 只承诺 stream flush，磁盘指标不代表系统级 I/O；独立进程不主动清空 OS 文件缓存或 GPU context。

每个进程默认 120 秒超时、每路输出 1 MiB 限额，保留 `stdout.log`、`stderr.log`、`process.json`、`sample.json`、`validation.json` 及负载目录。失败仍保留可用证据，整体保留 `run.json` 与 `statistics.json`；runner 不递归删除证据。Serving 示例自行清理其生成的临时 bundle。

统计包含中位数、nearest-rank p95、均值、原始样本和 CV。规定的重复资源指标要求 CV 不超过 10%；短启动、预热和转换指标作为 `gate:false` 的诊断值，仍要求有限正数。不静默重试或丢弃测量离群值，不稳定采样不能生成 reference。

首次稳定采样可传 `--write-reference out/baselines/resources/windows-reference.json`，之后用 `--reference` 比较；reference 不覆盖已有文件。相同指标契约与环境下，回归上限为 `max(10%, 3 × reference CV)`。环境身份绑定硬件、OS、编译参数、ABI/依赖锁、设备/驱动、GPU 控制、存储声明、线程与相关环境变量、固定负载及指标可用性。源码身份另保留准确 commit/tree、干净状态、二进制与 runner/cache hash、完整 fingerprint；必须精确匹配 profile 集合。更换源码提交不改变环境键，不能通过改写历史身份替代新测量。历史二进制无需留在原路径，未签名 JSON 的结构验证也不提供防篡改认证。旧 reference 缺少必需字段时应重新采样。

runner 的集中维护测试不编译 C++，也不要求硬件：

```sh
python3 -m unittest discover -s tests/scripts -p test_resource_baseline.py
```
