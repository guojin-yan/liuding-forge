# 教程 2：数据加载与断点恢复

本教程使用六丁炼算的显式 epoch `DataPipeline`，将 Tensor 数据集组装为带 schema、样本 ID、epoch 和批次序号的输入。完成后可以运行串行组批、双 worker 预取，以及退出进程后从已提交批次继续读取。所有示例均为原生 C++，不需要 Python、下载数据或训练模型。

## 1. 构建完整示例

先准备与编译器、LibTorch 匹配的已安装 SDK，具体组合见[构建说明](../../CONTRIBUTING.md#构建与验证)。在仓库根目录运行以下命令，将路径替换为实际 SDK 和 LibTorch；复用兼容的构建目录，不要把已有其他源码入口的 CMake cache 改成此入口：

```sh
cmake -S examples/data -B out/build/data-examples -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/path/to/sdk;/path/to/libtorch"
cmake --build out/build/data-examples --config Release --target jyppx_basic_data_pipeline --parallel 1
ctest --test-dir out/build/data-examples -C Release --output-on-failure --parallel 1
```

此独立 CMake 工程只要求 `data` 和 `interop` 组件。源码集中在 [examples/data/basic_data_pipeline.cpp](../../examples/data/basic_data_pipeline.cpp)。Linux 单配置可执行文件位于 `out/build/data-examples/`，Windows 多配置位于其 `Release/` 子目录；Windows 还需将匹配的 LibTorch DLL 目录加入 `PATH`，Linux 按 SDK 安装方式配置运行库搜索路径。

## 2. 从 TensorDataset 到批次

示例创建 10 条 CPU 样本，样本 `i` 的输入为两个 float32 特征 `[i, i+10]`，int64 标签为 `i % 3`。`TensorDataset::get()` 返回含 `data`、`target` 的 `Example`；`RandomSampler(10, 7)` 决定样本 ID 顺序，batch factory 读取样本后分别 `torch::stack`，返回二元 tuple。

`batch_schema()` 声明输入为 `[动态批长,2]` 的 float32 CPU Tensor，标签为 `[动态批长]` 的 int64 CPU Tensor，并标明 `features` 和 `target` 语义。动态批长允许最后一个不完整批次；有已知结构时应使用这种明确 schema，使错误 dtype、设备或形状在批次提交前暴露。`BatchSchema::any` 适合确实无法约束结构的值，不负责验证这些 Tensor 契约。

在可执行文件所在目录运行：

```sh
./liuding-forge-example-basic-data-pipeline inline
./liuding-forge-example-basic-data-pipeline workers
```

两次均显示 3 批、10 条样本，批长依次为 4、4、2；每批打印实际样本 ID。默认不带参数也运行 `inline`。程序逐样本检查输入和标签与 ID 对应，检查整轮 ID 无遗漏、无重复。`set_epoch(3)` 只选择这一轮，耗尽后 `next()` 返回空值，不会隐式开始下一轮；设置 `drop_last=true` 才会丢弃不足一批的尾部。

## 3. 启用 worker 预取

`make_pipeline()` 中的关键配置如下；`workers` 为 true 时使用两个进程内线程，否则在消费者线程组批：

```cpp
      .batch_size = 4,
      .num_workers = workers ? 2U : 0U,
      .prefetch_batches = 3,
      .persistent_workers = workers,
      .worker_seed = 7};
```

`prefetch_batches` 限制已调度但尚未提交的批次数，不是字节数或整个进程的内存上限。worker 可乱序完成，消费者仍按 sampler 顺序取得批次。此例只并发读取不变 Tensor，factory 在样本循环中检查 stop token，`close()` 会停止并回收 worker；该实现不是进程 worker。

自定义数据集应继承 `Dataset<T>`，提供 `get(index)` 和 `size()`，明确拒绝越界，且避免 worker 修改共享数据。factory 的捕获对象必须可安全并发访问；随机变换须从 `DataWorkerContext` 的稳定 epoch、sequence、worker_seed 或显式样本流派生，不能依赖可变全局 RNG。`persistent_workers` 只决定线程是否跨 epoch 保留，调用 `close()` 后仍会回收。

## 4. 退出进程后恢复

在同一可执行文件目录依次运行下列命令。每一行都会启动一个新进程，状态文件放在可写目录：

```sh
./liuding-forge-example-basic-data-pipeline save ./pipeline-state/workers.bin workers
./liuding-forge-example-basic-data-pipeline restore ./pipeline-state/workers.bin workers
./liuding-forge-example-basic-data-pipeline failure ./pipeline-state/workers.bin workers
```

`save` 消费第一批后输出 `saved cursor=4 sequence=1`，仅保存已经提交的 4 条样本。核心保存步骤为：

```cpp
  interop::FileStateArchive archive(path, archive_options(false));
  archive.begin({pipeline->describe_state()});
  pipeline->save_state(archive.writer_for(pipeline->participant_id()));
  archive.commit();
```

`restore` 用只读 `FileStateArchive` 打开完整性校验后的文件，以 `RestorePolicy::exact_only` 加载。它重建未提交的预取工作，读取余下 6 条样本，逐批比较样本 ID、序号、epoch 和全部 Tensor 值与连续执行的后缀，最后输出 `restored batches=2 samples=6 tail=2 exact=true`。将三条命令末尾及文件名改为 `inline`，可体验相同流程的串行版本。

`failure` 故意让下一批 factory 抛出异常；消费者收到 `BackendError`，已提交边界仍为 cursor 4、sequence 1。示例关闭故障开关后显式重载同一检查点，再验证余下批次，输出 `failure preserved cursor=4 sequence=1`。实际应用须处理故障原因，然后显式恢复或重新选择 epoch，不能把异常当作已经消费了失败批次。

状态恢复要求 pipeline ID、数据版本、batch schema、factory ID、sampler 配置和 worker 拓扑等与保存时兼容；修改影响重放的逻辑时同步更新其身份。示例检查当前管线具备 `RestoreLevel::exact`，恢复请求也明确要求 exact。归档字节限制还必须容纳参与者声明的字段上限，因此本例设置 64MiB 制品、32MiB 单字段上限；实际 10 样本状态远小于该上限，限制不是预分配。

这里保存的只有数据参与者。`next()` 成功返回代表数据批次已经提交，不代表模型更新、优化器、回调或外部写入也一起提交；完整训练检查点需把这些参与者交给[检查点协调器](../api/checkpoint_coordinator.md)，不能单独用本例保证训练的恰好一次更新。公共管线方法由单个 owner 调用，捕获状态时暂停该 owner 的修改。

## 5. 更换数据与采样方式

`random_split(dataset, lengths, seed)` 创建可重现的子集视图；训练和验证分别创建 sampler、pipeline ID 与数据版本，避免检查点参与者身份冲突。`SequentialSampler` 用于固定顺序，`RandomSampler` 用于带种子的 epoch 排列；加权、子集和分布式采样的配置及状态约束见[数据 API](../api/data.md)。直接给管线基础 sampler，再用 `batch_size`、`drop_last` 组批；`BatchSampler` 的直接接入限制以 API 说明为准。

图像解码、标注和变换接入见[数据示例](../../examples/data/README.md)与[图像分类教程](image_classification_recipe.md)。可在 `get()` 或 factory 中执行变换，并遵守相同的并发、取消和确定性要求；设备传输后、提交前的变换见[批次变换 API](../api/batch_consumer_transform.md)。顺序流使用 `IterableDataPipeline`，按来源实际能力声明 exact、bounded 或 best-effort，不能通过选择恢复策略把较弱来源提升为 exact。
