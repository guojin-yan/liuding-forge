# 持久化训练检查点

启用 interop 后，`jyppx::liuding::training::FileCheckpointStore` 将训练协调器连接到 `jyppx::liuding::interop::FileStateArchive`。禁用 interop 时，安装的 training 头文件不提供该文件适配器。自定义训练循环可直接使用 `FileStateArchive` 和 foundation 的 participant、reader/writer 契约。

```cpp
#include <jyppx/liuding/training/file_checkpoint_store.hpp>

jyppx::liuding::training::FileCheckpointStore destination("checkpoints/last.bin");
coordinator.save(destination);  // 在受支持的已提交边界捕获状态。

// 先重建完整训练域，再恢复所有已登记的状态。
jyppx::liuding::training::FileCheckpointStore source("checkpoints/last.bin");
new_coordinator.restore(source, jyppx::liuding::RestorePolicy::exact_only);
```

保存与重建时必须登记所有实际参与的状态所有者：模型、优化器、调度器、精度缩放器、RNG、数据管线、指标、回调、dispatcher、事件 outbox、strategy 和 Trainer。协调器先验证完整训练域，再执行恢复；失败时回滚 participant 的修改。文件存储不会发现漏登的状态，不支持 batch 中点或未提交梯度的恢复。

## 自定义状态示例

[checkpoint_state.cpp](../../examples/training/checkpoint_state.cpp)是完整C++程序，演示自定义计数器参与者与文件存储，不执行模型训练。启用training、interop和 `JYPPX_LIUDING_BUILD_EXAMPLES` 后，在现有兼容构建目录编译 `jyppx_checkpoint_state`；以下命令以Linux Release目录为例，运行时加载路径须指向构建所用LibTorch共享库：

```sh
cmake --build out/build/rebuilt-cpu-release --target jyppx_checkpoint_state --parallel 1
out/build/rebuilt-cpu-release/examples/training/liuding-forge-example-checkpoint-state save out/checkpoints/counters.bin
out/build/rebuilt-cpu-release/examples/training/liuding-forge-example-checkpoint-state restore out/checkpoints/counters.bin
out/build/rebuilt-cpu-release/examples/training/liuding-forge-example-checkpoint-state rollback out/checkpoints/counters.bin
out/build/rebuilt-cpu-release/examples/training/liuding-forge-example-checkpoint-state budget out/checkpoints/counters.bin
```

`save`以与ID排序相反的登记顺序保存42和84；下一条命令是新进程，`restore`重建两个状态所有者并恢复42/84。`rollback`让第二个参与者在修改前拒绝84，验证第一个已载入参与者与第二个参与者都回到操作前111/222；`budget`将16字节声明总量的回滚预算设为15，验证未调用任何参与者载入。保存文件上限64KiB、每字段8字节，格式显式采用小端编码。示例状态不是模型、优化器或完整Trainer域；实际训练恢复按[训练API](training.md)登记全部状态并验证任务结果。

安装SDK用户可直接用同目录的[独立CMake工程](../../examples/training/CMakeLists.txt)编译，配置 `CMAKE_PREFIX_PATH` 指向已安装jyppx和匹配LibTorch的前缀。此方式只编译示例，不重新构建库；例如在现有构建大目录下使用一个消费者子目录：

```sh
cmake -S examples/training -B out/build/rebuilt-cpu-release/_ic/checkpoint-state/Release/b -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/path/to/jyppx;/path/to/libtorch"
cmake --build out/build/rebuilt-cpu-release/_ic/checkpoint-state/Release/b --parallel 1
ctest --test-dir out/build/rebuilt-cpu-release/_ic/checkpoint-state/Release/b --output-on-failure --parallel 1
```

该工程使用 `find_package(jyppx CONFIG REQUIRED COMPONENTS training interop)`，链接 `jyppx::liuding::training` 和 `jyppx::liuding::interop`；两个组件的开关及安装资格须与所用SDK一致。独立构建的可执行文件位于消费者构建目录；Windows还须让进程能加载匹配的LibTorch DLL，主工程构建则复用已有运行库部署规则。

`TrainerOptions::checkpoint` 配置协调器的 participant 数量和恢复回滚预算。`max_rollback_bytes` 默认 1 GiB，按恢复前所有 participant 字段声明的最大字节数合计，包含可选字段；大型模型须显式配置足够的预算，例如 `.checkpoint = {.max_rollback_bytes = 3ULL << 30U}`。不足时在修改状态前拒绝恢复。该运行资源配置不属于训练拓扑，可在恢复时调整；实际 RSS 还包括模型、已读取检查点及后端临时量。

## 格式与完整性

v1 容器包含八字节 magic、小端版本号和 manifest 长度、规范化 JSON manifest、字段内容及完整 payload 的 SHA-256 尾部。manifest 记录构建身份、participant schema/恢复等级、字段大小与摘要。字段按名称排序，通过名称和 schema 属性匹配；调用方写入顺序不构成身份。读取端在暴露 reader 前校验完整文件及每个字段的摘要、格式和限额。SHA-256 检测字节变化，来源认证需要独立的可信签名；历史 participant schema 由对应 reader 迁移。

## 发布与恢复

提交按规范顺序逐段写入同目录临时文件，同时计算完整文件摘要，不再拼接额外的完整编码缓冲区。完成写入后才原子替换目标；每个路径的操作系统锁拒绝并发 writer，reader 保留打开时的完整文件代际，并允许其他 writer 替换路径。reader 不清理活动 writer 的临时文件；锁空闲时可清理遗留临时文件，锁侧文件保留供复用。

`synchronize=true` 请求内容和目录持久化：Windows 使用 `FlushFileBuffers`、`MOVEFILE_WRITE_THROUGH`，POSIX 使用 `fsync`、`rename`，新建父目录条目也会同步。替换前失败保留旧文件。替换后目录同步失败时，调用报错，当前句柄仍保留旧 descriptor/reader；重新打开路径可检查已发布的完整代际，断电后的结果取决于文件系统保证。Windows 对共享、锁和访问冲突最多重试 500 毫秒，持续冲突时保留旧目标。只有文件或路径缺失视为空的新归档，其他打开错误仍会报错。

## 限额与只读检查

`max_field_bytes` 限制单字段，`max_artifact_bytes` 包含全部 framing 和摘要；manifest parser 限额和 `min_free_storage_bytes` 分别约束元数据及写入后的剩余空间。读取时整份制品驻留内存，字段直接引用同一份已验证快照，避免逐字段再复制。reader 和返回的字节视图有效至该对象下一次成功提交或销毁；回滚、提交失败及其他句柄修改路径均不改变已打开的快照。捕获时仍复制调用方输入，这些限额不构成进程 RSS 上限。截断、摘要不符、未知版本、缺少必需字段、锁冲突、空间不足和文件系统失败均返回结构化 foundation 错误。

检查工具可设置 `FileStateArchiveOptions::read_only=true`：必须打开已存在的有效快照，禁止捕获，不创建目录、不获取 writer 锁，也不清理临时/锁文件。CLI 的 `checkpoint inspect/verify/diff` 使用此模式。对象不具备线程安全性，捕获时须暂停 participant 修改；独立读取使用不同句柄。
