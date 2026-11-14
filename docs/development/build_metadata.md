# 构建来源与增量编译

安装后的 `jyppx/liuding/foundation/build_config.hpp` 保存语义版本和已启用组件，描述公开头文件合同。提交、工作树状态和依赖指纹存放在私有生成头 `generated-private/jyppx/liuding/foundation/build_metadata.hpp`，仅 `src/jyppx/liuding/foundation/version.cpp` 包含它；公开配置值不变时，`configure_file` 保留公开头的时间戳，避免提交变化触发全部消费者重新编译。

通过 `<jyppx/liuding/foundation/version.hpp>` 中的 `jyppx::liuding::build_info()`、`jyppx::liuding::current_build_fingerprint()` 或 `jyppx::liuding::build_fingerprint_json()` 读取提交、编译器、系统架构、LibTorch 版本与 ABI/构建变体、加速器和依赖清单 SHA-256。运行时加速器可用性仍由原接口报告；多配置构建通过私有 `JYPPX_LIUDING_BUILD_TYPE` 编译定义记录实际编译配置。

CMake 还安装 `share/jyppx/build-provenance.json`，固定五个字段：整数 `schema_version=1`、`project_version`、含可选 `+dirty` 的原始 `git_commit`、`dependency_lock_sha256` 和 `configuration`。其来源与私有头相同，配置值由 `$<CONFIG>` 展开。只有 foundation 归档成功构建后的 `POST_BUILD` 步骤才保存对应配置的记录；安装只复制该记录。重新配置而未构建时保留上次记录，Debug 构建也不会更新 Release 记录；没有成功构建记录的配置安装失败。删除记录后，仅重新配置不会补回它。

来源值取自 CMake 配置时的快照。改变提交、工作树状态或依赖清单后，重新配置同一个构建目录并增量构建；来源变化会重编 `version.cpp`、更新 foundation 并重链接相关二进制。Git 状态包括未跟踪文件；存在改动或状态查询失败时追加 `+dirty`，提交不可用时记录 `unknown`，缺少依赖清单时记录 `unlocked`。普通源码、公开版本和组件变化仍按正常依赖关系重编。

离线发布准备和独立验包均要求安装来源文件完整、结构严格，提交及 dirty 状态与发布声明一致，并核对实际依赖清单原始字节的 SHA-256；`unknown`、`unlocked`、缺失或畸形元数据均拒绝。验包在有界安全解包后重新核对安装记录；发布版本为 RC 时使用数字基础版本匹配 `project_version`，保留原有 RC 和 dirty 发布门槛。来源文件记录构建身份，独立构建证据和真实制品签名仍按[发布流程](releasing.md)完成。
