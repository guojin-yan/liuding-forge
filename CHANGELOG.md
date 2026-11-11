# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- 图像训练基准接入完整 DINOv3 H+ 并报告参数元素数；Ubuntu CPU FP32/CUDA FP16 参数存储各完成 224×224 输入的两次无动量 SGD 更新。

- 增加完整 DINOv3 H+ 的 224×224 CUDA FP16 参数存储恢复显式门禁；当前 6 GiB 显卡在首次 SGD 动量分配时不足，保留多步训练与恢复未验边界。

- 模型检查点恢复将完整候选值和独立回滚副本暂存于 CPU，降低 CUDA 恢复的额外显存占用；保留原张量拓扑、完整校验和失败拷贝的逆序回滚。

- 验收 Ubuntu 上完整 DINOv3 H+ 的 224×224 CUDA FP16 参数存储前向、反向与一次 SGD 更新；显式门禁检查 520 项梯度、输入梯度及 head/patch embedding 更新、资源峰值和 cuBLAS 工作缓冲释放。

- 本地运行和搜索可显式固定 CPU 亲和性，在用户程序执行前设置并严格回读，结果保存实际放置收据；新增只读 `local cpu-affinity` 查询，重试和 worker 拒绝不可用集合，终态重放保持。

- 本地运行和搜索可显式绑定完整有效环境摘要，环境增删改在准入、重试和实际启动前拒绝；新增只返回摘要的 `local environment` 命令，原生启动复用同一份已核验快照，终态重放保持。

- 验收 Ubuntu CPU/CUDA 上 `paired-affine` 的多 trial 串行 grid/random、精确续训和完整历史备份恢复；相同设备与学习率的最终模型和检查点逐字节一致。

- 验收 Ubuntu CPU 安装 SDK 上两份不同训练权重的已签名 ONNX 版本切换与回退；并发请求及保留的租约分别核对各自版本的原生参考。

- 本地文件发布支持直接从编码字节准备持久化意图，无需额外输入文件。CIFAR-10 预览示例可用该模式完成文件、CAS 引用和事件，并在新进程中恢复已准备的预览。

- 增加原生ImageFolder训练与输出示例：显式CUDA拷贝或当前流传输接入Trainer，在SGD更新后发布预览并登记文件证据，源批次准入保持至异步证据完成，支持取消与失败释放。

- 图像批次增加显式、有界的CUDA拷贝stream与单次消费ticket；消费stream通过事件等待，源存储和预算保留到拷贝完成，目标存储登记消费stream，并可查询事件时间。CPU-only构建明确拒绝CUDA请求。

- 本地文件发布新增持久化意图和内容保留；生产者及输入文件丢失后，可发现原意图、从CAS重建最终文件并幂等提交引用与事件，覆盖冲突拒绝，复用现有GC和备份。

- 增加原生CIFAR-10有界输出示例，将整批准入保留到预览文件和原子证据登记结束；读取、变换、前向、绘图、编码及异步写入共用预算，逐批流式保存结果并核对最终资源释放。

- 本地存储新增显式 CAS 文件证据恢复：原文件丢失时核验已发布字节，在同一事务中补齐引用与事件；缺失或损坏对象、身份/大小/配额错误拒绝，保留文件与数据库独立发布边界。

- Windows图像输出在支持硬链接的卷上使用关闭时删除的暂存句柄，强杀后自动回收暂存文件，已发布文件保留并支持原身份重放；覆盖写入和不支持硬链接的卷保留普通具名暂存边界。

- 验收 Windows CPU 上已训练 ONNX 模型的并发负载、版本切换与关闭回收；明确跨平台派生包须绑定实际 Session 环境锁并重新签署和验证。

- 图像输出新增先预约容量、再调用编码生产者的原生接口，约束本地与共享队列中的生产并发；取消、关闭和生产失败释放预约，保留已有已编码提交接口。 已补齐Windows CPU路径的并发合同与独立安装消费验收；符号链接权限限制单列。

- 补齐 DINOv3 H+ 的 CPU FP32 常用分辨率前后向、SGD 更新和固定随机参考门禁；校正 H+/7B 参数计数并保留 7B 的独立局部 CLS 归一化状态。

- 完成CIFAR-10 ResNet18在CPU上的完整训练、暂停/强杀后精确恢复及安装SDK模型重载验收；新增可运行的C++模型重载示例，严格导入参数与buffer并评估完整验证集。

- 补齐已训练CIFAR-10 ONNX模型在Ubuntu CPU安装SDK的四路请求、版本发布/回退、取消/超时与关闭回收验收；逐请求独立核对输出，并以独立进程cgroup实测资源边界，临时示例签名不代表正式发布权限。

- 补齐完整CIFAR-10已训练ResNet18在Windows CPU/CUDA与Ubuntu CPU的安装SDK独立重载验收：各验证全部10000图、保持122项参数/缓冲字节，独立核对实际输入、混淆矩阵及准确率；仅授模型评估资格，不替代完整训练恢复或发布验收。

- 增加两套真实SDK的离线切换/回退维护验收：跨构建exact续训在状态修改前拒绝，切回原SDK后按完整双输入任务核模型、十个参与者、检查点及RNG精确恢复；开发版演练不自动授予正式Beta/RC资格。

- 刷新干净提交 `93da49f` 的Windows Release SDK：18库/246件安装、独立八Core消费者、选定2123项CTest分阶段（2044通过/79条件跳过）及CPU/CUDA doctor通过；一次LibTorch退出清理挂起经定向复验正常退出，原异常及其未修复边界保留；两组件矩阵、暂停传输合同与完整发布资格单列。

- 缩短构建来源生命周期测试的临时目录，恢复Windows嵌套MSBuild的Debug/Release来源验证。

- 修正Windows子进程资源测试的解释器身份，并检查被测PID；空环境子夹具不加载父测试框架，本地执行核心与资源合同分别注册，原CPU/内存断言保持。

- 显式关闭图像输出进程恢复测试中的SQLite连接，避免Windows临时目录清理时文件仍被占用。

- 为会嵌套启动原生CLI的Windows搜索成功夹具声明1GiB提交预算，避免CUDA关联DLL初始化超出原512MiB夹具预算；Linux512MiB及独立超限拒绝合同保持。

- 刷新干净提交 `9fd2763` 的 Ubuntu CPU Release SDK：18库/231件安装与独立消费者通过，完整选定2085项合同分阶段为1896通过、189条件跳过、0未解决失败；保留维护中断，2073项完成合同不重复、仅补验12项，暂停项/矩阵与其他设备/正式发布资格单列。

- 本地执行 CLI 合同按执行、环境、运行文件和主机准入分别注册，完整保留测试方法与子进程边界，并校验分组无遗漏或重复，避免不断增长的聚合测试先于具体合同超时。

- 修正 iterable 流水线生命周期测试：分别跟踪消费者批次与 EOF 的独立预约，强制验证 EOF 完成和取消两种交错；保留数据先释放、预约后归还的严格断言。

- 更新 SDK 候选范围说明，与已实现的原生 CLI、遥测、本地搜索及 ONNX/ORT 保持一致，并明确正式发布和设备资格边界。

- 覆盖率维护脚本支持显式排除测试名称正则，在元数据中记录筛选范围并使用独立的 filtered profile；原覆盖率策略保持不变。

- 修复 JPEG 在标记解析失败或缺少 scan 时仍可能返回像素的问题；畸形输入现在明确失败并释放解码资源，补充 JPEG/16 位 PNM 边界及失败恢复回归。

- 补录干净提交 `57e304c` 的 Ubuntu CPU Release SDK 验收：18 库/231 条目安装与独立消费者通过，完整选定 2076 项合同分阶段为 1887 通过、189 条件跳过、零未解决失败；保留维护中断、组件矩阵、其他设备和正式发布边界。

- 新增 Linux 完整双输入/双输出配方资源联验，覆盖 CPU/CUDA 的主机准入、逻辑预约、委派 cgroup 内存计数、本地精确续训、有界搜索与结果重放。

- 已验证 Ubuntu CUDA 的完整 CIFAR-10 ResNet18 分类配方：50 轮连续训练、独立模型重载、暂停恢复及强杀恢复通过，三条训练路线的最终模型和完整检查点逐字节一致；配置、准确率与设备边界见 compatibility.json。

- 已验证当前 Ubuntu 22.04.5 / GCC 11.4 / LibTorch 2.14.1+cu126 Release SDK：2111 项选定合同中 2033 通过、78 条件跳过，独立安装消费者及 CPU/CUDA 图像训练恢复通过；完整支持边界见 compatibility.json。

- 修正组件矩阵的嵌套构建：继承父配置的离线依赖及锁定源码，支持显式编译任务预算，并核实非法组合因指定依赖缺失而被拒绝。


### Added

- 完整双输入输出配方的资源联验扩展到 Windows CPU/CUDA：连续训练、本地精确续训和单 trial 搜索同时核验主机准入、逻辑预约及实际 Job 提交量限制，保留 Linux cgroup 原计费边界。

- 图像输出 sink 可显式等待多个 sink 共享的编码队列预算；等待支持调用方取消、关闭和错误唤醒，并在释放编码缓冲后归还额度，默认仍立即拒绝共享预算不足。

- 发布候选新增完整第三方许可原文集合及索引绑定；独立验证内容、来源、依赖状态和覆盖，历史仅含摘要的候选即使签名有效也继续阻断 GA。
- 发布维护工具可显式将安装目录内同目录文件别名实体化为普通 ZIP 条目；验证器逐件解包核验后及时清理，减少临时磁盘占用。

- 新增原生 C++ CIFAR-10 交付案例：准备真实图片 ONNX 输入、为导出包添加内存临时密钥 Ed25519 签名，并通过可信 CPU Serving 评估完整验证集；提供独立摘要绑定和任务结果文件，不持久化私钥。

- 本地执行和搜索可声明主机最低在线 CPU 数、物理总内存、可用内存及工作目录磁盘空间；原生准入与 worker 启动前复核，资源不足保持已确认重试前缀，完成态继续只读重放。

- 本地执行与搜索可用 `runtime_files` 冻结显式运行库、脚本和资源的文件身份；准入及 worker 启动前有界流式复核，文件漂移阻止新启动，已完成结果保持只读重放。

- 新增原生 C++ 行人检测示例，连接直接框预测、完整训练恢复和实际 AP 质量报告；可选独立 checker 从保存的模型输出重算解码、NMS 及 COCO AP。

- 本地训练与普通 typed 搜索可用 `training.artifacts` 固定初始模型或辅助制品的 CAS 身份；准入、worker、exact 续训和选优共同核对，原生 C++ 示例实际消费初态并将摘要纳入 checkpoint。

- 本地执行与搜索的环境引用可显式绑定值的 SHA-256，在准入及每个 worker 启动前拒绝值漂移；保留未绑定引用的原有语义和已终态任务的只读重放。

- 补齐精确 LibTorch 2.14.1+cpu / GCC 11.4 / Ubuntu Debug coverage profile 的覆盖率、真实内存边界与独立安装消费者证据，并同步兼容矩阵；保留条件跳过、组件范围与正式发布边界。

- 本地执行新增显式 `local abandon`：操作者以原 attempt 摘要和外部清理证据放弃未知结果，幂等释放预约并保留失败与未知更新数；迟到 worker、自动 retry 和搜索选优继续受围栏约束。

- 普通 typed grid/random 可显式启用有界 exact checkpoint 续训，并确认最后 attempt 的有效输入、清理及输出收据；原生 C++ 学习示例提供 20→60 步恢复，早期 attempt 的输出不参与最终选择。

- 图像输出支持显式重放相同的已有文件，按有界分块比较并同步后再次执行证据回调；拒绝不同内容、符号链接及冲突覆盖策略，可在文件发布或证据提交后进程退出时使用原身份恢复。
- Linux 图像不覆盖发布在目标文件系统支持时使用无名暂存，进程在发布前退出会由系统回收暂存内容；不支持时保持具名暂存行为，不自动删除历史残留。

- 基础组件开关矩阵顺序复用一套编译树和安装消费者，减少重复对象、依赖构建及 Windows DLL 暂存；各组合仍刷新安装目录并检查关闭隔离与非法依赖。

- 补录干净提交 `98e96c0` 的 CPU Release SDK 验收：18 库安装、独立消费者和真实配方内存合同通过，完整选定 2064 项合同分阶段闭合为 1877 通过、187 条件跳过、零未解决失败；保留首次监督中断及硬件与发布边界。

- 本地执行与搜索可显式设置原生进程树的 OS 内存上限，使用 Windows Job 提交量或 Linux 委派 cgroup v2，后端不可用时拒绝启动，并保存独立的限制与 OOM 计数收据。

- 图像批次内存、数值和几何变换保留共享 Tensor 存储或不可变目标对象的源预算租约，原批次释放后仍保持计费；完全独立的输出及时归还源预约。

- 补录干净提交 `d841b54` 的 CUDA Release SDK 验证：18 库安装、独立消费者及 CPU/CUDA 真实图像训练恢复通过，选定 2086 项合同分阶段闭合为 2014 通过、72 条件跳过、零未解决失败；保留首批证据目录冲突失败，完整模型矩阵和正式发布仍待验收。

- 流水线新增 `stage_resource_factory`，图像配方在锁页和设备复制分配前检查共享预算，保持源与目标的重叠计费；消费变换输出在替换输入前预约，失败不推进已提交位置。

- 修复 `DecodedImage` 与 `VisionSample` 整体赋值时旧像素租约的释放顺序，样本 mask 调色板复制失败时保留完整目标；两类型改为默认构造后逐字段设置，不再接受聚合初始化。

- 批次销毁、替换、预取丢弃和取消时先释放数据再归还资源租约；map 消费者直接移交 worker 值，避免变换后仍保留旧存储别名。`BatchEnvelope` 使用默认构造后设置字段。

- 安装验收消费者复用组件矩阵的编译预算：显式设置优先，其次继承环境；非法预算在刷新安装目录前拒绝，安装命令与构建并发参数分别传递。

- 补录干净提交 `5434b34` 的完整 CPU Release SDK 验证：18 库安装与独立消费者通过，选定 2008 项合同为 1823 通过、185 条件跳过、零失败；保留其他设备和正式发布边界。

- 离散 mask 在分配前计入 RGB 查表及返回调色板容量，复制 palette 独立预约并保持赋值异常安全；移除随 IDAT 数增长的块索引和动态 Adam7 pass 表。

- 图像编码在分配前计入 stb 私有堆缓冲和编码结果容量，扩容保留重叠预约，异常路径统一回收；移动、析构和异步消费保持缓冲与资源租约的释放顺序。

- 图像解码的共享资源预算纳入 stb 格式探测和私有堆缓冲，在实际分配前拒绝超限，保留扩容重叠计费及失败回收语义。

- 补录原生 CUDA ResNet18 的真实 batch=4 合批、16 种输入的响应映射、输入/输出快照及取消/身份隔离验证，保留原数值容差与单次观测边界。

- 补录 Ubuntu / RTX 2060 的原生 CUDA ResNet18 实际负载、版本切换回退、队列生命周期与 RSS/显存观测，明确单请求后端、同权重参考及单次负载的验证边界。

- 登记 CPU libgomp 离线源码伴随材料包的精确身份与发布证据用法，保留完整对应源码分发、安全和再分发审查的待审状态。

- 补录干净提交 `4953ef96` 的完整 CPU Release SDK 与安装消费者验证：选定 1997 项合同为 1812 通过、185 条件跳过、零失败，保留其他配置、设备及正式发布边界。

- 本地任务与搜索在创建存储或启动任务前拒绝显式环境目标名的 ASCII 大小写别名，避免同一计划在 Windows 和 Linux 上出现不同含义。

- 增加 OTLP 认证挑战、Cookie 与请求连接隔离检查，更新 CPU libgomp 和 oneMKL 已确认来源说明，保留完整安全及再分发审查边界。

- 补齐 BatchNorm 统计重估与 OTLP 编码、HTTP 发送选项的公共 API 参数、返回值及资源含义说明。

- 原生 BatchNorm2d 训练后统计重估支持同设备 CUDA FP32 模型与图像分类配方，显式保存并恢复 CPU/CUDA RNG；保留单遍训练数据、快照预算、取消回滚和 READY/COMPLETED 恢复语义，默认关闭。

- 补录干净提交 `570ca267` 的 Ubuntu 组件开关与安装隔离验证：两组矩阵通过，十种独立组件消费者及六轮 Preview 配置中的二十次消费者执行通过，保留完整 SDK 与设备资格边界。

- 修正离线发布指南的安装清单命令，统一中文说明，并区分实际 SDK 版本、RC 候选与正式签名验证。

- 补录重装 Ubuntu 后干净提交 `3c862070` 的完整 CPU Release SDK 与安装消费者事实：选定 1989 项合同为 1810 通过、179 条件跳过、零失败，保留原监督失败及独立设备和正式发布边界。

- 本地执行与搜索支持只保存来源名称的环境变量引用，原生启动前解析并检查缺失、名称冲突与总大小；已确认结果可在来源变量撤除后重放。

- 本地搜索在查询和恢复时复核可选 trial 的执行终态、真实指标及输入输出内容，拒绝损坏产物或与训练证据不符的缓存结果，保持已确认任务不重复执行。

- 本地执行结果保存原生 CPU 时间与峰值驻留内存，标明 Windows 直接进程和 Linux wait4 的统计范围，支持缺失观测与旧结果的原样重放。

- 绑定实际 Linux CPU LibTorch 2.14.1 内嵌 oneMKL 的版本 2024.2 与构建号 20240605，保留完整静态组成、许可和安全分发审查。

- 补录干净提交 `7b4e9d8c` 的完整 Linux CPU Release SDK 与安装消费者事实：选定 1987 项合同为 1808 通过、179 条件跳过、零失败，保留准备阶段原失败及其他配置/设备/发布边界。

- 本地配方暂停时保存绑定 checkpoint 的工作目录与训练记录快照；已确认的精确续训可从 CAS 重建丢失的目录，保留原身份并拒绝覆盖现存目录。

- 补齐 Linux CPU LibTorch 随包 libgomp 的固定官方打包脚本及 CPU 调用链，记录 `$ORIGIN` RPATH 修改，保留历史构建绑定和完整对应源码交付待办。

- 补齐 Linux CPU LibTorch 随包 libgomp 的精确官方 GCC 源包、源码与补丁身份，并验证源包及运行库包的签名；完整对应源码交付仍待完成。

- 补存 Linux CPU LibTorch 随包 libgomp 的 GPLv3 与 GCC Runtime Library Exception 原文，并核实实际运行库对应的 AlmaLinux 构建及源码版本，保留完整分发审查状态。

- 补存 Linux CPU LibTorch 2.14.1 的固定版本第三方许可原文及实际包成员来源，区分各组件条款，保留随包运行库和完整再分发审查。

- CPU ResNet50 冻结骨干迁移配方接入有界原生搜索，冻结完整权重描述与输入身份，支持精确续训、真实准确率选优及发布恢复。

- ImageFolder / ResNet18 分类配方接入有界原生搜索，支持精确续训、真实准确率选优和发布恢复。

- 多输入输出配方接入有界原生搜索，支持完整 checkpoint 精确续训、真实指标选优及发布中断恢复。

- 迁移学习配方接入本地任务、完整权重描述输入冻结和精确续训，完成态确认 checkpoint、权重及结果产物。

- 多输入输出 `paired-affine` 配方支持本地计划、拥有任务身份的原生执行、暂停后精确续训及 checkpoint、权重和结果三类产物确认。

- 补齐实际 Linux CPU LibTorch 2.14.1 的独立原包锁及对应构建提交的主许可来源，保留 CPU 2.5.1 候选和随包 notice 待审状态。

- 安装 foundation 实际构建后的配置隔离来源记录；离线发布准备和独立安全验包严格核对提交、dirty 状态、基础版本及依赖清单原始字节 SHA-256，拒绝重标旧安装目录。
- 补存两份 torchvision 参考归档中的 TensorFlow 测试资产告知，并补齐 timm 参考源码的 MIT 许可标识；原许可审查和分发门槛保持。

- 增加真实 ResNet18 完整检查点后的进程超时与 worker 强杀门禁，检查原生清理、逻辑预约释放条件及未知 attempt 的恢复围栏。

- Linux 本地 attempt 的直接子进程随持久 worker 死亡而终止；dispatch 调用者退出后 worker 仍可继续提交结果，未知 attempt 继续保留资源围栏。

- 增加已发布文件的 CAS 运行引用与制品事件联合提交接口；未提交的新增引用与事件一起回滚，提交期 I/O 错误按结果未知以原身份恢复，原有分阶段接口保持，提供原生 C++ 图像回调与重启重放示例。

- 增加真实 Collector 已接受但响应丢失的三信号门禁，覆盖指标与审计两个请求边界，明确原生未确认事件重试后的外部重复语义。

- 增加真实 Collector 队列满、原生已确认前缀保留和原字节重试验证；明确指标与审计部分成功后的重复交付语义。

- Collector 持久队列配置与崩溃恢复门禁覆盖日志、指标及其完整源审计、追踪；教学配置直接包含三个信号管线。

- 修正 DINOv2 的独立 K/V/Q 投影与确定性位置插值：CUDA 保留原生 bicubic 前向舍入，反向使用确定性系数矩阵；检查点参数布局保持兼容。

- 移除旧 Python 遥测进程入口和安装选项；TensorBoard/OTLP 导出使用原生 C++ 实现，官方 Python 映射仅保留为独立测试参考。

- 保留 Windows cu126 随包 libuv 1.40.0、zlib 1.2.3/MiniZip、CUDA/cuDNN、旧 NvToolsExt 和 Intel OpenMP 的 15 份原始许可与通知，并纳入相应 SDK 安装；明确 DINOv3 官方源码使用及后续适配事实，保留独立来源、完整运行库许可与再分发条件的待判定边界。

- 保留固定 torchvision、timm、Meta DINOv3 的许可原文与模型来源声明，并在启用 models 的 SDK 安装中附带；对支持的 LibTorch 版本安装对应 PyTorch 主许可，对 Windows 2.14.1 cu126 另附 NVIDIA 协议文本。源码来源和完整随包组件审查继续保留待核查状态。

- 兼容矩阵增加 LibTorch 2.14.1 CPU / GCC 11.4.0 / Debug coverage 的显式未验证候选，并校验待验条目不能夹带验证成绩或授予支持；补充 Windows MSVC 19.44 与 Ubuntu cu126 开发切片的精确源码、组件和验证边界，保留旧兼容事实与版本选择。

- 本地 typed 训练可声明有界输出引用，在原生成功退出且进程树清理确认后核对本 attempt 的类型、角色、实际大小与 CAS 内容；确认收据与原生结果和双流原子提交，重放只读验证。



- 增加有界原生 C++ BatchNorm2d 训练后校准，以及 CPU ResNet18 配方的显式开关、完整检查点阶段恢复和最终正常评估；只读取训练数据，不增加优化器更新。

- ORT Session 增加带类型的 CUDA Provider 配置与明确的不支持错误，设备执行资格仍需独立验证。

- ImageFolder 分类与迁移配方增加原生 C++ 零步完整模型导出，以及绑定 SHA-256、架构和有序类别的严格初始化；全部参数和缓冲区在优化器创建前导入，运行、续训与模型包保留相同来源。

- `RandomSampler` 改用版本明确的可移植整数排列算法，固定 seed/epoch 时不再受 Windows/Linux 标准库差异影响；ImageFolder 与迁移配方记录采样来源，检查点明确拒绝旧算法状态。

- ImageFolder 分类与迁移学习配方增加显式 AdamW、可选全局 L2 梯度裁剪和按成功更新线性降至零的学习率调度；优化器、参数组与调度状态随完整检查点恢复。

- ONNX Runtime 的普通推理和仅带取消令牌的调用省去未使用的截止计时等待对象；带截止时间的调用仍独立回收计时线程。

- 原生 OTLP HTTP 请求的停滞等待可被取消令牌或绝对截止时间中断，覆盖响应头、部分正文与 TLS 握手等待；未确认事件仍可原样重试。

- ImageFolder ResNet18 与迁移学习配方可显式选择逐成功优化器更新的线性预热与衰减，保持参数组学习率比例，并与模型、优化器及训练进度一起完整恢复；原逐 epoch StepLR 默认行为保留。

- OTLP C++ 教学例补齐指标、日志和实际计算 span，可先写入 SQLite run 再发送，并通过原生 CLI 重放；增加固定官方 Collector 与 protobuf reader 的独立 API/CLI 互操作验收入口。

- 线性、预热、常量和多项式学习率调度器可显式按成功的优化器更新推进，支持组合与精确恢复；基础训练示例展示纯 C++ 逐更新预热。

- 本地任务可显式启用失败 checkpoint 自动续训；恢复前核验训练输入、检查点和原进程清理证据，并提供模型、SGD 动量与步数一致的 C++ 学习流程。

- TensorBoard CLI 增加可选原生 C++ 隔离 worker，复用现有事件 writer 和进程 transport；配置模式与直接输出互斥，检查存储外输出、配置身份和协议容量，并提供完整学习步骤。

- DINOv2/DINOv3 ViT 支持选择中间 block，按请求顺序返回 CLS、patch、grid 或完整 tokens；复用骨干状态和 autograd，在最后选定层后停止，提供纯 C++ 多层特征示例。

- 本地执行可声明同机同存储的 CPU/GPU 逻辑容量与请求，在启动前持久准入；只有原生清理与结果双流均确认后释放，未知状态保留占用，系统级内存和显存硬限额继续开发。

- 扩展本地执行与遥测预览：grid/random 搜索固定每个 trial 的训练输入并支持显式恢复，增加原生 C++ OTLP CLI；Windows 进程管道采用同步非阻塞 I/O，提供冻结骨干、有界 CPU 特征缓存及分类头 SGD 更新的纯 C++ 学习示例。

- 增加可选原生 C++ OTLP HTTP 发送后端，使用系统 WinHTTP 或共享 curl，校验完整协议响应后记录进程内去重；提供指标与审计日志教学示例，强制进程终止及完整 HTTPS Collector 资格继续开发。
- 本地任务可冻结模型、配置、数据清单和显式 checkpoint 引用；扩展 C++ 教学示例，在新任务中恢复模型、SGD 动量、更新步数和输入身份后继续训练。
- 增加可选原生 C++ TensorBoard 事件文件后端与 OTLP 三信号离线编码、响应校验及教学示例。
- 增加原生 C++ MNIST 本地 gzip 准备与显式 HTTPS 下载，校验固定来源、大小、摘要和 IDX 内容；提供数据准备示例，训练示例使用本地 IDX。
- 增加可选的原生 C++ Fashion-MNIST/CIFAR-10 离线准备入口及教学示例，按固定来源和资源配额流式生成 PNG、索引与内容摘要；构建时显式配置静态 codec SDK。
- 增加固定 HF DINOv2-S 权重的原生 C++ 转换工具与 CLI，严格验证参数布局和来源摘要，生成权重及转换报告。
- 迁移学习配方增加可选的骨干和分类头独立学习率，沿用现有 SGD、调度器与完整检查点恢复，并将参数分组写入结果与模型包来源。
- 增加可选原生 C++ Ed25519 模型包签名，提供绑定原始 SHA-256 的外部信任策略、PKCS#8 私钥导入、全部声明签名校验、ONNX bundle 推理前验签，以及 CLI 与学习示例。
- 增加有界、默认串行的 grid/random 搜索，冻结候选配置、seed 与执行身份，复用持久本地任务；按真实 run 指标与 CAS 产物选择 best，并导出带完整搜索历史的存储备份。
- 增加独立可选 `onnx_serving` 组件与公共 ORT 环境锁校验，接入具名多输入输出、候选 warmup、完整图边界比较和版本 lease；Provider 显式选择，外部 Tensor 使用 CPU FP32。
- 增加原生 C++ 搜索训练与 ONNX Serving 示例，展示实际训练、最佳模型快照核对、多动态 batch 推理与旧版本排空。
- 增加本地任务 `local execute/show/recover/cancel`，用现有 run/CAS 保存冻结 spec、隔离 attempt 和双流结果，按显式退出码有限重试，无法确认进程归属时拒绝重复启动。
- 增加同 runtime、设备与 schema 的 Serving 候选校验、warmup 和原子切换，支持请求版本 lease、旧版本 drain、失败保留活动实例及双实例逻辑预算。
- 增加纯 C++ 部署和 Serving 切换学习示例，展示检查点恢复、通用 ONNX 导出、动态 batch、多输入输出核对与请求版本固定。
- 增加原生 C++ `model onnx export`：从原 run 的模型定义和检查点引用严格恢复状态，经通用导出与实际 ORT 数值/profile 验证后原子组包，保存可独立复验的输入、参考输出和来源关联；同时提供自动组包与环境锁检查。
- 增加有界模型包快照读取，直接返回同次完整校验中的模型字节；ONNX 验证命令可按模型 ID 消费 bundle，并保留原始 manifest 与实际执行证据的关联。
- 增加原生 C++ ONNX Runtime 数值验证、实际 Provider/kernel trace 汇总及 CLI 验证制品关联；数值失败保留失败报告与 run，不声明任务质量或性能资格。

### Fixed

- ResNet 分类前向及时释放不再使用的 stem 与中间级特征，减少推理的峰值显存；完整多级特征接口、反向梯度及 BatchNorm 状态保持一致。

- 图像配方搜索在创建 trial 前拒绝 CUDA BatchNorm 校准，保持搜索已声明的 CPU 校准边界；原生配方自身的 CUDA 校准入口继续可用，并增加真实训练的 OS 内存限制联验。

- 本地训练与搜索拒绝引用清理确认缺失或为 false 的旧 attempt 输入；历史终态结果仍可只读查看和重放，未声明资源的任务同样执行来源检查。

- 覆盖率采集保留普通异常计数上限，校验 SHA256 固定循环的原始控制流和精确命中次数；拒绝其他越限计数及采集期间变化的源码或计数，不截断命中或放宽模块覆盖率规则。

- 本地执行计划在创建任务存储或启动进程前拒绝保留环境变量前缀的 ASCII 大小写变体，防止 Windows 环境合并时覆盖任务身份。

- Linux 安装后的 CLI 通过相对运行库路径加载随包 ONNX Runtime，支持迁移安装目录；补齐默认构建中的本地训练输出测试夹具，官方遥测参考摘要按 Git 换行格式核对，Windows 安装消费者统一复用有所有权记录的 Torch 运行库部署，避免重复复制大型 DLL。

- 原生 OTLP HTTP 校验分块响应的 `Trailer` 字段名声明，拒绝在尾部声明内容类型、内容编码或传输长度等确认元数据；失败事件可保持原字节重试。

- 明确 Apache-2.0 声明适用于项目自有贡献，第三方材料保留各自条款；修正 Pillow 13.0.0 主许可为 MIT-CMU，并补充代码、权重、数据集与运行库的授权边界。

- SHA-256 按完整数据块处理输入，减少大检查点保存、读取和内容校验的开销；保持标准摘要、流式分块和非破坏性摘要读取语义。
- 已发布文件登记在等待 SQLite writer 或同实例锁时响应取消与绝对截止时间，失败后仍可用原引用身份重试并重放确认结果。
- 安装组件矩阵沿用根构建显式选择的 curl SDK，迁移消费者与 SDK 消费者使用独立构建缓存。
- 启用测试时默认构建 CLI 配方门禁程序；coverage owner 清单纳入 `.inc` 实现，图像 benchmark runner 测试补齐诊断策略与采样计数合同。

- coverage 的纯聚合头例外显式使用 LF 指纹，Windows CRLF 与 Linux LF 可验证为相同内容；内容改动、裸 CR 和可执行代码仍被拒绝。

- 补齐 zlib 1.3.2 完整许可原文与精确归档来源索引，独立登记图像 writer stb_image_write 1.16，并将安装 notice 替换为完整 MIT/Unlicense 原文；依赖安全和正式发布资格继续独立审查。

- Serving 的 queue 时间累计真实执行锁等待及 OOM 拆分重试调度等待，排除合批/快照/切片复制，并保留公开 OOM 异常与取消/关闭合同。

- 显式图像批次 prepare/transfer 在分配 pinned/设备输出之前预约共享预算，并按真实 CPU 别名或 CPU:0 pageable 复制分类；后端失败回滚，源与结果租约独立保留。

- 本地执行的未终态任务在进程树清理未确认时统一保留围栏，不再因未预约资源或未启用 checkpoint 续训而继续重试或确认成功。
- 本地执行的已确认结果可在旧程序或工作目录移除后按相同计划重放；新建任务与实际重试仍在启动前核验路径和可执行文件身份。

- POSIX 本地进程清理共用首次宽限期，等待主进程与原进程组全部回收后再确认释放；未回收后代继续保留未确认状态。
- 原生 OTLP HTTPS 将证书校验失败归类为 `IoError` 的 `verify_peer_certificate` 操作，保留系统 WinHTTP/curl 原因和严格信任校验。
- DINOv2 中间层提取按请求顺序创建所选输出的归一化与视图节点，保持层选择、共享状态和提前停止语义。
- Linux 私有静态签名加密库隐藏其归档符号，避免与系统 TLS 库发生符号冲突；安装消费者支持显式启用同进程签名与 TLS 回归。
- Windows 本地 CAS 统一处理长路径，覆盖目录创建、对象发布与读取、核验、临时恢复及垃圾回收，保留公开根路径身份。
- DINOv2 的 token 投影使用 ATen 融合 Linear，修正三维输入在 C++ 与独立 PyTorch 参考之间的梯度累加路径差异，保留原有参数与配置。
- 修复配方在非周期步暂停后额外增加检查点与回调计数的问题；Trainer 增加可续训的 `fit_until()` 安全边界，暂停后由调用方显式保存完整状态。
- DINOv2 在 CUDA 严格确定性模式下使用可求导的分离矩阵运算缩放位置编码，支持非原生分辨率的反向传播，保留参数梯度与 state 布局。
- CIFAR-10 离线准备接受官方类别表末尾的空行，继续严格核对十个类别及其顺序。
- 修复多 participant 检查点恢复失败时 strategy 的回滚计数检查，支持优化器先恢复或后恢复，并继续拒绝外部更新。
- SGD 检查点接受单元素维度上等价的 momentum stride，修复 DINO CLS token 使用 momentum 后无法保存训练状态的问题；参数布局、数值类型、设备和形状仍严格校验。

### Changed

- 修正 iterable 流水线组批失败和取消时的准入释放顺序：先释放本批源样本与输出存储，再归还额度，避免共享预算过早允许下一批分配。

- 将图像 writer 的来源固定到与入库 header 逐字节一致的上游提交，保留原许可和依赖安全状态。

- INT8 量化功能延期并从当前源码、构建和安装接口撤出。

- 双输入示例数据通过原生 C++ `data prepare --name paired-affine` 离线生成，保留固定样本内容和摘要；增加可直接运行的数据准备示例。
- 通用 C++ ONNX 导出生成版本化报告，记录输入输出、参数/buffer 字节摘要、构建身份和动态 batch；CLI 校验模型/报告与训练来源后，以独立关联 run 原子登记模型和报告，支持幂等重试。
- ONNX Runtime 支持按名称绑定多个 FP32 输入并返回多个具名输出，验证共享或独立 batch 符号；一次调用统一执行输入/输出总字节门禁，保留单输入输出简便接口。
- 增加 Fashion-MNIST 离线 ImageFolder 准备工具：固定来源与许可，支持完整数据和每类前 100 张子集，流式生成精确 RGB PNG，保留原始索引及内容摘要。
- 增加官方 CIFAR-10 binary 压缩包的离线校验与逐样本无损 ImageFolder 准备工具，以及完整 split 的 ResNet18 配置模板。
- 增加 ImageFolder ResNet18 分类 CLI 配方，使用现有配置解析、逐样本图像管线、任务评估与 run/CAS；支持完整训练状态的独立进程恢复，并发布显式预处理的 safetensors 模型包。
- DINOv2 增加显式位置编码插值 `position_antialias` 设置，默认沿用 timm 语义，`false` 可匹配官方 Transformers DINOv2 路径。
- 权重记录可关联显式图像预处理、来源版本和有序类别表，支持无像素分配的 resize/crop 坐标解析与绑定权重字节身份的 SHA-256；缺失输入契约明确报错，不按模型名推断。
- 图像联合几何支持显式 antialiased bicubic resize，保持离散 mask 的整数 nearest-exact；CPU/CUDA 均保留原照片 dtype，CUDA bilinear/bicubic 的半精度输入先升到 float32 计算，并纳入已知工作量预算。
- 全部视觉模型支持统一分类头查询、替换与 `num_classes=0` 的 embedding；替换时刷新传入 adapter 的 schema，保留骨干参数身份、冻结标志、设备、dtype 和模式。MobileNetV3 显式替换完整分类 MLP。
- 统一特征提取覆盖全部 33 个视觉变体，新增按配置查询的特征约定；MobileNetV3 宽度、ViT/Swin 动态尺寸与 CNN 最小尺寸均同步到输出或输入校验，保留各家族原有的尺寸和 padding 规则。
- 持久化检查点读取时，字段共享同一份已验证文件快照，省去逐字段的额外字节副本；回滚、提交失败和独立路径替换保留旧 reader 与字节视图。
- 持久化检查点提交按字段逐段写入临时文件并累计 SHA-256，省去额外的完整编码缓冲区；容器格式、大小限制及原子发布行为保持一致。
- Public branding is now 六丁炼算 (Liuding Forge), with `liuding-forge-*` library, example, benchmark, fuzzer, archive and CLI output names. The historical JYPPX Training Framework name and all `jyppx` C++/CMake/package compatibility surfaces remain available.
- The documentation index is now a concise Chinese task-based entry point; tutorials remain the detailed learning path and one-time development reports stay outside the source repository.

### Added

- 在 `deployment/` 下增加独立可选的原生 C++ ONNX 导出与 ONNX Runtime 组件：通用导出接受用户模型和 callable，经追踪、可扩展算子转换与 LibTorch serializer 生成 opset 17 自包含图，支持多输入输出、动态 batch、制品摘要与资源边界。Runtime 显式选择 Execution Provider，非 CPU 默认禁止 CPU fallback，Python 仅用于独立验证。
- 增加工具层私有的本地进程运行器：直接传递参数、显式子环境、双管道有界输出、原生退出状态及取消/超时后的进程树清理；持久 attempt、重试与搜索入口尚未提供。
- 增加显式本地预训练 ResNet50/DINOv2-S 的迁移学习配方，复用 Trainer、任务评估和检查点；支持冻结骨干或完整微调、独立 BN/dropout 策略、权重/预处理身份校验以及含来源的模型包。
- 增加双输入分类/回归 C++ 配方及有界离线数据准备：支持双输出损失与任务评估、完整训练状态的独立进程恢复、配置/内容身份校验和纯数据模型包。
- 增加独立训练诊断 JSON 报告：关联 run/attempt、模型/配置/数据身份与实际构建，保留精确计数，区分未启用、无样本和已测量，支持有界导出及既有 run/CAS 保存。
- 增加可恢复分类评估 session：固定输入身份、原子批次统计、索引游标、完成态检查点及最终报告幂等重试。
- 增加固定外部 Penn-Fudan 图像任务验证：原生 C++ 分割训练、留出集 IoU/Dice、预测掩码候选框 AP、拆批等价及持久续评；数据与权重不随仓库分发。
- 增加语义分割 IoU/Dice 和检测 AP 的评估 session：按已提交批次恢复，校验图片/像素总量及输入身份，支持完成态报告重试；检测复用原子匹配，不逐批复制历史数据集状态。
- 增加离线轴对齐框数据集 AP、紧凑匹配状态及实际推理/报告入口；明确 IoU、101 点插值、面积、每图每类 maxDet、crowd、并列与无 GT 类规则，支持来源切片、无重复图像合并及 exact 状态恢复。
- 增加可采样、有资源限额并可恢复的训练诊断：loss NaN/Inf、非有限梯度、AMP 跳步、裁剪前全局 L2 和数据读取计时；Trainer 当前检查点更新为 schema 4。
- 增加单标签语义分割 IoU/Dice、线性大小的可合并/恢复计数及实际数据集推理入口，明确 ignore、背景和空类策略，并输出类别/来源统计与质量报告观测值。
- 增加逐批实际推理的分类评估入口，输出准确率、宏平均 precision/recall/F1、逐类计数、来源切片和可接入质量报告的观测值；支持显式 ignore/空类策略、取消与逻辑资源限制，并恢复各子模块原模式。
- 增加固定官方 DINOv2-S/14 safetensors 的离线流式转换工具，记录全部源参数、Q/K/V 合并和省略的 mask token；外部实际权重验证入口覆盖 ResNet50 与 DINOv2-S，完整数据集和完整微调资格仍待验证。
- 新增 DINO 统一特征提取入口，支持查询布局/步幅/token 元数据及按顺序选择输出，复用 ModelAdapter 并保留 autograd 与共享存储。
- 补充 DINOv2 ViT-L 的 CPU FP32 完整训练状态恢复验证，使用独立资源门禁和显式回滚预算；CUDA 完整恢复仍待验证。
- 补充 DINOv3 ViT-L/L+ 的 CPU FP32/CUDA AMP 完整训练状态恢复验证，使用独立资源门禁和显式回滚预算。
- `TrainerOptions::checkpoint` 开放协调器 participant 和回滚快照预算，支持大型模型显式配置；默认仍为 1024 个 participant 和 1 GiB 回滚预算。
- 补充 DINOv3 ConvNeXt-L 的 CPU FP32/CUDA AMP 完整训练状态恢复验证；大容量恢复采用独立的资源开关和检查点 I/O 超时，默认跳过。
- 补充 DINOv3 ConvNeXt-B 的 CPU FP32/CUDA AMP 完整训练状态恢复验证，采用独立的显式高容量门禁。
- 补充 DINOv2 ViT-B、DINOv3 ViT-S+/B、ConvNeXt-S 的 CPU FP32/CUDA AMP 完整训练状态恢复验证，覆盖模型、SGD momentum、调度器、GradScaler、RNG 与数据进度。
- 图像训练基准增加 DINOv3 ViT-L/L+、ConvNeXt-Base/Large，可通过公共模型工厂进入 ImageFolder、Trainer、反向传播与 SGD 测量链。
- 图像训练基准支持 DINOv2 Large/Giant，以及 CUDA FP16 autocast/动态 GradScaler 和显式 FP16 参数存储；报告核对精度、参数 dtype 与实际 SGD 更新次数。
- Repository-local `.agents/skills` workflows for project builds and tests, model onboarding, and documentation maintenance.
- An independently exported `jyppx::liuding::interop` target with a schema-v1 checkpoint artifact manifest codec, deterministic JSON, strict field and digest validation, portable path checks, resource limits, capability metadata, and an installed-package consumer.
- An installed-package component matrix that configures, builds, links, and runs a consumer against each exported target independently.
- A legacy-surface gate that rejects retired headers, sources, package names, and build target names in both the source tree and staged installation.
- A persistent optional-component matrix that configures, builds, installs, and consumes each current component with unrelated components disabled.
- Build-level capability queries that report enabled and disabled entity components and bind the result to the current build fingerprint.

### Changed
- The remaining in-tree SDK target and project identifiers now use canonical
  `jyppx` names; the retired package name remains only in enforcement checks
  and historical release notes.
- Component options now control target creation, tests, exported package
  components, public headers, and installed libraries as one atomic surface.
- The installed CMake package now uses the canonical `jyppx` package name,
  validates requested components, and rejects unknown components. Legacy
  package and target entry points are not retained.

### Removed
- The legacy global text `Logger` and hand-encoded TensorBoard protobuf writer.
  Structured event sinks remain available; a future TensorBoard adapter must
  live in its own component and use reviewed generated schemas.
- The v1.0 single-process distributed-process-group and DDP placeholders. The
  framework will not publish distributed APIs until they provide real
  multi-process collectives, rank-safe training, and checkpoint semantics.

### Added
- A real-GPU CUDA Trainer benchmark and repeated-sampling runner covering
  FP32/single-SGD and FP16/two-SGD/AMP execution, explicit synchronization,
  correctness counters, peak allocated/reserved VRAM, environment identity,
  stability checks, and environment-bound regression references.
- Foundation-owned schema-v1 capability manifests with bounded validation,
  deterministic JSON, exact build fingerprints, and configured data-pipeline
  capability descriptors.
- Explicit `IterableDataPipeline` schema-v2 to schema-v3 checkpoint migration
  with historical exact, bounded, and best-effort fixtures, strict manifest
  compatibility, and failure-atomic source cursor replacement.
- Checkpoint-stable logical data-worker shard identities and an explicit
  caller-current CUDA transfer-stream policy with two-phase background
  preparation and consumer-thread publication.
- Current-only multi-optimizer Trainer execution with disjoint complete
  parameter ownership, canonical update order, independent scheduler cursors,
  exact coordinated recovery, and all-or-none AMP overflow handling.
- A current-only transactional `Trainer` that directly composes map/iterable
  pipelines, model/optimizer/RNG checkpoint participants, scheduler events,
  callbacks, the bounded event outbox, and a training strategy. It supports
  accumulation tail policies, validation, gradient clipping, cancellation,
  and safe-boundary checkpoint save/restore.
- A shared `BatchPipeline` contract implemented by both map-style and iterable
  data pipelines, removing data-style branching from the training loop.
- Descriptor-first `SequentialLrScheduler` and `ChainedLrScheduler` with unique
  child ownership, ordered typed descriptors, pure full-tree preflight, one
  atomic optimizer update, and exact recursive checkpoint recovery.
- A common `LrScheduler` transaction surface and generic scheduler checkpoint
  participant shared by leaf and composite schedulers.
- A descriptor-first `StepLrScheduler` with typed immutable configuration,
  stable optimizer/group identity, atomic rate control, and bounded exact
  checkpoint recovery without legacy scheduler maps or native archives.
- A narrow `OptimizerRateControl` contract implemented by the delegated SGD
  adapter so schedulers do not depend on algorithm-specific native objects.
- An independently exported `jyppx::liuding::optim` target with a LibTorch-delegated
  SGD adapter, stable group/parameter identities, fixed versioned raw state
  fields, bounded preflight, and atomic exact checkpoint restore.
- A LibTorch-delegated Adam adapter with stable groups and parameter IDs,
  long-run exact differential coverage, per-parameter moment/step recovery,
  AMSGrad state, and atomic storage-neutral checkpoint restore.
- A LibTorch-delegated AdamW adapter with decoupled weight-decay semantics,
  stable groups and parameter IDs, complete moment/step recovery, and atomic
  storage-neutral checkpoint restore.
- A LibTorch-delegated RMSprop adapter with typed centered/momentum topology,
  stable groups and parameter IDs, complete conditional state recovery, and
  atomic storage-neutral checkpoint restore.
- A LibTorch-delegated Adagrad adapter with typed decay and accumulator
  topology, stable identities, independent per-parameter step recovery, and
  atomic storage-neutral checkpoint restore.
- A current-only native NAdam/NAdamW adapter with the complete scheduled
  Nesterov update, typed stable topology, per-parameter momentum-product and
  step recovery, independent numerical coverage, and bounded atomic restore.
- A current-only native Adadelta adapter with typed rho/epsilon/coupled-decay
  semantics, two per-parameter accumulators, independent numerical coverage,
  and bounded atomic checkpoint restore.
- An exact `jyppx::liuding::nn` model checkpoint participant with versioned topology,
  bounded non-pickle tensor blobs, strict state-key/type/shape/alias preflight,
  and rollback-protected parameter and buffer restore.
- An independently exported `jyppx::liuding::training` checkpoint coordinator with a
  deterministic participant registry, canonical aggregate descriptors,
  transactional sink/source boundaries, restore preflight, bounded rollback
  snapshots, and reverse-order recovery after participant failure.
- A storage-neutral checkpoint participant contract in `jyppx::liuding::foundation`,
  including stable restore guarantees, validated state manifests, bounded named
  blob I/O, and an atomic per-participant load requirement.
- A schema-checked `ModelStateView`/`ModelAdapter` contract that separates
  shared LibTorch module state from explicit invocation, including typed
  tensor helpers and structured multi-input/output validation.
- A versioned `compatibility.json` source of truth and dependency-free
  validator. CI and benchmark workflows now read the candidate LibTorch version
  from this source instead of carrying an independent hard-coded version.
- A draft `dependency-manifest.json` with SHA-256-verified local archives and a
  validator that keeps unverified candidate downloads explicit.
- A reproducible local Release baseline runner covering clean configure/build,
  the full CTest suite, artifact sizes, throughput, optimizer convergence, and
  isolated process peak memory.
- A Linux coverage runner and unified GCC/Clang instrumentation helper covering
  framework libraries and test/benchmark executables, including UBSan support.
- An Ubuntu 22.04/GCC 11.4/LibTorch 2.5.1 CPU Debug evidence lane covering the
  full test suite, installed consumer, line/branch coverage, and ASan+UBSan.
- A strict foundation quality gate covering Google-style formatting,
  clang-tidy diagnostics, complete Doxygen documentation (including private
  declarations and enum values), and an installed-package consumer test.
- An independently exported `jyppx::liuding::nn` target, initially owning the audited
  label-smoothed cross-entropy adapter and its focused public header.
- An independently exported contrastive-loss target with strict pair-shape,
  device, dtype, finite-margin and binary-label validation; the duplicate SDK
  implementation and its compatibility surface were removed.
- An independently exported focal-loss target with explicit binary/multiclass
  shape semantics, finite alpha/gamma validation, class-index checks, and no
  duplicate SDK implementation.
- Exported `jyppx::liuding::foundation` and `jyppx::liuding::sdk` CMake targets for focused and
  full-framework consumers.
- Versioned DataLoader/sampler state for deterministic epoch-boundary resume.
- Transactional Trainer checkpoint restore covering model, optimizer,
  scheduler, AMP scaler, callbacks, RNG, loaders, and loop counters.
- Optimizer parameter-group insertion with topology validation.

### Changed
- Build fingerprints now derive CUDA LibTorch package identity from the
  official `build-version` tag when available, preventing CUDA distributions
  that omit `TORCH_CUDA_VERSION` from being mislabeled as CPU builds. Dirty
  state detection now includes untracked files.
- Dataset utilities, transforms, stress tests, examples, documentation, and
  throughput benchmarks now use the explicit-epoch `DataPipeline` contract.
  The baseline runner accepts an explicit external `LibTorchDir` so durable
  dependencies do not need to be copied into the source tree.
- Delegated SGD now preflights every gradient after an optional closure and
  rejects sparse layouts before any parameter update. CPU bfloat16, long-run
  LibTorch differential, CUDA capability, and adapter-overhead coverage are
  part of the optimizer quality matrix.
- Adam consumers now use `AdamOptimizerAdapter`; sparse gradients and invalid
  descriptors are rejected before mutation, while learning-rate control is
  exposed through the same narrow scheduler contract as delegated SGD.
- AdamW consumers now use `AdamWOptimizerAdapter`; immutable decoupled decay
  policy and complete state are represented without exposing mutable native
  options or pointer-keyed state.
- RMSprop consumers now use `RmspropOptimizerAdapter`; alpha, epsilon, coupled
  decay, momentum, and centered mode are immutable topology, and only learning
  rates remain mutable.
- Adagrad consumers now use `AdagradOptimizerAdapter`; learning-rate decay,
  coupled weight decay, initial accumulator value, and epsilon are immutable
  topology, while eager accumulators and per-parameter steps are recovered
  exactly.
- NAdam consumers now use `NAdamOptimizerAdapter`; coupled and decoupled
  weight decay, maximize mode, momentum decay, and all scheduled state are
  represented without mutable option maps or exposed pointer-keyed state.
- Adadelta consumers now use `AdadeltaOptimizerAdapter`; rho, epsilon, coupled
  weight decay, maximize mode, lazy steps, and both accumulator tensors are
  represented without mutable option maps or exposed pointer-keyed state.
- Delegated optimizers now synchronize model-owned dtype/device placement once
  during strategy preparation, then reject every placement change after native
  state or a successful step exists. This keeps CUDA Trainer checkpoints valid
  without exposing mutable parameter groups.
- Training examples, tutorials, the throughput benchmark, and the installed
  consumer now use explicit `ModelAdapter`, task step, participant, and
  `prepare()`/`fit()` contracts. Precision remains explicit task/strategy work;
the current Trainer does not retain the legacy implicit `use_amp` switch.
- Removed the replaced legacy composite scheduler implementation and its two forwarding headers; composite scheduling now has one direct public API.
- Windows tests, benchmarks, and examples now copy LibTorch runtime DLLs once per output directory, preventing parallel MSBuild copy races.
- Coverage and sanitizer options now fail clearly on unsupported compilers and reject incompatible instrumentation combinations.
- Linux coverage and sanitizer builds now bound parallelism to avoid exhausting memory on constrained runners; coverage exposes `JYPPX_LIUDING_BUILD_PARALLEL_LEVEL` for hosts that need a different budget.

### Removed
- The legacy iterator-style `DataLoader` and its private-purpose
  `ThreadSafeQueue` public header. Current consumers use `DataPipeline`
  directly; no forwarding header, target alias, or compatibility type remains.
- The hand-written `SGD` implementation and the old `optim/schedulers`
  hierarchy. Current consumers use `SgdOptimizerAdapter` and descriptor-first
  schedulers directly; no aliases or forwarding APIs remain.
- The hand-written `Adam` implementation and mutable options surface. Current
  consumers use the delegated adapter directly; no compatibility header,
  forwarding type, or legacy map checkpoint reader remains.
- The hand-written `AdamW` implementation and mutable options surface. Current
  consumers use the delegated adapter directly with no forwarding API.
- The hand-written `RMSprop` implementation and mutable options surface.
  Current consumers use the delegated adapter directly with no compatibility
  header, forwarding type, or legacy map checkpoint reader.
- The hand-written `Adagrad` implementation and mutable options surface.
  Current consumers use the delegated adapter directly with no compatibility
  header, forwarding type, or legacy map checkpoint reader.
- The incomplete hand-written `NAdam` implementation and mutable options
  surface. The native adapter is a direct replacement with no forwarding
  header, legacy map checkpoint reader, or compatibility alias.
- The legacy Trainer, mutable callback aggregate, callback implementations,
  compatibility headers, old callback tests, and the duplicate complete
  training example. No forwarding API or legacy state reader is retained.
- Foundation headers now define fully documented error, RNG, version, and
  build-fingerprint contracts; backend failures encountered while collecting
  the fingerprint are translated into the stable JYPPX error taxonomy with
  their original cause preserved.
- Removed the legacy exported CMake target names instead of retaining aliases
  or forwarding compatibility targets.
- Project licensing changed from MIT to Apache-2.0-only for the current and
  future source tree. Previously published MIT-licensed versions retain their
  original license.
- Checkpoint, optimizer, and DataLoader state readers now require their current
  explicit format instead of retaining unversioned compatibility branches.
- `AutocastGuard` validates device/dtype pairs and restores nested dtype state;
  Trainer supports CPU BFloat16 autocast without a loss scaler.
- Weighted sampling without replacement now uses a bounded algorithm and
  rejects input sets that cannot produce the requested draw.
- `Trainer` now steps metric-driven schedulers with the validation loss and
  rejects them when no validation loader is configured.
- Trainer checkpoints now preserve scheduler state, AMP `GradScaler` state,
  and the last train/validation loss values used by callbacks.
- Label-smoothed cross entropy now delegates to LibTorch for unbatched and
  spatial inputs, accepts the documented `smoothing = 1` boundary, reports
  structured JYPPX configuration errors, and has no duplicate SDK-owned
  implementation.
- The duplicate JYPPX triplet-margin wrapper was removed; callers use the
  complete LibTorch API directly, including its `p`, `eps` and `swap` options.
- The legacy `nn/losses.hpp` aggregate, public `apply_reduction` helper and
  policy-heavy `distillation_loss` were removed. The distillation example now
  states its hard/soft composition, teacher-gradient, reduction and dtype
  policy directly as a local training recipe.
- The duplicate `jyppx::liuding::nn::init` wrappers were removed; examples now use the
  complete typed LibTorch initialization API directly.
- The legacy `nn/quantization.hpp` storage-only prototype was removed from the
  default SDK. A future optional `jyppx::liuding::quantization` component must start
  from versioned formats, safe resource limits and separately proven storage,
  training and compute capabilities.
- `LRSchedulerCallback` follows the same metric-aware stepping rules as the
  Trainer loop.
- Release test coverage increased to 249 passing tests.

## [0.3.0] - 2026-10-11

Plan-completion release: closes every remaining Phase 1–4 item from the
project plan and lays the Phase 5 groundwork. 240 unit/integration tests
pass in Release and Debug configurations.

### Added
- **Plan-completion sweep** (closing every remaining Phase 1–4 item):
  - Callbacks: `LoggingCallback` (per-batch/epoch logging through the
    framework logger with optional log file), `LRMonitor` (per-epoch learning
    rate log), `TensorBoardCallback` (batch loss, epoch losses, learning
    rates). `on_train_batch_end` now receives the batch loss.
  - `utils::ProgressBar` (single-line console bar with description/postfix)
    and `utils::ProfilerReport`/`ScopedTimer` (named-scope wall-clock
    aggregation).
  - Metrics: `Precision`, `Recall`, `F1Score` (macro-averaged over the
    accumulated confusion matrix, lazily growing class count).
  - Samplers: `WeightedRandomSampler` (with/without replacement),
    `SubsetRandomSampler`, `DistributedSampler` (epoch-based shuffling,
    even padding, drop_last). Datasets: `ChainDataset`, `StackDataset`
    (variadic tuple zipping).
  - Transforms: `Pad`, `RandomRotation`, `RandomAffine` (grid_sample-based
    rotate/scale/translate), `Grayscale`/`RandomGrayscale` (ITU-R 601-2
    luma), `ColorJitter` (brightness/contrast/saturation/hue in random
    order).
  - Optimizers: `NAdam` (Nesterov momentum with the official mu schedule;
    `momentum_decay > 0` not supported) and `Lamb` (trust-ratio update with
    the reference weight-norm clamp).
  - TensorBoard: `utils::TensorBoardWriter` writes real event files
    (TFRecord framing with masked CRC32C, hand-encoded Event/Summary
    protobuf messages; scalars + text).
  - Model analysis: `utils::calculate_model_stats`/`model_summary`
    (Keras-style table) — plus a torch 2.14 API note: `named_children()`
    returns by value and must be kept alive while iterating `items()`.
  - Phase 5 groundwork: `jyppx::liuding::distributed` single-process process group
    (init/rank/world_size/barrier + `average_gradients`), a hook-based
    `DistributedDataParallel<ModuleT>` wrapper (no-op with world_size 1;
    cross-process all-reduce requires a c10d backend build),
    `functional::distillation_loss` (temperature-scaled KL), and
    `nn::quantization` weight-only int8 helpers (`quantize_symmetric`,
    `dequantize`, `QuantizedLinear`).
- **Examples**: `examples/mnist/mnist.cpp` — the plan's MNIST acceptance
  example with a self-contained gzip (RFC 1952) decoder and IDX reader
  (auto-download), CNN training through the Trainer with a 2-worker loader
  and TensorBoard logging (reaches 98.6% test accuracy in 3 epochs), and
  `examples/complete_training.cpp` — the full-featured flow (random split,
AdamW + cosine schedule, all callbacks, metrics, checkpoint best-model restore; 99.5% validation accuracy).

## [0.2.0] - 2026-10-11

First feature-complete release of the training framework: data loading (single- and multi-threaded), optimizers and schedulers, the Trainer with callbacks and checkpointing, mixed precision, transforms, losses, initialization, documentation and benchmarks. 195 unit/integration tests pass in Release and Debug configurations.

### Added
- **Loss functions** (`include/jyppx/liuding/nn/losses.hpp`): functional implementations in `jyppx::liuding::nn::functional` — `focal_loss` (multiclass cross-entropy path and binary sigmoid path, alpha weighting, gamma focusing), `dice_loss` (soft Dice with one-hot multiclass and binary paths, fully differentiable tensor ops), `label_smoothing_cross_entropy` (verified equal to torch's `label_smoothing` option), `contrastive_loss` (Euclidean pairs with margin hinge) and `triplet_margin_loss` (delegates to ATen for exact parity). Criterion modules `nn::FocalLoss`, `nn::DiceLoss`,
  `nn::LabelSmoothingCrossEntropy`, `nn::ContrastiveLoss`, `nn::TripletLoss`
  wrap the functions for forward-style use. 20 tests cover hand-computed
  values (1e-12), torch parity anchors, gradient flow, reduction semantics
  and argument validation.
- **Data transforms** (`jyppx::liuding::data::transforms`, header-only):
  `Transform<I, O>` base + `TensorTransform` alias, `Compose` (ordered
chaining with descriptive `name()`), `RandomChoice`, and six vision transforms — `Normalize` (per-channel, promotes integral inputs),
  `Resize` (bilinear/nearest, square or `{h, w}`), `CenterCrop`,
  `RandomCrop`, `RandomHorizontalFlip`, `RandomVerticalFlip`. Random
transforms draw from the global torch RNG (reproducible via
  `torch::manual_seed`, safe under the multi-worker DataLoader). 15 tests
  cover ordering, exact values, fan-free crop geometry, seeded
  determinism, probability bounds and DataLoader integration.
- **DataLoader benchmark** (`tests/benchmarks/bench_dataloader.cpp`, built
  with `JYPPX_LIUDING_BUILD_BENCHMARKS=ON`): a self-contained harness (no Google
  Benchmark dependency) measuring per-epoch loading time for 0/1/2/4/8
  workers over a synthetic dataset whose `get()` burns a configurable
  amount of CPU per sample (simulating decode/preprocessing cost), plus a
  one-epoch training-throughput reference line. Measured on the
  development machine (16 hardware threads, 2 ms/sample, 1024 samples):
  2 workers 2.0x, 4 workers 3.9x, 8 workers 7.1x speedup over the
  single-threaded loader — exceeding the plan's 2.5x/3.5x targets with
  near-linear scaling.
- **Documentation**: a docs tree (`docs/index.md`) with four API reference
  pages (`data`, `optim`, `training`, `utils`) and five tutorials (basic
  training, data loading, optimizers and schedulers, callbacks and
  checkpoints, mixed precision). Every code snippet was extracted and
  compile/run-checked, which fixed three reader-facing issues in the
  process: the README example fed int64 targets to the default MSE loss and
  used an invalid `make_shared<torch::nn::Linear>` construction, the
  ReduceLROnPlateau snippet relied on braced-init-list forwarding that
  `make_shared` does not support, and a TrainerOptions snippet used
out-of-order designated initializers (rejected by MSVC).
- **`nn::init` parameter initialization** (all 9 functions declared in the header are now implemented, delegating to `torch::nn::init` for exact PyTorch semantics): `uniform_`, `normal_`, `constant_`, `zeros_`, `ones_`,
  `xavier_uniform_`, `xavier_normal_`, `kaiming_uniform_`,
  `kaiming_normal_` (string-based `mode` = "fan_in"/"fan_out"; leaky_relu
gain). 13 tests cover seeded RNG parity with `torch::nn::init`, fan-based bounds/statistics for 2D and 4D weights, gain handling and argument validation. Platform notes: torch 2.14 C++ requires dim >= 2 for fan-based init (1D tensors are rejected, unlike Python) and the
  `NonlinearityType`/`FanModeType` parameters are `std::variant` enums.
- **Dataset utilities**:
  - `data::Subset<T>`: index-restricted view over an existing dataset.
  - `data::ConcatDataset<T>`: contiguous concatenation of same-typed
    datasets with cumulative-size indexing.
  - `data::random_split(dataset, lengths, seed)`: seeded train/validation
    splitting (mirrors `torch.utils.data.random_split`), returning
    heap-allocated subsets ready for DataLoader consumption.
- **Optimizers** (all validated numerically against `torch::optim`):
  - `optim::RMSprop` (alpha, eps, weight decay, momentum, centered).
    NOTE: momentum follows the `torch::optim` C++ semantics — the buffer
    accumulates the normalized gradient `grad / avg` and the step is
    `-lr * buffer`; this differs from `torch.optim.rmsprop` in Python
    (raw gradient in the buffer, divided by avg at update time) starting
    from the second step.
  - `optim::Adagrad` (lr_decay, weight_decay, per-parameter step counting).
- **Schedulers**:
  - `MultiStepLR`: decays by gamma at each milestone.
  - `ReduceLROnPlateau`: metric-driven (`step(metric)`) with mode, factor,
    patience, rel/abs threshold and min_lr; metric-less schedulers reject
    `step(metric)` and vice versa.
- **Metrics**:
  - `TopKAccuracy` (top-k hit rate over {batch, classes} scores),
    `MSE`, `MAE`, `R2Score` (exact whole-set coefficient of determination),
    `AverageMeter` (weighted running average), `MetricCollection`
    (named bundle with joint update/compute/reset).
- **Multi-threaded DataLoader** (`num_workers > 0`):
  - `jyppx::liuding::data::ThreadSafeQueue<T>`: bounded blocking queue for
    producer/consumer pipelines (blocking `put`/`get`, non-blocking
    `try_get`, close semantics that keep already-queued items retrievable).
  - Worker pool inside `DataLoader`: workers pull index batches, load and
    collate them, and push results through a sequence-tagged pipeline.
    Results are returned in submission order (identical to single-threaded
    loading), worker exceptions are rethrown on the consuming thread when
    their sequence is reached, and prefetching bounds the number of
    in-flight batches (`num_workers * prefetch_factor`).
  - Workers restart per epoch so a partially consumed epoch never leaks
    into the next one; the destructor drains in-flight work.
  - `DataLoaderOptions::prefetch_factor` (default 2).
  - Note: the seeded first-epoch shuffle no longer consumes an extra
    generator draw at construction, so a seeded loader's first epoch now
    matches its seed directly.
  - 16 new tests: queue semantics (ordering, blocking, close, move-only
    elements), multi-worker output equivalence, permutations, custom
    collate, exception propagation, early-break reuse, 20-epoch stress,
    and Trainer fit integration.
- **AMP (mixed-precision training)**:
  - `jyppx::liuding::training::amp::GradScaler` mirroring `torch.cuda.amp.GradScaler`
    semantics: loss scaling, gradient unscaling with inf/NaN detection, step
    skipping with scale backoff on overflow, scale growth after
    `growth_interval` clean steps, and a serializable state dict.
  - `jyppx::liuding::training::amp::AutocastGuard`, an RAII wrapper around the ATen
    autocast thread-local state (device-type/dtype scoped, restores prior
    state on destruction).
  - `Trainer` integration: `use_amp = true` (CUDA only, validated at
    construction) runs forward passes under autocast, scales the loss before
    backward, unscales before gradient clipping, and routes optimizer steps
    through the scaler (skipping steps that overflowed). Works together with
    gradient accumulation and gradient clipping.
- GoogleTest/CTest unit test infrastructure. `tests/CMakeLists.txt` fetches
  GoogleTest 1.14.0 from the local `third_party/downloads` cache (network
  fallback included) and registers 240 tests via `gtest_discover_tests`
  (PRE_TEST discovery). Suites cover:
  - `core`: device parsing, NaN/Inf helpers, tensor clone/device moves
  - `utils`: Logger levels, file output, unwritable path handling
  - `data`: TensorDataset, Sequential/Random/BatchSampler, DataLoader
    (ordering, seeded shuffling, drop_last, custom collate, tensor batching)
  - `optim`: SGD/Adam/AdamW validated numerically against `torch::optim`
    (momentum, dampening, weight decay, Nesterov, AMSGrad), state_dict
    round trips, gradient zeroing and learning-rate accessors, StepLR /
    ExponentialLR / CosineAnnealingLR trajectories
  - `training`: Trainer end-to-end (fit/validate/test, max_steps, gradient
    accumulation equivalence, gradient clipping, checkpoint round trip and
    resume), callback lifecycle ordering, EarlyStopping, ModelCheckpoint,
    Accuracy
- Shared test helpers in `tests/test_utils/jyppx_test_helpers.hpp`
  (deterministic linear model/data, leaf-tensor clones, torch::optim parity
  harness, callback event recorder).

### Fixed
- Adam / AdamW now match PyTorch's update exactly: eps is added to the
  bias-corrected denominator (`sqrt(v) / sqrt(bias_correction2) + eps` with
  `step_size = lr / bias_correction1`). The previous paper-style formula
(`sqrt(v) + eps` before bias correction) diverged from `torch::optim` by ~1e-7 per step.
- `Optimizer::state_dict()` serializes parameter-group options as float64.
  `torch::tensor(double)` defaults to float32, which corrupted lr/momentum
  values in checkpoint round trips.
- Trainer: the delegating (concrete-module) constructor no longer throws before
  it can install the type-erased forward function. The `forward_fn` check moved
  from `validate_configuration()` to the `fit`/`validate`/`test` entry points,
  so a Trainer built from a base `torch::nn::Module` plus `set_forward_fn()`
  is now a valid workflow. Previously this surfaced as an uncaught
  `std::invalid_argument` (Windows: silent crash `0xC0000409`).
- MSVC Debug builds: use the release CRT (`/MD` via
  `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL`) and
  `_ITERATOR_DEBUG_LEVEL=0` in every configuration. LibTorch's official
Windows binaries are Release-built with `/MD`; mixing them with a Debug runtime corrupted the heap across the DLL boundary (access violations,
  `STATUS_HEAP_CORRUPTION`) and mismatched STL layouts (segfaults inside
  `register_module`).
- Fixed MSVC Debug/Release flag handling for multi-config generators: per-config compiler flags in `cmake/CompilerOptions.cmake` now use generator expressions instead of `CMAKE_BUILD_TYPE` checks (previously building the Debug config failed with D8016 `/O2` vs `/RTC1`).
- Examples: the LibTorch POST_BUILD copy now copies only `*.dll` (`copy_if_different`) instead of the whole `lib` directory, which also contained large import libraries.

### Verified
- `ctest -C Release` and `ctest -C Debug`: 240/240 tests pass in both configurations. The CUDA AMP integration test skips gracefully when
  `torch::cuda::is_available()` is false; on this workstation the installed
  driver (576.02, up to CUDA 12.9) cannot run the cu132 LibTorch binaries, so
  the GPU path is compile-checked and unit-covered but not yet exercised
  end-to-end.
- `examples/basic_trainer`, `examples/basic_dataloader`,
  `examples/complete_training` (99.5% val accuracy),
  `examples/mnist` (98.6% test accuracy on the real dataset) and
  `examples/distillation` (teacher transfer into the label-scarce student)
run cleanly in both Release and Debug configurations.
- The installed CMake package was exercised by an out-of-tree consumer via
  `find_package(jyppx_liuding_training_framework CONFIG REQUIRED)` (build, link, run).
- DataLoader multi-worker scaling measured with the bundled benchmark:
  2 workers 2.0x, 4 workers 3.9x, 8 workers 7.1x over the single-threaded
  loader on the development machine.

## [0.1.0] - 2025-10-10

### Added
- Project initialization
- Basic architecture and build system
