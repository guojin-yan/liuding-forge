# 变更记录

项目尚未发布。本页记录当前未发布功能及用户可见修复；源码版本以根目录 `CMakeLists.txt` 为准。

## [Unreleased]

### 训练与数据

- 修复大容量恢复测试同时驻留两份检查点快照造成的额外主存峰值；顺序校验后有界逐字节比较。WSL Huge+ FP16存储在原10 GiB RSS及完整断言下通过。
- 修复 Windows 大容量检查点逐字节比较在 1 MiB 默认线程栈上分配双 1 MiB 缓冲区导致栈溢出；缓冲区改为堆上有界分配。W7900 DINOv3 Huge+ FP16 存储恢复在原 RSS、显存和完整字节断言下通过。
- 补齐 WSL DINOv3 ViT-large/large-plus CPU FP32、GPU FP16 autocast，DINOv2-large CPU FP32，以及 Huge+ GPU FP16 参数存储完整文件恢复；降低外层准入并保留原断言。修复测试比较时同时驻留两份大检查点的额外主存，Huge+ 现通过原10 GiB RSS门槛。
- 补齐 Windows DINOv3 ViT-large/large-plus 的 CPU FP32、GPU FP16 autocast 及 DINOv2-large CPU FP32 完整文件检查点恢复范围，保留原逐字段载荷、loss、随机数及资源断言。
- 补齐 WSL ROCm 分发的 DINOv2 Giant、DINOv3 Huge+ CPU FP32 反向与实际 SGD 更新，以及 Giant 固定 timm 架构数值验证；降低外层内存准入要求，保留原测试断言、RSS 上限及参考容差。
- 补齐 Windows ROCm 分发的 DINOv2 Giant CPU FP32 反向、实际 SGD 更新及固定 timm 随机架构数值验证，保留原资源上限、参考身份和数值容差。
- 增加默认关闭的 DINOv2 Giant CPU BF16 参数存储资源门禁，覆盖完整有限梯度、分类头及 patch 权重的实际 SGD 更新和 8 GiB RSS 上限；与原 FP32 反向门禁分别验证。
- 登记 ResNet18 完整小样本配方及已训练模型的原生 CPU 推理、EfficientNetV2 固定前向及 DINOv2 指定首批次张量与同 AMD 后端官方 PyTorch 实现的一致性范围，保留原固定参考、ORT 数值与任务质量门槛结果。
- DINOv2-small 真实首批次测试增加显式启用的完整张量采集，保存 logits、loss、六份梯度和六份更新参数；补齐 Windows 与 WSL AMD GPU 各自同后端的 14 份张量与官方 PyTorch 完整反向的逐字节对照，保留原数值断言和双方确定性设置边界。
- 增加默认关闭的原生 MIGraphX GPU 互操作测试，固定校验原生导出图、输入与参考 SHA、FP32 数值和实际 GPU 指令，保留完整输出及每次独立证据；不引入产品推理组件依赖。
- MIGraphX 开发测试补充编译程序在独立进程中的重载，核对程序 SHA、实际 GPU 指令、原 FP32 参考及首次执行输出的逐字节一致性。
- 增加默认关闭的原生 `migraphx_runtime` Preview 组件，显式选择 HIP 设备并编译固定形状 ONNX，支持有字节上限的同步主存 FP32 输入输出、串行 Session 调用及实际 GPU 路线检查；安装消费者显式提供外部 MIGraphX/HIP SDK。
- 原生 MIGraphX Session 增加有界环境锁接口，分别核对编译头文件版本、实际 C API/HIP 模块文件摘要、HIP runtime/driver、设备身份与固定输入输出形状；严格拒绝环境漂移，纯解析无需设备查询。
- 原生 MIGraphX Session 可使用外部 SDK 的可选指令名扩展，省去路线追踪的中间张量回拷及逐指令同步，保留每请求实际 GPU 路线校验、最终同步与错误清理；未扩展 SDK 保持原 trace 路径，另提供可复现的 MIT SDK 补丁。
- 增加默认关闭的 `migraphx_serving` Preview，将固定形状原生 AMD GPU Session 接入已验证 bundle、可信 pipeline、独立已知向量预热、请求生命周期与同 runtime 版本切换；公共输入输出为 CPU FP32，显式外部 SDK 和环境锁固定实际 GPU。
- 增加独立原生 CIFAR-10 MIGraphX 示例，复用已有 ONNX 图与 checkpoint 参考，分别打包固定 batch 128/16、临时 Ed25519 签名，并在新进程通过可信 Serving 评估全部验证图片；与原 ONNX 示例共用图片 codec。
- 注册 MIGraphX 图像示例的参数测试时，将其可执行程序纳入默认测试构建，避免默认构建后 CTest 缺少程序。
- 接入 AMD ROCm LibTorch/HIP 构建、GPU 设备发现与构建指纹，视觉 batch 预检使用 HIP 内存查询；AMD 继续使用 LibTorch 的 `cuda:<index>` 与既有训练/AMP/恢复接口。
- 增加默认关闭的 Windows ROCm 构建树运行库暂存选项，按 PE 导入依赖闭包暂存匹配 SDK DLL，并通过清单管理替换、冲突和旧文件；安装运行仍由应用提供匹配 SDK。
- 修复 Windows ROCm 可选 DLL 暂存后缺少数学库 `.kpack` 设备归档造成的 `hipErrorInvalidKernelFile`；构建树及视觉安装消费者保留 SDK 归档相对路径，增量更新和关闭选项时按清单管理归档。
- Windows/Linux ROCm 10.1.0 LibTorch 2.14.0 可启用原生 ONNX 导出；GPU 训练模型通过独立 CPU 副本导出，视觉 CLI 使用 CPU FP32，推理后端由 ONNX Runtime 单独选择。
- 补齐 AMD 开发组合的八类优化器、十五类调度器与 FP32/FP16/BF16 autocast 数值及精确续训回归；主权重使用 FP32。
- 补充 StepLR 显式 PyTorch 启动模式的完整状态恢复与模式不匹配拒绝验证，覆盖首次事件及首次衰减边界；记录选项和检查点兼容语义。
- 开发 benchmark 增加独立 ROCm 训练和八类优化器目标，使用真实 HIP 设备身份、同步计时和 allocator 峰值；开发 runner 保留独立进程原始样本并检查环境一致性与统计稳定性。
- ROCm 训练和优化器基准在缺少合格功耗/频率控制观测时拒绝发布或比较 reference，离线统计工具对历史 v1 样本执行同一限制；原样本和统计稳定性结果继续保留。
- 增加独立 ROCm 持续验证入口，覆盖固定负载的三精度训练、反复文件检查点精确续训、受控显存 payload 及进程 allocator 配额 OOM 后恢复，按实际时长记录资源与限制。
- ROCm 持续验证可显式请求每精度两小时及 2000 万更新上限，保留时长不足即失败和原资源限制；资格仍按实际执行结果登记。
- 行人检测和分割示例接入 HIP allocator，分别执行原有 reserved/allocated 显存预算并输出独立 ROCm 字段；检测入口在使用 LibTorch 前初始化运行时。
- 登记 WSL2/W7900 行人检测与分割示例分别在 CPU 和 AMD GPU 的完整训练、留出集质量及暂停/强杀恢复范围。
- 冻结特征与图像训练输出示例在配置 LibTorch 线程和执行张量操作前初始化运行时，接入公共 Windows 线程池处理及 ROCm 后端注册。
- 基础训练、数据管线、调度、梯度缩放、分类评估和蒸馏示例补齐运行时初始化；分割示例将初始化提前到设备查询和随机数设置之前，MNIST 示例将初始化提前到 IDX 张量转换之前。
- CPU 优化器、损失、数据管线、Trainer、内存和图像解码基准在首次使用 LibTorch 前调用公共运行时初始化，沿用 Windows 线程池处理与 ROCm 后端注册。
- 部署、Serving、发布文件、分组学习率与 CIFAR-10 推理示例在首次使用 LibTorch 前显式初始化运行时。
- 已训练 CIFAR-10 的 ONNX Serving 示例允许显式选择 DirectML，核对签名包中的 Provider 与选项并保留禁止 CPU fallback 的契约。
- 模型与 Serving 通用资源基准接入真实 HIP 计量和独立显存字段，采样前后核对原生身份；缺少 ROCm 功耗/频率控制观测时保留样本并拒绝生成可比较 reference。
- 图像训练开发基准接入 HIP 同步、设备身份与独立显存字段，保留有界子进程日志和独立样本，检查干净源码、数据、二进制及 10% CV 稳定性。
- 模型资源基准采用独立的 `v2` 负载，固定八次预热与 256 次前向，延长短负载计量区间；新旧版本的摘要和比较环境分离。
- 组件矩阵子配置保留父构建显式选择的编译器与 SDK 搜索根，防止 ROCm 等组合在隔离验证时切换工具链。
- 修复仅启用模型和图像数据时，诊断程序错误要求完整 SDK 而导致配置失败的问题。
- 明确 Linux 本地进程内核峰值可能包含执行新程序前继承的驻留页；资源计量测试保留执行后负载上限，并以子进程独立内核计数核对单位。
- 修复 GCC 13.3 下无效运行文件声明的异常清理崩溃，错误输入恢复为正常拒绝且不创建运行存储。
- Linux ROCm 初始化延后框架设备查询，并说明动态库加载阶段 profiler 可见性解析的启动设置。
- Linux ROCm 原生工具在构建与安装后保留所选 LibTorch/HIP SDK 的传递动态库搜索路径，修复默认空环境遥测 worker 在进入入口前缺少数学库而退出的问题。
- 登记 WSL ROCm 组合与锁定官方 Linux Collector 的原生 HTTP 互通、明确文件队列崩溃恢复、队列满和回复丢失边界。
- 新增 Linux 官方 Collector HTTPS 互通门禁，使用私有 CA 视图验证原 C++ API、SQLite/CLI 与 worker 三条交付路径和官方信号读回。
- 新增官方 Collector HTTPS 接收器与文件队列的独立强杀恢复门禁，保留三信号的持久/易失对照及原时间边界。
- 新增官方 Collector HTTPS 队列满与成功回复丢失门禁，验证原确认前缀、原字节重试和至少一次交付边界。
- 新增 Linux 原生 HTTPS 连接建立后的取消与截止门禁，保留停滞响应头/响应体后的原字节重试、进程内去重和严格文件描述符恢复检查。
- 登记 WSL2/W7900 本地进程实际 GPU 放置、重试/重放、原生资源计量及独立 cgroup v2 内存隔离验证范围。
- 提供显式任务 step、模型适配、多优化器、梯度累积、裁剪、AMP、回调及可恢复调度器。
- 提供 map/iterable 数据管线、确定性采样、线程 worker、结构化批次和游标恢复。
- 修复 Clang/MSVC 标准库组合下嵌套错误经 `exception_ptr` 保存和重新抛出时崩溃，保留跨线程传播的结构化诊断与原始原因。
- 协调保存模型、优化器、scheduler、RNG、数据、GradScaler、回调和训练控制状态，支持文件检查点及完整恢复。
- 提供图像解码、COCO/VOC/YOLO/LabelMe 导入、联合变换、数据增强、缓存、组批、设备传输、绘制及有界图像输出。
- 修复 Windows Clang/MSVC 组合下图像编码预算拒绝和输出回调异常未释放资源预约的问题，保留原编码字节及预算限制。

### 模型与视觉套件

- 对齐现代 YOLO 分布解码与 CIoU 的 FP32 运算顺序，共享损失分支的精度转换，避免 FP16 反向提前舍入；训练配方身份绑定新损失语义，阻止旧路径检查点直接精确续训。

- 修复 YOLO 旧版短 SPPF 配置遗漏首层 SiLU 激活的问题，与冻结上游图解析一致；YOLO26 扩展配置保留无激活语义。受影响模型使用新的架构 revision，区分修复前的恢复身份。

- 提供分类骨干、ViT/Swin、DINOv2/v3、统一特征提取、分类头替换与显式 safetensors 权重导入。
- 提供原始 YOLOv5 P5/P6、YOLOv8/11/12/26、Faster/Fast R-CNN、FCOS、RetinaNet、SSD、RT-DETR 与开放词汇模型的原生组件和统一 CLI。
- 提供分类、多标签、检测、实例分割、姿态、旋转框、语义分割、深度、SimMIM、ReID、蒸馏及序列跟踪工作流。
- 提供严格配方、数据预览、参数搜索、EMA、早停、best/last 保存、模型初始化及固定尺寸 ONNX bundle。
- 修复 VOC 图片扩展名与子目录导入、YOLOv5/SSD buffer 视图权重导入、完成态重复训练、扩展模型预测对训练标注的依赖及 attention 布局反向差异。
- 修复视觉任务 CUDA 输入与辅助张量设备不一致、结束多图混合后错误拒绝配对目标，以及大型检查点的元数据预算限制。
- 修复 RT-DETR 和 ViT-Adapter 在 FP16/BF16 autocast 下采样坐标与投影张量类型不一致，保留坐标精度及梯度。
- 修复 Linux 上跟踪 ReID 裁剪、开放词汇视觉提示及稠密检测正样本计数的整数类型不匹配，保留原边界与归一化公式。

### 实验、评估与交付

- 提供 SQLite/CAS 运行与制品存储、事务化发布、本地原生进程执行、有界搜索、原生 TensorBoard 与 OTLP 遥测。
- 本地进程执行增加 AMD 物理设备绑定、显存准入与子进程放置；Windows 使用 HIP/WDDM 全局显存计量，重试和 worker 启动前重新核对。
- Windows 子进程移除继承的 OpenMP 进程登记字段，修复启用该运行库时等价计划的环境绑定无法跨进程核对。
- 修复 Windows 缺失制品目录规范化意外添加驱动器环境伪变量、导致完整启动环境摘要不一致的问题，并保留长路径支持。
- 提供分类、检测和分割评估、检查点续评、质量规则及 JSON/HTML 报告。
- 提供通用 C++ ONNX 导出、独立 ONNX Runtime、模型 bundle、Ed25519 签名、本地 Serving 与受控版本切换。
- 修复 ONNX 归约类型提升、混合类型 cat/stack、可选仿射 BatchNorm 及 Double LayerNorm 方差计算。
- 增加 `initialize_runtime()`，规避适用 Windows LibTorch OpenMP 配置的辅助线程池退出等待；原生 HTTP 完成回调采用无等待事件通知。
- 将 Windows 辅助线程池退出处理扩展到 ROCm LibTorch 2.14.0，覆盖仅配置线程后退出的模型元数据与测试发现进程。

### 文档与仓库维护

- 修复独立 SDK 安装消费者未发现内部测试却报告通过的问题；安装消费者与组件矩阵在未发现测试时明确失败。
- 独立 DINOv2 随机架构参考生成器补齐 Base/Large/Giant 容量选择，保留原固定文件摘要与环境锁；大容量状态按张量写入，Giant 显式串行准入，已有参考不覆盖。
- 文档按当前公共接口、组件和可运行示例组织，移除开发阶段流水、测试成绩及无代码支持的功能说明。
- 视觉模型验证探针归入测试目录；示例目录保留教学程序，并提供集中索引。
- 移除一次性环境报告、未接入的预研脚本及依赖元数据中的本机路径和执行记录；增加嵌套缓存、生成物及可移植配置的卫生检查。

具体接口及限制见[文档索引](docs/index.md)。权重、任务精度和设备支持按当前构建与兼容矩阵分别判断。
