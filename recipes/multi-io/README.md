# 双输入、分类与回归数据

`paired-affine-v1` 是项目自行生成的离线示例数据，包含两个特征输入、一个二分类目标和一个回归目标，用于学习自定义多输入、多输出模型。它采用 Apache-2.0 许可，训练集 1,024 条、验证集 256 条，无需下载或 Python 第三方包。

在源码根目录执行下面的命令，将输出位置替换为仓库外的数据目录；也可以省略 `--data-root`，使用 `JYPPX_LIUDING_DATA_ROOT`。

```sh
python scripts/datasets/prepare_paired_affine.py --data-root /path/to/datasets
```

输出目录为 `paired-affine-v1/`，包含 `train.jsonl`、`validation.jsonl` 和 `preparation.json`。已有目标目录会被拒绝，准备失败只清理工具本次创建的暂存目录。调用者应保持输出父目录稳定；路径不能经过符号链接或 Windows reparse point。

每行是一个样本，`left` 和 `right` 分别作为两个模型输入；目标字段不应拼入输入。

| 字段 | 类型和形状 | 含义 |
| --- | --- | --- |
| `id` | 字符串 | split 和连续样本序号 |
| `left` | float32 `[2]` | 第一组特征，范围 `[-1,1]` |
| `right` | float32 `[2]` | 第二组特征，范围 `[-1,1]` |
| `class_id` | int64 标量 | `left[0] + left[1]/2 + right[0] - right[1]/4 >= 0` |
| `regression` | float32 `[1]` | `(4*left[0] - 2*left[1] + 3*right[0] + right[1] + 0.5)/8` |

两份数据文件合计 173,375 字节，每条记录不超过 512 字节；[数据合同](paired-affine.dataset.json)固定版本、种子、样本数、类别计数和文件 SHA-256。准备工具逐条生成并核对完整文件身份，不缓存整个数据集。准备完成后，按[双输入训练教程](../../docs/tutorials/multi_io_recipe.md)配置 C++ 训练、分类与回归评估、检查点和独立进程恢复。这些合成数据不能证明真实任务质量或性能优势。
