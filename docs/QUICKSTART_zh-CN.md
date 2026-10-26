# CESS v0.9.0 快速开始

[English](QUICKSTART.md) | 简体中文

本页给出常用操作的最短流程。逐项参数和产品定义见[用户指南](USER_GUIDE_zh-CN.md)，
路径、换行、日志及更多可复制例子见[Bash 命令指南](COMMANDS_zh-CN.md)。
`/path/to/...` 是占位路径，须替换为自己的实际位置；所有示例使用已激活的 Python3.12 环境。

## 1. 安装并检查环境

以下从已经下载或解压的 CESS 源码根目录安装，要求终端已能使用 Conda：

```bash
conda create -n cess python=3.12 pip
conda activate cess
cd /path/to/CESS
python -m pip install '.[tools,plot]'
python -m pip check
python -c 'import sys, cess; print(sys.executable, cess.__version__, cess.__file__)'
cess --version
cess --help
```

安装本版后软件版本应为 `0.9.0`。已有环境直接激活，不要重复创建。
不使用 Conda、从 wheel/sdist 安装或升级旧安装时，按[安装指南](INSTALL_zh-CN.md)操作。
Conda 和 venv 任选一种；venv 不出现在 `conda env list` 中。
本说明不要求 PyPI 上已经发布 CESS；源码与本地构件的在线发布状态以实际 GitHub release 为准。

## 2. 用微型示例熟悉输出

选择一个新的可写工作目录并进入它。以下文件名和目录均相对此工作目录：

```bash
python /path/to/CESS/examples/make_toy_input.py toy.hdf5 --sources 4 --invalid-first
cess simulate -i toy.hdf5 -o toy_results --all-effects --seed 1214
```

这会得到 `toy_results/CSST_grism_toy_4.hdf5`，包含4行；第0行为故意设置的无效源，仍保留ID与行号。
toy SED和参数是任意设定，仅供安装与接口演示，不用于科学定标。

## 3. 处理自己的完整星表

下面是不同用途的命令模板，按需要选择。它们使用绝对路径，可从任意工作目录执行。
先预览任务，再移除 `--dry-run` 运行；预览不要附加新的 `--manifest`：

```bash
cess simulate -i /path/to/input.hdf5 -o /path/to/new_results --all-effects --dry-run
cess simulate -i /path/to/input.hdf5 -o /path/to/new_results --all-effects
```

未设置源区间或chunk时，计算完整文件，一份输入对应一份包含全部源的光谱HDF5。
无效源保留原始行号、ID与质量标记，不修改原始星表。

目录输入会发现该目录直接包含的 `.hdf5` 和 `.h5` 文件。默认依次运行；
内存足以同时容纳两个任务时，使用两个worker：

```bash
cess simulate -i /path/to/catalogues -o /path/to/new_batch --all-effects -j 2
```

每个worker一次计算一个文件，完成后领取下一个文件，仍是一输入一输出。
`-j` 默认1，不自动使用全部CPU，也不自动拆分单个文件。
CLI默认 `--threads-per-worker 1` 请求各数值库单线程，实际运行会显示并发摘要、PID和耗时。
同时运行的大文件内存会累加；先用 `--dry-run` 查看估计，再提高并发，不能只按CPU数量设置worker。

## 4. 常用选择

| 选项 | 含义 |
| --- | --- |
| `--all-effects` | 同时开启形态、波长误差、七波段测光和发射线分析；四开关原默认均关闭 |
| `--config /path/to/parameters.json` | 使用外部JSON；相对资源路径按JSON所在目录解析 |
| `--source-start 100 --source-stop 120` | 只处理原始行100–119，零起点、stop不包含 |
| `--seed 1214` | 显式随机种子；默认0 |
| `--mode paired-wce` | 输出共享噪声的无/有波长误差对照 |
| `--output-layout legacy` | 输出兼容schema2.5；现代入口默认分组式schema3.0 |
| `--memory-budget-gib 16` | 按保守任务估计降低并发；不是实时RSS或硬内存限制 |

首次小样本与正式全量任务使用不同的输出目录。现有同名产品不会被覆盖；
如果一个任务的估计已超过预算，应减小chunk或使用内存足够的主机。

## 5. 保存清单并续跑

以下为同一任务的首次运行、停止后的只读预检和续跑；先确认旧worker全部停止。
输入、输出、配置、科学开关、种子、区间和chunk必须一致：

```bash
cess simulate -i /path/to/input.hdf5 -o /path/to/run_results \
  --all-effects --seed 1214 --manifest /path/to/run_results/run.json

cess simulate -i /path/to/input.hdf5 -o /path/to/run_results \
  --all-effects --seed 1214 --manifest /path/to/run_results/run.json --resume --dry-run

cess simulate -i /path/to/input.hdf5 -o /path/to/run_results \
  --all-effects --seed 1214 --manifest /path/to/run_results/run.json --resume
```

`--resume --dry-run` 是只读续跑核验；**新任务同时使用 `--dry-run --manifest` 会创建清单**。
不需要清单时直接省略 `--manifest`。已更新源码、资源或配置的任务不能直接接用旧manifest。
非常大的目录建议拆成独立批次或每输入一份manifest。

## 6. 单个大文件的分片并行与合并

以下以包含 `input.hdf5` 的工作目录为基准。先生成有限大小的分片，再合成一份完整文件：

```bash
cess simulate -i input.hdf5 -o chunks --all-effects --chunk-size 1000 -j 2
cess merge -i chunks -o merged/complete.hdf5 --input-source input.hdf5 --dry-run
cess merge -i chunks -o merged/complete.hdf5 --input-source input.hdf5
```

`merge` 按原始行顺序复制科学结果，校验覆盖与兼容性、重映射发射线源行引用，不重新模拟。
最终文件自包含，不依赖原分片；合并输出须为新路径，放在分片目录之外。
多个输入文件的分片必须分别选择和合并，不要把整目录不同星表的结果混为一个输入。

## 7. 读取、展示与后处理

在已安装环境中读取模拟文件：

```python
from cess.io.products import open_product, read_product_summary

print(read_product_summary('/path/to/results/simulation.hdf5'))
with open_product('/path/to/results/simulation.hdf5') as product:
    ids = product.catalog['ID'][:]
    wave = product.band('GV')['wave'][:]                # Angstrom
    flux = product.band('GV')['flux_ujy_with_noise'][0]  # microJy
    row = product.read_row(0)
```

在第2节同一toy工作目录执行，下例绘制有效的第1行，并对既有产品独立进行发射线后处理：

```bash
python -m cess.cli.demo toy_results/CSST_grism_toy_4.hdf5 1 emissionline --save spectrum.png
cess el-detect -i toy_results/CSST_grism_toy_4.hdf5 -o toy_lines
```

绘图需要 `plot` 依赖，抽样导出需要 `tools` 依赖。结果的字段、单位、掩膜和可用性按
[输出布局](OUTPUT_LAYOUT_zh-CN.md)与用户指南读取；不能靠固定列号或数组长度猜测物理量。

服务器使用独立Linux环境与[Gravity脚本](../scripts/gravity/README_zh-CN.md)，先申请对应CPU/内存并做预检。
PSF/LSF、背景、响应和科学选择函数的适用性仍需针对研究用途核实，见[科学模型说明](SCIENCE_MODELS_zh-CN.md)。
