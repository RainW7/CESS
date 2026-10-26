# CESS 0.9.0：Bash 命令手册

[English](COMMANDS.md) | [简体中文](COMMANDS_zh-CN.md)

本手册面向第一次使用 CESS 命令行的用户。适用于 macOS/Linux 的 Bash；常见单行命令也适用于 zsh。
软件目前准备 v0.9.0 本地发布候选构件，GitHub 发布状态见[发布清单](RELEASE_CHECKLIST_v0.9.0_zh-CN.md)。
科学量、输入 HDF5 字段和模型近似见[用户指南](USER_GUIDE_zh-CN.md)；这里只解释如何组织和运行命令。

## 1. 最常用的三种运行方式

先激活已经安装 CESS 的环境：

```bash
conda activate cess
```

下列 `/path/to/...` 是占位路径，执行前换成自己机器上的真实路径。使用绝对路径时，可以从任意工作目录运行。
每个例子都是独立任务，应选尚未包含同名结果的新输出位置。

**一个输入文件，生成一份完整结果：**

```bash
cess simulate -i /path/to/data/input.hdf5 -o /path/to/results/run001 --all-effects
```

**一个目录，依次处理其中所有输入：**

```bash
cess simulate -i /path/to/data/catalogues -o /path/to/results/batch001 --all-effects
```

**一个目录，同时用两个计算进程处理不同文件：**

```bash
cess simulate -i /path/to/data/catalogues -o /path/to/results/parallel001 --all-effects -j 2
```

未设 `--source-start/--source-stop` 时处理全部源；未设 `--chunk-size` 时，**每个输入文件对应一份完整光谱 HDF5**。
`-j 2` 要求内存足以同时容纳两个任务。默认 `-j 1` 会依次处理目录文件。
一个未分片的输入只有一个任务，单独把它的 `-j` 改成8不会让该文件自动分到8个进程。

| 参数 | 含义 |
| --- | --- |
| `simulate` | 从 BayeSED-shaped HDF5 生成一维模拟光谱 |
| `-i` / `--input` | 输入文件、目录或带引号的glob；可在同一个 `-i` 后列多个输入 |
| `-o` / `--output` | 模拟和后处理的输出目录；`merge -o` 则是具体输出文件 |
| `--all-effects` | 同时启用形态、波长定标误差、测光和发射线分析四个可选开关 |
| `-j` / `--workers` | 最多同时工作的计算进程数，默认1 |
| `--threads-per-worker` | 每个新计算进程的数值库线程请求，默认1 |

只运行 `cess simulate ...` 而不带 `--all-effects` 时，四个科学开关默认关闭。
`--all-effects` 不表示已经启用所有可能的仪器物理模型，例如它不会自动选择 MGE 或补出已知模板 LSF。

## 2. 安装、激活和确认当前版本

完整依赖及平台说明见[安装指南](INSTALL_zh-CN.md)。已能使用 Conda 时，从源码根目录安装：

```bash
conda create -n cess python=3.12 pip
conda activate cess
cd /path/to/CESS
python -m pip install '.[tools,plot]'
python -m pip check
```

`/path/to/CESS` 必须是含 `pyproject.toml` 和 `setup.cfg` 的源码根目录。
`.[tools,plot]` 中的点表示当前目录，引号让 Bash/zsh 把整个参数原样传给 pip。
这会安装 CESS、核心依赖、抽样和绘图依赖；不是从 PyPI 上寻找同名软件的安装命令。
若安装的是下载的本地 wheel，可在同一环境执行：

```bash
python -m pip install /path/to/downloads/cess-0.9.0-py3-none-any.whl
```

这个 wheel 命令只请求核心依赖；需要绘图/抽样功能时按安装指南安装相应 extras。
如果名为 `cess` 的环境已经存在且 Python 版本合适，直接激活，不必再次 `conda create`。
使用 venv 的用户按安装指南创建环境，以后改用：

```bash
source /path/to/cess-env/bin/activate
```

Conda 和 venv 方案选一种。每个新终端均需重新激活；venv 不出现在 `conda env list` 中。
激活后检查：

```bash
python --version
python -c 'import sys, cess; print(sys.executable); print(cess.__version__); print(cess.__file__)'
cess --version
cess --help
cess simulate --help
cess merge --help
```

Python 应为3.12.x；`cess --version` 显示当前命令的 CESS 版本，Python检查同时显示解释器和导入路径，
用于核对它们是否属于所选环境。版本路径不符合预期时，先处理环境或安装问题。
直接运行源码而不使用已安装入口时，以下写法可从任意目录使用，解释器仍需有 CESS 依赖：

```bash
env PYTHONPATH=/path/to/CESS/src python -s -B -m cess --help
```

## 3. Bash 的路径、引号和换行

### 3.1 先确定工作目录

```bash
pwd
cd /path/to/data
ls
```

`pwd` 显示工作目录，`cd` 改变它，`ls` 列出内容。若当前在 `/path/to/data`，则：

```bash
cess simulate -i input.hdf5 -o results --all-effects
```

这里输入是 `/path/to/data/input.hdf5`，输出目录是 `/path/to/data/results`。
命令不是必须在 CESS 源码目录运行；使用安装版时，数据目录和源码目录可以分开。

### 3.2 多行写法仍是一条命令

下面与第1节的单行命令等价：

```bash
cess simulate \
  -i /path/to/data/input.hdf5 \
  -o /path/to/results/run002 \
  --all-effects
```

- 行末的 `\` 表示“本条命令在下一行继续”；它后面不能再有空格或注释。
- 最后一行不加续行符；不要把多行文本粘成带有 `\   -i` 的一行。需要单行时直接删除续行符和换行。
- 不要复制终端提示符，例如 `$`、`%` 或 `(cess)`；文档代码块已经省略它们。
- 终端出现 `>` 等待续行时，通常是续行符、引号或括号尚未结束；可按 Ctrl+C 取消，再粘贴完整命令。

### 3.3 含空格的路径与普通变量

路径有空格时，用引号保护整个路径：

```bash
cess simulate -i "/path/to/my data/input.hdf5" -o "/path/to/my results/run001" --all-effects
```

也可在同一终端定义自己的变量，赋值时等号两侧没有空格；使用时保留双引号：

```bash
CESS_DOC_INPUT="/path/to/my data/input.hdf5"
CESS_DOC_OUTPUT="/path/to/my results/run002"
cess simulate -i "$CESS_DOC_INPUT" -o "$CESS_DOC_OUTPUT" --all-effects
```

普通变量不需要 `export`，因为 shell 会先把它们展开成命令参数。
不要用 `HOME`、`PATH` 等系统变量来保存任务路径。文件名以 `-` 开头时，使用它的绝对路径可避免被解析成选项。

## 4. 目录、多个文件与 glob

目录输入只发现其直接子级的 `.hdf5` 和 `.h5`，不递归扫描子目录。
多个输入放在同一个 `-i` 后，用空格分开，不写逗号或 Python 列表：

```bash
cess simulate -i /path/to/data/a.hdf5 /path/to/data/b.hdf5 \
  -o /path/to/results/two_files --all-effects -j 2
```

`-i` 会一直接收输入值，直到遇到下一个选项，如 `-o` 或 `-j`。
按文件名模式筛选时，推荐给整个 glob 加引号，由 CESS 解析；这样不依赖 Bash/zsh 对无匹配模式的不同处理：

```bash
cess simulate -i '/path/to/data/catalogues/*_MzLS_*.hdf5' \
  -o /path/to/results/mzls_batch --all-effects -j 2
```

若模式中还需要变量，使用双引号，让变量展开、让星号保留：

```bash
CESS_DOC_INPUT_DIR=/path/to/data/catalogues
cess simulate -i "${CESS_DOC_INPUT_DIR}/*_MzLS_*.hdf5" \
  -o /path/to/results/mzls_batch_002 --all-effects -j 2
```

不同输入的 stem（不含扩展名的文件名）必须互异。例如 `field.h5` 和 `field.hdf5` 可能产生同名输出，
需要放入不同批次运行。输入是多份文件时，每份仍有自己的光谱结果，不会把不同星表合成一张大目录。

## 5. 先看计划，再设置进程和内存

新任务的只读预览不带 manifest：

```bash
cess simulate -i /path/to/data/catalogues -o /path/to/results/planned_run \
  --all-effects -j 4 --dry-run
```

计划中的 `input_files`、`task_count`、`workers`、`threads_per_worker` 和每个任务的 `estimated_peak_gib`
分别说明文件数、任务数、规划并发上限、线程请求及估计内存。计划成功只说明已读到必要元数据并建立任务，
不表示全部 SED、资源或科学数值已经核验。普通CLI预览不加载所有运行资源。

内存充足、已分配4路计算资源时，可去掉 `--dry-run` 实际运行。进程池最多启用请求数量和任务数中的较小值；
设置内存预算后还可能降低并发：

```bash
cess simulate -i /path/to/data/catalogues -o /path/to/results/budgeted_run \
  --all-effects -j 4 --threads-per-worker 1 --memory-budget-gib 32
```

这里32 GiB只是预算写法示例，需按自己的主机或作业资源修改。
预算应为操作系统、父进程和其他程序保留余量，而不是直接填机器总内存。
当前预算按最大任务的估计内存保守计算可容纳的 worker 数，不是实时内存监控或操作系统硬限额；
若一个任务都放不下，会拒绝运行，需降低 chunk 或换用内存足够的资源。

默认每个 worker 的数值库线程请求为1，通常无需再手工设置 BLAS/OpenMP 变量。
这些变量在计算进程spawn前设置，不代表绑定物理CPU或自动申请服务器资源。
同时计算多个大文件会累加内存；并发数不能只依据CPU总数选择。
文件级并行缩短整批总耗时，一个未分片的大文件本身仍由一个 worker 处理。
单文件并行见第9节，服务器作业见第13节。

## 6. 源行区间、随机种子和输出格式

从一个文件选取原始第100–119行，共20行：

```bash
cess simulate -i /path/to/data/input.hdf5 -o /path/to/results/subset \
  --all-effects --source-start 100 --source-stop 120 --seed 1214
```

行号从0开始，起点包含、终点不包含；`--source-stop 120` 不是“运行120个源”。
默认起点0、终点为文件总行数。对目录传入源区间时，同一区间应用到每个文件，不能超过其中任何短文件的源数。
默认 `--seed 0`；固定 seed 便于复现。同一输入的逐源随机流不因 worker 数和 chunk 改变，
但不同依赖、平台或线程后端仍可能存在浮点舍入差异。

现代CLI默认 schema3.0；只有下游仍需旧磁盘布局时，显式选择 legacy，并使用独立输出：

```bash
cess simulate -i /path/to/data/input.hdf5 -o /path/to/results/legacy_run \
  --all-effects --output-layout legacy
```

`legacy` 指 schema2.5 输出，不是回到旧 Python 环境或旧科学算法。
文件命名会包含输入 stem 和所选源数，切片文件还包含原始行区间；每份光谱文件保存三个波段的同源数据。
源数、软件版本和HDF5 schema是不同概念，读取时使用[统一产品接口](OUTPUT_LAYOUT_zh-CN.md)。

## 7. 科学开关、WCE对照和外部配置

只打开部分可选功能，可以显式列出开关。例如仅请求形态和测光：

```bash
cess simulate -i /path/to/data/input.hdf5 -o /path/to/results/selected_effects \
  --morphology --photometry
```

下面生成共享噪声的无WCE/有WCE对照，并启用四个可选开关：

```bash
cess simulate -i /path/to/data/input.hdf5 -o /path/to/results/paired_run \
  --all-effects --mode paired-wce --seed 1214
```

默认模式是 `standard`。两分支共享同一次噪声 realization，不能当作两次独立曝光相加。
`--mode paired-wce` 本身不替代 `--wavelength-error`；使用 `--all-effects` 已包含这个开关。

已有完整配置文件时显式传入：

```bash
cess simulate --config /path/to/config/emulator_parameters.json \
  -i /path/to/data/input.hdf5 -o /path/to/results/custom_config --all-effects
```

配置优先级是 `--config`、非空 `CESS_CONFIG` 环境变量、包内默认JSON。
外部JSON中的相对资源路径以及其中输入/输出目录，相对JSON所在目录解析；
CLI中相对路径仍相对调用时的工作目录。包内默认资源由包自身定位，其默认运行目录相对当前工作目录。
复制默认JSON到新目录后，必须同步调整资源路径；几行不完整JSON不能作为完整仪器配置。
配置字段和近似见[用户指南](USER_GUIDE_zh-CN.md#5-配置文件资源与路径)与[科学模型说明](SCIENCE_MODELS_zh-CN.md)。

需要创建可编辑的完整配置时，可在已安装环境中导出包内默认值。下面是Bash调用Python的here-document：
`PY` 之间的文本交给Python，结束标记 `PY` 单独占一行，不在它前面缩进。

```bash
python - <<'PY'
import json
from pathlib import Path
from cess.config.loader import DEFAULT_CONFIG_PATH, load_config

destination = Path('/path/to/config/emulator_parameters.json')
destination.parent.mkdir(parents=True, exist_ok=True)
config = load_config(DEFAULT_CONFIG_PATH)
config.pop('_config_path', None)
config.pop('_config_dir', None)
with destination.open('x', encoding='utf-8') as stream:
    json.dump(config, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(destination)
PY
```

目标文件已存在时会拒绝覆盖。导出配置中的资源路径指向本环境的实际安装位置，运行目录也已解析为绝对路径；
在其他机器或环境使用前应重新生成或更新这些路径。编辑曝光、噪声、孔径等参数时，同时核对单位和科学定义，
并为修改后的配置使用新任务输出与manifest。

若此前设置了 `CESS_CONFIG`，可检查它；不希望使用时，在记录原值后取消：

```bash
printf '%s\n' "${CESS_CONFIG:-not set}"
unset CESS_CONFIG
```

## 8. 新任务、manifest和严格续跑

希望任务中断后能恢复时，首次运行就指定一份新的 manifest：

```bash
cess simulate -i /path/to/data/input.hdf5 -o /path/to/results/resumable \
  --all-effects --seed 1214 --manifest /path/to/manifests/run001.json
```

manifest记录完整任务计划及逐任务完成状态。没有设置chunk时，一个完整输入就是一个任务；
程序不能从该文件中间第几千行自动接着算。需要较细粒度恢复时，应在首次运行前显式设置chunk。
续跑前先确认原来的 worker 已停止，保留原来的输入、输出、配置和科学参数。

**先只读核验同一任务：**

```bash
cess simulate -i /path/to/data/input.hdf5 -o /path/to/results/resumable \
  --all-effects --seed 1214 --manifest /path/to/manifests/run001.json --resume --dry-run
```

**核验通过后执行剩余任务：**

```bash
cess simulate -i /path/to/data/input.hdf5 -o /path/to/results/resumable \
  --all-effects --seed 1214 --manifest /path/to/manifests/run001.json --resume
```

| 命令组合 | 文件行为 |
| --- | --- |
| 新任务 `--dry-run`，无 `--manifest` | 只打印计划，不创建科学产品或新manifest |
| 新任务 `--dry-run --manifest new.json` | 创建待运行清单及所需父目录/锁，不计算光谱 |
| 原任务 `--resume --dry-run --manifest existing.json` | 只读核验，不改写清单或产品 |

因此，如果已用第二种方式创建了清单，实际执行时应加 `--resume`。
输入内容/路径、完整配置、资源、源码、seed、模式、开关、布局、输出位置、源区间和chunk都受严格身份核验。
源码升级后不能把旧开发版manifest当作0.9.0同一任务续跑；旧的完整产品仍可按其schema读取，
不必仅为更换软件版本号重算已经完成的产品。
worker数、线程请求和内存预算属于调度设置，可以调整；它们不会取消对科学任务身份的核验。
清单损坏或身份不匹配时应查明原因，新配置使用新任务位置，不手改清单绕过检查。
不能同时启动两次占用同一manifest的任务。大型目录建议分批或按输入文件分别管理manifest，避免一个巨大JSON反复重写。

## 9. 单文件分片并行，最后仍得到一个完整文件

一个源数较多的文件可先分为每1000行一个任务，最多两个任务并发：

```bash
cess simulate -i /path/to/data/input.hdf5 -o /path/to/results/input_chunks \
  --all-effects --chunk-size 1000 -j 2 --seed 1214 \
  --manifest /path/to/manifests/input_chunks.json
```

1000是任务大小示例，应根据内存和耗时调整。它仍处理全部行，但会产生多份分片HDF5。
计算结束后只选择**同一输入、同一次模拟的分片**合并；先预检，再写一份新文件：

```bash
cess merge -i /path/to/results/input_chunks -o /path/to/results/merged/complete.hdf5 \
  --input-source /path/to/data/input.hdf5 --dry-run

cess merge -i /path/to/results/input_chunks -o /path/to/results/merged/complete.hdf5 \
  --input-source /path/to/data/input.hdf5
```

`merge -o` 是文件路径，和 `simulate -o` 的目录含义不同。
合并不重新模拟；它按原始行排序，核对配置、种子、版本、共享波长/响应及连续覆盖，重映射发射线表的源行引用。
`--input-source` 额外校验原始输入SHA、逐行ID和完整行覆盖，推荐保留。
分片必须连续、无重复、无重叠；不同输入、不同运行或不同schema不能混在一起。
普通和paired-wce模拟产品可合并；EL-only后处理不在此功能范围。

结果是自包含HDF5，不靠外链/VDS读取原分片。软件不会删除分片；建议先核对完整产品再自行安排存储。
输出放在分片目录外，避免以后把完整文件再次作为待合并输入。
`--copy-rows 128` 是默认复制块大小，不是模拟chunk；最终线表核验的内存仍随峰数量增长。
`merge` 没有manifest续跑参数；完成的合并文件也不能顶替原生模拟任务的manifest产品。

## 10. 对已有结果运行发射线后处理

```bash
cess el-detect -i /path/to/results/simulation.hdf5 -o /path/to/results/line_analysis \
  --seed 1214 --manifest /path/to/manifests/line_analysis.json
```

输入是已有CESS模拟产品，不是原始BayeSED库；支持schema2.5/3.0父产品。
该步骤保存发射线测量、质量和父来源，不复制整套父光谱大数组，也不重新生成噪声。
子集行号相对父产品，原星表行号另由映射保存。父产品缺少误差/认证协方差时，不能凭空得到有效线S/N。
`-j`、区间、chunk、dry-run及严格续跑的使用方式与模拟类似，但 `--all-effects` 属于 `simulate`，不加给 `el-detect`。

## 11. 长任务日志与返回状态

最容易理解的日志方式是将标准输出和错误输出写入一份新日志。以下命令可从任意目录执行：

```bash
mkdir -p /path/to/logs
cess simulate -i /path/to/data/catalogues -o /path/to/results/logged_batch \
  --all-effects -j 2 --manifest /path/to/manifests/logged_batch.json \
  > /path/to/logs/logged_batch.log 2>&1
echo "$?"
```

`>` 创建或截断日志；选择新的日志名，续跑另用新日志名保存记录。
`2>&1` 把错误输出并入同一文件。日志没有立即更新不必然表示卡死，可以结合任务开始/完成、PID及进程状态判断。
在另一个终端查看最后部分或持续跟踪：

```bash
tail -n 40 /path/to/logs/logged_batch.log
tail -f /path/to/logs/logged_batch.log
```

`tail -f` 用Ctrl+C退出只会结束查看，不会停止另一个终端中的CESS。
`echo "$?"` 必须紧接在CESS命令之后；插入别的命令后，它显示的是别的命令返回值。
通常0表示该命令成功，1表示任务执行存在失败，2表示参数或规划错误；应同时检查日志和manifest。
warning不自动等于失败，也不证明科学结果已正确；例如模板LSF未知的提示保留其物理边界。

### 同时看屏幕并保存日志：可选Bash脚本

管道末尾的 `tee` 成功不等于前面的CESS成功。若需要这种方式，将以下内容保存为 `run_cess.sh`，
替换路径，并在已激活CESS环境的终端、该脚本所在目录使用 `bash run_cess.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail
mkdir -p /path/to/logs
cess simulate -i /path/to/data/catalogues -o /path/to/results/tee_batch \
  --all-effects -j 2 --manifest /path/to/manifests/tee_batch.json \
  2>&1 | tee /path/to/logs/tee_batch.log
```

`pipefail` 使管道前段失败不会被 `tee` 的成功隐藏；`set -e` 使这个Bash脚本在失败时停止。
这里明确使用Bash脚本，避免混用Bash的 `PIPESTATUS` 和zsh的 `pipestatus` 等不同语法。
日志重定向/`tee` 本身不提供任务续跑；要续跑仍需首次就指定manifest。
普通前台命令不应当作“关掉终端也会继续”的作业；服务器长任务使用下一节PBS方式。

## 12. 已有结果、异常源和常见错误

CESS拒绝覆盖同名科学产品，包括已有目标路径或符号链接；输出目录本身可以已存在，只要本次目标产品不存在。
同样，首次运行的manifest路径必须是新的。要重新计算新配置，选择新输出和清单；要恢复原任务，使用原manifest与 `--resume`。
多进程目录任务失败时，部分其他文件可能已经完成，应根据清单处理，不把“没有覆盖”理解为所有任务都未运行。

| 现象 | 处理方法 |
| --- | --- |
| `command not found: cess` | 激活环境，确认已在该环境安装；检查 `python` 与 `cess.__file__` |
| 终端一直显示续行提示 | 检查引号/括号/行末反斜杠，Ctrl+C取消后重输完整命令 |
| 找不到输入 | 核对真实绝对路径、glob引号和目录直接子级；替换 `/path/to` 占位路径 |
| 两个输入同stem | 分成独立批次或使用不同输出目录，避免文件名冲突 |
| 找不到资源 | 核对外部JSON的路径基准；包内资源与自己拷贝的JSON不要混淆 |
| 有多个文件却仍串行 | 确认传了 `-j N`，查看规划的有效worker上限；任务数或预算可能降低并发 |
| 只传了一个文件，`-j 8`仍只有一个任务 | 选择文件级批量输入，或显式分片后合并 |
| 内存预算拒绝单任务 | 改小chunk或选择内存充足的主机；减少workers不能让一个过大的完整任务变小 |
| 输出/manifest已存在 | 区分新任务与续跑；新任务换新位置，相同任务用原参数resume |
| snapshot/plan不匹配 | 核对输入、配置、代码、资源、seed、模式、布局、路径及分片身份 |
| 谱为NaN、线表为空 | 检查开关、源标记、波段覆盖和阈值；不直接当成程序崩溃 |

无效源会保留原ID、原始行号和顺序，以 `data_mask=2`、非零 `input_quality_flags` 及不可用模拟数组标记。
程序不会为了跑通而修补原始HDF5或静默删除这些行。低信噪比观测的负流量可以合法出现，也不应统一截为0。
标记定义、输出单位与reader示例见[用户指南](USER_GUIDE_zh-CN.md)及[输出说明](OUTPUT_LAYOUT_zh-CN.md)。

## 13. Gravity / PBS运行

CESS的 `-j` 只决定程序使用的进程数，不替代服务器申请CPU/内存。
先按本站PBS规则申请单节点资源，再设置匹配的CESS参数。
例如已经获得同一节点4个CPU并确认内存足够时，在按[Gravity说明](../scripts/gravity/README_zh-CN.md)设置好所有绝对路径和科学开关后：

```bash
export CESS_INPUT=/path/to/data/catalogues
export CESS_OUTPUT=/path/to/results/gravity_batch
export CESS_MANIFEST=/path/to/manifests/gravity_batch.json
export CESS_CPUS=4 CESS_WORKERS=4
unset CESS_SOURCE_STOP CESS_CHUNK_SIZE
bash "$CESS_REPO/scripts/gravity/preflight.sh" --dry-run
```

此处 `CESS_REPO` 等其余必需变量须已按Gravity说明设置；不是仅执行这个片段就完成部署。
`CESS_CHUNK_SIZE` 未设/为空表示完整文件任务；正整数表示分片。
脚本固定数值库线程请求为1，检查workers不超过用户声明的CPU上限，支持可选估计内存预算。
`CESS_CPUS=4` 本身不会申请或自动认证实际分配了4个CPU。

PBS flavour、队列、内存与walltime语法必须以本站为准；完成资源头和预检后，由用户按本站方式提交。
本地脚本执行验收不等于Gravity调度验收。一个CESS进程池只在同一主机运行，没有MPI多节点分发；
多节点生产可划分为独立批次作业，各用自己的输出目录和manifest。

### 可选桌面浏览器

以下安装命令在CESS源码根目录、激活的环境中执行；viewer为可选功能。
Tcl/Tk属于Python/系统环境，不通过pip安装；Conda缺少时用 `conda install tk`，
用 `python -m tkinter` 检查。查看HDF5不会运行模拟。

```bash
python -m pip install '.[viewer]'
cess-viewer /path/to/product.hdf5 --lang zh
# 同等入口，适用于已安装CESS的环境：
python -m cess.viewer /path/to/product.hdf5 --lang en
# 不传文件，启动空窗口后选择文件：
cess-viewer
```

仓库脚本入口为 `python /path/to/CESS/cess_h5_viewer.py /path/to/product.hdf5`。
源行/ID切换、筛选、导出和无桌面服务器限制见[浏览器指南](VIEWER_zh-CN.md)。

## 14. 运行后应该检查什么

保留命令、工作目录、软件版本、输入/配置、日志和manifest，使用[统一reader](OUTPUT_LAYOUT_zh-CN.md)核对源数、ID/原始行映射、有效标记及所需科学字段。
开启测光或发射线分析后，还要区分model/observed、检测/未检测和无效状态。
参数表按字段名和单位读取；科学模型近似见[模型说明](SCIENCE_MODELS_zh-CN.md)，
进阶MGE提取见[形态模型](MGE_MODELS_zh-CN.md)和[库格式](MGE_BUNDLE_FORMAT.md)。
命令执行成功只说明程序完成相应任务，不替代仪器PSF/LSF、背景、灵敏度或样本选择函数的科学验收。
