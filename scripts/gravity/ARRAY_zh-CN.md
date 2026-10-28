# 在 Gravity 批量模拟星表

[English](ARRAY.md) | [简体中文](ARRAY_zh-CN.md)

`catalogue_array.py` 将输入星表编排成 MzLS 和 DECaLS 两个 Torque 作业数组。每个子作业使用 1 CPU，处理一个完整输入文件，生成一份完整光谱 HDF5；不按源分片，不需要事后 merge。默认配置针对 **1,385 个 MzLS 和 1,385 个 DECaLS，共 2,770 个文件**。

## 1. 资源与并行方式

Gravity 的[官方作业说明](https://jupyter.gravity.sjtu.edu.cn/doc/#/Basic/Job)采用 PBS/Torque，支持 `nodes=1:ppn=1` 和 `-t 1-N%并发数`。`normal` 队列的文档上限是 48 小时、每节点 376 GB；正式计算应在计算节点执行。以下脚本据此使用 Torque 语法。

| 类别 | 每子作业 CPU | 申请内存 | walltime | 数组同时运行上限 |
| --- | ---: | ---: | ---: | ---: |
| MzLS | 1 | 16 GB | 12 小时 | 32 |
| DECaLS | 1 | 48 GB | 24 小时 | 32 |

两个数组同时提交时，默认合计最多 64 个文件并行计算。调度器可能只运行其中一部分；这不是独占一个节点或保证获得 64 CPU。内存是每个子作业的申请量，不是整个数组合计。所有 BLAS/OpenMP 等数值库线程请求固定为 1，CESS 每子作业也只设置 `--workers 1 --threads-per-worker 1`。

本地同类输入的 CESS 工作内存估计约为 MzLS 8.9–9.1 GiB、DECaLS 29.2 GiB；它不是实测峰值或硬上界。16/48 GB 是留有余量的初始资源方案，须用下文的完整文件试跑确认。提交脚本同时传入申请内存 85% 的估计预算；预算只用于拒绝明显装不下的任务，实际内存仍由 PBS 管理。

不要把单个输入文件的约 0.535/1.764 GB 当作运行内存。解压后的模板、三波段光谱、噪声产品和输出组装会同时占用内存。每个文件单独启动和退出 Python，可在文件之间释放内存。

## 2. 在服务器生成提交脚本

先按[安装说明](../../docs/INSTALL_zh-CN.md)在 Gravity 创建 Linux 的 Python 3.12 `cess` 环境并安装 CESS。不要复制 Mac 的 Conda 环境。下面所有 `/path/to/...` 都须替换成实际服务器绝对路径。

```bash
conda activate cess
cd /path/to/CESS

python -s -B scripts/gravity/catalogue_array.py prepare \
  -i /path/to/CESS_input \
  --run-dir /path/to/runs/cess_v090_all
```

`--run-dir` 必须是尚不存在的新目录。默认使用 CESS 随包配置；使用自己的仪器配置时，在同一命令后增加 `--config /path/to/emulator_parameters.json`。如需将产品存到另一块存储，增加 `-o /path/to/CESS_output_all`；该输出目录也必须尚不存在，两个类别将写到它下面的子目录。脚本记录本次调用的 Python 解释器绝对路径，PBS 计算时直接使用它，不依赖非交互 shell 中的 `conda activate`。

准备阶段会检查以下内容：

- 输入文件总数、文件名中的 MzLS/DECaLS 分类，以及每类是否恰好为 1,385。
- HDF5 的源数、模板维度和估计内存；不把整个星表流量数组读入内存。
- 配置、CESS 版本、源码与资源身份，输入文件的大小和修改时间。

找不到分类、计数不符、同名输出 stem 冲突或估计内存超过预算时，准备阶段会拒绝继续。输入目录只扫描直接子级的 `.hdf5/.h5`；若输入分置多个目录，可以在 `-i` 后列出多个目录或带引号的通配符。

生成的主要文件为：

| 文件或目录 | 用途 |
| --- | --- |
| `MzLS.pbs`、`DECaLS.pbs` | 两类输入各自的提交脚本 |
| `plan.json`、`plan.sha256` | 固定本批输入顺序、资源参数和运行身份 |
| `config.json`、`catalogue_array.py` | 固定的有效配置和本批运行脚本 |
| `products/MzLS/`、`products/DECaLS/` | 完整光谱产品，运行后生成 |
| `manifests/<类别>/` | 每个输入独立的 CESS 恢复清单 |
| `logs/<类别>/`、`status/<类别>/` | 各次尝试的日志、资源记录及状态 |

同一类别内按估计内存从大到小排序，编号从 1 开始。第一项可用来检查该类较大的文件。准备后不要移动运行目录、替换输入、修改冻结配置，或更新正在使用的 CESS 源码、资源和 Python 依赖。完整输入内容哈希由每文件 CESS manifest 在计算节点上计算；准备阶段不会在登录节点顺序扫描约 3 TB 的输入内容。

只读预览某个任务：

```bash
python /path/to/runs/cess_v090_all/catalogue_array.py run \
  --plan /path/to/runs/cess_v090_all/plan.json \
  --group MzLS --index 1 --dry-run
```

新任务预览不创建科学产品或新的 CESS manifest。实际计算固定为 `standard` 模式、`structured` 输出、`--all-effects`、seed 1214；不更改现有仪器或物理模型。需要另一个随机种子时，在准备命令中设置 `--seed`。

## 3. 先试跑两个完整文件

```bash
qsub -t 1-1%1 /path/to/runs/cess_v090_all/MzLS.pbs
qsub -t 1-1%1 /path/to/runs/cess_v090_all/DECaLS.pbs
```

这里通过命令行将数组范围缩小为第一项，保留该类正式任务的资源申请。试跑使用 `normal` 队列；DECaLS 在本地已有约 4 小时的运行记录，不适合预设进 3 小时的 `debug` 队列。

等待这两个任务结束，检查退出状态、光谱源数、输出文件大小、耗时和内存余量。`logs/` 保留每次尝试；资源记录中的 child max RSS 是子进程树报告的最大驻留内存统计，不是所有同时存活进程的内存总和。作业仍可查询时，也应使用 `qstat -f JOB_ID` 查看 PBS 的 `resources_used.mem` 等统计。若作业记录已从 `qstat` 移除，保留运行日志并按站点提供的方式查询历史资源使用。

输入本身约占 **3.185 TB（2.897 TiB）**；按每对约十万个源，整批约有 **1.385 亿个源**，以准备阶段汇总的实际行数为准。输出和临时文件需要额外空间。用两类试跑产物的实际大小乘以各 1,385 估计输出空间，并预留临时写出和重试余量；不要用输入字节数代替输出空间估计。

需要更改初始申请时，在全量计算前选择新的运行目录重新准备，例如：

```bash
python -s -B /path/to/CESS/scripts/gravity/catalogue_array.py prepare \
  -i /path/to/CESS_input \
  --run-dir /path/to/runs/cess_v090_all_larger \
  --mzls-slots 32 --decals-slots 16 \
  --mzls-memory-gb 20 --decals-memory-gb 64 \
  --mzls-walltime 16:00:00 --decals-walltime 36:00:00
```

这些数值是格式示例，应依据试跑和账户资源选择。不要降低 `qsub -l mem=...` 来覆盖生成脚本中的申请而仍保留原估计预算。

## 4. 全量提交

先检查当前队列和账户的实际限制；文档并未保证每个账户都能提交含 1,385 项、32 并发的数组：

```bash
qstat -q
qstat -f -Q
qstat -Bf
```

确认试跑任务已经结束后，提交两个完整数组：

```bash
qsub /path/to/runs/cess_v090_all/MzLS.pbs
qsub /path/to/runs/cess_v090_all/DECaLS.pbs

qstat -tan -u "$USER"
```

试跑已完成的第一项会走原 CESS manifest 的严格恢复核验，并跳过已有完整产品。不要在试跑还运行时提交同一项。

若当前站点或账户对单个数组长度的限制更小，可以分段提交同一个脚本：

```bash
qsub -t 1-500%32 /path/to/runs/cess_v090_all/MzLS.pbs
# 等这一段全部结束后，再提交下一段：
qsub -t 501-1000%32 /path/to/runs/cess_v090_all/MzLS.pbs
# 等第二段全部结束后：
qsub -t 1001-1385%32 /path/to/runs/cess_v090_all/MzLS.pbs
```

DECaLS 同理。`%32` 的上限作用于每次提交的数组；同时提交三个区间会把该类上限提高到 96，而不是仍为 32。Torque 的数组长度、slot limit 和用户提交/运行数量可由管理员限制，见 [Torque 服务器参数](https://docs.adaptivecomputing.com/torque/6-1-1/adminGuide/Content/topics/torque/13-appendices/serverParameters.htm)。

CESS 的进程池只在当前节点运行。这里由 Torque 将独立文件分配到不同节点；不要给每个子作业申请多个节点，也不要将单文件的 `--workers` 提高到 32。运行器会核对 `PBS_NODEFILE` 为单主机单 slot，并检查可用的 PBS CPU/节点变量。

## 5. 查看进度与恢复

```bash
python /path/to/runs/cess_v090_all/catalogue_array.py status \
  --plan /path/to/runs/cess_v090_all/plan.json
```

状态命令汇总逐文件运行记录，并打印未完成项的 `retry_commands`。它不自行执行 `qsub`。状态摘要用于管理任务；真正的产品来源、校验和及结构核验由 CESS 的 `--resume` 完成。

每个输入有独立 manifest；某一文件失败不会阻止其他数组项继续。重新提交该项时，运行器自动为已有 manifest 加 `--resume`。有效完整输出会核验后跳过；损坏或来源不匹配的输出会报错，不会覆盖原文件。中断发生在写出期间时，可能留下 `.partial` 临时文件，它不是完成产品。

取消、超时或 OOM 后，先通过 PBS 确认旧作业及 worker 已停止，再执行打印出的补交命令。`SIGKILL` 不能被捕获，状态文件可能仍显示 `running`；不要只凭这一字段判断任务仍存活。运行器与 CESS 均使用文件锁；`.lock` 文件可长期存在，不要删锁文件来强制重跑。

脚本会向子进程转发可捕获的终止信号，保存已有日志和恢复清单。整文件任务中断后，未完成文件从头计算；已完成的其他文件无需重算。

提交语法的其他细节见 [Torque qsub 文档](https://docs.adaptivecomputing.com/torque/6-1-1/adminGuide/Content/topics/torque/commands/qsub.htm)。本地脚本验证不能代替 Gravity 上的试跑，也不能证明仪器物理模型已经完成科学定标。
