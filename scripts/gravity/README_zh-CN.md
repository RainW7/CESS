# Gravity / PBS 使用说明

[English](README.md) | [简体中文](README_zh-CN.md)

处理数千个独立星表时，推荐使用[整文件作业数组方案](ARRAY_zh-CN.md)：按 MzLS/DECaLS 分别申请内存，每文件独立输出、日志和恢复清单。下面的 `cess.pbs` 则适用于用户自行配置的单节点小批次。

这些脚本用于在PBS计算节点运行CESS。预检要求 `0.9.0 / 20261027-D5`；请在Linux的Python3.12环境部署相同源码版本，提交作业前先运行预检。

## 部署 D5 源码包

将 D5 的 `cess-0.9.0.tar.gz` 和对应的 `SHA256SUMS` 上传到服务器的独立部署目录。版本号仍为 0.9.0，升级时请核对科学 revision，不能仅凭文件名区分 D4 和 D5。以下命令从存放安装包的目录执行：

```bash
sha256sum --check --ignore-missing SHA256SUMS
tar -xzf cess-0.9.0.tar.gz
cd cess-0.9.0
conda create -n cess python=3.12 pip -y  # 仅首次创建环境时执行
conda activate cess
python -m pip install .
python -c 'from cess.contracts import REPAIR_REVISION; print(REPAIR_REVISION)'
```

最后一条应显示 `20261027-D5`。已安装 CESS 的环境应将安装命令改为 `python -m pip install --force-reinstall --no-deps .`，前提是现有依赖满足要求；随后运行 `python -m pip check`。在运行中的作业结束前，不要替换其源码或环境。Mac 的 Conda 环境不能直接复制到 Linux。

`cess-gravity-D5-20261028.tar.gz` 是同一份源码的 Gravity 便携包，解压目录为 `CESS-v0.9.0-D5`；也可进入该目录执行上述安装命令。它包含完整资源、示例、说明和作业脚本，无须另找旧版形态库或 throughput 文件。

D5 使用更新的三波段范围、中央分辨率和输出采样，见[仪器模型说明](../../docs/INSTRUMENT_MODEL_251106917_zh-CN.md)。默认 BayeSED 模板采用 Gaussian R=300，GU 红端的过宽模板按 `preserve` 保留并提示局限。旧 D4 清单不能用于 D5 续跑；请按[作业数组指南](ARRAY_zh-CN.md)重新 `prepare` 新目录，再做每类一个完整文件的试跑。D5 内部的意外中断仍可按原清单续跑。

## 单节点预检与小批次

`preflight.sh` 只读检查 Python 3.12、导入源码位置、版本/revision、配置及资源文件是否可读，再调用实际 CLI dry-run；不检验仪器科学模型，也不证明所有输入行均有效。`cess.pbs` 才执行模拟，首次运行创建新 manifest，失败保留日志及续跑清单，直接返回失败状态。输出已有产品时 CLI 拒绝覆盖。

以下`/path/to/...`均为通用占位路径；解释器、仓库、配置、星表、队列和资源必须替换为实际存在并获分配的值。不含 SSH、密码、依赖安装或自动 qsub。

```bash
# 在服务器终端设置为对应的绝对路径；本地 toy 预检时使用本地实际位置。
export CESS_PYTHON=/path/to/cess-env/bin/python
export CESS_REPO=/path/to/CESS
export CESS_CONFIG=/path/to/config/emulator_parameters.json
export CESS_INPUT=/path/to/data/catalogue_001.hdf5
export CESS_OUTPUT=/path/to/results/run001/catalogue_001
export CESS_MANIFEST=/path/to/manifests/run001/catalogue_001.json
export CESS_EXPECTED_VERSION=0.9.0 CESS_EXPECTED_REVISION=20261027-D5
export CESS_CPUS=1 CESS_WORKERS=1 CESS_SEED=1214
unset CESS_CHUNK_SIZE CESS_MEMORY_BUDGET_GIB
export CESS_MODE=standard CESS_OUTPUT_LAYOUT=structured CESS_RESUME=0
export CESS_MORPHOLOGY=1 CESS_WAVELENGTH_ERROR=1 CESS_PHOTOMETRY=1 CESS_EMISSION_LINES=1
export CESS_SOURCE_START=0 CESS_SOURCE_STOP=4  # 首次只检查/计算4行，停止位置不包含

bash -n "$CESS_REPO/scripts/gravity/preflight.sh"
bash -n "$CESS_REPO/scripts/gravity/cess.pbs"
bash "$CESS_REPO/scripts/gravity/preflight.sh" --help
bash "$CESS_REPO/scripts/gravity/preflight.sh" --dry-run
```

全部命令使用 Bash 数组保留路径边界，不使用 `eval`，兼容本机 Bash 3.2。预检和计算均将 BLAS/OpenMP/NumExpr 等线程固定为1，并向 CLI 明确传入 `--threads-per-worker 1`；默认仅1 worker。`CESS_CPUS` 是用户声明的已分配CPU上限，脚本不能替调度器申请资源；`CESS_WORKERS` 不得超过它。

`CESS_CHUNK_SIZE` 未设置或为空时，每个输入文件对应一个任务和一份完整HDF5。显式设置正整数（例如 `CESS_CHUNK_SIZE=100`）后，才按原始源行分片输出。`CESS_MEMORY_BUDGET_GIB` 可选，必须为正有限数值，转交 CLI 的估计内存预算来降低并发；它不是实际可用内存测量或操作系统硬限额，也不会自动申请调度器资源。

预检CLI与PBS计算都从`CESS_REPO`执行，保证包内默认配置的运行目录解析及resume snapshot一致；调用预检时所在的目录不影响这一约定。

模式选 `standard` 或 `paired-wce`；四开关分别取 `0/1`，默认全0。默认 structured=3.0，legacy=2.5。`CESS_INPUT` 接受一个 HDF5 或目录；目录只发现其直接子级的 `.hdf5/.h5`，同一批输入stem须互异。将 `CESS_SOURCE_STOP` 取消后才处理所选文件的全部剩余行；目录的显式源区间同时应用到所有文件，不能超过其中短文件的行数。小试验通过后，全量新任务须使用新的 output/manifest。配置中的仪器/库相对路径相对 JSON 所在目录，而非 qsub 工作目录。

例如，在已向调度器申请同一节点4个CPU、并确认内存能够容纳同时运行的整文件任务后，用4个worker处理一个输入目录；每个worker一次处理一个输入文件，完成后领取下一个文件：

```bash
export CESS_INPUT=/path/to/data/input_catalogues
export CESS_OUTPUT=/path/to/results/parallel_run
export CESS_MANIFEST=/path/to/manifests/parallel_run.json
export CESS_CPUS=4 CESS_WORKERS=4
export CESS_SOURCE_START=0
unset CESS_SOURCE_STOP CESS_CHUNK_SIZE
# 可选：按实际申请资源留足内存余量后设置估计预算，例如：
# export CESS_MEMORY_BUDGET_GIB=96
bash "$CESS_REPO/scripts/gravity/preflight.sh" --dry-run
```

worker数量最终受任务数和可选内存预算限制：目录有2个文件时，4个worker的请求仍只有2个可执行任务；单文件未分片时只有1个任务。worker不绑定某个物理CPU，不进行MPI跨节点分发。增大worker一般缩短整批总耗时，不保证单个输入文件更快；I/O、内存和较大文件的尾部计算都会限制加速。

一个目录批次共用一份独立manifest。当前完整JSON每完成任务都会重写，文件非常多时宜拆成若干目录/批次，或逐文件使用唯一output/manifest；不要让多个作业竞争同一manifest或同名输出。

新任务 dry-run 故意不传 `--manifest`：当前 CLI 的新任务 `--dry-run --manifest` 会创建清单。此处预览不创建目录、manifest或科学产品，但不等价于正式执行前的全部输出冲突/输入哈希核验；正式 CLI 仍检查并拒绝覆盖。预览会读取星表元数据及资源可读性，未实际加载全部科学输入/资源。

Gravity 的[官方作业说明](https://jupyter.gravity.sjtu.edu.cn/doc/#/Basic/Job)使用 PBS/Torque。小批次可按实际资源需要，将以下指令补入 `cess.pbs` 的头部；替换尖括号占位，并通过 `qstat -q`、`qstat -f -Q` 核对当前队列限制：

```text
# Torque；示例未填写实际 CPU、内存和时限，不可原样提交。
#PBS -q normal
#PBS -l nodes=1:ppn=<allocated_cpus>
#PBS -l mem=<memory>,walltime=<HH:MM:SS>
```

填好资源头并完成预检后，由用户执行 `qsub`。此小批次模板需要传入上述环境变量，例如：

```bash
qsub -V "$CESS_REPO/scripts/gravity/cess.pbs"
```

`-V` 是用户显式选择导出当前环境；本站若禁止或希望只传指定变量，应按本站方式导出上述 CESS 变量。脚本不自动调用 qsub。默认日志位于 manifest父目录的 `logs/`，也可设绝对 `CESS_LOG_DIR`；每次生成独立日志，PBS输出首先显示路径。CPU、内存和 walltime 仍由站点调度器约束，本草案没有声称服务器可运行或获得特定资源。

续跑前确认旧进程已停止，保留原 input/config/resources/source/seed/模式/开关/区间/chunk/output/manifest；只设 `CESS_RESUME=1`，先重跑 preflight dry-run，再由用户 qsub。此预览对既有 manifest走严格只读核验。跨代码/资源/schema身份不可续用旧清单；选择全新任务位置。不要同时提交两个使用同一 manifest 的作业。
