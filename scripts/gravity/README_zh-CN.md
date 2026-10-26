# Gravity / PBS 使用说明

[English](README.md) | [简体中文](README_zh-CN.md)
本目录只交付一维计算脚本；尚未登录 Gravity、安装 Linux 环境、提交作业或验证本站 PBS。当前预检身份为 `0.9.0 / 20261024-D4`；部署时须使用相同版本源码并重新预检。Mac 的 `.envs/cess` 和 ARM64 lock 不能直接移作 Linux 环境。

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
export CESS_EXPECTED_VERSION=0.9.0 CESS_EXPECTED_REVISION=20261024-D4
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

确认服务器 PBS flavour、可用队列、内存与 walltime 语法后，**只选一种**资源写法添加到 `cess.pbs` 的指令头，替换尖括号占位；不要把下面两组同时使用，也不要原样提交：

```text
# Torque 风格示意，须本站确认：
#PBS -q <queue>
#PBS -l nodes=1:ppn=<allocated_cpus>
#PBS -l mem=<memory>,walltime=<HH:MM:SS>

# PBS Pro/OpenPBS 风格示意，须本站确认：
#PBS -q <queue>
#PBS -l select=1:ncpus=<allocated_cpus>:mem=<memory>
#PBS -l walltime=<HH:MM:SS>
```

本站语法确认后，由用户执行 `qsub`，例如本站支持导出提交环境时：

```bash
qsub -V "$CESS_REPO/scripts/gravity/cess.pbs"
```

`-V` 是用户显式选择导出当前环境；本站若禁止或希望只传指定变量，应按本站方式导出上述 CESS 变量。脚本不自动调用 qsub。默认日志位于 manifest父目录的 `logs/`，也可设绝对 `CESS_LOG_DIR`；每次生成独立日志，PBS输出首先显示路径。CPU、内存和 walltime 仍由站点调度器约束，本草案没有声称服务器可运行或获得特定资源。

续跑前确认旧进程已停止，保留原 input/config/resources/source/seed/模式/开关/区间/chunk/output/manifest；只设 `CESS_RESUME=1`，先重跑 preflight dry-run，再由用户 qsub。此预览对既有 manifest走严格只读核验。跨代码/资源/schema身份不可续用旧清单；选择全新任务位置。不要同时提交两个使用同一 manifest 的作业。

本地验证仅使用临时toy输入和本地Bash执行，不代表本站PBS调度验收；具体工程记录单独保存。修改版本或脚本后须重新预检，不跨版本沿用旧manifest。
