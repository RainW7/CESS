# CESS v0.9.0 发布内容与目录

[English](RELEASE_CONTENTS_v0.9.0.md) | [简体中文](RELEASE_CONTENTS_v0.9.0_zh-CN.md)
本文件说明可公开的一维软件范围。版本号为0.9.0；本地发布构件准备、Git提交/推送、版本标签及GitHub Release是不同步骤，实际线上状态以仓库为准。

## 1. 可以发布的内容

| 内容 | 发布位置与用途 |
| --- | --- |
| 主程序 | `src/cess/`，按配置、光谱、仪器、形态、测光、发射线、I/O、流程和CLI分模块 |
| 默认资源 | `src/cess/data/`：默认JSON、7个测光响应、GU/GV/GI零/一级共6个响应、背景、GEMS及2个预存形态库 |
| 安装与依赖 | `pyproject.toml`、`setup.cfg`、`requirements/`、`MANIFEST.in` |
| 用户说明 | README、安装、快速开始、Bash命令、用户指南、输出与科学模型文档 |
| 可运行示例 | `examples/make_toy_input.py` 与说明；不含真实上游星表 |
| 服务器模板 | `scripts/gravity/`，含预检、PBS payload及站点资源配置说明 |
| 兼容入口 | 明确列入清单的 `CESS_v0.8.6` Python aliases、生成器和17份测试所需资源副本，以及根目录几个旧入口 |
| 测试 | `tests/` 及4个小型源码数值oracle；不是用户模拟输出或完整开发档案 |
| 许可与来源 | `LICENSE`（MIT）、`RESOURCE_NOTICE.md`、`docs/RESOURCE_MANIFEST.json`、变更说明与发布说明 |

源码发布保留兼容目录是为了现有脚本和测试闭包。它不是把整个历史v0.8.6研究目录公开；实际文件以构件清单为准。Python源中的历史作者/版本注释保留追溯价值，当前版本以 `cess.__version__` 和包声明为准。

随包18个资源文件包含默认配置，17个为校准/形态数据；逐文件字节大小与SHA见[资源清单](RESOURCE_MANIFEST.json)。维护者已同意随包分发，来源和权利边界见[资源说明](../RESOURCE_NOTICE_zh-CN.md)。不把“当前采用副本”写成已独立核对的官方统一release，也不替外部资源重新授予许可。

可选桌面浏览器也纳入本版：`src/cess/viewer/`、`cess-viewer`入口、
兼容启动脚本 `cess_h5_viewer.py`、`requirements/viewer-py312.txt` 和
[中英文浏览器指南](VIEWER_zh-CN.md)。Matplotlib通过viewer extra安装，Tk由运行环境提供。

## 2. 只保留本地的内容

- 真实BayeSED输入、完整/抽样科学输出、运行manifest、日志及私人路径信息。
- Conda/venv环境、缓存、构建临时目录、`.git`、本地应用/代理配置和凭据。
- 历史完整版本、notebook、PDF手册、论文草稿、开发记录与checkpoint。
- `CESS_kinematics/`、旧二维后端及尚未验收的CESS+研究材料。
- 未列入清单的个人查看器、临时脚本和独立数据表。

本地目录的存在不代表应将其上传。不要对研究工作区直接使用 `git add .`；使用已整理的干净源码目录或逐文件清单。原始数据和历史记录不因发布整理被删除或移动。

测试oracle只包含必要的小段历史源码。其再分发副本已去除本机绝对解释器路径，执行AST保持一致，原档案身份与清理后SHA分别记录于 `tests/baselines/PROVENANCE.json`；不要求发布或访问原始大型开发档案。

## 3. 三种交付物

| 交付物 | 包含内容 | 适合用途 |
| --- | --- | --- |
| wheel：`cess-0.9.0-py3-none-any.whl` | 当前Python包、18个默认资源、入口及许可元数据 | 安装运行；依赖由目标平台另行安装 |
| sdist：`cess-0.9.0.tar.gz` | 可构建源码、公开文档、示例、服务器模板、兼容和测试闭包 | 从源码安装或复现实验/工程测试 |
| 干净源码目录 | 公开文件及原始构建配置，不含本地研究工作区和构建元数据 | GitHub文件审阅与后续提交准备 |

`py3-none-any` 只表示CESS自身没有平台专用二进制扩展，不表示NumPy/SciPy等依赖可跨平台复制。Linux服务器需要自己的Python3.12环境；Mac环境目录不能直接搬过去。

本地整理会在构件旁提供文件清单、SHA256和验证摘要。源码包中的 `MANIFEST.in` 定义构建范围；wheel与sdist内容不同，不能只检查其中一个就宣称所有交付物齐全。真实运行输出的provenance记录生成版本，不因安装了0.9.0而被改写。

## 4. 使用文档入口

第一次使用按[安装指南](INSTALL_zh-CN.md)创建环境，然后依次阅读[快速开始](QUICKSTART_zh-CN.md)和[Bash命令指南](COMMANDS_zh-CN.md)。输入字段、物理量和进阶设置见[用户指南](USER_GUIDE_zh-CN.md)；字段布局见[输出说明](OUTPUT_LAYOUT_zh-CN.md)；模型假设见[科学模型](SCIENCE_MODELS_zh-CN.md)。

默认每输入文件一份完整输出；目录并发须显式设置 `-j`，且按任务内存选择进程数。使用分片时可再合并成自包含产品。Gravity的CPU/内存申请与CESS的worker参数分别设置，详见[服务器说明](../scripts/gravity/README_zh-CN.md)。

## 5. 发布边界

v0.9.0是一维软件发布，不包含二维CCD成像、源间谱污染提取或运动学反演的完成声明。已有用户整文件运行、工程测试和小样本并发比较不能替代仪器科学定标，也不代表Gravity已完成部署/调度验收。

正式上传时核对准确源码与构件、版本号和资源SHA，保留验证摘要，再按维护者授权创建Git提交、`v0.9.0`标签及GitHub Release。发布程序包不要求把真实输入或模拟结果一同上传。
