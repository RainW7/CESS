# 代码许可与随包资源说明

[English](RESOURCE_NOTICE.md) | [简体中文](RESOURCE_NOTICE_zh-CN.md)

CESS 自有源码和文档采用根目录 `LICENSE` 中的 MIT License。
依赖软件及外部资源保留其原有权利和适用条件；本项目的 MIT 声明不替它们重新授予许可。

## 本次发布的资源范围

2026-10-24，维护者确认随 CESS v0.9.0 发布当前使用的 CSST throughput、背景模板和形态资源。
维护者说明 throughput 为本次采用的最新 CSST 七测光波段与三个无缝光谱波段的 FITS 副本。
当前库没有可独立核对的统一上游 release 编号或日期，因此不将“最新”解释成已验证的官方全局版本。
文件身份以分发副本的路径、大小和 SHA256 为准，详见源码中的 `docs/RESOURCE_MANIFEST.json`。
清单共18个文件：17个校准/形态资源和1个默认配置。

| 资源组（相对 `src/cess/data`） | 文件数量 | 用途与来源说明 |
| --- | ---: | --- |
| `csst_phot_tp/csst_phot.{nuv,u,g,r,i,z,y}.fits` | 7 | 当前采用的 CSST 七波段测光响应，维护者确认随包发布 |
| `csst_sls_tp/*throughput*.fits` | 6 | GU/GV/GI 各含零级、一级响应；三个波段对应六个谱级文件，维护者确认随包发布 |
| `basic_file/updated_bkg.fits` | 1 | 当前背景模板，维护者确认随包发布；不据此认证随时间/太阳几何变化的背景模型 |
| `basic_file/gems_cat.fits` | 1 | 既有 GEMS 形态统计星表副本，维护者确认随形态资源发布；不补造具体上游 release 或重新署名 |
| `basic_file/{widthlib,heightlib}_20x20x10x10.pkl` | 2 | 维护者原形态生成程序生成的预存库，确认随包发布；历史 PSF 和归一化未知属性继续保留其原状态 |
| `emulator_parameters.json` | 1 | 默认配置及已有来源记录 |

`GV_throughput_0th_no56.fits` 和 `GV_throughput_1st_no1567.fits` 的文件名按历史资源保留。
不从文件名推断其处理步骤，也不补造提供方、官方日期、下载网址或数据许可证。
这些说明记录维护者的发布决定与当前已知来源，不把它们当作独立科学定标结果。
真实 BayeSED 输入星表、用户模拟输出、环境目录和开发 checkpoint 不属于随包资源。

## 第三方软件与旧兼容实现

NumPy、SciPy、Astropy、h5py、Specutils 等依赖由包管理器单独安装，其许可证随各自软件保留。
本项目不把依赖软件整体复制到 CESS 分发包。

原兼容 `varsmooth` 中沿用的 pPXF 实现已从本次可发布源码移除，
改为 CESS 自行实现的有限区间 Gaussian 积分；没有将 pPXF 代码重新许可为 MIT。
同名接口仍保留，但不承诺复现原实现的边界数值；`oversample` 的适用范围见接口文档。
它不参与当前生产的 `kernels_convolve` → `convolve_to_bins` 路径。
旧研究目录与冻结存档中的历史副本保留本地追溯，不纳入源码包或 wheel。

参考许可：[Grizli MIT](https://github.com/gbrammer/grizli/blob/master/LICENSE.txt)、
[Specutils BSD-3-Clause](https://github.com/astropy/specutils)、
[pPXF 上游许可](https://pypi.org/project/ppxf/9.0.2/#license)。
这些参考项目的选择不替代 CESS 维护者自己的许可决定。
