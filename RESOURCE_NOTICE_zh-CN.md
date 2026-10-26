# 代码许可与随包资源

[English](RESOURCE_NOTICE.md) | [简体中文](RESOURCE_NOTICE_zh-CN.md)

CESS 源码和文档采用 [MIT License](LICENSE)。依赖软件及外部数据保留其原有权利和适用条件。

## 数据文件

默认资源位于 `src/cess/data`。[资源清单](docs/RESOURCE_MANIFEST.json)记录文件路径、大小和 SHA256。

| 资源 | 文件数量 | 说明 |
| --- | ---: | --- |
| `csst_phot_tp/csst_phot.{nuv,u,g,r,i,z,y}.fits` | 7 | CSST 七波段测光响应 |
| `csst_sls_tp/*throughput*.fits` | 6 | GU/GV/GI 的零级和一级响应 |
| `basic_file/updated_bkg.fits` | 1 | 固定天空背景模板 |
| `basic_file/gems_cat.fits` | 1 | 用于经验形态统计的 GEMS 星表 |
| `basic_file/{widthlib,heightlib}_20x20x10x10.pkl` | 2 | CESS 形态渲染程序生成的预存 profile 库 |
| `emulator_parameters.json` | 1 | 默认配置 |

当前 throughput 副本不带统一的上游版本号或日期，使用时以文件哈希识别。
`GV_throughput_0th_no56.fits` 和 `GV_throughput_1st_no1567.fits` 沿用所提供响应的文件名。
预存形态库的历史 PSF 和归一化假设尚未完全记录，具体使用边界见[模型说明](docs/SCIENCE_MODELS_zh-CN.md)。

## 依赖软件

NumPy、SciPy、Astropy、h5py、Specutils 及可选包独立安装，保留各自许可证。
CESS 不分发 pPXF 源码；Gaussian 平滑辅助函数为独立实现，不是生产流程的 `convolve_to_bins` 算子。
