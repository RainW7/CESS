# CESS v0.9.0 的物理约定与适用边界

[English](SCIENCE_MODELS.md) | [简体中文](SCIENCE_MODELS_zh-CN.md)

本文说明 CESS `0.9.0`、科学 revision `20261027-D5` 采用的模型。参数可以复现计算，但不因软件发布就成为经独立定标的CSST官方仪器模型。普通运行命令见[命令指南](COMMANDS_zh-CN.md)，字段布局见[输出说明](OUTPUT_LAYOUT_zh-CN.md)。

## 1. 输入与单位

BayeSED输入的静止系波长单位为μm，进入算子后转换为Angstrom；`best_fit/z`用于将波长乘以 `1+z`。输入SED约定为上游已完成观测归一化的 `fnu`，runner按μJy解释，内部按需要转换为cgs；不能把本征光度密度直接作为输入，也不能再自动施加一次距离或红移流量因子。

HDF5中的源轴、波长轴、参数名和单位按[用户指南](USER_GUIDE_zh-CN.md)要求提供。源ID可重复，程序用原始行号维持对应关系，不依靠ID排序。计算所需的元数据、SED数值和波长支持接受安全检查；异常源保留ID/原行号，以 `data_mask=2` 和 `input_quality_flags` 标记，模拟数组无效，不修补源文件。这些范围用于数值安全，不是科学样本选择函数。

## 2. 响应积分、像元与分辨率

对已包含电子探测效率的响应T，忽略展宽时一个波长区间的源电子率与下式成正比：

$$
\dot N_e = \frac{A}{h}\int \frac{f_\nu(\lambda)T(\lambda)}{\lambda}\,d\lambda .
$$

A为计算所用集光面积，波长与积分变量采用一致单位。响应已经折叠的量不能再次乘throughput。实际算子在原生响应/SED节点上积分，并处理Gaussian展宽和输出像元；不能把响应简单替换为一个像元中心值。

输出 `flux_ujy` 是当前像元定义下的光子等价bin流量密度，不能把它无条件当作任意波长的连续函数点值。光子账目使用相应的 `pixel_width/wave` 权重；共享 `wave_edges` 和 `pixel_width` 明确像元几何。有限波段、核窗口和探测器边界的损失必须保留。

随包仪器参考采用 [Zhang 等，arXiv:2511.06917v2](https://arxiv.org/html/2511.06917#S2) 表1的中央波长与中央分辨本领。GU/GV/GI 输出边界为 2550–4100 / 4000–6400 / 6200–10000 Å。启用 `instrument_lsf.mode=central_resolving_power` 时，`*_central_wl` 与 `*_Res` 实际参与计算，按 `central_wl/Res` 得到目标 Gaussian FWHM，约为 11.74216 / 22.62931 / 39.13043 Å。

显式的 `wavelength_law=constant_fwhm` 将这个参考宽度延拓到各自波段；论文只规定中央波长处的 R，没有证明全带恒定 R。Gaussian 仪器轮廓与上述延拓均是一维近似，论文也没有明确分解 LSF 中的像元积分与 PSF。因此，目标 Gaussian 宽度不等于最终像元积分、形态展宽后分辨率的认证。[仪器模型说明](INSTRUMENT_MODEL_251106917_zh-CN.md)区分了参考值、推导量和保留假设。

默认 `spectral_sampling.mode=resolution_elements` 请求每个参考分辨单元四个像元。程序按 `ceil(band_span/(FWHM/4))` 设置 bin 数并保留精确波段边界，GU/GV/GI 分别为 529 / 425 / 389 bins。实际像元宽度约为 2.93006 / 5.64706 / 9.76864 Å；应读取 `wave_edges` 与 `pixel_width`，不要把采样等同于 FWHM。默认 `spectral_window.mode=bandpass` 依据实际波长网格选择探测器/S/N和发射线范围。缺少这些新模型键的旧外部配置仍沿用固定 `*_dlambda`、历史网格和索引窗口行为。

`template_lsf` 与波长采样独立：

| mode | 含义 |
| --- | --- |
| `unknown` | 模板LSF形状未知；当前继续施加完整仪器核，明确提示最终分辨率不能据此认证 |
| `delta` | 显式假设输入未含可观展宽 |
| `resolving_power` | 显式Gaussian近似及正的R值，按观测系匹配所需额外展宽 |
| `fwhm_rest` | 显式静止系Gaussian FWHM，单位Angstrom |

随包模板模型为 `resolving_power`、`value=300.0`，采用用户于 2026-10-27 为这批 BayeSED 库明确选择的 Gaussian R=300 近似。该假设不是上述仪器参考论文的测量结论；换用其他输入库时须相应设置模板 LSF。

默认 `template_lsf_broader_policy=preserve` 在模板已经比目标更宽的位置不再施加额外仪器展宽，并报告目标未达到。具体而言，R=300 模板在 GU 约 3522.65 Å 以上比参考目标更宽，程序保留已有宽度，不反卷积，也不把它标记为成功匹配。另选 `error` 策略可拒绝这种计算。模板 LSF 描述上游数值/仪器平滑，不能把真实天体的本征谱线宽度作为模板响应扣除。

## 3. 源形态、PSF与孔径

默认从GEMS统计及预存profile库获取一维形态。原始形态参数与实际选中的库参数分开理解；最近邻网格选择不是连续插值。用于一维投影的PA先按投影对称性折叠，原始PA仍保留；这不能用于替代二维速度场的有向PA或旋转符号。

`spectral_extraction` 提供三种明确约定：

| mode | 源光比例与噪声面积 |
| --- | --- |
| `legacy_full` | 继承历史全源信号近似；空间profile长度参与噪声面积，不把旧height权重和自动解释为全天体孔径比例 |
| `explicit` | 用户提供 `source_fraction` 和 `spatial_pixels`，两者为显式模型输入 |
| `mge_strip` | 从同一正值Gaussian源mixture计算居中空间strip的源光比例及条件色散核 |

旧库的可追溯范围以实际provenance为准；某个旧生成程序的设置不能自动认证既有pickle的全部物理属性。新库有独立manifest绑定，但数学一致性也不替代仪器PSF定标。MGE模型、配置和推导见[形态指南](MGE_MODELS_zh-CN.md)。

## 4. 电子计数、噪声和波长误差

源和背景光子计数使用Poisson涨落，另计暗电流与Gaussian读出噪声。等权抽取N个空间像元、叠加多次曝光时，背景、暗电流和读出方差按实际像元/曝光数累加。源孔径比例只作用于源信号，不应乘到均匀天空、暗电流或读出噪声上。

扣除平均背景和暗电流后，含噪源信号允许为负；不截零。当前模型不额外模拟由独立背景估计产生的误差、宇宙线、flat-field残差或通用探测器相关噪声。

默认 `background_mode=fixed_template` 使用随包固定背景谱及其配置缩放；不能据RA/DEC和背景星等标签推出有观测时间/太阳几何约束的黄道光模型。针对实际观测用途需独立核对背景归一化和变化范围。

WCE在标称像元网格上以线性重采样表达，误差按平方权重传播，同时保存相邻像元 `covariance_lag1`。在当前单次重采样约定下，它用于跨像元测量的协方差账目；不是完整的仪器波长解反演或所有相关噪声模型。`paired-wce` 两分支共享噪声，不是独立曝光，不能按独立观测相加。

零响应区可仍有有限的预期探测器电子数，但无法得到有限的响应校准流量。应读取 `detector_model_mask`、`calibration_support_mask`、`spec_mask`、`snr_mask` 各自定义，不能合并成一个含糊的“好像元”判据。

## 5. 测光、零级和发射线

测光采用光子计数响应权重：

$$
\langle f_\nu\rangle_T=
\frac{\int f_\nu(\lambda)T(\lambda)\lambda^{-1}d\lambda}
{\int T(\lambda)\lambda^{-1}d\lambda} .
$$

模型星等与含噪观测值、统计/系统误差、S/N和检出标记分别保存。非正含噪流量仍可存在，不能强行转换成有限AB星等。测光孔径源光比例与像元数均是当前配置约定，应与光谱提取定义一起核查。

standard模式中 `/zero_order` 是通带积分测光；历史谱数组的 `_0th` 后缀是既有一维诊断proxy，不是CCD上的真实二维零级图像。v0.9.0没有模拟源之间的二维谱重叠或污染。

发射线模块区分本征峰、含噪候选、匹配、误报和漏检；输出的窗口线通量与S/N使用旁带仿射连续谱及当前协方差模型。候选峰不自动成为已确认原子线。配置中的40像元线窗口和20像元旁带仍是分析选择，其物理宽度由实际输出网格决定，未由仪器论文定标。固定窗口可能丢失线翼、混入邻线；从同一含噪数据选择中心还存在选择/定心偏差。窗口S/N不等于独立检出显著度，也不是完备度或污染率的实测认证。

## 6. 科研使用

研究中应记录采用的版本、科学revision、输入SHA、有效配置、随机种子和具体字段定义，并核对适用于研究目标的PSF/LSF、响应、背景、孔径、灵敏度和选择函数。旧科学revision的manifest不能用于续跑D5计算；应保留旧产品，另建输出/manifest或Gravity运行目录。
