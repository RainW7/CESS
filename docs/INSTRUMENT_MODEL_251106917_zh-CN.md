# 仪器参考参数与一维光谱模型

[English](INSTRUMENT_MODEL_251106917.md) | [简体中文](INSTRUMENT_MODEL_251106917_zh-CN.md)

CESS `0.9.0`、科学 revision `20261027-D5` 的随包 GU/GV/GI 配置采用以下参考。波长范围与中央分辨本领来自 [Zhang 等《Mock Observations for the CSST Mission: Main Surveys – the Slitless Spectroscopy Simulation》，arXiv:2511.06917v2](https://arxiv.org/html/2511.06917#S2) 表1；它们是设计/模拟参考值，并非在轨实测仪器定标。

## 参考参数与推导线宽

| 波段 | 下边界（Å） | 上边界（Å） | 中央波长（Å） | 中央波长处的 R | 推导参考 FWHM（Å） |
| --- | ---: | ---: | ---: | ---: | ---: |
| GU | 2550 | 4100 | 3370 | 287 | 11.74216028 |
| GV | 4000 | 6400 | 5250 | 232 | 22.62931034 |
| GI | 6200 | 10000 | 8100 | 207 | 39.13043478 |

FWHM 一列由 CESS 按 `central_wl/Res` 计算，不是论文独立列出的测量量。论文将 R 定义在中央波长处，没有提供完整的波长相关 LSF、Gaussian 轮廓测量，或光学 PSF 与像元响应的明确分解。

有关配置片段如下，须放入完整仪器 JSON，不能作为独立配置文件使用。资源路径的处理见[用户指南](USER_GUIDE_zh-CN.md#5-配置文件资源与路径)。

```json
{
  "instrument_lsf": {
    "mode": "central_resolving_power",
    "wavelength_law": "constant_fwhm"
  },
  "spectral_sampling": {
    "mode": "resolution_elements",
    "pixels_per_resolution_element": 4.0
  },
  "spectral_window": {"mode": "bandpass"},
  "gu_central_wl": 3370.0,
  "gu_Res": 287.0,
  "gu_min_wl": 2550.0,
  "gu_max_wl": 4100.0,
  "gv_central_wl": 5250.0,
  "gv_Res": 232.0,
  "gv_min_wl": 4000.0,
  "gv_max_wl": 6400.0,
  "gi_central_wl": 8100.0,
  "gi_Res": 207.0,
  "gi_min_wl": 6200.0,
  "gi_max_wl": 10000.0,
  "template_lsf": {"mode": "resolving_power", "value": 300.0},
  "template_lsf_broader_policy": "preserve"
}
```

## 仪器 LSF 与采样分别设置

`instrument_lsf.mode=central_resolving_power` 使 `*_central_wl` 与 `*_Res` 实际参与计算；显式的 `wavelength_law=constant_fwhm` 将推导宽度延拓至各自波段。目标是输出像元积分之前的 Gaussian 响应，因此参考 FWHM 固定时，模型的目标 R 随波长改变。这是一维近似，不能从三个中央 R 值推定全带恒定 R 或完整的焦面位置相关光学模型。

加载器在有效配置的 `*_dlambda` 中写入推导宽度，供既有代码读取。在这个模式中，它们是推导量，不是独立控制参数；应修改参考字段和声明的模型，而非仅修改缓存的线宽。

参考论文第3节采用局部线性的波长—迹长关系，其系数随焦面位置变化；第4节的估算使用四像元分辨单元和 0.074 arcsec/pixel 的空间尺度。CESS 将每参考 FWHM 四个采样点作为独立采样约定，未加载论文的光学迹线系数，也不以空间像元尺度推定实测色散。

对每个波段，CESS 设置

$$
N=\left\lceil\frac{\lambda_{\max}-\lambda_{\min}}{\mathrm{FWHM}_{\mathrm{ref}}/4}\right\rceil
$$

随后将精确的波段外边界之间均匀分成 N 个 bin，像元中心取各 bin 中点。

| 波段 | Bin 数 | 实际像元宽度（Å） |
| --- | ---: | ---: |
| GU | 529 | 2.93005671 |
| GV | 425 | 5.64705882 |
| GI | 389 | 9.76863753 |

N 向上取整使实际间距略细于 `FWHM_ref/4`。应读取产品的 `wave`、`wave_edges`、`pixel_width`；首末中心位于所列外边界以内。像元积分与源形态还可能使最终响应宽于声明的像元积分前目标。

`spectral_window.mode=bandpass` 按实际输入的波长中心确定探测器/S/N和发射线分析范围；在此模式中，旧的 `*_init_idx/end_idx` 不再是独立窗口控制量。配置中的40像元线测量窗口、20像元旁带及检出阈值仍是独立分析参数，其波长宽度随输出网格改变，未由参考论文完成灵敏度或提取定标。

## 输入模板响应

随包 `template_lsf` 为所用 BayeSED 输入库采用 Gaussian R=300 近似。这是单独记录的输入库假设，不是 CSST 仪器论文的结论；换用其他输入库时须重新核对。

在该 Gaussian 模型下，观测系需要补充的仪器宽度为

$$
\mathrm{FWHM}_{\mathrm{extra}}(\lambda)
=\sqrt{\mathrm{FWHM}_{\mathrm{ref}}^2-(\lambda/300)^2}
$$

该式用于根号内非负的位置，扣除的是声明的上游平滑方差，不能扣除天体真实速度弥散或发射线本征宽度。

GU 中，模板在约 **3522.65 Å** 以上比目标更宽。默认 `template_lsf_broader_policy=preserve` 在这些位置不再增加额外展宽，保留输入宽度，并提示目标未达到；不会锐化输入。选择 `error` 可拒绝这种情况；若需要更窄的响应，必须提供分辨率足够高的输入。已声明模板模型后，“LSF未知”的警告会消失，但“模板比目标更宽”的另一条提示仍保留。

其余输入模式 `unknown`、`delta`、`fwhm_rest` 的定义见[科学模型](SCIENCE_MODELS_zh-CN.md#2-响应积分像元与分辨率)。仅凭波长采样间隔，不能判断模板属于其中哪一种响应。

## 保留的资源与观测约定

Throughput FITS、GEMS统计、预存形态profile和天空背景谱保留原有内容。论文中的光栅本身效率要求不能替代包含光学、滤光片、光栅及探测器 QE 的总 throughput；现有零级 FITS 也不按一级效率的固定比例重缩放。

集光半径仍为100 cm，光谱曝光仍为150 s × 4，空间尺度仍为0.074 arcsec/pixel。暗电流、读出噪声、提取孔径、天空归一化及测光/零级曝光设计不变。论文演示的一次150 s曝光不规定CESS批次应采用的曝光次数；其中的 R80 也未被当成 Gaussian PSF FWHM，或用于重新解释既有形态库。

像元积分前的整体 Gaussian LSF 不能明确一项文献分辨率是否包含特定 PSF 贡献。进阶形态模型仍须明确PSF/LSF约定，不能仅凭一个R值再增加一次色散方向PSF。最终分辨率与灵敏度仍取决于模板、仪器、像元、形态、提取及噪声约定。

## 既有配置、产品与续跑

没有新模型键的外部配置仍采用早期 `fixed_fwhm` 仪器、`legacy` 采样和 `indices` 窗口约定；在该仪器模式下，`*_dlambda` 控制宽度，旧 `*_Res` 仅作描述。随包新默认值通过显式配置选择，不会在读取旧配置时自动转换为本参考模型。

输出 schema 仍为 structured 3.0 或兼容2.5。模型改变由科学revision `20261027-D5`、有效配置及provenance标识，实际波长几何存入各产品。比较结果时须保留这些信息：软件版本同为 `0.9.0` 不代表科学设置相同。

保留既有产品。D5重算应使用新输出目录和新manifest，不能用改变后的代码续跑旧科学revision的清单。Gravity批次也应在新运行目录重新prepare，之后仅在输入、配置、资源和软件身份不变时续跑；操作见[Gravity作业数组指南](../scripts/gravity/ARRAY_zh-CN.md)。
