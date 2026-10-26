# 原生 HDF5 3.0 输出与兼容边界

[English](OUTPUT_LAYOUT.md) | [简体中文](OUTPUT_LAYOUT_zh-CN.md)

适用于CESS v0.9.0。schema3.0整理存储和访问接口；2.5兼容输出保留。
参数表与光谱字段按名称、单位、axis_order及definition属性读取；
常见科学量的解释见[用户指南](USER_GUIDE_zh-CN.md#9-读取产品及物理量)。

| 3.0 位置 | 2.5 来源及含义 |
| --- | --- |
| `/catalog` | ID、源/父行、参数表、data_mask、input_quality_flags、提取和 MGE 逐源诊断 |
| `/spectra/GU`、`GV`、`GI` | 原波段组中的波长、流量、电子计数、误差、S/N、响应、掩膜和 lag1 covariance；leaf 名称保留 |
| `/photometry` | 七波段完整 model/observed/error/SNR/status，开启时存在数据 |
| `/zero_order` | standard 的真实零级积分测光；不把历史 `_0th` 谱 proxy 改成真实零级谱 |
| `/lines/GV`、`GI` | 匹配峰、误报/漏检诊断和窗口线测量；旧 EL-only 根 `gv_*` / `gi_*` 去掉波段前缀后移入相应组 |
| `/provenance` | 原根 attrs、完整运行/配置/源码/资源/随机流身份，及本次布局映射 |

六根组都存在。`cess_layout_status` 明确 present、disabled 或 not_applicable；
N=0 仍保留共享波长及正确 `(0,W)` 数组。EL-only 不复制父光谱大数组。
GU/GV/GI 的 W 可以不同；谱轴波长为 Angstrom，流量/误差通常为 microJy，
电子率和电子计数需分别读取字段的单位/定义，不能仅凭名称推断。

根 attrs 声明物理外壳 schema 3.0、软件版本、mode 和 product_kind。
`/provenance.attrs['schema_version']='2.5'` 保留原科学 payload 的语义身份，
不是把文件重新声明为 2.5；严格 resume 同时检查外壳和 payload。
原 root attrs 的 dtype/shape 保留，谱和目录 dataset attrs 同样保留。

`/provenance/conversion` 中 `mapping_json` 明确 original→stored 路径。
原生写入先在未发布的临时文件内生成科学数据，再移动 HDF5 对象；不复制整套 N×W 数组，
也不先发布 2.5 文件再修改。`operation='native_layout_adapter'` 不伪造中间文件 SHA。
独立旧文件转换使用 `operation='legacy_file_conversion'`，记录真实源路径、SHA 和转换块大小。

未知旧字段按映射保存到 `/provenance/extensions`，不会仅因长度等于 N 就认定为源轴。
外链、软链、对象引用等不能可靠重映射的存储会显式拒绝，不静默丢字段。
`validate_structure(...)` 检查六根组、状态、header、映射全覆盖、重复/冲突和元数据一致性，
不读取全部谱数组，也不据此认证数值物理模型。

`open_product(...)` 返回只读统一接口；`catalog`、`band(...)`、`line_group(...)`、
`parameter_table(...)` 和 `read_row(...)` 读取 2.5 / 3.0。
`legacy_view()` 为已有算法提供老路径，返回实际 h5py Dataset；
EL 峰表可以通过原根名和便捷 band 名访问，同一个对象只遍历一次。
这属于逻辑访问，不在磁盘复制旧路径。必须在 open_product 上下文内使用这些对象。

`SourceSelection` 只切已声明的源字段。源/父行、参数表、谱数组、提取和 MGE 诊断按源切片；
共享 wave/response/bin geometry、稀疏/变长线表不根据相同长度猜测轴。
ID dtype/编码原样保留；不依靠 ID 唯一性重建原行。

manifest 同时绑定 output_layout、真实源 schema、配置、种子、源码/资源/输入 SHA、
行区间和期望路径；3.0 通过外壳验证后还需通过原有完整科学结构核验。
布局互换、缺协方差/错误 MGE 行、损坏映射等不会被兼容 view 掩盖。
拒绝已有输出，包括符号链接；不会自动修复、覆盖或收养未经认证的历史结果。

抽样产品继续使用 `extracted-mock-spectra-1.1`，不改成完整模拟的 3.0。
筛选沿用 GV/GI mean S/N 均至少 3，然后按红移 qcut 分层抽样。
保留实际可用的误差、掩膜、协方差及行来源，缺失字段使用 availability 标识；
父元数据和共享几何按父文件去重保存。不会为缺失父字段补造测量值。

命令用法见[用户指南](USER_GUIDE_zh-CN.md)。批量任务须使用明确的输出布局、源区间和运行清单。

`cess merge -i chunks -o merged/complete.hdf5` 将同一次模拟的连续分片合为一份自包含 HDF5。
它保留输入布局和原始模拟身份，不重新计算科学量；复制按有限行块进行，
只对稀疏发射线表中的局部源行引用加入合并后的行偏移。波长、响应等共享数组只保存一份。
`--copy-rows` 限制单次复制的行数；最终完整性检查仍需读取发射线表，内存会随峰数量增长。
分片的行区间必须连续且不重叠，配置、种子、运行版本和共享字段必须一致。
使用 `--input-source input.hdf5` 可额外核对原始输入 SHA、完整覆盖和逐行 ID。
schema3.0 中的合并历史经原布局映射保存在 `/provenance/extensions/root/merge_history`，
保留分片来源及原 attrs；schema2.5 对应根 `merge_history`。原始运行manifest继续描述原分片，
合并结果不冒充原来某个任务的产物；它也不依赖分片外链或 VDS。

逐源 `catalog/input_quality_flags` 为 uint32 位掩膜，0 表示通过当前输入安全检查，非零表示无效，且对应 `data_mask=2`。其 `flag_masks` / `flag_meanings` 属性给出各位含义，`policy_json` 保存具体范围、单位和处理策略。旧产品没有该字段时表示未记录该项检查，不能据缺失字段推断源已通过。EL 派生产品按源切片继承标志及其属性。
