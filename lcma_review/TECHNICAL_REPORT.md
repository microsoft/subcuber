# Cooperative FP16 LCMA：实现缺陷、因果验证与修复结果

## 版本和实验边界

原项目为 Microsoft SubCuber，实测历史提交 `b0672e829c49db76512096dc43d97ab1b733e02b`。保存的原版实验源码另外包含两处模板参数兼容修改；它们只补入 `ArchTag, OperatorClass`，与下文正确性修复分开提交。CUTLASS 是原项目依赖，原项目自带 `cutlass.patch`，并非本实验临时引入的替代算法。

原版保存的 1161 个文件经 Git blob、子模块提交、依赖补丁重建核验：288 个原项目文件直接一致，2 个文件仅有模板兼容改动，867 个 CUTLASS 文件直接一致，4 个 CUTLASS 文件与原项目依赖补丁重建结果一致。修复版在此基线上修改四个文件；完整检出同样对最终 1161 项 SHA256 核验通过。

唯一完成本轮修复/回归闭环的配置是 `HopperF16InterleavedPresumCooperativeMaxFusion_4x256_4x256_OptNo`：FP16 A/B/D 与预加和，FP32 累加，MMA tile 128×256×64，cluster 2×1×1，stage 数 4，A/B 预加和小块 4×256，单 stream，swizzle `{2,2,1,1,1,1,1}`，alpha=1、beta=0，无额外 padding 或输出转换。

七个子乘积 M0…M6 在此配置中融合为 G0={M0,M1}、G2={M2,M3,M6}、G4={M4,M5}。原调度 G0 后将 G2/G4 放在同一 launch，修复后为同一 stream 上 G0→G2→G4。组的编号不是 launch 顺序。

## 初始现象与分类

原版普通运行接受下列输入，返回成功且输出有限，但数值检查失败。初期观察的范围与最终发布回归采样不是同一批调用，不能将范围与调用次数混用。

| M×N×K | 初期超差输出比例 | 相对 L2 | 空间和重复性表现 |
|---|---:|---:|---|
| 4096³，两个种子 | 0.77%～0.93% | 0.1252～0.1518 | 首次为 256×512 局部矩形，重复调用可变化 |
| 4608³ | 9.64%～11.55% | 0.5663～0.5832 | 多处错误，定位运行中预加和顶部异常 |
| 8192²×4224 | 72.99%～73.08% | 11.9383～12.7069 | 大范围错误，预加和有周期带状异常 |
| 8192²×4352 | 72.25%～72.28% | 0.3788～0.3829 | B 预加和每 128 行前 8 行异常 |
| 8192²×4480 | 72.82%～72.87% | 13.2551～13.2679 | 大范围错误及预加和周期带状异常 |
| 8192²×4736 | 约 72.82% | 13.6580～13.6581 | 大范围错误及预加和周期带状异常 |

这些失败已定位为实现缺陷，包括未满足依赖的读取、共享存储过早复用、逻辑分区越界写回。既有局部输出错误，也有边界错误经乘法传播后的大范围错误。错误元素比例不代表独立 bug 数量；“73% 输出错”不能写成“存在大量独立 bug”。

在本次冻结 normal 输入域中，没有证据将以上失败归为算法固有的浮点稳定性问题。修复没有提高精度、放宽阈值或换成 Classical GEMM。其他输入分布是否数值稳定仍须另行实验。

诊断工具报告与普通数值检查分开解释：racecheck 的 hazard 指工具检测到的潜在共享内存访问冲突；0 hazard 不证明输出正确。未取得整块显存分配以外非法地址的确认报告，逻辑分区跨界仍可落在合法 workspace 内。

## 根因 1：G2→G4 的全局中间量依赖未排序

原配置 G2 中 M3 产生 T2，G4 中 M4 读取 T2；两者却处于同一个 `ParallelMiGroups`，同次 launch 不提供所需的跨 CTA 完成顺序。CTA 内同步或 TMA 完成等待不能替代另一个 CTA 的生产完成。

[原配置与错误调度，第 58～82 行](https://github.com/microsoft/subcuber/blob/b0672e829c49db76512096dc43d97ab1b733e02b/src/cuda/kernels/hopper/strassen_winograd/hopper_f16_sw_interleaved_presum_cooperative_max_fusion.cuh#L58-L82)。

4096³ 首次错误矩形为行 2048～2303、列 2048～2559；应有 D11=T2+M4，错误区域两个种子各约 99% 元素却与受控 M4 预测逐位吻合。它支持“漏读尚未就绪的 T2”，而非只依据大误差猜测访存错误。

修复仅拆分依赖组，保持组内融合、公式、tile 和精度。仅改调度的单因素版本，普通检查 3/3 通过，但工具运行仍有其他共享同步失败；不能把这一步说成最终完全正确。

## 根因 2：共享预加和缺少完整生命周期同步

[原 A 尾部读，第 1574 行起](https://github.com/microsoft/subcuber/blob/b0672e829c49db76512096dc43d97ab1b733e02b/include/cutlass/gemm/collective/sm90_strassen_mma_tma_gmma_ss_warpspecialized.hpp#L1574-L1590)、[B 尾部读，第 1666 行起](https://github.com/microsoft/subcuber/blob/b0672e829c49db76512096dc43d97ab1b733e02b/include/cutlass/gemm/collective/sm90_strassen_mma_tma_gmma_ss_warpspecialized.hpp#L1666-L1682)、[mma 尾部，第 1988 行起](https://github.com/microsoft/subcuber/blob/b0672e829c49db76512096dc43d97ab1b733e02b/include/cutlass/gemm/collective/sm90_strassen_mma_tma_gmma_ss_warpspecialized.hpp#L1988-L1996)。

线程各写 4 个 FP16，而尾部按 8 个 FP16 重新读取，读写分工不同，可能读取另一 consumer warp-group 尚未写完的数据；尾部未读完时，epilogue 又可复用同一共享地址。实际存储地址表明 presum stage3 完全落在输出 D 缓冲的复用范围内，不只是变量名称相近。

原 4096³ racecheck 报 8 组共享访问 hazard。只补 A/B 读前同步，工具报告归零但数值仍失败；结合调度修复后，捕获到 B31 中 2560 个错误值，最终输出有 2723151 个元素超差。用捕获的 ΔB31 预测 ΔM3=A02·ΔB31，与下方输出块变化高度吻合。

最终在 A/B 尾部读取前、mma 尾部完成后，条件化加入参与数为 256 的 consumer named barrier，分别建立“写完再读”和“读完再复用”顺序。增加尾读完成同步后，八份预加和恢复，指定普通/工具检查通过。三个位置均保留在累计修复中，不能只保留读前同步。

## 根因 3：4608³ 补齐 tile 未检查二维逻辑边界

半矩阵为 2304×2304，128×256 的逻辑 tile 网格为 18×9；当前 swizzle 与 cluster 要求将调度网格补齐为 20×10。额外 tile 可参与 pipeline，但不应写入真实逻辑子矩阵以外的区域。

[GlobalIterator 原版第 139 行](https://github.com/microsoft/subcuber/blob/b0672e829c49db76512096dc43d97ab1b733e02b/include/cutlass/gemm/threadblock/presum_detail.h#L139)：A 只检查列、B 只检查行，漏掉另一维。保留根因 1/2 修复后，两份 A 预加和顶部 256 行异常，数值与 A2、0、−A2 的覆盖预测一致。补齐二维预加和边界后，中间量恢复，但最终输出仍错。

进一步核对 [store_m2 第 1096 行](https://github.com/microsoft/subcuber/blob/b0672e829c49db76512096dc43d97ab1b733e02b/include/cutlass/epilogue/collective/sm90_strassen_epilogue_tma_warpspecialized.hpp#L1096) 与 [store 第 1423 行](https://github.com/microsoft/subcuber/blob/b0672e829c49db76512096dc43d97ab1b733e02b/include/cutlass/epilogue/collective/sm90_strassen_epilogue_tma_warpspecialized.hpp#L1423)，写回只选择发出线程，没有判断 m/n 是否在逻辑范围内。中间槽按 `m*9+n` 编址，使无效 (0,9) 与有效 (1,0) 同为索引 9；无效 (18,0) 更进入下一个中间槽。

修复包含预加和二维有效性检查和两处 epilogue 发出条件的 m/n 边界；不直接提前退出整个 CTA，保留 pipeline 消费和同步。4608³ 两个检查及两个控制场景各两次，共 6/6 通过。这不是六次都测 4608³。

## 根因 4：把迭代序号误当成环形 stage 位置

[原预加载第 1204 行](https://github.com/microsoft/subcuber/blob/b0672e829c49db76512096dc43d97ab1b733e02b/include/cutlass/gemm/collective/sm90_strassen_mma_tma_gmma_ss_warpspecialized.hpp#L1204)、[原等待第 1286 行](https://github.com/microsoft/subcuber/blob/b0672e829c49db76512096dc43d97ab1b733e02b/include/cutlass/gemm/collective/sm90_strassen_mma_tma_gmma_ss_warpspecialized.hpp#L1286)。

4 个共享 stage 循环使用，其中 stage2/3 与 epilogue 地址重叠。原实现先预加载两次，到第 3 次才等待，隐含前两次总写 stage0/1；实际 pipeline 状态从 M0 延续到 M1，没有回到 0。

| K | K/128 | 首个 M1 的前两个 stage | 固定“前两次不等”的结果 |
|---:|---:|---|---|
| 4096 | 32 | 0、1 | 这一相位条件不触发 |
| 4224 | 33 | 1、2 | 提前写入重叠 stage2 |
| 4352 | 34 | 2、3 | 一开始即写重叠区域 |
| 4480 | 35 | 3、0 | 一开始即写重叠 stage3 |
| 4608 | 36 | 0、1 | 这一相位条件不触发 |
| 4736 | 37 | 1、2 | 提前写入重叠 stage2 |

K4352 的 B 预加和每 128 行前 8 行坏，吻合前两次各 4 行的加载范围。修复让 prologue 只在实际写 stage<2 时提前加载，首次进入 stage>=2 前等待旧 epilogue 释放，并只 advance 一次。只改相位，K4352 两次通过；另外三个 K 的 A/B31 恢复，而 B 后三份仍错，说明还存在独立的根因 5。

## 根因 5：B 部分尾块写回跨越下一预加和分区

[原 B 写回循环，第 1258～1275 行](https://github.com/microsoft/subcuber/blob/b0672e829c49db76512096dc43d97ab1b733e02b/include/cutlass/gemm/collective/sm90_strassen_mma_tma_gmma_ss_warpspecialized.hpp#L1258-L1275)。旧代码只检查 128 行大 tile 起点，后续每个 4 行写回没有单份 `halfK` 边界。

四份 B 预加和连续存放，并共用覆盖四份的 TMA 描述符。K4224 时每份 halfK=2112 行，最后 tile 从 2048 行开始，只有 64 行有效；生产者实际发出了 116 行，额外 52 行写入下一份顶部。越过一份逻辑边界仍可能位于整块合法分配内。

| K | 尾 tile 实际写回 | 单份剩余有效行 | 下一分区覆盖行数 |
|---:|---:|---:|---:|
| 4224 | 116 | 64 | 52 |
| 4480 | 124 | 64 | 60 |
| 4736 | 128 | 64 | 64 |

修复根因 4 后，B 后三份顶部错误的行数与上述预测吻合，值又吻合 −B3、B3−B2、B2−B3 的覆盖预测。不是仅看形状猜测。

最终给 B 的每次 4 行 TMA 写回增加局部行起点 `< halfK` 条件。在本次 halfK 为 64 对齐的测试域内，4 行块起点合法即可保证整个 4 行块合法；不能将这一写法宣传为已支持任意非对齐残块。

在相位修复基础上增加此项，四个 K 失败场景各两次通过，八份预加和均恢复；4608³ 仍需根因 3 的独立边界修复，最终版本合并全部修改。

## 最终正确性、性能与限制

冻结判据：输出有限、逐元素 `|error|<=0.01+0.01|reference|`，相对 L2<=0.001。输入为 `normal_scaled_by_sqrtK`，历史数组文件以 SHA256 固定。参考为禁 TF32 的 FP32 GEMM，不称为数学精确值。

| 场景 | 原版 | 最终修复版 |
|---|---:|---:|
| A：原来正确的 7 个输入点 | 42/42 | 42/42 |
| B：原来失败的 7 个输入点 | 0/42 | 42/42 |

修复相对 L2 范围约 0.00082606～0.00082742。诊断候选另有 113/113 检查，包括跨 shape 状态复用和三个工具场景；这些不冒充最终 release 的逐点工具全覆盖。旧节点 8192³ 一次工具下数值失败未完整归因，保留为限制，不能删除反例后宣称任意情况正确。

性能在同一 H100 上、无其他 GPU 作业时交错测量 9 轮。主要指标 prepared GPU span 包括原生 launch 序列及其空隙，排除 workspace 分配和 initialize；完整 helper 的 host 时间另列。paired change 是每轮修复/原版延迟比的中位数变化，未必等于两列延迟中位数直接相除。

7 个 A 点延迟比等权几何平均 +1.60%；6144³ 从 0.756809 ms 至 0.833861 ms，配对 +10.37%，描述性 bootstrap 95% 区间 +9.47%～+10.88%。其他 A 点约 −0.26%～+1.27%；小变化结合波动解释。CV 为标准差/均值，衡量测量相对波动。

完整 helper 波动较大，等权平均 +4.65% 仅作描述；不能与 prepared 指标混用。workspace 在 14 个输入点上保持不变。各个修复项没有单独的性能归因实验，不能把最终额外耗时全部归于某一个 barrier 或额外 launch。

归档结果包含 504 条性能记录、8316 个计时调用。prepared 每批 30 次只检查批末输出，不能声称每次计时调用都单独完整评分。原版 B 的错误输出耗时只用于诊断，不参与有效吞吐/加速比结论。

本次交付恢复了完整仓库与固定依赖，逐项匹配实测源码；新脚本的检查属于本地交付验证。尚需在新授权 H100 上完成完整检出的新构建/运行验证。其他配置、任意 shape、混合尺度/相消输入、多 stream 均不在本次正确性保证内。
