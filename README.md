# FlashAttention 的 Tensor、Vector 算力与加载带宽配比

以 vLLM v0.11.2 V1 的 NVIDIA `FLASH_ATTN` 分页 decoder 前向为例。输入为 FP16/BF16，score、分母和输出累加器按 FP32 处理。主算例取 FA2 实际调度的 $64\times128$ tile，Tensor 目标取本项目已有的 **128 TFLOP/s**。这里的带宽是为给定计算速率配套的**有效载荷速率**，用十进制 GB/s、TB/s。容量用 KiB，矩阵 FMA 计 2 FLOP。检索源码、数值对照和性能测量各自的范围见第 7 节。[项目已有资源模型](../01_pingpang_bufer_perf_analyze/inference_resource_model.md) · [固定版本源码附录](kernel_sources.md)

| 设计问题 | 在 $d_k=d_v=128$ 主算例中的答案 |
|---|---|
| Tensor 每产生多大工作量，需要 Vector 处理一个 score？ | QK 做 128 MAC，PV 做 128 MAC，合计 **512 FLOP/score**。128 TFLOP/s 对应 **250 Gscore/s**。 |
| Vector 要配多少算力？ | 独立 MUL、ADD/SUB、REDMAX、REDSUM、EXP 的参考映射分别需要 **501.89、253.78、248.05、248.05、251.89 G/s**，另计行 MAX、倒数、转换及 LSE。MUL 中恰有 **250 G/s** 用于跨块缩放输出及最终归一化。 |
| Tensor 加载数据要多快？ | 单个 KV 头供 64 行复用时，K、V 各 **1.00 TB/s**。Q 从源存储取一次只需 **15.625 GB/s**。若从片上 Q 缓冲每 KV 块重新装填一次，那个接口需要 **0.50 TB/s**。PV 还须消费 **0.50 TB/s** 的逻辑 P 载荷。 |
| Vector 的 LD/ST 要多快？ | 假设独立引擎经共享 SRAM 交接 S/P，Vector 读 S **1.00 TB/s**、写 P **0.50 TB/s**。两端都计，SRAM 为此服务 **3.00 TB/s**。本例 vLLM FA2 通过寄存器 fragment 交付 S/P。若旧输出也通过 SRAM 交给 Vector 重标并送回，Vector 读、写每方向再加 **0.969 TB/s**。 |
| 换成 decode 还需要这组带宽吗？ | 单 token、GQA 每份 KV 供 4 个头复用时，BF16 的理想 KV 算术强度降到 **4 有效 FLOP/B**。512 GB/s 的冷 KV 源带宽最多支撑 **2.048 有效 TFLOP/s**。矩形填充、split 与有效吞吐应分别计算。 |

## 模型前提、参数与计算式

**适用范围。** 主算例固定 vLLM v0.11.2、单 GPU、分页 FA2、因果 decoder 前向、单分区。只计从已就位的 Q/K/V 到 O 和 LSE 的核心计算。QKV 投影、RoPE、cache 追加和输出投影单独计费。关闭 dropout、softcap、额外 bias、滑窗与 cascade。[版本及内核分派](kernel_sources.md)

**执行假定。** QK 和 PV 共用同一 Tensor 阵列，一个 MAC 计 2 FLOP。Vector 按独立 MUL、ADD/SUB、二元归约和指数计数，每行每个非首块计算一份重标系数并广播。旧输出经缩放后直接作为 PV 的累加输入，末块结束才做最终归一化。带宽按单份有效载荷计，Q 跨 KV 块驻留，K/V 在一个 Q 块内复用，外存示例假定跨 CTA 冷读。S/P 经共享 SRAM 是独立双引擎设计假设，核对的 FA2 使用寄存器 fragment。吞吐预算假定有足够独立工作，使用可持续矩阵速率；单 CTA 延迟与瞬时峰值另行核算。

| 参数 | 含义 | 主算例 |
|---|---|---|
| $p,q,L$ | 已缓存前缀、本轮新增 query 数、最终 KV 长度，$L=p+q$ | 4032、64、4096 token |
| $H_q,H_{kv},G$ | Q/KV 头数、组头比 $G=H_q/H_{kv}$ | 预算按一个 Q 头，无跨头复用 |
| $d_k,d_v,s$ | Q/K 头维、V 头维、score 缩放 $s=1/\sqrt{d_k}$ | 128、128、$1/\sqrt{128}$ |
| $M,N,P_{\rm page},T$ | Q 块行数、KV 计算块宽、cache 页长、该 Q 块扫描的 KV 块数 | 64、128、16、32 |
| $b_Q,b_K,b_V,b_P,b_O$；$b_S,b_U$ | Q/K/V、PV 输入与最终 O 的字节数；score 与未归一化累加器的字节数 | 前五项 2 B，后两项 4 B，行统计量 FP32 |
| $m_{\rm reuse},T_Q$ | 指定存储边界上一份 KV 服务的行数、一份 Q 装入后服务的 KV 块数 | KV 供 64 行；Q 源端取 32，逐块装填端取 1 |
| $P_T,C_i,B_e$ | Tensor 持续 FLOP/s、Vector 单元 $i$ 的服务率、接口 $e$ 的有效 B/s | Tensor 目标 128 TFLOP/s，冷外存例取 512 GB/s，Vector 需求由下式反推 |

**从请求得到计数。** 单头的因果有效配对为 $A_{\rm valid}=pq+q(q+1)/2$。对实际发射的每个矩形块 $b$，记执行行、列数为 $M_b,N_b$（包含发射的填充槽），则 $A=\sum_b M_bN_b$ 为执行槽数，$J=\sum_b M_b$ 为行块访问数，$R$ 为各独立扫描首次处理的执行行总数，$H=J-R$ 为非首块行访问数。本例 $A_{\rm valid}=260128$，而 $A=MTN=262144,\ J=MT=2048,\ R=M=64,\ H=1984$。

| 工作量 | 数学表达 |
|---|---|
| Tensor FLOP | $F_{QK}=2Ad_k,\quad F_{PV}=2Ad_v,\quad F_T=2A(d_k+d_v)$ |
| Vector 乘、加减 | $W_{\rm MUL}=A+H+d_vJ,\quad W_{\rm ADD/SUB}=A+2H$ |
| Vector 归约和指数 | $W_{\rm REDMAX}=W_{\rm REDSUM}=A-J$（各按二元操作），$W_{\rm MAX合并}=H,\quad W_{\rm EXP}=A+H$ |
| 收尾与转换 | RCP、LSE 的 LOG 和 ADD 各 $R$ 次。P 转换 $A$ 次，最终 O 转换为有效输出行数乘 $d_v$ |
| 时间、所需吞吐和算力比 | $t_T=F_T/P_T,\quad x=A/t_T,\quad C_{i,\rm req}=W_i/t_T,\quad \text{Tensor∶Vector}_i=(F_T/W_i)∶1$（FLOP∶元素操作） |

**从算力得到带宽。** 在满 tile 的稳态下，Q、K、V 各自所需的读带宽为 $x b_Qd_k/(NT_Q)$、$x b_Kd_k/m_{\rm reuse}$、$x b_Vd_v/m_{\rm reuse}$。S/P 各交接一次时，Tensor 写 S 与 Vector 读 S 均需 $xb_S$，Vector 写 P 与 Tensor 读 P 均需 $xb_P$。因此仅计该交接的两侧读口比为 $b_P∶b_S$，写口比为 $b_S∶b_P$。若旧输出经 Vector 重标再回送，每方向另需 $b_Ud_vH/t_T$。对任意接口按 $B_{e,\rm req}=D_e/t_T$ 折算，$D_e$ 只累计穿过该接口的字节。满足这些平均需求是持续喂满 Tensor 的必要条件，实际还需满足依赖、容量和端口突发要求。

## 1. 从标准 attention 到逐块在线计算

对一个 Q 头，标准前向为 $O=\mathrm{softmax}(sQK^\top+\mathrm{mask})V$，通常 $s=1/\sqrt{d_k}$。GQA 中多个 Q 头映射到同一个 KV 头，每个 Q 头仍有自己的 softmax。以 vLLM 的 Qwen2 为例，整层按 QKV 投影 → 拆出 Q/K/V → Q/K 做 RoPE → attention → 输出投影执行。两侧投影也属于 Tensor 工作，RoPE 包含逐元素乘加，cache 追加包含 slot 映射和 K/V 读写。下面的配比以 attention 核的 Q、已就位的 KV cache 为输入，cache 追加单列在第 4 节。这样可以分别估算 attention 核与整层时间。[vLLM Qwen2Attention.forward](https://github.com/vllm-project/vllm/blob/275de34170654274616082721348b7edd9741d32/vllm/model_executor/models/qwen2.py) · [vLLM cache 更新及 attention 调用](kernel_sources.md)

取 $M$ 行 query 和 $N$ 列 key，$Q[M,d_k]$、$K[N,d_k]$、$V[N,d_v]$。省略行下标，跨 KV 块保留最大值 $m$、分母 $\ell$、未归一化输出向量 $U\in\mathbb{R}^{d_v}$。处理到第 $t$ 块时，记已访问的有效 key 集合为 $\mathcal K_t$，则 $\ell=\sum_{j\in\mathcal K_t}e^{S_j-m}$，$U=\sum_{j\in\mathcal K_t} e^{S_j-m}V_j$。新块改变最大值时，历史分母和历史分子的每一项都要换到新的指数基准。采用 FA2 的延迟归一化形式，每行只在末尾除一次。[FlashAttention 原论文](https://arxiv.org/abs/2205.14135) · [FlashAttention-2 算法 1](https://arxiv.org/pdf/2307.08691)

| 顺序 | 一块的操作 | 运算与数据依赖 |
|---|---|---|
| 1 | $X=QK^\top$，$S=sX$，无效位置置 $-\infty$ | Tensor GEMM，$2MNd_k$ FLOP，FP32 score。scale 可移位或融合。分页寻址、边界判定、掩码选择由相应整数/谓词通路执行。 |
| 2 | $m'=\max(m,\mathrm{rowmax}(S))$ | 沿 KV 维对每行归约，再合并历史 max。后续指数依赖 $m'$。 |
| 3 | $P_j=\exp(S_j-m')$，$\alpha=\exp(m-m')$ | 一个指数对应一个 score，另一个指数对应一行的一个非首块。$P$ 是相对当前最大值的未归一化权重。 |
| 4 | $\ell'=\alpha\ell+\mathrm{rowsum}(P)$ | FP32 行求和及旧分母缩放。求和使用 FP32 的 $P$，送往 PV 的副本可转 BF16/FP16。 |
| 5 | $U'=\mathrm{MMA}(P,V,\alpha U)$ | 先把旧输出的 $d_v$ 个分量乘 $\alpha$，再由 Tensor 做 $2MNd_v$ FLOP 的 PV 并累加。$P$ 就绪即可开始 PV，完整行分母只在归一化时需要。 |
| 6 | 扫完该行后，$O=U/\ell$，按需生成 $\mathrm{LSE}=m+\log\ell$ | 每行一次倒数、每个输出分量一次乘法，输出转换和写回。本文核对的 FA2/FA3 常规路径还会写 FP32 LSE。 |

首个有效块直接初始化 $m,\ell,U$，省去旧状态重标。数学模型可跳过全 mask 块，全空行单独处理，避免 $-\infty-(-\infty)$。实际 kernel 依靠块裁剪、掩码、无穷值和零分母分支处理边界，FA2 与 FA3、单分区与多分区的 LSE 哨兵有差别。若一行最终仍为空，应按选定后端的输出约定校验。[FA2 在线更新与收尾](https://github.com/vllm-project/flash-attention/blob/58e0626a692f09241182582659e3bf8f16472659/csrc/flash_attn/src/softmax.h#L137-L185) · [FA3 在线更新与收尾](https://github.com/vllm-project/flash-attention/blob/58e0626a692f09241182582659e3bf8f16472659/hopper/softmax.h)

## 2. 全过程的执行单元工作量

令 $A$ 为参与计算的 score 槽数，$J$ 为「一行访问一个 KV 块」的总次数，$R$ 为扫描的行数，$H=J-R$ 为首块以后的行块数。例如 $M$ 行都扫描 $T$ 个完整的 $N$ 列块，则 $A=MTN,\ J=MT,\ R=M,\ H=M(T-1)$。做数学有效工作分析时，只计非空行和有效列。做实际发射预算时，按各操作真正执行的槽数统计；下表的矩形模型对访问的 tile 按完整形状计。尤其要把输出有效行与执行的填充行分开。

| 资源 | 总服务量 | 折算口径 |
|---|---:|---|
| Tensor QK / PV | $2Ad_k$ / $2Ad_v$ FLOP | 两者共用一套矩阵阵列时，服务时间相加。若吞吐分别为 $P_{QK},P_{PV}$，则 $t_T=2Ad_k/P_{QK}+2Ad_v/P_{PV}$。 |
| MUL | $A+H+d_vJ$ | $A$ 次 score 缩放，$H$ 次缩放旧分母，$d_vH$ 次缩放旧输出，$d_vR$ 次最终归一化。 |
| ADD/SUB | $A+2H$ | score 减行最大值 $A$，旧/新最大值之差 $H$，旧/新分母相加 $H$。假定 PV 吃得下已重标的累加器。 |
| 行 MAX 合并 | $H$ 次比较 | 可并入 REDMAX 或比较 ALU，计在实际占用的一侧。 |
| 块内 REDMAX / REDSUM | 各 $A-J$ 个二元操作 | 一行 $n$ 个输入各需 $n-1$ 次比较/加法。若归约器规格是输入元素/s，则两种操作各吞入 $A$ 个元素、各产生 $J$ 个块结果。 |
| EXP | $A+H$ | 逐 score 与行系数共用指数资源。若实现用 EXP2，则用 EXP2 的可持续吞吐。 |
| RCP，LOG 与 LSE | RCP 为 $R$，产出 LSE 时另有 $R$ 次 LOG 和 $R$ 次 ADD | 定义已缩放 $S$。源码若保存未缩放的 max，LSE 输出还含行 scale。RCP、LOG、EXP 共用 SFU 时累加各自占用。 |
| CVT、打包、布局 | $A$ 次 P 降精度，有效输出行数 $\times d_v$ 次 O 降精度 | 半精度打包指令可能一次转换多个元素。重排、跨 lane shuffle、广播、掩码、整数寻址、异步拷贝和同步另按实现计。 |

表内 MUL/ADD 是**可拆分运算**。例如芯片用 FFMA 完成 score 的乘常数再减 max，应把相应的 $A$ 次 MUL 和 $A$ 次 SUB 换成 $A$ 次 FFMA。若 scale 预乘 Q 并跨块复用，score 缩放改为每个 Q 元素一次，约 $Rd_k$ 次，同时考虑额外舍入。若 PV 先产生独立部分向量再由 Vector 合并，额外计 $d_vH$ 次向量 ADD 及部分向量交接；项目原资源模型采用了这种计数。最大值不变时 $\alpha=1$，固定版 FA2/FA3 所核对的常规循环仍计算这个系数。[旧模型第 4 节](../01_pingpang_bufer_perf_analyze/inference_resource_model.md) · [FA2 的 PV 累加落点](kernel_sources.md)

固定目标 $P_T$ 且两种 GEMM 同速，令 $x$ 为每秒执行的 score 槽数，则
$$
x=\frac{P_T}{2(d_k+d_v)},\qquad r_i=x\,\frac{W_i}{A}.
$$
其中 $W_i$ 是该单元总服务量。对完整 tile，$\mathrm{EXP}/\mathrm{score}=1+(1-1/T)/N$，$\mathrm{MUL}/\mathrm{score}=1+d_v/N+(1-1/T)/N$。**头维控制每个 score 对应多少 Tensor 工作，KV 块宽控制每个 score 分摊多少行状态和输出重标工作。** $d_v/N$ 可以接近或超过 1。此时只给每个 score 配一次乘法会低估 MUL 需求。

## 3. vLLM 的实际设置如何进入模型

固定 vLLM 提交 `275de341` 与默认构建的 FlashAttention fork `58e0626a`。V1 在后端、设备能力及功能检查后选择实现。选中 `FLASH_ATTN` 时，此版本在支持 FA3 的 SM90 上默认用 FA3，其他受支持配置通常用 FA2，版本覆盖和 ALiBi 等要求会影响选择。两者均可从 `flash_attn_varlen_func` 进入。下表中的计算块宽、KV cache 页长、一次请求的 Q 长度，是三个独立量。[后端选择、版本及完整调用链](kernel_sources.md)

在此版本复现分派时，可用 `VLLM_ATTENTION_BACKEND=FLASH_ATTN` 选后端，`VLLM_FLASH_ATTN_VERSION=2` 选 FA2，`--block-size 16` 指定 KV cache 每页 16 个 token。计算 tile 由编译好的 kernel 依 head dim 和分派条件选择。每轮实际 Q/K 长度来自调度元数据，不能用配置的最大上下文替代。FA3 的普通路径传 `num_splits=0` 以使用启发式，整图 CUDA Graph 路径可供给 split 设置，batch-invariant 模式设为 1。[vLLM 环境参数](https://github.com/vllm-project/vllm/blob/275de34170654274616082721348b7edd9741d32/vllm/envs.py) · [cache 命令行参数](https://github.com/vllm-project/vllm/blob/275de34170654274616082721348b7edd9741d32/vllm/engine/arg_utils.py) · [split 元数据构建](https://github.com/vllm-project/vllm/blob/275de34170654274616082721348b7edd9741d32/vllm/v1/attention/backends/flash_attn.py#L332-L341)

| vLLM / kernel 设置 | 对计算、加载或配比的影响 |
|---|---|
| 分页 FA2，$d=64,128,256$ | 分派选择 $B_M=64$，相应 $B_N=256,128,64$，四个 warp。Q/K/V 从源内存进入片上缓冲。QK 的 FP32 fragment 直接进入在线 softmax，转窄后的 P 在寄存器中送 PV，旧 O 在寄存器中重标。用到 `splitkv` 内核族，也可以只产生一份最终输出。[FA2 分派与代码落点](kernel_sources.md) |
| 分页 FA3，SM90，BF16，$d=128$ | causal 或 non-TMA paged 时 $B_N=128$，单 MMA warp-group 配 $B_M=64$，另一配置为 128。该 head dim 下的启发式考察 `max_seqlen_q × (Hq/Hkv)`。这两种配置 P 在寄存器中，其他 tile 可从共享内存向 PV 供 P。TMA 选用另受页长与块宽的整除关系、形状影响。相同数学量对应不同 SMEM 与发射需求。[FA3 tile、GQA 与数据路径](kernel_sources.md) |
| scale、max 与 exp | 正 scale 下可先在原始 QK 域做 row max。源码令 $c=s\log_2e$，逐元素计算 $\exp_2(cX-cm_{\rm raw})$。这给出逐 score FFMA 的机会，还要算行级 $cm_{\rm raw}$，非首块计算 $\exp_2((m_{\rm old}-m_{\rm new})c)$。用自然指数参考表给 GPU 配指令时须完成这一步映射。[FA2 softmax](https://github.com/vllm-project/flash-attention/blob/58e0626a692f09241182582659e3bf8f16472659/csrc/flash_attn/src/softmax.h) |
| reduce 与输出 | FA2 每块先在线程内求 max，再做四 lane allreduce。各 lane 保留自己的部分分母，整次扫描结束才做跨 lane sum。P 的 FP32 值供求和，半精度值供 PV。包装层只把 O 返回给上层时，所核对的扩展路径仍可定位到 LSE 的计算和全局写出。[缩放、归约、P 转换与 LSE 落点](kernel_sources.md) |

**同一个算例可以得到两种有不同用途的 EXP 规格。** 若独立 Vector 每行只算一份 $\alpha$ 后广播，EXP 需求为 $A+H$。若照 FA2 四 lane 各自保存相同 max、各算一次 $\alpha$，完整执行行上的源码级 lane 表达式为 $A+4H$。第 4 节算例分别给出 **251.89** 与 **257.57 Gexp/s**。同理，每个四 lane 的 XOR allreduce 有两轮 shuffle，max 每行块执行，最终 sum 每行执行，共 $8(J+R)$ 个 lane-shuffle，约 **16.11 G lane-shuffle/s**。这些是按源级布局推导的服务量，warp 指令数、有效发射间隔和寄存器 bank 冲突需看编译结果与计数器。[FA2 的归约布局](https://github.com/vllm-project/flash-attention/blob/58e0626a692f09241182582659e3bf8f16472659/csrc/flash_attn/src/softmax.h#L23-L89) · [四 lane shuffle 实现](https://github.com/vllm-project/flash-attention/blob/58e0626a692f09241182582659e3bf8f16472659/csrc/flash_attn/src/utils.h#L111-L131)

## 4. 64×128 tile 的算力与带宽预算

取单请求的一个 Q 头，已有 4032 个前缀 token，本轮追加 64 个 query，最终 KV 长 4096。取 FA2 分页、页长 16、BF16、$d_k=d_v=128$、无 split、无额外 bias。64 行共享一次 KV 扫描，cache 内包含本轮新 token。执行 $32$ 个 $64\times128$ 矩形 tile。有效配对数 $64\times4032+64\times65/2=260128$，执行槽数 $A=262144$，相差 2016 个因果掩码位置。$J=2048,\ R=64,\ H=1984$。两次矩阵乘合计 **134,217,728 FLOP**。以整芯片持续 128 TFLOP/s 分摊，这些工作占 **1.048576 µs** 的 Tensor 服务时间。它对应有足够多独立工作填满芯片时的份额。单 CTA 实际延迟另受每个 SM 的资源与流水限制。

### 4.1 Tensor 与 Vector 的算力配比

统一按整段 32 个 KV 块的服务时间求平均。Tensor 行分别列出 QK、PV 与同一 score 对应的 Vector 操作。Vector 各行以 **QK+PV 的总 FLOP** 为分子，以该项 Vector 元素操作次数为分母。例如 $134217728/264128=508.15$，表示每做 508.15 个 Tensor FLOP，需要 1 次 EXP。FMA 计 2 FLOP，EXP、转换和二元归约各按一次元素操作计。

| 执行单元与操作 | 整段服务量 | 全循环平均有效吞吐 | Tensor ∶ Vector（FLOP ∶ 次） |
|---|---:|---:|---|
| **Tensor QK**，产生 score | 67,108,864 FLOP | **64 TFLOP/s** | QK ∶ Vector 逐 score 缩放 = **256∶1** |
| **Tensor PV**，消费 P | 67,108,864 FLOP | **64 TFLOP/s** | PV ∶ Vector 将 P 转为 BF16 = **256∶1** |
| **Tensor 合计**，QK+PV | **134,217,728 FLOP** | **128 TFLOP/s** | QK+PV ∶ Vector 逐 score 指数 = **512∶1**，行系数指数另计 |
| Vector MUL | 526,272 次 | **501.892 G 元素乘/s** | QK+PV ∶ MUL = **255.03∶1** |
| Vector ADD/SUB | 266,112 次 | **253.784 G 元素加减/s** | QK+PV ∶ ADD/SUB = **504.37∶1** |
| Vector REDMAX，块内 | 260,096 次 | **248.047 G 二元比较/s** | QK+PV ∶ REDMAX = **516.03∶1** |
| Vector REDSUM，块内 | 260,096 次 | **248.047 G 二元加法/s** | QK+PV ∶ REDSUM = **516.03∶1** |
| Vector 行 MAX 合并 | 1,984 次 | 1.892 G 比较/s | QK+PV ∶ 行 MAX = **67,650.06∶1** |
| Vector EXP，逐 score 与行系数 | 264,128 次 | **251.892 Gexp/s** | QK+PV ∶ 全部 EXP = **508.15∶1** |
| Vector RCP、LOG、LSE 的 ADD | 每项 64 次 | 各 0.0610 G/s | QK+PV ∶ **每一项** = **2,097,152∶1** |
| Vector CVT，P 转 BF16 送 PV | 262,144 次 | 250.000 G 元素/s | QK+PV ∶ P 转换 = **512∶1** |
| Vector CVT，最终 O 转 BF16 | 8,192 次 | 7.8125 G 元素/s | QK+PV ∶ O 转换 = **16,384∶1** |

QK 和 PV 各占共享 Tensor 阵列一半服务时间，各自独占阵列时的目标都是 128 TFLOP/s。若归约器规格按输入元素计，REDMAX 和 REDSUM 各需要 250 G 输入元素/s，对应 **512 Tensor FLOP ∶ 1 个归约输入元素**。

MUL 的 501.892 可进一步拆为 score 缩放 **250**、旧分母缩放 **1.892**、旧输出重标 **242.1875**、最终归一化 **7.8125**。若提供独立逐行广播缩放器给 Tensor 累加器，后两项合计 **250 G 元素乘/s** 可在累加器侧完成。若 MUL 与 ADD 实际合用 FMA 管线，应按选定指令序列重新合计。若水平 max/sum 共用一个归约器，它要同时承受 **496.094 G 二元操作/s**，再把它承担的历史 max 比较计入。[项目 Vector 已定义的 ALU、MUL、EXP、LD、ST](../arch.r0.md)

折算到单个执行 score，这个 tile 和上下文的配比为 **512 Tensor FLOP ∶ 2.00757 MUL ∶ 1.01514 ADD/SUB ∶ 0.99219 块内 REDMAX ∶ 0.99219 块内 REDSUM ∶ 1.00757 EXP**。其中归约按二元操作计。该写法可以直接按目标 score/s 缩放，各条物理管线合用时再按上面的归属合并。

### 4.2 Tensor 与 Vector 的带宽配比

取一个满 tile。Q 为 16 KiB，K、V 各 32 KiB，FP32 score S 为 32 KiB，BF16 权重 P 为 16 KiB，FP32 输出累加器 U 为 32 KiB。表内速率均按全循环平均。配比列固定 **Tensor 在左、Vector 在右**，两侧都按有效载荷字节计。S/P 的 LD、ST 名称采用独立引擎经共享 SRAM 交接的映射；在该版 FA2 上，相同的值经寄存器 fragment 交付。

| 数据与接口 | 每块载荷或频次 | Tensor 侧带宽（TB/s） | Vector 侧带宽（TB/s） | 对应项目的 Tensor ∶ Vector（B ∶ B） |
|---|---|---:|---:|---|
| 源存储 → Q 缓冲 | 整段读 16 KiB 一次 | Q 输入 **0.015625** | — | 单侧供 QK，由搬运通路读 Q |
| 源存储 → K/V 缓冲 | 每块 K、V 各读 32 KiB | K **1.000**，V **1.000** | — | 单侧供 QK/PV，K/V 各供 64 行复用 |
| 片上缓冲 → Tensor Q/K/V 操作数 | 每块分别装填 16/32/32 KiB | Q **0.500**，K **1.000**，V **1.000** | — | 单侧装填，Q 的源读取摊到 32 块 |
| FP32 score S，Tensor → Vector | 每块交付 32 KiB | ST(S) **1.000** | LD(S) **1.000** | Tensor 写 S ∶ Vector 读 S = **1∶1** |
| BF16 权重 P，Vector → Tensor | 每块交付 16 KiB | LD(P) **0.500** | ST(P) **0.500** | Tensor 读 P ∶ Vector 写 P = **1∶1** |
| **S/P 的读口需求** | 每块 Tensor 读 P，Vector 读 S | LD(P) **0.500** | LD(S) **1.000** | **Tensor LD ∶ Vector LD = 1∶2** |
| **S/P 的写口需求** | 每块 Tensor 写 S，Vector 写 P | ST(S) **1.000** | ST(P) **0.500** | **Tensor ST ∶ Vector ST = 2∶1** |
| **计入 Q/K/V 装填的逻辑读带宽** | 每块 Tensor 读 Q/K/V/P 共 96 KiB，Vector 读 S 32 KiB | LD(Q/K/V/P) **3.000** | LD(S) **1.000** | **Tensor LD ∶ Vector LD = 3∶1** |
| 旧 U 送 Vector 重标，再送回 Tensor（可选） | 后续 31 块每方向各 32 KiB | 发 **0.96875**，收 **0.96875** | 收 **0.96875**，发 **0.96875** | Tensor 发 ∶ Vector 收 = **1∶1**，Tensor 收 ∶ Vector 发 = **1∶1** |
| 末块原始 U 送 Vector 归一化（可选） | 整段交付 32 KiB 一次 | 发 **0.03125** | 收 **0.03125** | Tensor 发 ∶ Vector 收 = **1∶1** |
| O、LSE → 源存储 | 整段写 16 KiB、256 B | — | 归一化后写 O **0.015625**，写 LSE **0.000244** | 单侧最终写回，按 Vector 承担归一化与写出计 |

读口、写口汇总行复用上方的 S/P 传输量。若将 Q/K/V/P 装填和 S 导出都计在 Tensor 端，Vector 端计读 S 和写 P，则这组逻辑读写的合计配比为 **Tensor ∶ Vector = (3.00+1.00) ∶ (1.00+0.50) = 8∶3**。其中只有 S/P 经共享 SRAM 时，该 SRAM 的总事务为 **3.00 TB/s**。从算力折算到 Vector 带宽，**128 TFLOP/s ∶ 1.00 TB/s LD(S) = 128 FLOP/B**，**128 TFLOP/s ∶ 0.50 TB/s ST(P) = 256 FLOP/B**。

Tensor 的 Q/K/P/V 四个逻辑输入，若全部在同一 tile 级读接口各取一次，共需 **3.00 TB/s**。其中 Q/K/V 为 2.50 TB/s，P 为 0.50 TB/s。FA2 实际 P 走寄存器，SMEM→Tensor 的物理读量须按 Q/K/V 对应的 warp 搬运组织重算。仅核算源内存时，Q、K、V、O、LSE 的总载荷为 **2,130,176 B**。在无跨 CTA 缓存复用、512 GB/s 有效带宽下，搬运至少 **4.1605 µs**，已超过上述 Tensor 服务份额约四倍。即使把所有 Vector 单元配满，这个外存条件也无法持续供给 128 TFLOP/s。[FA2 片上 gemm 装填](https://github.com/vllm-project/flash-attention/blob/58e0626a692f09241182582659e3bf8f16472659/csrc/flash_attn/src/utils.h#L134-L183) · [FA2 的 P 与 V 输入位置](kernel_sources.md)

**独立 Vector 的 LD/ST 如何定规格。** 若 S 由 Tensor 写 SRAM、Vector 读 SRAM，P 由 Vector 写 SRAM、Tensor 读 SRAM，每个 score 的 Vector LD/ST 分别是 **4/2 B**，Tensor 端写/读是 **4/2 B**。单个共享 SRAM 总读和总写各 **6 B/score**，在 250 Gscore/s 下读、写各 **1.50 TB/s**。前提是读到的 S 能在求 max 之后继续用于减 max 和指数。若 Vector 缓冲容不下 S，求 max 扫一次，指数阶段再从同一 SRAM 扫一次，Vector LD 加 **4 B/score，即 1.00 TB/s**。P 或部分分母溢写也按真实写读逐项增加。直连或共享寄存器映射可去掉 S/P 在该 SRAM 的事务，但应给 fragment 读写端口、广播与寄存器文件配吞吐。

**旧 U 的放置决定另一组带宽。** 若采用上表的私有累加器回送方案，并使两个方向均经 SRAM 中转，则除 S/P 外，SRAM 还要服务每方向的一写一读，合计再加 **3.875 TB/s**，未含末尾交付。另一种设计把 $\alpha$ 和最后的 $1/\ell$ 送到有行广播乘法的累加器端，本例每个行块传一个 FP32 标量，共 $4J$ B，约 **7.8125 GB/s**。两种做法的差额取决于重标在哪个单元执行、旧 U 是否可原位读取。若 Vector 私有 RF 承载 U，重标本身仍需读、写 FP32 分量各一次，旁路可改变 RF 端口事务。

**瞬时需求高于平均值。** 若 QK 独占同一 Tensor 阵列以 128 TFLOP/s 运行，QK 阶段的 S 产生率为 $P_T/(2d_k)=500$ Gscore/s，即 **2 TB/s**。同理 PV 独占阵列时 P 消费率为 **1 TB/s**。整个循环才摊成 1.00/0.50 TB/s。没有吸收突发的 FIFO 时，应按阶段发射率校验。有缓冲时，可用任意时刻「累计生产字节 − 累计消费字节」的最大积压定容量，并按单向端口、bank 仲裁和可并发的 tile 核查。QK → rowmax → EXP → PV 的同块依赖仍然存在。

**KV cache 追加和页表。** 本例页长为 16，一个 128 列计算块跨 8 页，4096 个 key 共 256 个页号，单请求 32 位页号的不同表项合计 1024 B。各 warp、各核可能重复取表，非连续页也会影响有效带宽。cache 更新核另外读取新 K/V 并写到 slot，BF16 每新 token、每 KV 头读写各 $2(d_k+d_v)$ B，本例 64 token 合计读、写各 **32 KiB**，另加 slot 元数据。后续扫描已计入这些新 K/V 的读取。页号、slot、cache 追加、attention 主循环应各在对应边界计一次。[分页、cache kernel 与 block table](kernel_sources.md)

## 5. 哪些设置会改变配比

### 5.1 同一 vLLM 分派下改变 head dim

保持上例请求、BF16 和 Tensor 目标，使用 FA2 分页路径对三个 head dim 的真实块宽。表中 Vector 吞吐单位为 G 次/s，RED 表示块内 REDMAX 或 REDSUM 的单项二元操作。最后一列按表头顺序逐项列出「多少个 Tensor FLOP 对应 1 次该项 Vector 操作」。

| $d_k=d_v$ | $B_M\times B_N$ | Tensor QK+PV（TFLOP/s） | Gscore/s | EXP | MUL | ADD/SUB | RED（单项） | K+V 源读 | Tensor ∶ Vector（EXP / MUL / ADD/SUB / RED，各对 1 次） |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 64×256 | 128 | 500 | 501.83 | 626.83 | 503.66 | 498.05 | 2.00 TB/s | 255.07∶1 / 204.20∶1 / 254.14∶1 / 257.00∶1 |
| 128 | 64×128 | 128 | 250 | 251.89 | 501.89 | 253.78 | 248.05 | 2.00 TB/s | 508.15∶1 / 255.03∶1 / 504.37∶1 / 516.03∶1 |
| 256 | 64×64 | 128 | 125 | 126.92 | 626.92 | 128.85 | 123.05 | 2.00 TB/s | 1,008.49∶1 / 204.17∶1 / 993.44∶1 / 1,040.25∶1 |

$d=256$ 时，每个 score 分摊 $256/64=4$ 个输出分量的乘法，因而 MUL 需求回升。K/V 每个 token 的字节随 $d$ 增长，但每个 score 所需矩阵 FLOP 也随 $d$ 增长，在相同 Tensor 目标和行复用下二者抵消。若芯片在不同 head dim 或计算精度下达到的矩阵吞吐不同，应对每行代入其自身的 $P_T$。[FA2 计算 tile 分派](https://github.com/vllm-project/flash-attention/blob/58e0626a692f09241182582659e3bf8f16472659/csrc/flash_attn/src/flash_fwd_launch_template.h#L162-L171)

### 5.2 从 prefill 到 decode，变化最大的是 KV 复用

在指定存储边界，定义 $m_{\rm eff}$ 为一份 K/V 读入后服务的**有效 query-head 行数**。忽略 Q/O、额外复制与 split，在每个被复用 KV token 上，
$$
I_{\rm KV}=\frac{2m_{\rm eff}(d_k+d_v)}{b_Kd_k+b_Vd_v}\quad\mathrm{FLOP/B},\qquad
P_{\rm useful}\le B_{\rm KV}\,I_{\rm KV}.
$$
相同 BF16 位宽时 $I_{\rm KV}=m_{\rm eff}$。满块内部的 prefill 取 $m_{\rm eff}=64$，128 TFLOP/s 要 **2 TB/s** 的 K+V；有限因果边界还要扣除掩码。单 token MHA 取 $m_{\rm eff}=1$，同一**有效**算力目标要 **128 TB/s**。若单 token GQA 的 4 个查询头确实共用该次物理读取，需求降为 **32 TB/s**。以 512 GB/s 供这类 GQA decode，最多约 **4 G 有效 score/s**，即 **2.048 有效 TFLOP/s**。若 kernel 仍填满 64 行 MMA 槽，实际发射的 Tensor FLOP 和 EXP 可大于有效工作；上述带宽界约束服务给用户的有效结果。

同样是单 token decode，把 batch 增大可提供更多 CTA、更多未决访存并摊薄启动开销。不同请求通常访问各自的 KV，batch 大小本身不能当作 $m_{\rm eff}$。长公共前缀、同请求组头打包、跨 CTA 的 L2 命中分别改变不同层的复用。FA2 在单 token GQA、全窗口、无 ALiBi 等条件下把同一 KV 头的多个 Q 头换到行维，部分配置还会把临时 O 拷回调用者传入的输出。混合 prefill/decode 批次的 `max_seqlen_q` 变化可使这个分支不再命中。[GQA 打包与内部 split](kernel_sources.md) · [FA2 输出布局恢复](https://github.com/vllm-project/flash-attention/blob/58e0626a692f09241182582659e3bf8f16472659/csrc/flash_attn/flash_api.cpp)

### 5.3 请求、分页、并行和硬件的影响

| 改哪个因素 | 应重算什么 |
|---|---|
| 上下文、chunk 长度、因果与滑窗 | 单请求原有前缀 $p$、新 query 数 $q$，每头因果有效槽为 $qp+q(q+1)/2$。滑窗按每行实际可见集合取交集。将 Q 分块后，对每个 CTA 枚举相交的 KV tile，再按发射的矩形求 $A,J,R$。短序列、尾块、混合长度会放大填充、倒数、收尾以及 mask 开销占比。 |
| Q 头数、KV 头数、GQA 是否打包 | $H_q$ 倍增 QK、PV、softmax，KV 容量随 $H_{kv}$。在每层存储边界分别估计真实复用；若 $G=H_q/H_{kv}$ 个 Q 头各自从该层读 KV，不能把读量除以 $G$。 |
| cache 页长与放置 | 同一 $B_N$ 可跨多页，页长影响地址计算、表项、尾页、连续事务与可用搬运路径。页长 16、32、64 与计算块宽独立。FA3 还要根据分页条件选 TMA 或线程搬运，改变带宽利用率与发射占用。[页长与后端检查](https://github.com/vllm-project/vllm/blob/275de34170654274616082721348b7edd9741d32/vllm/v1/attention/backends/flash_attn.py#L61-L100) · [FA3 分派](kernel_sources.md) |
| split-KV 数 $s$、cascade | $s$ 个局部分区各自做本地归一化，之后合并。以部分 FP32 的归一化输出 $O_i$ 和 $L_i=\mathrm{LSE}_i$ 为例，$w_i=e^{L_i-\max L}/\sum_j e^{L_j-\max L}$，最终 $O=\sum_i w_i O_i$。合并引入约 $s$ 个 EXP、$s-1$ 个 max 比较、$s-1$ 个求和加法、一次倒数、$s d_v$ 个加权乘法及 $(s-1)d_v$ 个累加，还要归一化权重并生成 LSE。FP32 部分结果各写读一次，另加 **$8s(d_v+1)$ B/有效 query-head**；例 $s=4,d_v=128$ 为 **4128 B/行**。更多分区可能改善短 Q 的占用率。cascade 的公共前缀与后缀亦须合并，另有不同的 KV 复用收益。[实际分区、合并与 cascade 源码](kernel_sources.md) |
| 精度、变体和 mask | KV 位宽直接改变 K/V 读量，P 位宽改变回送量。Q/K/V/P 的转换、量化 scale、FP32 累加和实际 Tensor 峰值要一起调整。ALiBi 加位置相关 bias，softcap 引入 tanh，attention sink 增加分母贡献。RoPE 和 QKV 投影位于本例核心外。 |
| 执行单元组织、RF/共享 SRAM 容量 | 独立 MUL、融合 FMA、原位缩放累加器、水平归约器、SFU 共用情况改变物理单元工作量。S、P、U 能否驻留决定 LD/ST。例中双缓冲 K/V 需 $2(32+32)=128$ KiB，单缓冲 Q/S/P/U 若同时保留再需 $16+32+16+32=96$ KiB，另有行状态等；此假设共 **224 KiB 加行状态/并发 tile**。实际可复用存储且分属 RF/SMEM，多 CTA 并发也会争用容量。 |
| 发射、时延、重叠和目标利用率 | 同一 Tensor 阵列需排 QK 与 PV。EXP 的发射间隔影响稳态，rowmax→EXP 的延迟影响启动、短行与跨块依赖。前端、RF 读写口、barrier、warp/CTA 数和访存延迟可能限制三方并行。硬件宽度换算还须明确时钟、每条指令的有效 lane 数和共享端口效率。 |

## 6. 从目标吞吐反推硬件端口与执行宽度

对满矩形内块，把对应边界的一次读入所服务的 query-head 行数写成 $m_{\rm reuse}$，把一份 Q 在该边界缓存后使用的 KV 块数写成 $T_Q$。目标 score 速率为 $x$，则每秒要供给的逻辑有效载荷为

$$
\begin{aligned}
B_Q&=x\,b_Qd_k/(N T_Q), &
B_K&=x\,b_Kd_k/m_{\rm reuse}, &
B_V&=x\,b_Vd_v/m_{\rm reuse},\\
B_{S:T\to V}&=x\,b_S, &
B_{P:V\to T}&=x\,b_P, &
B_{U:\text{单方向}}&=(x/A)\,b_Ud_v H .
\end{aligned}
$$

$T_Q=1$ 表示每块重新装 Q，$T_Q=T$ 表示整段只装一次。对实际批次应分别求和各 tile 的字节，尾块用真实加载长度，算术按发射形状。$B_U$ 对应旧 U 全量往返方案的每个方向。若只有 S/P 走共享 SRAM，则 $B_{\rm VectorLD}=xb_S,\ B_{\rm VectorST}=xb_P$；同一 SRAM 的总事务是 $2x(b_S+b_P)$。如果 DMA、Tensor 和 Vector 共用同一读写口，须把 Q/K/V 的落地和读取、U 的读写、最终输出等穿过此端口的事务一起加上。

设资源 $j$ 的操作或字节服务量为 $W_j$，持续吞吐为 $C_j$。为给定工作量提供速率的必要条件是

$$
t_{\rm kernel}\ \ge\
\max\left(t_T,\ \max_j W_j/C_j,\ t_{\rm dependency}\right).
$$

同一管线有多类操作时，先按各自发射能力合并成 $\sum_i W_i/C_i$ 的资源占用时间。若 Tensor、Vector、搬运只能分时使用，共享的服务时间也须相加。对近期可重复调度的一批 tile，最后再检查依赖链、启动排空、端口峰值、片上空间与独立工作数，才能从必要吞吐推到可达时间。

**把规格换成硬件宽度。** 例如以 1 GHz、16 个 FP32 lane、每个 lane 每次接收一个 EXP 操作为假设。若启动间隔 II=1 且 lane 都有效，每套管线为 16 Gexp/s。主算例的 251.892 Gexp/s 至少需 16 套同宽管线；若有效发射率只有 80%，则至少需 20 套。类似地，1 TB/s 的 LD 在 1 GHz 下等于全芯片 **1000 B/cycle**；以 64 B/次计，平均需 15.625 次/cycle。水平归约若每次消费 16 个输入，还要规定每行的部分结果如何跨指令合并，以及结果广播、跨 lane 交换和反馈延迟。这些条件会影响同一行能否按目标 II 继续推进。

**判断本项目哪个端口先成为瓶颈。** 在 $d=128$ 的主算例，若 Q/K/V 已就位且主瓶颈是矩阵，先用第 4.1 节的服务率校核 ALU/MUL/EXP，再指定 REDMAX、REDSUM、RCP、LOG、CVT 各由哪条物理管线执行。若冷 KV 源只有 512 GB/s，按第 4.2 节的总字节，整段平均 score 槽率上界先降为约 **63.0 Gscore/s**，对应约 **63.5 Gexp/s**（行系数广播）。这只降低该场景的平均需求，K/V 有缓存命中的 prefill 波段或片上富复用场景仍可能接近 251.9 Gexp/s。[项目硬件功能边界](../arch.r0.md)

若目标扩到整个 Qwen2 attention 子层，可在核心预算两侧增加投影。取本轮 token 数 $q$、隐藏维 $h$、头数 $H_q,H_{kv}$。QKV 投影约 $2qh[H_qd_k+H_{kv}(d_k+d_v)]$ FLOP，输出投影约 $2q(H_qd_v)h$ FLOP；QKV bias 每个投影输出元素一次 ADD。对旋转维 $d_{\rm rot}$，每对 Q/K 分量的一次旋转由 4 次 MUL 与 2 次 ADD/SUB 构成，涉及 $q(H_q+H_{kv})d_{\rm rot}/2$ 对；查 sin/cos、加载存储和能否与前后融合另计。是否计前置 RMSNorm、残差或跨卡归并取决于选择的层边界。[vLLM Qwen2 层定义](https://github.com/vllm-project/vllm/blob/275de34170654274616082721348b7edd9741d32/vllm/model_executor/models/qwen2.py)

## 7. 核对与复算

**源码和公开测量。** 对应 vLLM CMake 固定依赖逐层核对了后端、C++ 分派、tile、主循环、softmax、写回和 merge。[vLLM 源码附录](kernel_sources.md) 中的链接均固定到提交。独立的 FA3 论文也用 $d=128$ 得到每个逐 score 指数约 512 个矩阵 FLOP。其 H100 实验在相同 FP16 非因果形状（batch 4、序列长 8448、16 头、头维 128）下，同时使用 warp specialization 和两阶段流水为 **661 TFLOP/s**，保留 warp specialization 但移除 GEMM-softmax 流水为 **582 TFLOP/s**。这个对照解释了为何相同算术配比仍会产生不同的实际吞吐。实验由论文作者执行，使用其 FA3 内核。[FA3 第 3.1 节与表 2](https://arxiv.org/html/2407.08608v2)

**数值核对。** 在 WSL 的 NumPy FP64 参考计算中，直接计算整行稳定 softmax，再分别与逆序逐块在线更新、三分区 LSE 加权合并对照。6 组形状共 93 个 query 行，覆盖 $d_k\ne d_v$、单 token、单块、多块、因果尾块、滑窗及全 mask 行，两个结果相对直接计算的最大绝对误差为 $8.88\times10^{-16}$。主算例另逐行枚举得到 260128 个有效配对和 2048 次行块访问。该检查验证递推与计数；BF16 舍入、CUDA 指令条数与真实 GPU 端口吞吐仍需单独测量。

复算器另按 query 行和 KV tile 独立枚举，核对了 378 组长度、组头数与 head dim 的有效配对、发射槽、行块访问和 K/V 载荷，并通过 6 个命令行算例。

**运行复算器。** 在本目录用 WSL Python 运行，无第三方依赖。第一条给出主算例，第二条改为 FA2 的 $d=256$ 分派，第三条给出单 token、4 个 Q 头共享一个 KV 头时的有效配对与矩形填充。[参数化复算脚本](attention_balance.py)

```bash
python3 attention_balance.py
python3 attention_balance.py --head-dim 256
python3 attention_balance.py --q-tokens 1 --prefix 4095 --group 4
```

脚本模型以单请求、单 KV 头、单分区、无窗口的 BF16 FA2 为边界，`--group` 扩展该 KV 头对应的全部 Q 头。`--kv-tile` 可覆盖计算块宽做设计探索。脚本按完整 64 行矩形估算被调度 tile 的算术，外存载荷按真实有效行列统计，假设不同 CTA 之间 K/V 冷读。打印的 Vector 吞吐采用每行一份状态、独立 MUL/ADD 的参考映射。真实归约复制、可选 split、页表、cache 追加、GQA 额外拷贝与 SRAM 多次扫描应按正文相应分支叠加。

因果边界会让「有用的 key」与「实际从该请求加载的 key」分开。例如无前缀、128 个新 token、64×128 tile，第一块 Q 的最后一行只能看见 64 个 key，进入计算的 K/V 块却可以覆盖 128 个已存在的 key。复算器先取与可见范围相交的 KV 块，再按整条请求真实长度裁掉越界的源加载；本例两个 Q 块分别可加载 128 个 key。对尾行做 mask 并不会自动少读对应的整段 K/V。[FA2 块范围](https://github.com/vllm-project/flash-attention/blob/58e0626a692f09241182582659e3bf8f16472659/csrc/flash_attn/src/flash_fwd_kernel.h#L520-L541) · [FA2 尾块取数谓词](https://github.com/vllm-project/flash-attention/blob/58e0626a692f09241182582659e3bf8f16472659/csrc/flash_attn/src/flash_fwd_kernel.h#L888-L898)

**实际部署的验证量。** 固定 GPU、vLLM/扩展版本、backend/FA 版本、dtype、页长、Q/K 实际长度、头数、mask 和 split，记录实际 kernel 名与 grid。prefill 和 decode 分开观测主核、cache 核、combine 核。先对照有效配对与发射矩形，再核对 global 读写请求、L2 命中与 DRAM 字节、SMEM 事务、MMA/SFU/ALU 指令、寄存器溢出及 kernel 时长。对自研双引擎另采集 S/P/U 每方向字节、LD/ST 每周期接受量、FIFO 高水位和等待周期。本次分析未采集 GPU profiler 或目标芯片计数器。
