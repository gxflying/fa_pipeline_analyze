# JSON 参数化 FlashAttention 稳态评估

在 WSL 的 Python 3 环境运行，需要 numpy、matplotlib，以及 Noto Sans CJK 或 Windows 的 Microsoft YaHei 中文字体。脚本支持从 WSL 访问 Windows 字体。默认配置来自 00.param.txt，补充了逐类型结果延迟、Tensor 端口分配和图片尺寸。

在本目录执行

    python3 flashattention_pipeline.py --config flashattention_pipeline.default.json --output-dir artifacts/flashattention_pipeline

生成三张全尺寸 **7200×2400 px** PNG 和 report.json，终端同时输出时间表与所有文件路径。

| 文件 | 内容 |
|---|---|
| pipeline.png | 稳态泳道图及各引擎合并忙碌区间 |
| unit_statistics.png | 各计算单元的占用/发射时间，以及各读写接口的活动时间 |
| engine_statistics.png | Tensor、Vector、DMA 含读写与纯计算两种口径的时间并集 |
| report.json | 全部数值、配置、排程与图片路径 |

三张 PNG 使用 JSON 的 width_px、height_px、dpi，保存时不裁切画布。时间统一为 cycle / clock_ghz 纳秒。全部统计图直接读取同一份 report 数据，对应相同窗口。纯计算口径保留计算指令因供数不足而持有单元的时间，实际发射时间另列。

## 参数

| JSON 分组 | 含义 |
|---|---|
| shape | Q 行数、KV tile 列数、QK/V 头维，支持非整齐微块与尾组 |
| clock_ghz、dtype_bytes | 时钟和各数据对象的元素字节数 |
| tensor | M/K/N 微块并行度、微步结果延迟、L1 读写分配上限 |
| ddr | 读写额定 GB/s、有效系数 |
| l1 | KiB 容量、读写 GB/s |
| vector | 读写 GB/s、输入/输出接口延迟，各类型的 parallelism、latency_cycles |
| scheduler.max_ii_cycles | 周期排程搜索上限，找不到可行解时报告原因 |
| plot | 展示的完整周期数、PNG 尺寸、DPI、是否强制显示隐藏的 Vector 读写 |

所有字段必填，未知字段、非正数和无穷值会报错。各类型执行单元均为一套。parallelism 表示每周期接收的元素数，REDMAX/REDSUM 按二元操作数计。latency_cycles 表示一组从发射到结果可见的周期数，默认普通 Vector 操作为 1，Tensor 为 15。读首拍取 max(首组传输周期, input_latency_cycles)，写首组在计算结果可见后再等 output_latency_cycles−1，随后通过写口发送。

## 时间与统计

- units 中每一项给出单周期及整个绘图窗口的占用周期、纳秒、利用率、发射周期和资源需求。欠供造成向量操作停顿时，操作仍持有单元，占用时间包含这一停顿，发射周期另列。
- engines 中 Tensor、Vector、DMA 的主时间是该引擎内部所有单元和读写接口忙碌区间的**并集**。被图形隐藏的 Vector 读写也参与统计。
- compute_only 另给纯计算区间并集。三类引擎可重叠执行，各自时间相加不等于窗口长度。
- operations 保存每项操作的工作量、公式、起止时刻、依赖、发射组和各物理资源占用区间，可独立复算。
- RCP、DIV、SQRT 与 DMA_WR 在中间 KV 稳态中为零。初始化、最终归一化和输出写回发生于 Q 块边界，当前统计窗口专注于中间稳态。

默认值的资源下界是 520 cycle/tile。脚本从资源下界开始，以固定拓扑顺序寻找最早可行的周期相位，再检查跨 tile 的 m/ℓ/A 反馈与 L1 槽位。如果结果达到下界，proven_optimal_in_model 为 true。超过下界时输出已验证的可行排程，固定顺序的搜索不保证全局最优。

## 访存与容量

Q 预先驻留 L1，每 tile 从 L1 重新读入 Tensor。K/V 每 tile 由 DDR 装载。S、P、Y 经共享 L1 交接。Tensor 内部依靠本地操作数复用，Vector 中间值和跨 tile 状态保存在本地，假定这些本地空间和端口足够。

Vector SCALE 和 ADD_A 逐组读取 S/Y，CVT_P 逐组输出 P，读写与计算重叠。共享 L1 的所有访问仍参与周期级容量约束。带宽充足时图中仅标输入、输出边界，发生按组供数或写回限速时显示 Vector 读写泳道和延长后的计算占用。

归约保留整行结果依赖。连续向量操作默认整块结果交付，只有上述外部读写与相应算子实施逐组重叠。每条向量指令持有其功能单元至末组发射，允许已发射结果继续在内部流水线排空。

L1 容量采用 Q 常驻加 K/V/S/P/Y 各自固定槽池，按生命周期与 II 算槽数。槽位不足会继续尝试更大的 II，连单份对象都装不下时直接报错。没有模拟具体 SRAM bank、DDR 请求启动和动态 cache 行为。

执行回归检查

    python3 -m unittest -v test_flashattention_pipeline.py
