"""重算 vLLM v0.11.2 FA2 分页、无 split 的 BF16 因果前向预算。"""

import argparse
import math
from collections import Counter


def tile_counts(q_tokens, prefix, group, dim, block_n):
    """单请求、单 KV 头。prefill 每个 Q 头各扫一次，单 token GQA 打包组头。"""
    block_m = 64
    decode = q_tokens == 1
    rows = group if decode else q_tokens
    copies = 1 if decode else group
    counts = Counter()
    for start in range(0, rows, block_m):
        end = min(start + block_m, rows)
        visible = prefix + (1 if decode else end)
        panels = (visible + block_n - 1) // block_n
        # 因果边界截到 KV 块。在块内，访存谓词只排除超出整条序列的列。
        loaded_keys = min(prefix + q_tokens, panels * block_n)
        counts["slots"] += copies * block_m * block_n * panels
        counts["row_visits"] += copies * block_m * panels
        counts["issued_rows"] += copies * block_m
        counts["kv_bytes"] += copies * loaded_keys * 2 * dim * 2
    counts["valid_rows"] = q_tokens * group
    counts["valid_pairs"] = group * (q_tokens * prefix + q_tokens * (q_tokens + 1) // 2)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-tokens", type=int, default=64, help="本轮 query token 数")
    parser.add_argument("--prefix", type=int, default=4032, help="本轮之前的 KV 长度")
    parser.add_argument("--group", type=int, default=1, help="每个 KV 头的 Q 头数")
    parser.add_argument("--head-dim", type=int, choices=(64, 128, 256), default=128)
    parser.add_argument("--kv-tile", type=int, help="覆盖源码块宽，用于假设的数据流")
    parser.add_argument("--tensor-tflops", type=float, default=128, help="可持续的矩阵计算目标")
    parser.add_argument("--dram-gbps", type=float, default=512, help="十进制有效 GB/s")
    args = parser.parse_args()
    if min(args.q_tokens, args.group) <= 0 or args.prefix < 0:
        parser.error("q-tokens、group 须为正，prefix 须非负")
    if args.kv_tile is not None and args.kv_tile <= 0:
        parser.error("kv-tile 须为正")
    for value in (args.tensor_tflops, args.dram_gbps):
        if not math.isfinite(value) or value <= 0:
            parser.error("吞吐与带宽须为有限正数")

    d = args.head_dim
    n = args.kv_tile or {64: 256, 128: 128, 256: 64}[d]
    c = tile_counts(args.q_tokens, args.prefix, args.group, d, n)
    a, j, r = c["slots"], c["row_visits"], c["issued_rows"]
    u = j - r
    real_rows = c["valid_rows"]
    flops = 4 * d * a
    tensor_time = flops / (args.tensor_tflops * 1e12)
    services = {
        "MUL，含 score 缩放和输出归一化": a + u + d * j,
        "ADD/SUB，不含块内归约或 LSE": a + 2 * u,
        "MAX，新旧行最大值": u,
        "REDMAX，块内二元比较": a - j,
        "REDSUM，块内二元加法": a - j,
        "EXP，逐 score 和行系数": a + u,
        "RCP，行倒数": r,
        "CVT，P 送 PV": a,
        "CVT，输出转 BF16，按有效行": real_rows * d,
    }

    print(f"FA2 分页布局 B_M=64, B_N={n}, d={d}, G={args.group}, split=1")
    print(f"有效配对 {c['valid_pairs']:,}，矩形槽 {a:,}，占比 {c['valid_pairs']/a:.3%}")
    print(f"执行行 R={r:,}，行块访问 J={j:,}，后续行块 U={u:,}，有效输出行 {real_rows:,}")
    print(f"矩阵 {flops:,} FLOP，整芯片 Tensor 服务时间份额 {tensor_time*1e6:.6f} µs")
    print("未融合参考映射，按发射矩形为各执行行维护一份状态")
    for name, count in services.items():
        print(f"  {name:<25} {count:>12,}   {count/tensor_time/1e9:9.3f} G/s")

    traffic = {
        "Q 源读取": real_rows*d*2,
        "K 源读取": c["kv_bytes"]//2,
        "V 源读取": c["kv_bytes"]//2,
        "O 最终写": real_rows*d*2,
        "LSE 最终写": real_rows*4,
    }
    print("各 CTA 之间无 K/V 缓存命中，cache 追加、页元数据和额外拷贝另计")
    for name, amount in traffic.items():
        print(f"  {name:<20} {amount:>12,} B   {amount/tensor_time/1e12:.6f} TB/s")
    memory_time = sum(traffic.values()) / (args.dram_gbps*1e9)
    print(f"上述有效载荷在给定外存带宽下至少 {memory_time*1e6:.6f} µs")
    lower_time = max(memory_time, tensor_time)
    useful_flops = 4 * d * c["valid_pairs"]
    print(f"只考虑 Tensor 与外存，有效算力上界 {useful_flops/lower_time/1e12:.6f} TFLOP/s")
    print("Tensor 输入，按满 tile 的片上逻辑操作数各装一次；物理端口依布局而定")
    feed = {"Q": 2*d*j, "K": 2*d*a//64, "P": 2*a, "V": 2*d*a//64}
    for name, amount in feed.items():
        print(f"  {name} {amount:>12,} B   {amount/tensor_time/1e12:.6f} TB/s")
    print("独立引擎假设，S/P 各经一次 SRAM 交接，块内保留中间值")
    print(f"  Vector LD(S) {4*a/tensor_time/1e12:.6f} TB/s")
    print(f"  Vector ST(P) {2*a/tensor_time/1e12:.6f} TB/s")
    print(f"  两端合计 SRAM 读写 {12*a/tensor_time/1e12:.6f} TB/s")
    print(f"  如需送回旧输出重标，每方向另加 {4*d*u/tensor_time/1e12:.6f} TB/s")


if __name__ == "__main__":
    main()
