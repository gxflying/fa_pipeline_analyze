"""输出 DeepSeek-V3 两种 MLA 路径的逐步预算表，使用标准库。"""
from dataclasses import dataclass


@dataclass
class Scan:
    tiles: int
    rows: int
    dk: int
    dv: int
    diagonal: int = 0


def report(name, scan):
    m = n = 64
    c, r, k, v = scan.tiles, scan.rows, scan.dk, scan.dv
    a, j = c*m*n, c*m
    h = j-r
    tensor_us = 2*a*(k+v)/128e6
    # 操作名、输入/输出有效字节、服务次数、持续服务率（次/微秒）。
    rows = [
        ("Tensor QK", 2*c*(m+n)*k, 4*a, 2*a*k, 128e6),
        ("Vector score 缩放", 4*a, 4*a, a, 1e6),
        ("Vector causal 选择", 4*scan.diagonal, 4*scan.diagonal, scan.diagonal, 1e6),
        ("Vector 块内 REDMAX", 4*a, 4*j, a-j, 0.5e6),
        ("Vector 合并历史 max", 8*h, 4*h, h, 1e6),
        ("Vector 中心化 S−m", 4*(a+j), 4*a, a, 1e6),
        ("Vector 指数 P", 4*a, 4*a, a, 0.25e6),
        ("Vector 跨块差值", 8*h, 4*h, h, 1e6),
        ("Vector 指数 α", 4*h, 4*h, h, 0.25e6),
        ("Vector 块内 REDSUM", 4*a, 4*j, a-j, 0.5e6),
        ("Vector 缩放旧分母", 8*h, 4*h, h, 1e6),
        ("Vector 合并分母", 8*h, 4*h, h, 1e6),
        ("Vector 重标旧 U", 4*h*(v+1), 4*h*v, h*v, 1e6),
        ("Vector P 转 BF16", 4*a, 2*a, a, 1e6),
        ("Tensor PV 累加", 2*a+2*c*n*v+4*h*v, 4*j*v, 2*a*v, 128e6),
        ("Vector 分母倒数", 4*r, 4*r, r, 0.125e6),
        ("Vector 归一化 U", 4*r*(v+1), 4*r*v, r*v, 1e6),
        ("Vector 输出转 BF16", 4*r*v, 2*r*v, r*v, 1e6),
        ("Vector LSE 的 LOG", 4*r, 4*r, r, 0.125e6),
        ("Vector LSE 的 ADD", 8*r, 4*r, r, 1e6),
    ]
    print(f"### {name}\n")
    print(f"块数 {c:,}，A={a:,}，J={j:,}，R={r:,}，H={h:,}。矩阵服务预算 τ={tensor_us:.6f} µs。\n")
    print("| 执行步骤 | 输入 KiB | 输出 KiB | 工作量 | Tensor FLOP ∶ 本项操作 | 所需读 / 写 TB/s | 算力时间 µs | 单步时间下界 µs |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for op, rd, wr, work, throughput in rows:
        time = work/throughput
        ratio = f"{2*a*(k+v)/work:,.2f}∶1" if work and op.startswith("Vector") else "—"
        print(f"| {op} | {rd/1024:,.3f} | {wr/1024:,.3f} | {work:,} | {ratio} | {rd/tensor_us/1e6:.4f} / {wr/tensor_us/1e6:.4f} | {time:.6f} | {max(time,rd/8e6,wr/8e6):.6f} |")
    print()


if __name__ == "__main__":
    report("首次 prefill，256 个 token", Scan(128*10, 128*256, 192, 128, 128*4*64*64))
    report("单 token decode，4096 个可见 KV", Scan(2*64, 128, 576, 512))
