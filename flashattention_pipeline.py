"""JSON 驱动的 FlashAttention 稳态周期排程、资源统计与全尺寸泳道图。

采用完整有效 KV tile、独立 Vector 功能单元、共享 L1 和逐组 Vector
读写。Q 在 L1 驻留，Tensor 局部复用、Vector 中间值和状态驻留本地。
只统计中间稳态。初始化、末块归一化和最终写回不摊入每个 KV tile。
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


DEFAULT_CONFIG = Path(__file__).with_name("flashattention_pipeline.default.json")
VECTOR_UNITS = ("MUL", "ADD", "SUB", "MAX", "CVT", "EXP", "DIV", "RCP",
                "SQRT", "REDMAX", "REDSUM")
ENGINE_RESOURCES = {
    "Tensor": ("TENSOR", "TENSOR_LD", "TENSOR_ST"),
    "Vector": VECTOR_UNITS + ("VECTOR_LD", "VECTOR_ST"),
    "DMA": ("DMA_RD", "DMA_WR"),
}


def validate_config(config):
    """严格核对键和值，避免拼写错误被静默忽略。"""
    template = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))

    def check(value, reference, path):
        if isinstance(reference, dict):
            if not isinstance(value, dict) or value.keys() != reference.keys():
                raise ValueError(f"{path} 的键必须为 {list(reference)}")
            for key in reference:
                check(value[key], reference[key], f"{path}.{key}")
        elif isinstance(reference, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{path} 必须是布尔值")
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path} 必须是数值")
        elif not math.isfinite(value) or value <= 0:
            raise ValueError(f"{path} 必须是有限正数")
        elif isinstance(reference, int) and not isinstance(value, int):
            raise ValueError(f"{path} 必须是整数")

    check(config, template, "config")
    if config["ddr"]["efficiency"] > 1:
        raise ValueError("ddr.efficiency 必须在 (0, 1] 内")
    if min(config["plot"]["width_px"], config["plot"]["height_px"]) < 800:
        raise ValueError("图片宽高须至少 800 px，以显示完整泳道")
    return config


@dataclass
class Operation:
    name: str
    unit: str
    profiles: dict
    ready_offset: int
    deps: tuple = ()
    formula: str = ""
    work: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    issue_offsets: list = field(default_factory=list)
    read_end: int = 0
    write_start: int = 0
    start: int = 0

    @property
    def ready(self):
        return self.start + self.ready_offset

    def runs(self, resource):
        """本操作在指定资源上的非零区间，含起止周期及每周期需求。"""
        values = self.profiles.get(resource, np.zeros(0))
        if not len(values):
            return []
        boundaries = np.flatnonzero(np.diff(np.r_[0, values, 0]) != 0)
        result = []
        for begin, end in zip(boundaries, boundaries[1:]):
            if values[begin] > 0:
                result.append((self.start + int(begin), self.start + int(end),
                               float(values[begin])))
        return result


def byte_profile(size, rate):
    cycles = math.ceil(size / rate)
    values = np.full(cycles, rate, dtype=float)
    values[-1] = size - rate * (cycles - 1)
    return values


def make_model(config):
    shape, sizes = config["shape"], config["dtype_bytes"]
    m, n = shape["query_rows"], shape["kv_cols"]
    dk, dv = shape["head_dim_qk"], shape["head_dim_v"]
    f = config["clock_ghz"]
    tensor, vector = config["tensor"], config["vector"]
    capacities = {unit: 1.0 for unit in ("TENSOR",) + VECTOR_UNITS}
    for group, prefix in [("l1", "L1"), ("tensor", "TENSOR"),
                          ("vector", "VECTOR"), ("ddr", "DMA")]:
        efficiency = config["ddr"]["efficiency"] if group == "ddr" else 1
        for direction, suffix in [("read", "R"), ("write", "W")]:
            name = f"{prefix}_{suffix}"
            if prefix in ("TENSOR", "VECTOR"):
                name = f"{prefix}_{'LD' if suffix == 'R' else 'ST'}"
            elif prefix == "DMA":
                name = f"DMA_{'RD' if suffix == 'R' else 'WR'}"
            capacities[name] = config[group][f"{direction}_gbps"] * efficiency / f
    operations = []

    def transfer(name, unit, size, direction, deps=()):
        interface = "L1_R" if direction == "read" else "L1_W"
        rate = min(capacities[unit], capacities[interface])
        data = byte_profile(size, rate)
        op = Operation(name, unit, {unit: data, interface: data.copy()}, len(data),
                       deps, f"ceil({size} B / {rate:g} B/c) = {len(data)} c",
                       bytes_read=size if direction == "read" else 0,
                       bytes_written=size if direction == "write" else 0)
        op.read_end = len(data)
        operations.append(op)

    def compute(name, unit, work, deps=(), read_element_bytes=0, write_element_bytes=0):
        if work == 0:
            operations.append(Operation(name, unit, {}, 0, deps,
                                        "single-element reduction: no binary operation"))
            return
        spec = vector["units"][unit]
        width = spec["parallelism"]
        latency = spec["latency_cycles"]
        groups = math.ceil(work / width)
        read_rate = min(capacities["VECTOR_LD"], capacities["L1_R"])
        write_rate = min(capacities["VECTOR_ST"], capacities["L1_W"])
        read_cycles = math.ceil(width * read_element_bytes / read_rate)
        write_cycles = math.ceil(width * write_element_bytes / write_rate)
        stride = max(1, read_cycles, write_cycles)
        input_delay = max(read_cycles, vector["input_latency_cycles"]) if read_element_bytes else 0
        output_extra = vector["output_latency_cycles"] - 1
        issues = [input_delay + group * stride for group in range(groups)]
        end = issues[-1] + latency
        profile_size = end + output_extra + write_cycles + 1
        profiles = {unit: np.zeros(profile_size)}
        # 一条向量操作持有单元至末组发射；欠供期间在此单元产生空转占用。
        profiles[unit][issues[0]:issues[-1] + 1] = 1
        read_end = 0
        first_write = 0
        if read_element_bytes:
            profiles["VECTOR_LD"] = np.zeros(profile_size)
        if write_element_bytes:
            profiles["VECTOR_ST"] = np.zeros(profile_size)
        for group, issue in enumerate(issues):
            count = min(width, work - group * width)
            if read_element_bytes:
                data = byte_profile(count * read_element_bytes, read_rate)
                begin = group * stride
                profiles["VECTOR_LD"][begin:begin + len(data)] += data
                read_end = begin + len(data)
                assert read_end <= issue
            if write_element_bytes:
                data = byte_profile(count * write_element_bytes, write_rate)
                begin = issue + latency + output_extra
                profiles["VECTOR_ST"][begin:begin + len(data)] += data
                if group == 0:
                    first_write = begin
                end = max(end, begin + len(data))
        if read_element_bytes:
            profiles["L1_R"] = profiles["VECTOR_LD"].copy()
        if write_element_bytes:
            profiles["L1_W"] = profiles["VECTOR_ST"].copy()
        operations.append(Operation(
            name, unit, profiles, end, deps,
            f"{work} ops / {width} lanes; {groups} groups; stride={stride} c; latency={latency} c",
            work, work * read_element_bytes, work * write_element_bytes,
            issues, read_end, first_write))

    def gemm(name, rows, cols, reduction, deps):
        blocks = math.ceil(rows / tensor["micro_m"]) * math.ceil(cols / tensor["micro_n"])
        rounds = math.ceil(reduction / tensor["micro_k"])
        # 同一累加器的相邻 K 微块须等待之前结果；小输出 tile 会产生间隙。
        gap = max(0, tensor["latency_cycles"] - blocks)
        length = blocks * rounds + gap * (rounds - 1)
        issues = [r * (blocks + gap) + b for r in range(rounds) for b in range(blocks)]
        operations.append(Operation(
            name, "TENSOR", {"TENSOR": np.ones(length)},
            length + tensor["latency_cycles"] - 1, deps,
            f"{blocks} output blocks × {rounds} K rounds; inter-round gap={gap} c",
            rows * cols * reduction, issue_offsets=issues))

    q, k, v = m * dk * sizes["q"], n * dk * sizes["k"], n * dv * sizes["v"]
    s, p, y = m * n * sizes["s"], m * n * sizes["p"], m * dv * sizes["y"]
    transfer("K_DMA", "DMA_RD", k, "write")
    transfer("V_DMA", "DMA_RD", v, "write")
    transfer("Q_LD", "TENSOR_LD", q, "read")
    transfer("K_LD", "TENSOR_LD", k, "read", ("K_DMA",))
    gemm("QK", m, n, dk, ("Q_LD", "K_LD"))
    transfer("S_ST", "TENSOR_ST", s, "write", ("QK",))
    compute("SCALE", "MUL", m * n, ("S_ST",), read_element_bytes=sizes["s"])
    compute("ROWMAX", "REDMAX", m * (n - 1), ("SCALE",))
    compute("MAX", "MAX", m, ("ROWMAX",))
    compute("SUB_S", "SUB", m * n, ("MAX",))
    compute("DELTA", "SUB", m, ("MAX",))
    compute("EXP_P", "EXP", m * n, ("SUB_S",))
    compute("ALPHA", "EXP", m, ("DELTA",))
    compute("ROWSUM", "REDSUM", m * (n - 1), ("EXP_P",))
    compute("CVT_P", "CVT", m * n, ("EXP_P",), write_element_bytes=sizes["p"])
    transfer("P_LD", "TENSOR_LD", p, "read", ("CVT_P",))
    transfer("V_LD", "TENSOR_LD", v, "read", ("V_DMA",))
    gemm("PV", m, dv, n, ("P_LD", "V_LD"))
    transfer("Y_ST", "TENSOR_ST", y, "write", ("PV",))
    compute("SCALE_A", "MUL", m * dv, ("ALPHA", "Y_ST"))
    compute("ADD_A", "ADD", m * dv, ("SCALE_A", "Y_ST"), read_element_bytes=sizes["y"])
    compute("SCALE_L", "MUL", m, ("ALPHA", "ROWSUM"))
    compute("ADD_L", "ADD", m, ("SCALE_L", "ROWSUM"))
    lifetimes = [(k, "K_DMA", "K_LD"), (v, "V_DMA", "V_LD"),
                 (s, "S_ST", "SCALE"), (p, "CVT_P", "P_LD"),
                 (y, "Y_ST", "ADD_A")]
    return operations, capacities, lifetimes, q


def folded_profile(profile, ii):
    return np.bincount(np.arange(len(profile)) % ii, weights=profile, minlength=ii)


def feasible_phases(patterns, occupied, capacities, ii):
    """用循环区间前缀和一次检查全部相位，避免逐相位复制长向量。"""
    valid = np.ones(ii, dtype=bool)
    phases = np.arange(ii)
    for resource, pattern in patterns.items():
        boundaries = np.flatnonzero(np.diff(np.r_[0, pattern, 0]) != 0)
        for begin, end in zip(boundaries, boundaries[1:]):
            amount = pattern[begin]
            if amount == 0:
                continue
            blocked = occupied[resource] + amount > capacities[resource] + 1e-8
            cumulative = np.r_[0, np.cumsum(np.tile(blocked, 2))]
            starts = (phases + begin) % ii
            valid &= cumulative[starts + end - begin] == cumulative[starts]
        if not np.any(valid):
            break
    return valid


def buffer_usage(operations, lifetimes, resident, ii):
    by_name = {op.name: op for op in operations}
    pools = {}
    for size, producer, consumer in lifetimes:
        first, last = by_name[producer], by_name[consumer]
        end = last.start + last.read_end
        if not last.read_end:
            end = last.ready
        # 为生产操作开始时预留整个输出槽，便于保守地避免覆写。
        lifetime = end - first.start
        slots = max(1, math.ceil(lifetime / ii))
        pools[producer] = {"bytes_per_slot": size, "slots": slots, "lifetime_cycles": lifetime}
    return resident + sum(p["slots"] * p["bytes_per_slot"] for p in pools.values()), pools


def solve(config):
    operations, capacities, lifetimes, resident = make_model(config)
    budget = config["l1"]["size_kib"] * 1024
    minimum = resident + sum(size for size, _, _ in lifetimes)
    if minimum > budget:
        raise ValueError(f"当前独立槽布局至少需要 {minimum / 1024:g} KiB L1，配置仅 {budget / 1024:g} KiB")
    requirements = {
        resource: sum(float(op.profiles.get(resource, np.zeros(0)).sum()) for op in operations)
        for resource in capacities
    }
    lower_bound = max(1, math.ceil(max(requirements[r] / capacities[r] for r in capacities)))
    if lower_bound > config["scheduler"]["max_ii_cycles"]:
        raise ValueError(f"资源下界 {lower_bound} cycle 已超过 max_ii_cycles")
    by_name = {op.name: op for op in operations}
    last_failure = ""
    for ii in range(lower_bound, config["scheduler"]["max_ii_cycles"] + 1):
        occupied = {r: np.zeros(ii) for r in capacities}
        failed = False
        for op in operations:
            patterns = {r: folded_profile(p, ii) for r, p in op.profiles.items() if np.any(p)}
            earliest = max((by_name[name].ready for name in op.deps), default=0)
            valid = feasible_phases(patterns, occupied, capacities, ii)
            offsets = np.flatnonzero(valid[(earliest + np.arange(ii)) % ii])
            if not len(offsets):
                last_failure = f"{op.name} 无可用相位"
                failed = True
                break
            op.start = earliest + int(offsets[0])
            for resource, pattern in patterns.items():
                occupied[resource] += np.roll(pattern, op.start % ii)
        if failed:
            continue
        recurrences = (("MAX", "MAX"), ("ADD_L", "SCALE_L"), ("ADD_A", "SCALE_A"))
        if any(by_name[a].ready > by_name[b].start + ii for a, b in recurrences):
            last_failure = "跨 tile 的 m/ℓ/A 反馈无法在此节拍内完成"
            continue
        total, pools = buffer_usage(operations, lifetimes, resident, ii)
        if total > budget:
            last_failure = f"L1 槽位需要 {total / 1024:g} KiB"
            continue
        verify_schedule(operations, capacities, ii)
        return operations, capacities, ii, lower_bound, total, pools
    raise ValueError(f"在 II≤{config['scheduler']['max_ii_cycles']} 内未找到可行排程。"
                     f"最后障碍为 {last_failure}。可调整容量、最大搜索周期或硬件参数。")


def verify_schedule(operations, capacities, ii):
    """用展开三个周期的区间扫描检查，不复用调度器的模周期累加。"""
    by_name = {op.name: op for op in operations}
    for op in operations:
        for dep in op.deps:
            if by_name[dep].ready > op.start:
                raise AssertionError(f"依赖违规 {dep} → {op.name}")
    for a, b in (("MAX", "MAX"), ("ADD_L", "SCALE_L"), ("ADD_A", "SCALE_A")):
        if by_name[a].ready > by_name[b].start + ii:
            raise AssertionError(f"反馈违规 {a} → {b}")
    # 每个 profile 的任意长度都可覆盖多个周期，包含前后所有相交实例。
    for resource, capacity in capacities.items():
        changes = {}
        for op in operations:
            for begin, end, amount in op.runs(resource):
                first_tile = math.floor(-end / ii) + 1
                last_tile = math.ceil((3 * ii - begin) / ii)
                for tile in range(first_tile, last_tile):
                    a, b = max(0, begin + tile * ii), min(3 * ii, end + tile * ii)
                    if a < b:
                        changes[a] = changes.get(a, 0) + amount
                        changes[b] = changes.get(b, 0) - amount
        current = 0
        for cycle, change in sorted(changes.items()):
            current += change
            if current > capacity + 1e-7:
                raise AssertionError(f"{resource} 在 {cycle} 超出容量")


def make_report(config, operations, capacities, ii, lower_bound, total, pools):
    periods, frequency = config["plot"]["periods"], config["clock_ghz"]
    busy = {r: np.zeros(ii, dtype=bool) for r in capacities}
    amounts = {r: np.zeros(ii) for r in capacities}
    issued = {r: 0 for r in capacities}
    for op in operations:
        for resource, profile in op.profiles.items():
            load = np.roll(folded_profile(profile, ii), op.start % ii)
            amounts[resource] += load
            busy[resource] |= load > 0
        issued[op.unit] += len(op.issue_offsets)

    def timing(cycles):
        return {"per_period_cycles": cycles, "window_cycles": cycles * periods,
                "window_ns": cycles * periods / frequency,
                "utilization": cycles / ii}

    unit_times = {}
    for resource in capacities:
        unit_times[resource] = {
            **timing(int(busy[resource].sum())),
            "issue_cycles_per_period": issued[resource],
            "peak_demand_per_cycle": float(amounts[resource].max()),
            "demand_per_period": float(amounts[resource].sum()),
        }
    engine_times = {}
    for engine, resources in ENGINE_RESOURCES.items():
        union = np.logical_or.reduce([busy[r] for r in resources])
        compute = [r for r in resources if r == "TENSOR" or r in VECTOR_UNITS]
        compute_union = np.zeros(ii, dtype=bool)
        for resource in compute:
            compute_union |= busy[resource]
        engine_times[engine] = {
            **timing(int(union.sum())),
            "compute_only": timing(int(compute_union.sum())),
            "definition": "内部物理单元及接口占用区间的并集，包含图中隐藏的Vector访存",
        }
    return {
        "config": config, "ii_cycles": ii, "resource_lower_bound_cycles": lower_bound,
        "proven_optimal_in_model": ii == lower_bound,
        "scheduler": "确定性拓扑顺序、最早可行相位的周期排程；下界以上的结果为可行解",
        "window_cycles": ii * periods, "window_ns": ii * periods / frequency,
        "l1_reserved_bytes": total, "l1_pools": pools,
        "units": unit_times, "engines": engine_times,
        "scope": "完整有效KV tile的中间稳态，Q已在L1；不含首块初始化、最终归一化与写回",
        "assumptions": [
            "Tensor局部操作数复用，Vector中间值、m/ℓ/A及其在途版本有足够本地空间",
            "同类型单元各一份，不同类型并发；向量指令欠供期间保持单元占用",
            "归约工作量按二元操作计，归约结果整行就绪后交付",
            "L1读写独立；DMA与Tensor接口的传输按整块完成；不模拟bank映射",
            "DDR带宽含有效系数；没有额外请求启动延迟；采用独立固定L1槽池",
        ],
        "operations": [{
            "name": op.name, "unit": op.unit, "start": op.start, "ready": op.ready,
            "dependencies": list(op.deps), "formula": op.formula, "work": op.work,
            "bytes_read": op.bytes_read, "bytes_written": op.bytes_written,
            "issue_offsets": op.issue_offsets,
            "resource_intervals": {r: op.runs(r) for r in op.profiles},
        } for op in operations],
    }


def configure_plot_fonts():
    import matplotlib as mpl
    from matplotlib import font_manager

    chinese_font = next((font for font in font_manager.fontManager.ttflist
                         if "Noto Sans CJK" in font.name or "Microsoft YaHei" in font.name), None)
    if chinese_font is None:
        # WSL 已安装字体可能尚未进入 matplotlib 的缓存，按路径显式注册。
        for path in (Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
                     Path("/mnt/c/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/msyh.ttc")):
            if path.exists():
                font_manager.fontManager.addfont(str(path))
                name = font_manager.FontProperties(fname=str(path)).get_name()
                mpl.rcParams["font.family"] = name
                break
        else:
            raise ValueError("中文统计图需要 Noto Sans CJK 或 Microsoft YaHei 字体")
    else:
        mpl.rcParams["font.family"] = chinese_font.name
    mpl.rcParams["axes.unicode_minus"] = False


def draw(config, operations, report, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_plot_fonts()
    plot = config["plot"]
    ii, window = report["ii_cycles"], report["window_cycles"]
    lanes = ["DMA_RD", "DMA_WR", "TENSOR_LD", "TENSOR", "TENSOR_ST"] + list(VECTOR_UNITS)
    throttled = any(op.issue_offsets and len(op.issue_offsets) > 1
                    and op.issue_offsets[1] - op.issue_offsets[0] > 1
                    for op in operations if op.bytes_read or op.bytes_written)
    show_io = plot["show_hidden_vector_io"] or throttled
    if show_io:
        lanes += ["VECTOR_LD", "VECTOR_ST"]
    lanes += ["Tensor union", "Vector union", "DMA union"]
    fig, ax = plt.subplots(figsize=(plot["width_px"] / plot["dpi"],
                                    plot["height_px"] / plot["dpi"]), dpi=plot["dpi"])
    fig.subplots_adjust(left=0.08, right=0.985, top=0.87, bottom=0.11)
    colors = plt.get_cmap("Pastel1").colors
    for row, resource in enumerate(lanes):
        ax.axhspan(row - 0.45, row + 0.45, color="#f4f7fa", zorder=0)
        if resource.endswith(" union"):
            continue
        for op in operations:
            intervals = op.runs(resource)
            if not intervals:
                continue
            # 合并相邻的非零占用，尾拍字节不足不应切碎图上的同一传输。
            merged = []
            for begin, end, _ in intervals:
                if merged and merged[-1][1] == begin:
                    merged[-1] = (merged[-1][0], end)
                else:
                    merged.append((begin, end))
            for begin, end in merged:
                for tile in range(math.floor(-end / ii) + 1, math.ceil((window - begin) / ii)):
                    a, b = max(0, begin + tile * ii), min(window, end + tile * ii)
                    if a >= b:
                        continue
                    ax.broken_barh([(a, b-a)], (row-0.3, 0.6),
                                   facecolors=colors[tile % len(colors)], edgecolors="#64758b", linewidth=0.5)
                    if b - a > ii * 0.035:
                        ax.text((a+b)/2, row, f"{op.name} [{tile}]\n{end-begin} c",
                                ha="center", va="center", fontsize=8)
                    elif begin + tile * ii >= 0 and end + tile * ii <= window:
                        ax.annotate(f"{op.name} {end-begin}c", ((a+b)/2, row-0.3),
                                    xytext=(2, 3), textcoords="offset points",
                                    fontsize=6, color="#33465b")
            if resource == op.unit and op.issue_offsets:
                for tile in range(-math.ceil(op.ready / ii), plot["periods"] + 1):
                    issue_end = op.start + op.issue_offsets[-1] + 1 + tile*ii
                    result = op.ready + tile*ii
                    if 0 <= issue_end < result <= window:
                        ax.plot([issue_end, result], [row+0.34, row+0.34], "--",
                                color="#64758b", linewidth=0.8)
            if not show_io and resource == op.unit and op.unit in VECTOR_UNITS:
                marker = op.start + (op.issue_offsets[0] if op.issue_offsets else 0)
                if op.bytes_read:
                    for tile in range(-math.ceil(marker/ii), plot["periods"]+1):
                        t = marker + tile*ii
                        if 0 <= t <= window:
                            ax.plot(t, row-0.37, marker="v", color="#bb741a", markersize=4)
                if op.bytes_written:
                    for tile in range(-math.ceil(op.ready/ii), plot["periods"]+1):
                        t = op.ready + tile*ii
                        if 0 <= t <= window:
                            ax.plot(t, row+0.37, marker="^", color="#bb741a", markersize=4)
    for engine, resources in ENGINE_RESOURCES.items():
        row = lanes.index(engine+" union")
        mask = np.zeros(window, dtype=bool)
        for op in operations:
            for resource in resources:
                for begin, end, _ in op.runs(resource):
                    for tile in range(math.floor(-end/ii)+1, math.ceil((window-begin)/ii)):
                        a, b = max(0, begin+tile*ii), min(window, end+tile*ii)
                        if a < b:
                            mask[a:b] = True
        edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
        ax.broken_barh([(int(a), int(b-a)) for a,b in zip(edges[::2],edges[1::2])],
                       (row-0.3,0.6), facecolors="#8198b3")
    ax.set_xlim(0, window)
    ax.set_ylim(len(lanes)-0.5, -0.8)
    ax.set_yticks(range(len(lanes)), lanes, fontsize=10)
    ax.set_xticks(np.arange(plot["periods"]+1)*ii)
    ax.grid(axis="x", color="#cad4df", linewidth=0.7)
    ax.set_xlabel(f"Time (cycles); 1 cycle = {1/config['clock_ghz']:g} ns", fontsize=11)
    title = f"FlashAttention steady state | II = {ii} cycles/tile | lower bound = {report['resource_lower_bound_cycles']} cycles"
    fig.suptitle(title, fontsize=19, y=0.97)
    fig.text(0.08,0.925,
             f"L1 reserved {report['l1_reserved_bytes']/1024:g} KiB | one unit per type | "
             f"Vector I/O {'shown (requested or bandwidth-limited)' if show_io else 'overlapped; orange markers = first input / final output'}",
             fontsize=11)
    totals = " | ".join(f"{name} union {data['window_cycles']} c ({data['window_ns']:.1f} ns)"
                       for name,data in report["engines"].items())
    fig.text(0.08,0.035,totals+"\nSteady-state window only; dashed segments show result latency. Engine unions include hidden I/O.",fontsize=10)
    # 明确指定像素尺寸，避免 bbox_inches='tight' 改变请求的全尺寸输出。
    fig.savefig(destination, dpi=plot["dpi"])
    plt.close(fig)


def draw_statistics(config, report, output_dir):
    """直接使用同一份报告绘制统计，所有数值均对应完整绘图窗口。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_plot_fonts()
    plot = config["plot"]
    figsize = (plot["width_px"] / plot["dpi"], plot["height_px"] / plot["dpi"])
    window = report["window_ns"]
    frequency = config["clock_ghz"]
    periods = plot["periods"]
    busy_color, issue_color = "#54799f", "#d59b50"
    subtitle = (f"{periods} 个稳态周期 · 窗口 {report['window_cycles']} cycle / {window:g} ns"
                f" · II = {report['ii_cycles']} cycle/tile · {frequency:g} GHz")

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=plot["dpi"], sharex=True)
    fig.subplots_adjust(left=0.07, right=0.95, top=0.82, bottom=0.13, wspace=0.25)
    compute_units = ["TENSOR"] + list(VECTOR_UNITS)
    io_units = ["TENSOR_LD", "TENSOR_ST", "VECTOR_LD", "VECTOR_ST",
                "DMA_RD", "DMA_WR", "L1_R", "L1_W"]
    for ax, units, title in zip(axes, (compute_units, io_units),
                               ("各计算单元", "读写接口（包含隐藏的 Vector 访存）")):
        positions = np.arange(len(units))
        times = [report["units"][unit]["window_ns"] for unit in units]
        ax.set_title(title, fontsize=22, pad=20)
        if units is compute_units:
            issued = [report["units"][unit]["issue_cycles_per_period"] * periods / frequency
                      for unit in units]
            bars = ax.barh(positions - 0.18, times, height=0.32, color=busy_color, label="占用时间")
            issue_bars = ax.barh(positions + 0.18, issued, height=0.32,
                                color=issue_color, label="发射时间")
            ax.bar_label(issue_bars, labels=[f"{value:g} ns" for value in issued],
                         padding=7, fontsize=14)
            ax.legend(loc="lower right", fontsize=16, frameon=False)
        else:
            bars = ax.barh(positions, times, height=0.55, color=busy_color)
        ax.bar_label(bars, labels=[f"{value:g} ns  ({value / window:.2%})" for value in times],
                     padding=7, fontsize=14)
        ax.set_yticks(positions, units, fontsize=16)
        ax.invert_yaxis()
        ax.set_xlim(0, window * 1.24)
        ax.axvline(window, color="#9ca9b5", linestyle="--", linewidth=1)
        ax.set_xlabel("时间 / ns", fontsize=17)
        ax.grid(axis="x", color="#e3e8ee", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=14)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("FlashAttention · 各执行单元时间统计", fontsize=29, y=0.965)
    fig.text(0.07, 0.90, subtitle, fontsize=19)
    fig.text(0.07, 0.045,
             "占用时间包含指令持有单元期间的供数停顿；发射时间仅计实际发射组。\n"
             "各单元可以并发，条形长度分别相对于同一个统计窗口。L1 接口按活动周期统计。",
             fontsize=16)
    units_path = output_dir / "unit_statistics.png"
    fig.savefig(units_path, dpi=plot["dpi"])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize, dpi=plot["dpi"])
    fig.subplots_adjust(left=0.10, right=0.94, top=0.78, bottom=0.20)
    engines = list(ENGINE_RESOURCES)
    positions = np.arange(len(engines))
    busy = [report["engines"][engine]["window_ns"] for engine in engines]
    compute = [report["engines"][engine]["compute_only"]["window_ns"] for engine in engines]
    for offset, values, color, label in (
            (-0.18, busy, busy_color, "含读写的忙碌时间并集"),
            (0.18, compute, issue_color, "纯计算占用时间并集")):
        bars = ax.barh(positions + offset, values, height=0.30, color=color, label=label)
        labels = [f"{value:g} ns   ({value * frequency:g} cycle，{value / window:.2%})"
                  for value in values]
        ax.bar_label(bars, labels=labels, padding=13, fontsize=22)
    ax.set_yticks(positions, engines, fontsize=26)
    ax.invert_yaxis()
    ax.set_xlim(0, window * 1.32)
    ax.axvline(window, color="#9ca9b5", linestyle="--", linewidth=1)
    ax.set_xlabel("时间 / ns", fontsize=22)
    ax.grid(axis="x", color="#e3e8ee", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=21, frameon=False)
    fig.suptitle("FlashAttention · Tensor / Vector / DMA 两种时间口径", fontsize=29, y=0.965)
    fig.text(0.10, 0.88, subtitle, fontsize=20)
    fig.text(0.10, 0.065,
             "每类引擎内部按时间区间取并集，内部重叠只计一次。三类引擎之间允许重叠。\n"
             "纯计算占用保留计算指令的供数停顿；DMA 没有算术单元，该口径为 0。\n"
             "统计覆盖中间稳态，初始化、最终归一化和输出写回发生在窗口之外。",
             fontsize=17)
    engines_path = output_dir / "engine_statistics.png"
    fig.savefig(engines_path, dpi=plot["dpi"])
    plt.close(fig)
    return {"unit_statistics": units_path, "engine_statistics": engines_path}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/flashattention_pipeline"))
    args = parser.parse_args()
    try:
        config = validate_config(json.loads(args.config.read_text(encoding="utf-8-sig")))
        operations, capacities, ii, lower, total, pools = solve(config)
        report = make_report(config, operations, capacities, ii, lower, total, pools)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    png = args.output_dir / "pipeline.png"
    draw(config, operations, report, png)
    statistics_paths = draw_statistics(config, report, args.output_dir)
    report["png"] = {"path": str(png.resolve()), "width_px": config["plot"]["width_px"],
                     "height_px": config["plot"]["height_px"]}
    report["statistics_png"] = {
        name: {"path": str(path.resolve()), "width_px": config["plot"]["width_px"],
               "height_px": config["plot"]["height_px"]}
        for name, path in statistics_paths.items()
    }
    report_file = args.output_dir / "report.json"
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"II={ii} cycle/tile；资源下界={lower}；L1槽位={total/1024:g} KiB")
    print(f"统计窗口={report['window_cycles']} cycle / {report['window_ns']:.3f} ns")
    print(f"{'执行单元':18} {'窗口占用/cycle':>16} {'窗口占用/ns':>16} {'利用率':>10}")
    for resource, values in report["units"].items():
        if resource.startswith("L1_"):
            continue
        print(f"{resource:18} {values['window_cycles']:16} {values['window_ns']:16.3f} {values['utilization']:10.2%}")
    for engine, values in report["engines"].items():
        print(f"{engine} 合并忙碌 {values['window_cycles']} cycle / {values['window_ns']:.3f} ns；"
              f"纯计算并集 {values['compute_only']['window_cycles']} cycle")
    print(f"PNG {png.resolve()}\nJSON {report_file.resolve()}")
    for name, path in statistics_paths.items():
        print(f"{name} PNG {path.resolve()}")


if __name__ == "__main__":
    main()
