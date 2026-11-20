"""JSON 驱动的 FlashAttention 稳态周期排程、资源统计与全尺寸泳道图。

采用完整有效 KV tile、独立 Vector 功能单元、共享 L1 和逐组 Vector
读写。Q/A 在 L1 驻留，Tensor 局部复用；Vector 小状态在寄存器，score 工作区另列。
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
# 同一资源上的操作以标签区分，带波浪号的量是后续加法的中间结果。
ATTENTION_EQUATIONS = {
    "QK": r"X_j = QK_j^{\mathsf{T}}",
    "SCALE": r"H_j = X_j c",
    "MASK": r"S_j = H_j + \mathrm{mask}_j",
    "ROWMAX": r"r_j = \mathrm{rowmax}(S_j)",
    "MAX": r"m_j = \max(m_{j-1}, r_j)",
    "SUB_S": r"U_j = S_j-m_j",
    "DELTA": r"\Delta m_j = m_{j-1}-m_j",
    "EXP_P": r"P_j = \exp(U_j)",
    "ALPHA": r"\alpha_j = \exp(\Delta m_j)",
    "ROWSUM": r"z_j = \mathrm{rowsum}(P_j)",
    "PV": r"Y_j = P_jV_j",
    "SCALE_L": r"\widetilde{\ell}_j = \alpha_j\ell_{j-1}",
    "ADD_L": r"\ell_j = \widetilde{\ell}_j+z_j",
    "SCALE_A": r"\widetilde{A}_j = \alpha_j A_{j-1}",
    "ADD_A": r"A_j = \widetilde{A}_j+Y_j",
    "CVT_S": r"S_j^{(\mathrm{V})} = \mathrm{cast}(S_j)",
    "CVT_P": r"P_j^{(\mathrm{T})} = \mathrm{cast}(P_j)",
    "CVT_Y": r"Y_j^{(\mathrm{V})} = \mathrm{cast}(Y_j)",
}
FORMAT_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4}
NUMERIC_DEFAULTS = {
    "tensor_input_dtype": "bf16",
    "tensor_output_dtype": "fp32",
    "vector_dtype": "fp32",
    "ddr_k_dtype": "tensor",
    "ddr_v_dtype": "tensor",
    "conversion": {
        "mode": "explicit",
        "vector_load_extra_cycles": 2,
        "vector_store_extra_cycles": 2,
        "dma_extra_cycles": 2,
    },
}


def parameter_caption(config):
    """所有图共用的有效参数摘要；带宽是十进制 GB/s，不是 Gb/s。"""
    f = config['clock_ghz']
    sizes, formats = numeric_layout(config)
    lines = [f"参数快照 | {f:g} GHz | shape {config['shape']} | L1 {config['l1']['size_kib']:g} KiB | 窗口 {config['plot']['periods']} periods"]
    for group in ('vector', 'tensor', 'l1', 'ddr'):
        spec = config[group]
        efficiency = spec.get('efficiency', 1)
        rates = ' / '.join(
            (f"{spec[d+'_port_gbps']:g} GB/s/port × {spec[d+'_ports']} ports → {spec[d+'_port_gbps']*spec[d+'_ports']/f:g} B/cycle" if group == 'vector' else
             f"{spec[d+'_gbps']:g} GB/s → {spec[d+'_gbps']*efficiency/f:g} B/cycle") for d in ('read', 'write'))
        lines.append(f"{group.upper()} 读 / 写容量: {rates}" + (f"；DDR efficiency={efficiency:g}（箭头后为有效值）" if group == 'ddr' else ''))
    lines.append(f"有效数据字节数 {sizes}；格式 {formats or '由 dtype_bytes 指定'}；mask {config['mask']}")
    lines.append(f"Tensor micro(M,K,N)=({config['tensor']['micro_m']},{config['tensor']['micro_k']},{config['tensor']['micro_n']})；计算延迟 {config['tensor']['latency_cycles']} c；" + '；'.join(f"{g} 输入/输出延迟 {config[g]['input_latency_cycles']}/{config[g]['output_latency_cycles']} c" for g in ('tensor', 'vector')))
    lines.append('Vector 各类型一份；并行度/延迟(c): ' + ' | '.join(f"{k} {v['parallelism']}/{v['latency_cycles']}" for k,v in config['vector']['units'].items()))
    lines.append(f"转换 {config.get('numeric', {}).get('conversion', '未启用 numeric')}；排程最大 II {config['scheduler']['max_ii_cycles']}；非活跃参数不会强制改变时序。")
    lines.append('L1 实际占用 = Vector 整端口拍 + Tensor/DMA 原流式需求；每条 Vector 操作每周期最多一个端口，端口逐拍释放，不模拟 AXI burst 锁定。')
    return lines


def save_parameterized_figure(fig, destination, config):
    """追加独立参数区，保持原绘图区尺寸，避免挤压或遮盖已有标注。"""
    import io
    import matplotlib.pyplot as plt
    from PIL import Image
    dpi = config['plot']['dpi']
    plot_buffer = io.BytesIO()
    fig.savefig(plot_buffer, dpi=dpi)
    lines = parameter_caption(config)
    footer = plt.figure(figsize=(fig.get_figwidth(), 2.4), dpi=dpi)
    for i, line in enumerate(lines):
        footer.text(.02, .96-i*.095, line, va='top', fontsize=9)
    footer_buffer = io.BytesIO()
    footer.savefig(footer_buffer, dpi=dpi)
    plt.close(footer)
    plot_buffer.seek(0); footer_buffer.seek(0)
    with Image.open(plot_buffer) as main, Image.open(footer_buffer) as tail:
        combined = Image.new('RGB', (main.width, main.height+tail.height), 'white')
        combined.paste(main, (0, 0)); combined.paste(tail, (0, main.height))
        combined.save(destination)


def validate_config(config):
    """严格核对键和值，避免拼写错误被静默忽略。"""
    template = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))

    def check(value, reference, path):
        if isinstance(reference, dict):
            if not isinstance(value, dict) or value.keys() != reference.keys():
                raise ValueError(f"{path} 的键必须为 {list(reference)}")
            for key in reference:
                check(value[key], reference[key], f"{path}.{key}")
        elif isinstance(reference, str):
            if not isinstance(value, str):
                raise ValueError(f"{path} 必须是字符串")
        elif isinstance(reference, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{path} 必须是布尔值")
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path} 必须是数值")
        elif not math.isfinite(value) or value <= 0:
            raise ValueError(f"{path} 必须是有限正数")
        elif isinstance(reference, int) and not isinstance(value, int):
            raise ValueError(f"{path} 必须是整数")

    if not isinstance(config, dict):
        raise ValueError("config 必须是对象")
    check({key: value for key, value in config.items() if key != "numeric"}, template, "config")
    if "numeric" in config:
        numeric = config["numeric"]
        check(numeric, NUMERIC_DEFAULTS, "config.numeric")
        for key in ("tensor_input_dtype", "tensor_output_dtype", "vector_dtype",
                    "ddr_k_dtype", "ddr_v_dtype"):
            allowed = set(FORMAT_BYTES)
            if key.startswith("ddr_"):
                allowed.add("tensor")
            if numeric[key] not in allowed:
                raise ValueError(f"numeric.{key} 必须是 {sorted(allowed)} 中的一项")
        conversion = numeric["conversion"]
        if conversion["mode"] not in ("explicit", "inline"):
            raise ValueError("numeric.conversion.mode 必须为 explicit 或 inline")
        for key in ("vector_load_extra_cycles", "vector_store_extra_cycles", "dma_extra_cycles"):
            if conversion[key] not in (1, 2):
                raise ValueError(f"numeric.conversion.{key} 必须为 1 或 2")
        if conversion["mode"] == "explicit" and any(
                numeric[key] not in ("tensor", numeric["tensor_input_dtype"])
                for key in ("ddr_k_dtype", "ddr_v_dtype")):
            raise ValueError("DDR K/V 与 Tensor 格式不同时需要 inline DMA 转换")
    if config["ddr"]["efficiency"] > 1:
        raise ValueError("ddr.efficiency 必须在 (0, 1] 内")
    for direction in ('read', 'write'):
        if config['vector'][direction+'_port_gbps'] > config['l1'][direction+'_gbps']:
            raise ValueError('L1 带宽必须至少覆盖一个 Vector 端口；不支持跨周期拆分总线拍')
    if config["mask"] != {"source": "ddr", "dtype": "vector"}:
        raise ValueError("当前mask模型仅支持source=ddr、dtype=vector（稠密加性mask）")
    if min(config["plot"]["width_px"], config["plot"]["height_px"]) < 800:
        raise ValueError("图片宽高须至少 800 px，以显示完整泳道")
    return config


def numeric_layout(config):
    sizes = dict(config["dtype_bytes"])
    numeric = config.get("numeric")
    if numeric is None:
        return sizes, None
    operand, result, vector = (numeric[key] for key in
                               ("tensor_input_dtype", "tensor_output_dtype", "vector_dtype"))
    formats = dict.fromkeys(("q", "k", "v", "p"), operand)
    formats.update(s=result, y=result, state=vector)
    for obj in ("k", "v"):
        fmt = numeric[f"ddr_{obj}_dtype"]
        formats[f"ddr_{obj}"] = operand if fmt == "tensor" else fmt
    sizes.update({obj: FORMAT_BYTES[fmt] for obj, fmt in formats.items()})
    return sizes, formats


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
    conversion: dict | None = None
    result_offset: int | None = None
    local_deps: tuple = ()

    @property
    def ready(self):
        return self.start + self.ready_offset

    def dependency_ready(self, predecessor):
        if predecessor.name in self.local_deps and predecessor.result_offset is not None:
            return predecessor.start + predecessor.result_offset
        return predecessor.ready

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
    shape = config["shape"]
    sizes, formats = numeric_layout(config)
    conversion = config.get("numeric", NUMERIC_DEFAULTS)["conversion"]
    inline = formats is not None and conversion["mode"] == "inline"
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
            rate = (config[group][direction+'_port_gbps'] * config[group][direction+'_ports']
                    if group == 'vector' else config[group][direction+'_gbps'])
            capacities[name] = rate * efficiency / f
    capacities['L1_R_BUS'] = capacities['L1_R']
    capacities['L1_W_BUS'] = capacities['L1_W']
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

    def cast(source, destination, path, extra_cycles=0):
        return {"source": source, "destination": destination,
                "path": path, "extra_cycles": extra_cycles}

    def dma_load(obj, elements):
        name = f"{obj.upper()}_DMA"
        if formats is None or formats[f"ddr_{obj}"] == formats[obj]:
            transfer(name, "DMA_RD", elements * sizes[obj], "write")
            operations[-1].bytes_read = elements * sizes[obj]
            return
        if not inline:
            raise ValueError("DDR K/V 与 Tensor 格式不同时需要 inline DMA 转换")
        source_width, destination_width = sizes[f"ddr_{obj}"], sizes[obj]
        rate = min(capacities["DMA_RD"], capacities["L1_W"] * source_width / destination_width)
        source = byte_profile(elements * source_width, rate)
        extra = conversion["dma_extra_cycles"]
        destination = np.pad(source * (destination_width / source_width), (extra, 0))
        operations.append(Operation(
            name, "DMA_RD", {"DMA_RD": source, "L1_W": destination}, len(destination),
            formula=f"{elements} elements; {source_width} -> {destination_width} B; {len(source)}+{extra} c",
            bytes_read=elements * source_width, bytes_written=elements * destination_width,
            write_start=extra,
            conversion=cast(formats[f"ddr_{obj}"], formats[obj], "DMA_RD", extra)))

    def compute(name, unit, work, deps=(), read_element_bytes=0,
                write_element_bytes=0, format_cast=None, local_deps=()):
        if work == 0:
            operations.append(Operation(name, unit, {}, 0, deps,
                                        "single-element reduction: no binary operation",
                                        local_deps=local_deps))
            return
        spec = vector["units"][unit]
        width = spec["parallelism"]
        latency = spec["latency_cycles"]
        groups = math.ceil(work / width)
        # 一个操作每周期使用至多一个端口；多端口允许不同操作并发。
        read_rate = vector['read_port_gbps'] / f
        write_rate = vector['write_port_gbps'] / f
        read_cycles = math.ceil(min(width, work) * read_element_bytes / read_rate)
        write_cycles = math.ceil(min(width, work) * write_element_bytes / write_rate)
        stride = max(1, read_cycles, write_cycles)
        input_delay = read_cycles + vector["input_latency_cycles"] - 1 if read_element_bytes else 0
        output_extra = vector["output_latency_cycles"] - 1
        if format_cast and format_cast["path"] == "VECTOR_LD":
            input_delay += format_cast["extra_cycles"]
        if format_cast and format_cast["path"] == "VECTOR_ST":
            output_extra += format_cast["extra_cycles"]
        issues = [input_delay + group * stride for group in range(groups)]
        result_end = end = issues[-1] + latency
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
            issues, read_end, first_write, conversion=format_cast,
            result_offset=result_end, local_deps=local_deps))

    def gemm(name, rows, cols, reduction, deps, input_bytes, output_bytes):
        blocks = math.ceil(rows / tensor["micro_m"]) * math.ceil(cols / tensor["micro_n"])
        rounds = math.ceil(reduction / tensor["micro_k"])
        # 理想本地复用：唯一输入字节按发射组均匀供给，不建模 SRAM bank/具体复用顺序。
        # 只有最后一个 K round 产生最终输出，不能提前写出部分和。
        read_rate = min(capacities["TENSOR_LD"], capacities["L1_R"])
        write_rate = min(capacities["TENSOR_ST"], capacities["L1_W"])
        chunk = input_bytes / (blocks * rounds)
        output_width = output_bytes / (rows * cols)
        outputs = [min(tensor["micro_m"], rows-i)*min(tensor["micro_n"], cols-j)*output_width
                   for i in range(0, rows, tensor["micro_m"])
                   for j in range(0, cols, tensor["micro_n"])]
        read_cycles = math.ceil(chunk/read_rate)
        stride = max(1, read_cycles, math.ceil(max(outputs)/write_rate))
        first = read_cycles + tensor["input_latency_cycles"] - 1
        gap = max(0, tensor["latency_cycles"] - blocks*stride)
        reads = [r*(blocks*stride+gap)+b*stride for r in range(rounds) for b in range(blocks)]
        issues = [t+first for t in reads]
        result = issues[-1]+tensor["latency_cycles"]
        writes = [t+tensor["latency_cycles"]+tensor["output_latency_cycles"]-1
                  for t in issues[-blocks:]]
        ready = max(result, max(t+math.ceil(size/write_rate) for t,size in zip(writes,outputs)))
        profiles = {r: np.zeros(ready) for r in ("TENSOR", "TENSOR_LD", "TENSOR_ST", "L1_R", "L1_W")}
        # 带宽供数停顿持有阵列；独立结果尾延迟不算发射占用。
        profiles["TENSOR"][issues[0]:issues[-1]+1] = 1
        for t in reads:
            data = byte_profile(chunk, read_rate)
            profiles["TENSOR_LD"][t:t+len(data)] += data
        for t,size in zip(writes,outputs):
            data = byte_profile(size, write_rate)
            profiles["TENSOR_ST"][t:t+len(data)] += data
        profiles["L1_R"] = profiles["TENSOR_LD"].copy()
        profiles["L1_W"] = profiles["TENSOR_ST"].copy()
        operations.append(Operation(
            name, "TENSOR", profiles, ready, deps,
            f"{blocks} output blocks × {rounds} K rounds; stream stride={stride}; "
            f"read latency={tensor['input_latency_cycles']}; write latency={tensor['output_latency_cycles']}",
            rows * cols * reduction, bytes_read=input_bytes, bytes_written=output_bytes,
            issue_offsets=issues, read_end=reads[-1]+read_cycles,
            write_start=writes[0], result_offset=result))

    q, k, v = m * dk * sizes["q"], n * dk * sizes["k"], n * dv * sizes["v"]
    s, p, y = m * n * sizes["s"], m * n * sizes["p"], m * dv * sizes["y"]
    dma_load("k", n * dk)
    dma_load("v", n * dv)
    mask_bytes = m*n*sizes["state"]
    transfer("MASK_DMA", "DMA_RD", mask_bytes, "write")
    operations[-1].bytes_read = mask_bytes
    gemm("QK", m, n, dk, ("K_DMA",), q+k, s)
    def vector_input(obj, elements, producer):
        if formats is None or formats[obj] == formats["state"]:
            return producer, sizes[obj], None
        if inline:
            return producer, sizes[obj], cast(
                formats[obj], formats["state"], "VECTOR_LD", conversion["vector_load_extra_cycles"])
        name = f"CVT_{obj.upper()}"
        compute(name, "CVT", elements, (producer,), read_element_bytes=sizes[obj],
                format_cast=cast(formats[obj], formats["state"], "CVT"))
        return name, 0, None

    s_input, s_read, s_cast = vector_input("s", m * n, "QK")
    compute("SCALE", "MUL", m * n, (s_input,), read_element_bytes=s_read, format_cast=s_cast)
    # c=1/sqrt(d_k) 在初始化预计算；稠密mask从L1流式读取。
    # 即使完整有效 tile 的 mask 为零，也显式执行加法以保留硬件开销。
    compute("MASK", "ADD", m * n, ("SCALE", "MASK_DMA"), read_element_bytes=sizes["state"])
    compute("ROWMAX", "REDMAX", m * (n - 1), ("MASK",))
    compute("MAX", "MAX", m, ("ROWMAX",))
    compute("SUB_S", "SUB", m * n, ("MAX",))
    compute("DELTA", "SUB", m, ("MAX",))
    p_store = "EXP_P" if inline or (formats and formats["p"] == formats["state"]) else "CVT_P"
    p_cast = None
    if formats and formats["state"] != formats["p"]:
        p_path = "VECTOR_ST" if inline else "CVT"
        p_cast = cast(formats["state"], formats["p"], p_path,
                      conversion["vector_store_extra_cycles"] if inline else 0)
    compute("EXP_P", "EXP", m * n, ("SUB_S",),
            write_element_bytes=sizes["p"] if p_store == "EXP_P" else 0,
            format_cast=p_cast if p_store == "EXP_P" else None)
    compute("ALPHA", "EXP", m, ("DELTA",))
    compute("ROWSUM", "REDSUM", m * (n - 1), ("EXP_P",), local_deps=("EXP_P",))
    if p_store == "CVT_P":
        compute(p_store, "CVT", m * n, ("EXP_P",), write_element_bytes=sizes["p"], format_cast=p_cast)
    gemm("PV", m, dv, n, (p_store, "V_DMA"), p+v, y)
    compute("SCALE_A", "MUL", m * dv, ("ALPHA", "PV"),
            read_element_bytes=sizes["state"], write_element_bytes=sizes["state"])
    y_input, y_read, y_cast = vector_input("y", m * dv, "PV")
    compute("ADD_A", "ADD", m * dv, ("SCALE_A", y_input),
            read_element_bytes=y_read+sizes["state"], write_element_bytes=sizes["state"],
            format_cast=y_cast)
    compute("SCALE_L", "MUL", m, ("ALPHA", "ROWSUM"))
    compute("ADD_L", "ADD", m, ("SCALE_L", "ROWSUM"))
    lifetimes = [(k, "K_DMA", "QK"), (v, "V_DMA", "PV"),
                 (mask_bytes, "MASK_DMA", "MASK"),
                 (s, "QK", s_input if s_input != "QK" else "SCALE"),
                 (p, p_store, "PV"),
                 (y, "PV", y_input if y_input != "PV" else "ADD_A")]
    for op in operations:
        for effective, bus, port, direction in (
                ('L1_R', 'L1_R_BUS', 'VECTOR_LD', 'read'),
                ('L1_W', 'L1_W_BUS', 'VECTOR_ST', 'write')):
            if effective in op.profiles:
                op.profiles[bus] = op.profiles[effective].copy()
            if port in op.profiles:
                beat = vector[direction+'_port_gbps'] / f
                # 有效数据保留在 L1_R/W；接口与总线按完整端口拍占用。
                op.profiles[port] = (op.profiles[port] > 0).astype(float) * beat
                op.profiles[bus] = op.profiles[port].copy()
    return operations, capacities, lifetimes, q + m*dv*sizes["state"]


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
        pools[producer] = {"bytes_per_slot": size, "slots": slots, "lifetime_cycles": lifetime,
                           "allocate_cycle": first.start, "release_cycle": end,
                           "consumer": consumer}
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
            earliest = max((op.dependency_ready(by_name[name]) for name in op.deps), default=0)
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
            if op.dependency_ready(by_name[dep]) > op.start:
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


def capacity_report(config, operations, pools, ii):
    sizes, _ = numeric_layout(config)
    m,n,dk,dv = (config["shape"][k] for k in ("query_rows","kv_cols","head_dim_qk","head_dim_v"))
    shared = {"Q":m*dk*sizes["q"], "A":m*dv*sizes["state"]}
    names = {"K_DMA":"K","V_DMA":"V","MASK_DMA":"mask","QK":"S","EXP_P":"P","CVT_P":"P","PV":"Y"}
    count_period = {}
    for producer,pool in pools.items():
        a,b = pool["allocate_cycle"],pool["release_cycle"]
        count_period[names[producer]] = np.bincount(
            np.arange(a,b) % ii,minlength=ii)*pool["bytes_per_slot"]
    series = {"Q+A":np.full(ii,sum(shared.values()),dtype=int),**count_period}
    series["总存活"] = sum(series.values())
    intervals = {}
    for name,period in series.items():
        values = np.tile(period,config["plot"]["periods"])
        edges = np.r_[0,np.flatnonzero(np.diff(values))+1,len(values)]
        intervals[name] = [{"start":int(a),"end":int(b),"bytes":int(values[a])}
                           for a,b in zip(edges,edges[1:])]
    named = {o.name:o for o in operations}
    begin = named.get("CVT_S",named["SCALE"]).start
    end = max(named["ROWSUM"].ready,named.get("CVT_P",named["EXP_P"]).ready)
    scratch_slots = max(1,math.ceil((end-begin)/ii))
    scratch = {"score_work_bytes_per_slot":m*n*sizes["state"],
               "score_work_slots":scratch_slots,
               "score_work_reserved_bytes":scratch_slots*m*n*sizes["state"],
               "score_lifetime_cycles":end-begin,
               "note":"H/S/U/P宽精度工作区可就地复用；P必须保留至ROWSUM及窄化结束；不计入L1"}
    if "CVT_Y" in named:
        life = named["ADD_A"].ready-named["CVT_Y"].start
        scratch["explicit_Y_reserved_bytes"] = math.ceil(life/ii)*m*dv*sizes["state"]
    return {"shared_bytes":shared,"pool_names":names,"intervals":intervals,
            "live_peak_bytes":int(series["总存活"].max()),"local_scratch":scratch,
            "register_state_bytes_m_l":2*m*sizes["state"],
            "scope":"中间稳态；最终O可在A区域内逐组归一化转换后写出，未另预留O"}


def bandwidth_samples(report, resource):
    values = np.zeros(report["window_cycles"])
    for interval in report["l1_bandwidth_intervals"][resource]:
        values[interval["start"]:interval["end"]] = interval["bytes_per_cycle"]
    return values


def bandwidth_regions(values):
    """标记连续密集跳变区，不改变任何原始采样值。"""
    edges = np.r_[0,np.flatnonzero(np.diff(values))+1,len(values)]
    groups, begin = [], None
    for a,b in zip(edges,edges[1:]):
        if b-a <= 3:
            if begin is None:
                begin = int(a)
        else:
            if begin is not None and a-begin >= 8:
                groups.append((begin,int(a)))
            begin = None
    if begin is not None and len(values)-begin >= 8:
        groups.append((begin,len(values)))
    return groups


def draw_bandwidth_lane(ax, report, resource, row):
    values = bandwidth_samples(report, resource)
    cfg = report["config"]
    direction = "read" if resource.startswith("L1_R") else "write"
    cap = cfg["l1"][direction+"_gbps"]/cfg["clock_ghz"]
    color = "#246a9e" if direction == "read" else "#23854b"
    base, height = row+.36, .64
    edges = np.arange(len(values)+1)
    ys = base-height*np.r_[values,values[-1]]/cap
    ax.fill_between(edges,ys,base,step="post",color=color,alpha=.17)
    ax.step(edges,ys,where="post",color=color,linewidth=.55)
    ax.axhline(base-height,color=color,linestyle="--",linewidth=.6,alpha=.6)
    renderer = ax.figure.canvas.get_renderer()
    boxes = []
    def label(x,y,text,size=7):
        item = ax.text(x,y,text,fontsize=size,ha="center",va="center",color="#183242",
                       bbox=dict(facecolor="white",edgecolor="none",alpha=.8,pad=.3))
        box = item.get_window_extent(renderer).expanded(1.12,1.05)
        if any(box.overlaps(old) for old in boxes) or box.x0<ax.bbox.x0 or box.x1>ax.bbox.x1:
            item.remove()
        else:
            boxes.append(box)
    # 区域编号在每个周期重复；明细图使用相同编号及周期内坐标。
    prefix = ("R" if direction == "read" else "W") + ("B" if resource.endswith('_BUS') else '')
    for index,(a,b) in enumerate(bandwidth_regions(values[:report["ii_cycles"]]),1):
        for offset in range(0,len(values),report["ii_cycles"]):
            label((a+b)/2+offset,row-.02,f"{prefix}{index}")
    for item in report["l1_bandwidth_intervals"][resource]:
        a,b,v = item["start"],item["end"],item["bytes_per_cycle"]
        width = (b-a)*ax.bbox.width/len(values)
        if width >= 38:
            text = f"{100*v/cap:.1f}%" if resource.endswith('_BUS') else f"{v:g}"
            label((a+b)/2,row+.07,text)
    kind = '实际占用' if resource.endswith('_BUS') else '有效数据'
    if resource.endswith('_BUS'):
        return f"{'L1 读' if direction=='read' else 'L1 写'} {kind} (%)\n均 {100*values.mean()/cap:.1f}% / 峰 {100*values.max()/cap:.1f}% / 限 100%"
    return f"{'L1 读' if direction=='read' else 'L1 写'} {kind} B/c\n均 {values.mean():.1f} / 峰 {values.max():g} / 限 {cap:g}"


def bandwidth_contributors(report, resource):
    """展开所有与一个稳态周期相交的操作实例，保留来源而非只按总量合并。"""
    ii = report["ii_cycles"]
    sources = {}
    for op in report["operations"]:
        for a,b,rate in op["resource_intervals"].get(resource,[]):
            for tile in range(math.floor(-b/ii)+1,math.ceil((ii-a)/ii)):
                left,right = max(0,a+tile*ii),min(ii,b+tile*ii)
                if left < right:
                    key = f"{op['name']}[t{tile:+d}]"
                    sources.setdefault(key,np.zeros(ii))[left:right] += rate
    total = sum(sources.values(),np.zeros(ii))
    np.testing.assert_allclose(total,bandwidth_samples(report,resource)[:ii],rtol=0,atol=1e-8)
    return sources


def draw_bandwidth_details(config, report, output_dir):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    configure_plot_fonts()
    ii = report["ii_cycles"]
    chunks = math.ceil(ii/64)
    rows = chunks+1
    plot = config["plot"]
    height = max(plot["height_px"],rows*430+240)
    fig = plt.figure(figsize=(plot["width_px"]/plot["dpi"],height/plot["dpi"]),dpi=plot["dpi"])
    grid = fig.add_gridspec(rows,2,left=.045,right=.985,top=.975,bottom=.018,hspace=.95,wspace=.12)
    report["bandwidth_contributors"] = {}
    op_names = [o["name"] for o in report["operations"]]
    palette = plt.get_cmap("tab20")
    for column,(resource,direction) in enumerate((("L1_R","read"),("L1_W","write"))):
        samples = bandwidth_samples(report,resource)
        values = samples[:ii]
        if not np.allclose(samples,np.tile(values,config["plot"]["periods"])):
            raise ValueError("带宽明细的周期折叠与原始窗口不一致")
        sources = bandwidth_contributors(report,resource)
        source_records = {}
        for key,series in sources.items():
            edges = np.r_[0,np.flatnonzero(np.diff(series))+1,ii]
            source_records[key] = [dict(start=int(a),end=int(b),bytes_per_cycle=float(series[a]))
                                   for a,b in zip(edges,edges[1:]) if series[a]>0]
        report["bandwidth_contributors"][resource] = source_records
        cap = config["l1"][direction+"_gbps"]/config["clock_ghz"]
        regions = bandwidth_regions(values)
        prefix = "R" if resource=="L1_R" else "W"
        def panel(ax,a,b,annotate):
            v = values[a:b]
            ax.set_xlim(a,b); ax.set_ylim(-cap*.05,cap*1.25)
            baseline = np.zeros(b-a)
            handles = []
            label_regions = []
            number_labels = []
            for key,series in sources.items():
                part = series[a:b]
                if not np.any(part):
                    continue
                name = key.split("[")[0]
                color = palette(op_names.index(name)%20)
                upper = baseline+part
                ax.fill_between(np.arange(a,b+1),np.r_[baseline,baseline[-1]],
                                np.r_[upper,upper[-1]],step="post",color=color,alpha=.75)
                handles.append(Patch(facecolor=color,label=key))
                if annotate:
                    # 来源数值或堆叠位置改变都必须重新分段。
                    edges = np.r_[0,np.flatnonzero((np.diff(part)!=0)|(np.diff(baseline)!=0))+1,len(part)]
                    for x,y in zip(edges,edges[1:]):
                        if part[x] > 0:
                            number_labels.append(ax.text(a+(x+y)/2,baseline[x]+part[x]/2,f"{part[x]:g}",
                                    fontsize=7,ha="center",va="center",color="#102432"))
                active = np.flatnonzero(part>0)
                # 隔拍写出的同一操作只标一次；长时间空闲则分段标注。
                cuts = np.r_[0,np.flatnonzero(np.diff(active)>3)+1,len(active)]
                for left,right in zip(cuts,cuts[1:]):
                    positions = active[left:right]
                    label_regions.append((key,positions,baseline.copy(),part.copy()))
                baseline = upper
            ax.stairs(v,np.arange(a,b+1),color="#152b3c",linewidth=.9)
            bus = bandwidth_samples(report, resource+'_BUS')[a:b]
            ax.stairs(bus,np.arange(a,b+1),color="#8b3293",linestyle='--',linewidth=1)
            ax.axhline(cap,color="#b35d4f",linestyle="--",linewidth=.8)
            ax.set_xlim(a,b); ax.set_ylim(-cap*.05,cap*1.25)
            ax.set_ylabel("B/cycle",fontsize=9)
            ax.set_xlabel("cycle（周期内坐标）",fontsize=9)
            ax.grid(alpha=.18)
            if handles:
                ax.legend(handles=handles,loc="upper left",bbox_to_anchor=(0,-.32),
                          ncol=4,fontsize=8,frameon=False,borderaxespad=0)
            if annotate:
                edges = np.r_[0,np.flatnonzero(np.diff(v))+1,len(v)]
                for x,y in zip(edges,edges[1:]):
                    ax.text(a+(x+y)/2,v[x]+cap*.035,f"{v[x]:g}",ha="center",va="bottom",fontsize=8)
            renderer = fig.canvas.get_renderer()
            occupied = [text.get_window_extent(renderer).expanded(1.05,1.05) for text in number_labels]
            for key,positions,base,part in label_regions:
                # 在来源区域内寻找不覆盖数值的放置点；薄层改用引线标注。
                placed = False
                for fraction in (.5,.3,.7):
                    x = int(positions[min(len(positions)-1,int((len(positions)-1)*fraction))])
                    for vertical in (.78,.22,.5):
                        item = ax.text(a+x+.5,base[x]+part[x]*vertical,key,fontsize=9 if not annotate else 8,
                                       fontweight="bold",ha="center",va="center",zorder=5)
                        box = item.get_window_extent(renderer).expanded(1.07,1.08)
                        low,high = ax.transData.transform((a+x,base[x]))[1],ax.transData.transform((a+x,base[x]+part[x]))[1]
                        left = ax.transData.transform((a+positions[0],0))[0]
                        right = ax.transData.transform((a+positions[-1]+1,0))[0]
                        if (box.x0>=left and box.x1<=right and box.y0>=low and box.y1<=high
                                and not any(box.overlaps(old) for old in occupied)):
                            occupied.append(box); placed=True
                            break
                        item.remove()
                    if placed:
                        break
                if not placed:
                    x = int(positions[len(positions)//2])
                    for level in (1.18,1.07,.96):
                        item = ax.annotate(key,xy=(a+x+.5,base[x]+part[x]/2),
                                           xytext=(a+x+.5,cap*level),fontsize=7,fontweight="bold",
                                           ha="center",va="center",arrowprops=dict(arrowstyle="-",lw=.5),
                                           bbox=dict(facecolor="white",edgecolor="none",alpha=.8,pad=.4))
                        box = item.get_window_extent(renderer)
                        if not any(box.overlaps(old) for old in occupied):
                            occupied.append(box); break
                        item.remove()
            ids = [f"{prefix}{j}=[{s},{e})" for j,(s,e) in enumerate(regions,1) if s<b and e>a]
            ax.set_title(f"{'读' if column==0 else '写'} [{a}, {b})  "+ "  ".join(ids),fontsize=11,loc="left")
        full = fig.add_subplot(grid[0,column]); panel(full,0,ii,False)
        for j,(a,b) in enumerate(regions,1):
            full.text((a+b)/2,cap*1.1,f"{prefix}{j}",ha="center",fontsize=11)
        for index,a in enumerate(range(0,ii,64)):
            panel(fig.add_subplot(grid[index+1,column]),a,min(a+64,ii),True)
    fig.suptitle(f"L1 读写带宽及操作贡献 | 左读、右写，同一行对应同一时间区间 | II={ii} cycle",fontsize=22,y=.997)
    fig.text(.045,.987,"黑线/数字/彩色层=有效数据带宽；紫色虚线=L1实际占用（Vector整端口拍）；红虚线=L1容量。单位 B/cycle；图例为操作[tile偏移]。总图第k周期坐标=此处坐标+k×II。",fontsize=12)
    path = output_dir/"bandwidth_detail.png"
    save_parameterized_figure(fig, path, config); plt.close(fig)
    return {"bandwidth_detail":path}


def draw_capacity_rows(ax, report, first_row):
    import matplotlib.pyplot as plt
    info = report["l1_capacity"]
    for offset,(name,intervals) in enumerate(info["intervals"].items()):
        row = first_row+offset
        peak = max(v["bytes"] for v in intervals)
        for v in intervals:
            a,b,size = v["start"],v["end"],v["bytes"]
            ax.broken_barh([(a,b-a)],(row-.28,.56),
                          facecolors=plt.get_cmap("Oranges")(.1+.55*size/max(1,peak)),
                          edgecolors="white",linewidth=.3)
            if b-a >= report["ii_cycles"]*.055:
                ax.text((a+b)/2,row,f"{size/1024:g}",ha="center",va="center",fontsize=7)
    return [f"{name} (KiB)" for name in info["intervals"]]


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
            "capacity_per_cycle": capacities[resource],
            "mean_demand_per_cycle": float(amounts[resource].mean()),
            "capacity_utilization": float(amounts[resource].mean()/capacities[resource]),
            **timing(int(busy[resource].sum())),
            "issue_cycles_per_period": issued[resource],
            "peak_demand_per_cycle": float(amounts[resource].max()),
            "demand_per_period": float(amounts[resource].sum()),
        }
        if resource in ('VECTOR_LD', 'VECTOR_ST'):
            direction = 'read' if resource == 'VECTOR_LD' else 'write'
            beat = config['vector'][direction+'_port_gbps']/frequency
            unit_times[resource].update(
                port_count=config['vector'][direction+'_ports'],
                port_bytes_per_cycle=beat,
                occupied_port_cycles_per_period=float(amounts[resource].sum()/beat),
                peak_ports_in_use=float(amounts[resource].max()/beat))
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
    bandwidth = {}
    for resource in ("L1_R", "L1_W", "L1_R_BUS", "L1_W_BUS"):
        samples = np.tile(amounts[resource], periods)
        edges = np.r_[0, np.flatnonzero(np.diff(samples)) + 1, len(samples)]
        bandwidth[resource] = [
            {"start": int(a), "end": int(b), "bytes_per_cycle": float(samples[a])}
            for a, b in zip(edges, edges[1:])
        ]
    sizes, formats = numeric_layout(config)
    return {
        "config": config, "ii_cycles": ii, "resource_lower_bound_cycles": lower_bound,
        "effective_formats": formats, "effective_dtype_bytes": sizes,
        "tensor_streaming": {
            "input_latency_cycles": config["tensor"]["input_latency_cycles"],
            "output_latency_cycles": config["tensor"]["output_latency_cycles"],
            "input_model": "unique operand bytes evenly supplied per micro-GEMM issue with ideal local reuse",
            "output_model": "final K round only; each output block waits for compute result then streams to L1",
        },
        "proven_optimal_in_model": ii == lower_bound,
        "scheduler": "确定性拓扑顺序、最早可行相位的周期排程；下界以上的结果为可行解",
        "window_cycles": ii * periods, "window_ns": ii * periods / frequency,
        "l1_reserved_bytes": total, "l1_pools": pools,
        "l1_capacity": capacity_report(config, operations, pools, ii),
        "units": unit_times, "engines": engine_times,
        "l1_bandwidth_intervals": bandwidth,
        "scope": "完整有效KV tile的中间稳态，Q已在L1；不含首块初始化、最终归一化与写回",
        "assumptions": [
            "Vector LD/ST 按单端口完整拍占用；同操作每周期最多一个端口，多端口仅允许不同操作并发；按拍释放，不模拟burst锁定或bank映射",
            "L1_R/W为有效数据，L1_R_BUS/W_BUS为Vector整拍加Tensor/DMA流式需求；实际占用参与共享L1资源约束",
            "缩放常数c=1/sqrt(d_k)在初始化预计算；稠密mask从DDR经DMA加载，Vector从L1读取后执行独立ADD",
            "Q/A驻留L1；A由SCALE_A和ADD_A各读写一次并原地更新；m/ℓ及行向量保留寄存器",
            "Vector的FP32 score工作区仍需本地scratch，容量另列；不将其视为免费寄存器或计入L1",
            "同类型单元各一份，不同类型并发；向量指令欠供期间保持单元占用",
            "归约工作量按二元操作计，归约结果整行就绪后交付",
            "L1读写独立；Tensor流式读写，唯一输入按发射组均匀供给且本地复用，最后K轮逐输出块写回；不模拟bank映射及具体缓存读取顺序",
            "DDR带宽含有效系数；没有额外请求启动延迟；采用独立固定L1槽池",
        ],
        "operations": [{
            "name": op.name, "unit": op.unit, "start": op.start, "ready": op.ready,
            "result_ready": op.start + op.result_offset if op.result_offset is not None else op.ready,
            "dependencies": list(op.deps), "local_dependencies": list(op.local_deps),
            "formula": op.formula, "work": op.work, "conversion": op.conversion,
            "attention_equation": ATTENTION_EQUATIONS.get(op.name),
            "bytes_read": op.bytes_read, "bytes_written": op.bytes_written,
            "issue_offsets": op.issue_offsets,
            "resource_intervals": {r: op.runs(r) for r in op.profiles},
        } for op in operations],
    }


def execution_overview(operations, ii, window):
    """从实际资源区间重建单 tile 和周期窗口，不以操作时长之和代替并集。"""
    end = max(op.ready for op in operations)
    resources = {r for rs in ENGINE_RESOURCES.values() for r in rs}
    single = {r: np.zeros(end, dtype=bool) for r in resources}
    steady = {r: np.zeros(window, dtype=bool) for r in resources}
    for op in operations:
        for r in resources:
            for a, b, _ in op.runs(r):
                single[r][a:b] = True
                for tile in range(math.floor(-b / ii) + 1, math.ceil((window-a) / ii)):
                    left, right = max(0, a+tile*ii), min(window, b+tile*ii)
                    steady[r][left:right] = True

    def summarize(masks, selected):
        count = np.sum([masks[r].astype(int) for r in selected], axis=0)
        active = np.flatnonzero(count)
        busy = len(active)
        return dict(work_resource_cycles=int(count.sum()), busy_cycles=busy,
                    occupancy_span_cycles=int(active[-1]-active[0]+1) if busy else 0,
                    average_parallelism=float(count.sum()/busy) if busy else 0,
                    peak_parallelism=int(count.max()))

    data = {"single_tile": {}, "steady_window": {}, "joins": []}
    for engine, rs in ENGINE_RESOURCES.items():
        primary = tuple(r for r in rs if r == "TENSOR" or r in VECTOR_UNITS)
        if engine == "DMA":
            primary = rs
        owned = [op for op in operations if op.unit in rs]
        for name, masks in (("single_tile", single), ("steady_window", steady)):
            data[name][engine] = {"primary": summarize(masks, primary),
                                  "including_io": summarize(masks, rs)}
        data["single_tile"][engine]["completion_span_cycles"] = (
            max(op.ready for op in owned)-min(op.start for op in owned) if owned else 0)
    by_name = {op.name: op for op in operations}
    for name in ("ADD_L", "ADD_A"):
        op = by_name[name]
        ready = {dep: op.dependency_ready(by_name[dep]) for dep in op.deps}
        gate = max(ready.values())
        data["joins"].append(dict(operation=name, input_ready=ready,
                                  dependency_gate=gate, scheduled_start=op.start,
                                  scheduled_wait_cycles=op.start-gate))
    return data, single, steady


def draw_execution_overview(config, operations, report, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_plot_fonts()
    data, single, steady = execution_overview(
        operations, report["ii_cycles"], report["window_cycles"])
    report["execution_overview"] = data
    plot = config["plot"]
    fig = plt.figure(figsize=(plot["width_px"]/plot["dpi"],
                              plot["height_px"]/plot["dpi"]), dpi=plot["dpi"])
    grid = fig.add_gridspec(2, 2, width_ratios=[3.6, 1.4], height_ratios=[1.4, 1.5],
                           left=.055, right=.985, top=.89, bottom=.09,
                           hspace=.42, wspace=.08)
    top, bottom = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])
    notes = fig.add_subplot(grid[:, 1]); notes.axis("off")
    active_units = [r for r in VECTOR_UNITS if single[r].any()]
    lanes = ["DMA", "Tensor", "Vector"] + active_units
    colors = plt.get_cmap("Blues")

    def blocks(ax, values, row, cmap=colors):
        edges = np.r_[0, np.flatnonzero(np.diff(values))+1, len(values)]
        for a, b in zip(edges, edges[1:]):
            if values[a]:
                ax.broken_barh([(int(a), int(b-a))], (row-.29, .58),
                               facecolors=cmap(.28+.65*values[a]/max(1, values.max())),
                               edgecolors="white", linewidth=.25)

    def label_dma(ax, row, window, periodic=False):
        # 按操作边界覆盖并集条，尾拍字节不足仍属于同一笔搬运。
        for op in operations:
            if op.unit not in ENGINE_RESOURCES["DMA"]:
                continue
            spans = []
            for a, b, _ in op.runs(op.unit):
                if spans and spans[-1][1] == a:
                    spans[-1] = (spans[-1][0], b)
                else:
                    spans.append((a, b))
            label = ("load " if op.unit == "DMA_RD" else "store ") + op.name.split("_")[0]
            color = {"K_DMA":"#285f91","V_DMA":"#438a9d","MASK_DMA":"#875c91"}.get(op.name,"#596c7a")
            for begin, end in spans:
                tiles = (range(math.floor(-end/report["ii_cycles"])+1,
                               math.ceil((window-begin)/report["ii_cycles"])) if periodic else (0,))
                for tile in tiles:
                    a = max(0, begin+tile*report["ii_cycles"])
                    b = min(window, end+tile*report["ii_cycles"])
                    if a >= b:
                        continue
                    ax.broken_barh([(a,b-a)], (row-.29,.58), facecolors=color,
                                   edgecolors="white", linewidth=1, zorder=3)
                    text = ax.text((a+b)/2,row,label,ha="center",va="center",
                                   color="white",fontsize=11,zorder=4)
                    if text.get_window_extent(fig.canvas.get_renderer()).width+6 > (b-a)*ax.bbox.width/window:
                        text.set_y(row-.31); text.set_va("bottom"); text.set_color(color)

    for row, lane in enumerate(lanes):
        top.axhspan(row-.43, row+.43, color="#f2f5f8", zorder=0)
        if lane in ENGINE_RESOURCES:
            rs = ("TENSOR",) if lane == "Tensor" else ENGINE_RESOURCES[lane]
            # 主泳道统一包含接口；内部资源行仅显示计算。
            count = np.sum([single[r].astype(int) for r in rs], axis=0)
            blocks(top, count, row)
            if lane == "Tensor":
                for op in operations:
                    if op.unit != "TENSOR":
                        continue
                    a = op.start+op.issue_offsets[0]
                    b = op.start+op.issue_offsets[-1]+1
                    top.text((a+b)/2,row,"$"+ATTENTION_EQUATIONS[op.name]+"$",
                             ha="center",va="center",fontsize=10)
                    top.plot([a,op.ready],[row-.34,row+.34],linestyle="none",
                             marker="|",color="#bb741a",markersize=8)
        else:
            last_label_end = -float("inf")
            for op in operations:
                if op.unit != lane:
                    continue
                intervals = op.runs(lane)
                for a, b, _ in intervals:
                    top.broken_barh([(a, b-a)], (row-.26, .52), facecolors="#9bc9e2")
                if intervals:
                    a, b = intervals[0][0], intervals[-1][1]
                    equation = ATTENTION_EQUATIONS.get(op.name, op.name)
                    # 相邻短操作分别放在上下侧，避免两个 SUB 的公式相互覆盖。
                    middle = (a+b)/2
                    below = middle-last_label_end < max(op.ready for op in operations)*.065
                    top.text(middle, row+.29 if below else row-.29, "$"+equation+"$",
                             fontsize=9, ha="center", va="top" if below else "bottom")
                    last_label_end = middle
    for join in data["joins"]:
        row = lanes.index("ADD")
        top.plot(join["scheduled_start"], row, "v", color="#b76426", markersize=6)
    top.set_yticks(range(len(lanes)), lanes, fontsize=11)
    top.set_ylim(len(lanes)-.5, -.9)
    top.set_xlim(0, max(op.ready for op in operations))
    label_dma(top, lanes.index("DMA"), max(op.ready for op in operations))
    top.set_title("单 tile：Tensor 行仅计算（QK / PV），其余主行含 I/O；颜色越深同时活动资源越多",
                  loc="left", fontsize=15, pad=15)
    top.set_xlabel("实际排程时间 / cycle（计算块上方为公式；橙色标记为 ADD 汇合）", fontsize=11)
    for row, engine in enumerate(("DMA", "Tensor", "Vector")):
        count = np.sum([steady[r].astype(int) for r in ENGINE_RESOURCES[engine]], axis=0)
        blocks(bottom, count, row)
        # 精确并行数轮廓，区别连续满忙与内部并行。
        bottom.plot(np.arange(len(count)), row+.32-.1*count/max(1, count.max()),
                    color="#143d59", linewidth=.7)
    bottom.set_xlim(0,report["window_cycles"])
    bw_labels = [draw_bandwidth_lane(bottom,report,r,row)
                 for row,r in enumerate(("L1_R","L1_W","L1_R_BUS","L1_W_BUS"),3)]
    capacity_labels = draw_capacity_rows(bottom,report,7)
    bottom.set_yticks(range(7+len(capacity_labels)), ["DMA", "Tensor", "Vector"]+bw_labels+capacity_labels, fontsize=8)
    bottom.set_ylim(6.5+len(capacity_labels),-.7); bottom.set_xlim(0,report["window_cycles"])
    label_dma(bottom, 0, report["window_cycles"], periodic=True)
    bottom.set_xticks(np.arange(config["plot"]["periods"]+1)*report["ii_cycles"])
    bottom.set_title("满流水：三引擎均为含 I/O 并集，包含跨 tile 重叠；竖线间隔为 II", loc="left", fontsize=15)
    bottom.set_xlabel("稳态时间 / cycle", fontsize=11)
    for ax in (top,bottom):
        ax.grid(axis="x", alpha=.25); ax.set_axisbelow(True)
    lines = ["时间统计（单位 cycle；W 为 resource-cycle）",
             "W = 各资源占用累计；B = 占用区间并集",
             "C = 首次启动至最终结果就绪，包含等待", "",
             "单 tile：计算口径（DMA 按读写接口）"]
    for engine in ("DMA", "Tensor", "Vector"):
        item = data["single_tile"][engine]; p = item["primary"]
        lines += [f"{engine}: W={p['work_resource_cycles']}   B={p['busy_cycles']}   C={item['completion_span_cycles']}",
                  f"    忙碌期间平均并行度 {p['average_parallelism']:.2f}；峰值 {p['peak_parallelism']}"]
    lines += ["", "稳态窗口：计算 W / B；含 I/O 并集"]
    for engine in ("DMA", "Tensor", "Vector"):
        item = data["steady_window"][engine]; p = item["primary"]
        lines += [f"{engine}: {p['work_resource_cycles']} / {p['busy_cycles']}；{item['including_io']['busy_cycles']}"]
        lines += [f"    计算平均并行度 {p['average_parallelism']:.2f}；峰值 {p['peak_parallelism']}"]
    lines += ["",f"L1 固定预留 {report['l1_reserved_bytes']/1024:g} KiB",
              f"存活峰值 {report['l1_capacity']['live_peak_bytes']/1024:g} KiB；上限 {config['l1']['size_kib']} KiB",
              "各数据独立槽池；Q/A 常驻，m/ℓ 在寄存器",
              "", "实际汇合（按输入可消费时刻取 max）"]
    for join in data["joins"]:
        values = ", ".join(str(v) for v in join["input_ready"].values())
        lines += [f"{join['operation']}: max({values}) = {join['dependency_gate']}",
                  f"    实际启动 {join['scheduled_start']}；排程等待 {join['scheduled_wait_cycles']}"]
    lines += ["", "同类 SUB / EXP / MUL / ADD 各一份，必须分时。",
              "REDSUM 与 Tensor 可重叠，仍需满足数据和带宽约束。",
              "排程等待不单独归因于某资源；还包含周期相位约束。",
              "分支同起点且资源无冲突时，历时才等于 max(T1,T2)。"]
    notes.text(0,1,"\n".join(lines), va="top", fontsize=12, linespacing=1.5)
    fig.suptitle(f"FlashAttention · DMA / Tensor / Vector 实际执行时间 | II={report['ii_cycles']} cycle/tile",
                 fontsize=23, y=.97)
    fig.text(.055,.025,"mask 从 DDR 加载，A 在 L1 原地更新；容量数字为各槽池当前存活 KiB，固定预留另列。FP32 score scratch 另行核算，不属于 L1。稳态不含初始化与最终归一化。",
             fontsize=12)
    save_parameterized_figure(fig, destination, config); plt.close(fig)


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
    tensor_throttled = any(op.unit == "TENSOR" and len(op.issue_offsets)>1
                           and op.issue_offsets[1]-op.issue_offsets[0]>1 for op in operations)
    lanes = ["DMA_RD", "DMA_WR"] + (["TENSOR_LD"] if tensor_throttled else [])
    lanes += ["TENSOR"] + (["TENSOR_ST"] if tensor_throttled else []) + list(VECTOR_UNITS)
    throttled = any(op.issue_offsets and len(op.issue_offsets) > 1
                    and op.issue_offsets[1] - op.issue_offsets[0] > 1
                    for op in operations if op.bytes_read or op.bytes_written)
    show_io = plot["show_hidden_vector_io"] or throttled
    if show_io:
        lanes += ["VECTOR_LD", "VECTOR_ST"]
    lanes += ["Tensor union", "Vector union", "DMA union", "L1_R", "L1_W", "L1_R_BUS", "L1_W_BUS"]
    capacity_start = len(lanes)
    lanes += [f"{name} (KiB)" for name in report["l1_capacity"]["intervals"]]
    figsize = (plot["width_px"] / plot["dpi"], plot["height_px"] / plot["dpi"])
    fig, ax = plt.subplots(figsize=figsize, dpi=plot["dpi"])
    fig.subplots_adjust(left=0.08, right=0.98, top=0.835, bottom=0.11)
    ax.set_xlim(0, window)
    ax.set_ylim(len(lanes)-0.5, -0.8)
    renderer = fig.canvas.get_renderer()
    # 同号表示在算法依赖图中可以并行，物理资源冲突由下方排程体现。
    stages = [
        (r"X_j=QK_j^{\mathsf{T}}",),
        (r"H_j=X_jc",),
        (r"S_j=H_j+\mathrm{mask}_j",),
        (r"r_j=\mathrm{rowmax}(S_j)",),
        (r"m_j=\max(m_{j-1},r_j)",),
        (r"\Delta m_j=m_{j-1}-m_j", r"U_j=S_j-m_j"),
        (r"\alpha_j=\exp(\Delta m_j)", r"P_j=\exp(U_j)"),
        (r"z_j=\mathrm{rowsum}(P_j)", r"Y_j=P_jV_j",
         r"\widetilde{\ell}_j=\alpha_j\ell_{j-1}", r"\widetilde{A}_j=\alpha_j A_{j-1}"),
        (r"\ell_j=\widetilde{\ell}_j+z_j", r"A_j=\widetilde{A}_j+Y_j"),
    ]
    steps = "     ".join(
        f"[{number}]  $" + equation + "$"
        for number, stage in enumerate(stages, start=1) for equation in stage
    )
    caption = fig.text(0.08, 0.852, "单 tile 计算步骤（同号可并行）    " + steps,
                       va="bottom", fontsize=15, color="#19324a")
    available_width = ax.bbox.width
    caption_width = caption.get_window_extent(renderer).width
    if caption_width > available_width:
        caption.set_fontsize(caption.get_fontsize() * available_width / caption_width)
    outside_boxes = []
    bandwidth_labels = {}
    colors = plt.get_cmap("Pastel1").colors
    for row, resource in enumerate(lanes):
        if row >= capacity_start:
            continue
        ax.axhspan(row - 0.45, row + 0.45, color="#f4f7fa", zorder=0)
        if resource in ("L1_R", "L1_W", "L1_R_BUS", "L1_W_BUS"):
            bandwidth_labels[resource] = draw_bandwidth_lane(ax,report,resource,row)
            continue
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
                    equation = ATTENTION_EQUATIONS.get(op.name) if resource == op.unit else None
                    if equation:
                        middle = (a + b) / 2
                        math_label = "$" + equation + "$"
                        label = ax.text(middle, row, math_label,
                                        ha="center", va="center", fontsize=9)
                        bounds = label.get_window_extent(renderer)
                        width = (b-a) * ax.bbox.width / window
                        height = 0.6 * ax.bbox.height / (len(lanes) + 0.3)
                        if bounds.width + 6 > width or bounds.height + 3 > height:
                            label.set_fontsize(7)
                            half = (label.get_window_extent(renderer).width / 2 + 3) * window / ax.bbox.width
                            x = min(max(middle, half), window - half)
                            for y, align in ((row-0.32, "bottom"), (row+0.32, "top")):
                                label.set_position((x, y))
                                label.set_va(align)
                                bounds = label.get_window_extent(renderer)
                                if not any(bounds.overlaps(previous) for previous in outside_boxes):
                                    break
                            outside_boxes.append(bounds)
                    elif b - a > ii * 0.035:
                        # 搬运泳道的方向由行名给出，块内仅标数据对象。
                        obj = op.name.split("_")[0]
                        if resource in ("DMA_RD", "DMA_WR"):
                            obj = ("load " if resource == "DMA_RD" else "store ") + obj
                        if resource == "VECTOR_LD":
                            obj = {"ADD_A":"A,Y","CVT_Y":"Y","SCALE_A":"A","MASK":"mask"}.get(op.name,"S")
                        elif resource == "VECTOR_ST":
                            obj = "A" if op.name in ("SCALE_A","ADD_A") else "P"
                        elif resource == "TENSOR_LD":
                            obj = "Q,K" if op.name == "QK" else "P,V"
                        elif resource == "TENSOR_ST":
                            obj = "S" if op.name == "QK" else "Y"
                        ax.text((a+b)/2, row, obj, ha="center", va="center", fontsize=8)
            if resource == op.unit and op.issue_offsets:
                for tile in range(-math.ceil(op.ready / ii), plot["periods"] + 1):
                    issue_end = op.start + op.issue_offsets[-1] + 1 + tile*ii
                    result = op.ready + tile*ii
                    if 0 <= issue_end < result <= window:
                        ax.plot([issue_end, result], [row+0.34, row+0.34], "--",
                                color="#64758b", linewidth=0.8)
            if resource == op.unit and op.conversion and op.conversion["extra_cycles"]:
                path = op.conversion["path"]
                end = op.start + op.issue_offsets[0] if path == "VECTOR_LD" else op.ready
                begin = end - op.conversion["extra_cycles"]
                for tile in range(math.floor(-end / ii) + 1, math.ceil((window - begin) / ii)):
                    a, b = max(0, begin + tile * ii), min(window, end + tile * ii)
                    ax.plot([a, b], [row-0.35, row-0.35], color="#9b4c80", linewidth=2)
            if resource == op.unit and ((not show_io and op.unit in VECTOR_UNITS) or op.unit == "TENSOR"):
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
    draw_capacity_rows(ax, report, capacity_start)
    ax.set_ylim(len(lanes)-0.5, -0.8)
    ax.set_yticks(range(len(lanes)), [bandwidth_labels.get(lane, lane) for lane in lanes], fontsize=9)
    ax.set_xticks(np.arange(plot["periods"]+1)*ii)
    ax.grid(axis="x", color="#cad4df", linewidth=0.7)
    ax.set_xlabel(f"Time (cycles); 1 cycle = {1/config['clock_ghz']:g} ns", fontsize=11)
    title = f"FlashAttention steady state | II = {ii} cycles/tile | lower bound = {report['resource_lower_bound_cycles']} cycles"
    fig.suptitle(title, fontsize=19, y=0.97)
    fig.text(0.08,0.925,
             f"L1 reserved {report['l1_reserved_bytes']/1024:g} KiB | one unit per type | "
             f"Vector I/O {'shown' if show_io else 'overlapped'} | Tensor streaming I/O: "
             f"read latency {config['tensor']['input_latency_cycles']} c, write latency {config['tensor']['output_latency_cycles']} c"
             " | orange = first input / final output",
             fontsize=11)
    if "numeric" in config:
        numeric = config["numeric"]
        fig.text(0.08, 0.894,
                 f"Tensor {numeric['tensor_input_dtype'].upper()} -> {numeric['tensor_output_dtype'].upper()} | "
                 f"Vector {numeric['vector_dtype'].upper()} | {numeric['conversion']['mode']} conversion"
                 + (" | purple = conversion latency" if numeric["conversion"]["mode"] == "inline" else ""),
                 fontsize=11)
    totals = " | ".join(f"{name} union {data['window_cycles']} c ({data['window_ns']:.1f} ns)"
                       for name,data in report["engines"].items())
    fig.text(0.08, 0.025, totals +
             f"\nmask 从DDR加载；A驻留L1。L1固定预留 {report['l1_reserved_bytes']/1024:g} KiB，存活峰值 {report['l1_capacity']['live_peak_bytes']/1024:g} KiB；有效带宽 B/cycle，实际占用为L1对应读/写上限的百分比，容量 KiB。"
             "\n同号表示数据依赖允许并行。指数两支共用 EXP，实际发射时间以泳道为准。虚线表示结果延迟。", fontsize=10)
    # 明确指定像素尺寸，避免 bbox_inches='tight' 改变请求的全尺寸输出。
    save_parameterized_figure(fig, destination, config)
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
                "DMA_RD", "DMA_WR", "L1_R", "L1_W", "L1_R_BUS", "L1_W_BUS"]
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
        labels = units if units is compute_units else [
            f"{unit}\n上限 {report['units'][unit]['capacity_per_cycle']:g} B/c = "
            f"{report['units'][unit]['capacity_per_cycle']*frequency:g} GB/s"
            for unit in units]
        ax.set_yticks(positions, labels, fontsize=12 if units is io_units else 16)
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
    save_parameterized_figure(fig, units_path, config)
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
    save_parameterized_figure(fig, engines_path, config)
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
    overview_path = args.output_dir / "execution_overview.png"
    draw_execution_overview(config, operations, report, overview_path)
    statistics_paths["execution_overview"] = overview_path
    statistics_paths.update(draw_bandwidth_details(config,report,args.output_dir))
    report["png"] = {"path": str(png.resolve()), "width_px": config["plot"]["width_px"],
                     "height_px": config["plot"]["height_px"]}
    from PIL import Image
    with Image.open(png) as im:
        report['png'].update(width_px=im.width, height_px=im.height)
    report["statistics_png"] = {}
    for name,path in statistics_paths.items():
        with Image.open(path) as im:
            report["statistics_png"][name] = dict(path=str(path.resolve()),width_px=im.width,height_px=im.height)
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
