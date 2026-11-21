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


legacy_main = main

# 新后端不调用旧 solve；以上保留复制时的旧实现作为可比较基线。
from collections import Counter, defaultdict, deque


DETAIL_DEFAULTS = {
    "group_size": 4, "register": {"elements": 64, "dtype": "fp32",
    "capacity_registers": None, "read_ports": None, "write_ports": None},
    "forwarding": True, "writeback_delay_cycles": 1,
    "queue_depth": 8, "dispatch_width": 1, "branch_latency_cycles": 1,
    "tiles": 6, "max_cycles": 200000, "fusion_regions": [
        ["SCALE", "MASK", "ROWMAX", "MAX", "SUB_S", "DELTA", "EXP_P",
         "ALPHA", "ROWSUM"], ["SCALE_L", "ADD_L"], ["SCALE_A", "ADD_A"]]
}

MACRO_DEFAULTS = {"frontend_dispatch_width": 1, "queues": {
    engine: {"entries": 4, "policy": "fifo"} for engine in ("Tensor", "Vector", "DMA")}}


def vector_phase(op):
    if op in ("SCALE_A", "ADD_A"): return "A_UPDATE"
    if op in ("SCALE_L", "ADD_L"): return "DENOMINATOR"
    return "P_PRODUCER"


class DetailProgram:
    """显式指令流；values 是版本化值，物理分配由运行时统一完成。"""
    def __init__(self, config):
        self.config = config
        self.detail = config["vector_detail"]
        self.instructions = []
        self.values = {}
        self.serial = 0
        self.spills = {}
        self.l1_sizes = {}

    def value(self, count=1, op="", tile=0, elements=None):
        name = f"x{len(self.values)}"
        g = self.detail["group_size"]
        allocation = g
        self.values[name] = dict(count=count, allocation=allocation, op=op, tile=tile,
                                 elements=elements or [self.detail["register"]["elements"]]*count)
        return [(name, i) for i in range(count)]

    def emit(self, unit, label, sources=(), destinations=(), deps=(), tile=0, **kw):
        i = len(self.instructions)
        owner=kw.pop("owner_op", {"load score":"SCALE", "load mask":"MASK",
                    "load A":"SCALE_A", "load Y":"ADD_A", "store A":"ADD_A",
                    "store P":"EXP_P"}.get(label,label))
        shape=self.config["shape"]
        sizes={"K":shape["kv_cols"]*shape["head_dim_qk"]*2,
               "V":shape["kv_cols"]*shape["head_dim_v"]*2,
               "mask":shape["query_rows"]*shape["kv_cols"]*4,
               "S":shape["query_rows"]*shape["kv_cols"]*2,
               "P":shape["query_rows"]*shape["kv_cols"]*2,
               "Y":shape["query_rows"]*shape["head_dim_v"]*2}
        produces={"load K":"K","load V":"V","QK":"S","store P":"P","PV":"Y"}
        if unit=="DMA_RD" and label=="load mask": produces[label]="mask"
        consumes={"QK":["K"],"PV":["P","V"],"load score":["S"],"load Y":["Y"]}
        if unit=="LD" and label=="load mask": consumes[label]=["mask"]
        alloc=[]; uses=[]
        if label in produces:
            name=f"{tile}:{produces[label]}"; self.l1_sizes[name]=sizes[produces[label]]; alloc=[name]
        uses=[f"{tile}:{name}" for name in consumes.get(label,[])]
        if label.startswith("spill "):
            name=f"{tile}:spill:{i}"; self.l1_sizes[name]=sum(self.values[r[0]]["elements"][r[1]]*4 for src in sources for r in src); alloc=[name]
        if label.startswith("reload "):
            uses=list(self.instructions[list(deps)[0]]["l1_allocate"])
        self.instructions.append(dict(id=i, unit=unit, label=label, sources=list(sources),
            destinations=list(destinations), deps=list(deps), tile=tile,owner_op=owner,
            row=getattr(self,'current_row',-1),l1_allocate=alloc,l1_use=uses, **kw))
        return i

    def region(self, op):
        region=next((i for i,r in enumerate(self.detail["fusion_regions"]) if op in r), op)
        return vector_phase(op),region

    def materialize(self, refs, consumer, tile):
        if not refs:
            return refs
        producer = self.values[refs[0][0]]["op"]
        if (producer in ("LOAD", "STATE") or self.values[refs[0][0]].get("small_state")
                or self.region(producer) == self.region(consumer)):
            return refs
        key = tuple(refs)
        if key not in self.spills:
            token = self.emit("ST", "spill "+producer, [[r] for r in refs], tile=tile, width=4,owner_op=producer)
            self.spills[key] = token
        out = self.value(len(refs), "LOAD", tile, [self.values[n]["elements"][k] for n,k in refs])
        self.emit("LD", "reload "+producer, destinations=out, deps=[self.spills[key]], tile=tile, width=4,owner_op=consumer)
        return out

    def arithmetic(self, op, unit, inputs, tile, scalar=False):
        inputs = [self.materialize(r, op, tile) for r in inputs]
        count = len(inputs[0])
        sizes = [1]*count if scalar else [self.values[n]["elements"][k] for n,k in inputs[0]]
        out = self.value(count, op, tile, sizes)
        if scalar:
            self.values[out[0][0]]["allocation"] = 1
            self.values[out[0][0]]["small_state"] = True
        self.emit(unit, op, [list(x) for x in zip(*inputs)], out, tile=tile,
                  reduction=unit in ("REDMAX", "REDSUM"))
        return out


def detail_program(config):
    p = DetailProgram(config)
    d = p.detail
    m,n,dk,dv = [config["shape"][k] for k in ("query_rows","kv_cols","head_dim_qk","head_dim_v")]
    e,g = d["register"]["elements"],d["group_size"]
    # 当前编译器按行构造真正的水平归约；长行分段合并，不跨行做归约。
    old_m,old_l = {},{}
    for row in range(m):
        for states in (old_m,old_l):
            ref=p.value(1,"STATE",-1,[1])[0]
            p.values[ref[0]].update(allocation=1,initial=True)
            states[row]=ref
    a_done = {}
    ops,_,_,_ = make_model(config)
    macro = {o.name:o for o in ops}
    for tile in range(d["tiles"]):
        p.current_row=-1
        k = p.emit("DMA_RD", "load K", tile=tile, macro=macro["K_DMA"])
        v = p.emit("DMA_RD", "load V", tile=tile, macro=macro["V_DMA"])
        mask = p.emit("DMA_RD", "load mask", tile=tile, macro=macro["MASK_DMA"])
        qk = p.emit("TENSOR", "QK", deps=[k], tile=tile, macro=macro["QK"])
        p_stores, alphas, next_l = [],{},{}
        for row in range(m):
            p.current_row=row
            scores = []
            for col in range(0,n,e*g):
                sizes = [min(e,n-x) for x in range(col,min(n,col+e*g),e)]
                x = p.value(len(sizes), "LOAD",tile,sizes)
                z = p.value(len(sizes), "LOAD",tile,sizes)
                p.emit("LD", "load score", destinations=x, deps=[qk],tile=tile,width=2)
                p.emit("LD", "load mask", destinations=z, deps=[mask],tile=tile,width=4)
                h = p.arithmetic("SCALE","MUL",[x],tile)
                scores.extend(p.arithmetic("MASK","ADD",[h,z],tile))
            reductions = []
            for ref in scores:
                reductions.extend(p.arithmetic("ROWMAX","REDMAX",[[ref]],tile,True))
            r = [reductions[0]]
            for part in reductions[1:]:
                r = p.arithmetic("ROWMAX","MAX",[r,[part]],tile,True)
            mj = p.arithmetic("MAX","MAX",[r]+([[old_m[row]]] if row in old_m else []),tile,True)
            delta = p.arithmetic("DELTA","SUB",[[old_m[row]],mj],tile,True)
            alpha = p.arithmetic("ALPHA","EXP",[delta],tile,True)
            old_m[row],alphas[row] = mj[0],alpha[0]
            sums=[]
            for offset in range(0,len(scores),g):
                ss=scores[offset:offset+g]
                u=p.arithmetic("SUB_S","SUB",[ss,mj*len(ss)],tile)
                prob=p.arithmetic("EXP_P","EXP",[u],tile)
                # 分母使用宽精度 P；写 Tensor 的副本才窄化。
                for ref in prob:
                    sums.extend(p.arithmetic("ROWSUM","REDSUM",[[ref]],tile,True))
                p_stores.append(p.emit("ST","store P",[[r] for r in prob],tile=tile,width=2))
            z=[sums[0]]
            for part in sums[1:]:
                z=p.arithmetic("ROWSUM","ADD",[z,[part]],tile,True)
            sl=p.arithmetic("SCALE_L","MUL",[alpha]+([[old_l[row]]] if row in old_l else []),tile,True)
            next_l[row]=p.arithmetic("ADD_L","ADD",[sl,z],tile,True)[0]
            p.emit("BR","row loop",tile=tile)
        old_l=next_l
        pv=p.emit("TENSOR","PV",deps=p_stores+[v],tile=tile,macro=macro["PV"])
        for row in range(m):
            p.current_row=row
            stores=[]
            for col in range(0,dv,e*g):
                sizes=[min(e,dv-x) for x in range(col,min(dv,col+e*g),e)]
                a=p.value(len(sizes),"LOAD",tile,sizes)
                y=p.value(len(sizes),"LOAD",tile,sizes)
                p.emit("LD","load A",destinations=a,deps=a_done.get(row,[]),tile=tile,width=4)
                p.emit("LD","load Y",destinations=y,deps=[pv],tile=tile,width=2)
                scaled=p.arithmetic("SCALE_A","MUL",[a,[alphas[row]]*len(a)],tile)
                out=p.arithmetic("ADD_A","ADD",[scaled,y],tile)
                stores.append(p.emit("ST","store A",[[r] for r in out],tile=tile,width=4))
            a_done[row]=stores
            p.emit("BR","A row loop",tile=tile)
        p.emit("BR","tile loop",tile=tile)
    for ref in list(old_m.values())+list(old_l.values()):
        p.values[ref[0]]["retain_final"] = True
    return p


class DetailSimulator:
    def __init__(self, program):
        self.p=program; self.c=program.config; self.d=program.detail
        self.used=defaultdict(Counter); self.events=[]; self.done={}; self.ready={}; self.wb={}
        self.queues=defaultdict(deque); self.cursor={}; self.collected={}
        self.references=Counter(r for i in program.instructions for s in i["sources"] for r in s)
        self.alloc={}; self.pending=defaultdict(int); self.allocated_series=[]
        self.pc=0; self.blocked=None; self.stalls=Counter(); self.dispatch={}
        self.ref_reads=Counter(); self.forwarded=Counter()
        sh=self.c["shape"]
        self.l1_resident=sh["query_rows"]*(sh["head_dim_qk"]*2+sh["head_dim_v"]*4)
        self.l1_live={}; self.l1_history={}; self.l1_finished={}
        self.l1_refs=Counter(x for i in program.instructions for x in i["l1_use"])
        self.l1_producers=Counter(x for i in program.instructions for x in i["l1_allocate"])
        _,self.cap,_,_=make_model(self.c)
        self.cap.update({"RF_R":self.d["register"]["read_ports"],
                         "RF_W":self.d["register"]["write_ports"]})
        for unit in VECTOR_UNITS: self.cap[unit]=1
        self.cap.update(LD=self.c["vector"]["read_ports"],ST=self.c["vector"]["write_ports"],BR=1)
        for name,spec in self.p.values.items():
            if spec.get("initial"):
                if not self.allocate(name): raise ValueError("寄存器容量不足以保存初始 m/l 行状态")
                self.ready[name,0]=0; self.wb[name,0]=0

    def can(self, requests):
        return all(self.cap.get(r) is None or self.used[r][t]+v<=self.cap[r]+1e-8
                   for (r,t),v in requests.items())

    def reserve(self, requests):
        for (r,t),v in requests.items(): self.used[r][t]+=v

    def l1_accept(self,i,t):
        new=[x for x in i["l1_allocate"] if x not in self.l1_live]
        required=self.l1_resident+sum(self.p.l1_sizes[x] for x in self.l1_live)+sum(self.p.l1_sizes[x] for x in new)
        if required>self.c["l1"]["size_kib"]*1024:
            self.stalls["L1_capacity"]+=1; return False
        for x in new:
            self.l1_live[x]=t; self.l1_history[x]=dict(start=t,end=None,bytes=self.p.l1_sizes[x])
        return True

    def complete(self,i,end):
        self.done[i["id"]]=end
        for x in i["l1_allocate"]: self.l1_producers[x]-=1
        for x in i["l1_use"]:
            self.l1_refs[x]-=1; self.l1_finished[x]=max(end,self.l1_finished.get(x,0))

    def allocate(self,name):
        if name in self.alloc: return True
        size=self.p.values[name]["allocation"]
        limit=self.d["register"]["capacity_registers"]
        occupied={r for start,n in self.alloc.values() for r in range(start,start+n)}
        base=0
        while any(r in occupied for r in range(base,base+size)): base+=size
        if limit is not None and base+size>limit: return False
        self.alloc[name]=(base,size)
        return True

    def release(self,t):
        for x in list(self.l1_live):
            if self.l1_refs[x]==0 and self.l1_producers[x]==0 and self.l1_finished.get(x,float("inf"))<=t:
                self.l1_history[x]["end"]=t; del self.l1_live[x]
        for name in list(self.alloc):
            if self.p.values[name].get("retain_final"): continue
            refs=[(name,k) for k in range(self.p.values[name]["count"])]
            if all(r in self.wb and self.wb[r]<=t and self.references[r]==0 for r in refs):
                del self.alloc[name]

    def step_instruction(self,i,t):
        if any(self.done.get(x,float("inf"))>t for x in i["deps"]): return False
        idx=self.cursor.get(i["id"],0)
        if "macro" in i:
            op=i["macro"]; req=Counter()
            for resource,profile in op.profiles.items():
                for dt,v in enumerate(profile):
                    if v: req[resource,t+dt]+=float(v)
            if not self.can(req) or not self.l1_accept(i,t): return False
            self.reserve(req); end=t+op.ready_offset
            self.events.append(dict(instruction=i["id"],tile=i["tile"],unit=i["unit"],label=i["label"],start=t,end=end,result=end,writeback=None,register=None))
            self.complete(i,end)
            return True
        if i["unit"]=="BR":
            end=t+self.d["branch_latency_cycles"]
            self.done[i["id"]]=end
            self.events.append(dict(instruction=i["id"],tile=i["tile"],unit="BR",label=i["label"],start=t,end=end,result=end,writeback=None,register=None))
            return True
        dest=i["destinations"][idx] if i["destinations"] else None
        src=i["sources"][idx] if i["sources"] else []
        if dest and not self.allocate(dest[0]): self.stalls["register_capacity"]+=1; return False
        captured=self.collected.setdefault((i["id"],idx),{})
        for operand,r in enumerate(src):
            if operand in captured: continue
            if self.ready.get(r,float("inf"))>t: continue
            fwd=self.d["forwarding"] and t<self.wb[r]
            if not fwd and self.wb[r]>t: continue
            if not fwd and not self.can({("RF_R",t):1}): self.stalls["RF_read_ports"]+=1; continue
            if not fwd: self.reserve({("RF_R",t):1})
            else: self.forwarded[t]+=1
            self.ref_reads[t]+=1
            captured[operand]=dict(ref=r,cycle=t,path="forward" if fwd else "RF")
            self.references[r]-=1
        if len(captured)!=len(src): return False
        unit=i["unit"]; req=Counter()
        ref=dest or src[0]
        elements=self.p.values[ref[0]]["elements"][ref[1]]
        if i.get("reduction"):
            elements=self.p.values[src[0][0]]["elements"][src[0][1]]
        if unit in ("LD","ST"):
            direction="read" if unit=="LD" else "write"
            port=self.c["vector"][direction+"_port_gbps"]/self.c["clock_ghz"]
            size=elements*i["width"]; beats=math.ceil(size/port)
            extra=self.c["numeric"]["conversion"]["vector_load_extra_cycles" if unit=="LD" else "vector_store_extra_cycles"] if i["width"]!=4 else 0
            latency=self.c["vector"]["input_latency_cycles" if unit=="LD" else "output_latency_cycles"]
            offset=extra+latency-1 if unit=="ST" else 0
            for b in range(beats):
                req[unit,t+offset+b]+=1
                req["L1_R_BUS" if unit=="LD" else "L1_W_BUS",t+offset+b]+=port
                req["L1_R" if unit=="LD" else "L1_W",t+offset+b]+=min(port,size-b*port)
            result=t+beats+extra+latency-1
            issue_end=t+beats+offset
        else:
            spec=self.c["vector"]["units"][unit]
            batches=math.ceil(elements/spec["parallelism"])
            for b in range(batches): req[unit,t+b]+=1
            result=t+batches-1+spec["latency_cycles"]
            issue_end=t+batches
        writeback=result+self.d["writeback_delay_cycles"] if dest else None
        if dest: req["RF_W",writeback]+=1
        if not self.can(req): self.stalls["unit_bus_or_writeback"]+=1; return False
        if not self.l1_accept(i,t): return False
        self.reserve(req)
        if dest: self.ready[dest]=result; self.wb[dest]=writeback
        self.events.append(dict(instruction=i["id"],tile=i["tile"],unit=unit,label=i["label"],start=t,end=issue_end,
            result=result,writeback=writeback,register=(self.alloc[dest[0]][0]+dest[1] if dest else None),
            destination=dest,sources=list(captured.values()),elements=elements))
        self.cursor[i["id"]]=idx+1
        total=max(len(i["sources"]),len(i["destinations"]))
        if idx+1==total:
            self.complete(i,max(e["result"] for e in self.events if e["instruction"]==i["id"]))
            return True
        return False

    def run(self):
        instructions=self.p.instructions
        for t in range(self.d["max_cycles"]):
            self.release(t)
            if self.blocked is not None and self.done.get(self.blocked,float("inf"))<=t: self.blocked=None
            for _ in range(self.d["dispatch_width"]):
                if self.pc>=len(instructions) or self.blocked is not None: break
                i=instructions[self.pc]; q=self.queues[i["unit"]]
                if len(q)>=self.d["queue_depth"]: self.stalls["fifo_full"]+=1; break
                q.append(i); self.dispatch[i["id"]]=t; self.pc+=1
                if i["unit"]=="BR": self.blocked=i["id"]; break
            # 单周期快照的结果由 future timestamp 发布；L>=1，不可同拍穿透算术链。
            for unit,q in list(self.queues.items()):
                slots=self.cap[unit] if unit in ("LD","ST") else 1
                for i in list(q)[:int(slots)]:
                    if self.step_instruction(i,t): q.remove(i)
            self.allocated_series.append(sum(n for _,n in self.alloc.values()))
            if self.pc==len(instructions) and not any(self.queues.values()):
                final=max([t]+list(self.done.values())+list(self.wb.values()))
                for tail in range(t+1,final+1):
                    self.release(tail); self.allocated_series.append(sum(n for _,n in self.alloc.values()))
                return self.report(final+1)
        waiting=[(u,q[0]["label"]) for u,q in self.queues.items() if q]
        raise ValueError(f"在 max_cycles 内未完成；可能容量不足或队头阻塞：{waiting}；停顿={dict(self.stalls)}")

    def report(self,end):
        for ev in self.events: ev['row']=self.p.instructions[ev['instruction']]['row']
        tile_ends=[max(self.done[i["id"]] for i in self.p.instructions if i["tile"]==k)
                   for k in range(self.d["tiles"])]
        intervals=np.diff(tile_ends).tolist()
        steady=len(intervals)>=3 and len(set(intervals[-3:]))==1
        groups={"Tensor":["TENSOR","TENSOR_LD","TENSOR_ST"],"DMA":["DMA_RD","DMA_WR"],"Vector":list(VECTOR_UNITS)+["LD","ST","BR"]}
        engines={name:len(set(t for r in resources for t,v in self.used[r].items() if v)) for name,resources in groups.items()}
        # BR 也计入 Vector 控制执行占用。
        for ev in self.events:
            if ev["unit"]=="BR":
                for t in range(ev["start"],ev["end"]): self.used["BR"][t]=1
        engines["Vector"]=len(set(t for r in groups["Vector"] for t,v in self.used[r].items() if v))
        l1_curve=[self.l1_resident]*end
        pools=defaultdict(lambda:np.zeros(end,dtype=int))
        for name,life in self.l1_history.items():
            b=life["end"] if life["end"] is not None else end
            typ=name.split(":")[1]
            pools[typ][life["start"]:b]+=life["bytes"]
        for curve in pools.values(): l1_curve=(np.asarray(l1_curve)+curve).tolist()
        pool_reserved=self.l1_resident+sum(int(v.max()) for v in pools.values())
        if pool_reserved>self.c["l1"]["size_kib"]*1024:
            raise ValueError(f"逐类独立 L1 槽池峰值之和 {pool_reserved} B 超过容量；当前排程不可接受")
        operand_curve=np.zeros(end,dtype=int)
        for ev in self.events:
            for src in ev.get("sources",[]):
                operand_curve[src["cycle"]:ev["end"]]+=1
        return dict(config=self.c,cycles=end,events=self.events,dispatch=self.dispatch,
            operand_latch_registers=operand_curve.tolist(),peak_operand_latch_registers=int(operand_curve.max()),
            unit_busy_cycles={r:sum(v>0 for v in counts.values()) for r,counts in self.used.items()},
            l1_live_bytes=l1_curve,l1_pool_reserved_bytes=pool_reserved,l1_lifetimes=self.l1_history,
            resources={r:dict(v) for r,v in self.used.items()},allocated_registers=self.allocated_series,
            source_operand_requests=dict(self.ref_reads),forwarding=dict(self.forwarded),
            peak_registers=max(self.allocated_series),peak_RF_reads=max(self.used["RF_R"].values(),default=0),
            peak_RF_writes=max(self.used["RF_W"].values(),default=0),stalls=dict(self.stalls),
            engine_busy_cycles=engines,tile_completion_cycles=tile_ends,tile_intervals=intervals,
            observed_steady_II=intervals[-1] if steady else None,
            scope=["时序仿真，不计算浮点数值；初始 m/l 为已有历史状态，计入 RF 分配和访问。",
                   "每行 m/l/alpha 独立占用一个 vreg，未做跨行标量打包。",
                   "每个行块内部 group chaining；BR 是已知循环回跳，无预测。",
                   "RF 操作数锁存每个活动指令的当前 vreg；forward 网络理想带宽。",
                   "Tensor 使用旧模型的流式微块资源曲线，Vector 事件与其共享 L1 带宽。",
                   "L1 按对象生命周期限制实时占用，交付前另检查逐类槽池峰值之和；释放保守取消费者完成。",
                   "逐行 strip mining；短行 group 尾寄存器仍预留，不自动跨行打包。",
                   "连续三个 tile 间隔相等仅标记观测到的稳定间隔，不构成周期状态或最优性证明。"])


def macro_program(config):
    """按 tile/融合区域形成宏 DAG；跨 Tensor 边界强制拆分，无宏内外部同步。"""
    p=detail_program(config)
    original=p.instructions
    retained=[i for i in original if i['unit']!='BR']
    remap={i['id']:k for k,i in enumerate(retained)}
    p.instructions=[]
    for old in retained:
        item=dict(old,id=remap[old['id']],deps=[remap[d] for d in old['deps']])
        p.instructions.append(item)
    groups={}
    for i in p.instructions:
        engine='Tensor' if i['unit']=='TENSOR' else 'DMA' if i['unit'].startswith('DMA_') else 'Vector'
        key=(i['tile'],engine,p.region(i['owner_op']) if engine=='Vector' else i['id'])
        groups.setdefault(key,[]).append(i['id'])
    temporary=[]; owner={}
    for key,body in groups.items():
        tile,engine,region=key
        operations=list(dict.fromkeys(p.instructions[n]['owner_op'] for n in body))
        label=p.instructions[body[0]]['label'] if engine!='Vector' else '+'.join(operations)
        if engine=='Vector' and 'EXP_P' in operations and 'ROWSUM' in operations: label='P_PRODUCER + ROWSUM'
        if engine=='Vector' and set(operations)=={'SCALE_L','ADD_L'}: label='DENOMINATOR'
        if engine=='Vector' and set(operations)=={'SCALE_A','ADD_A'}: label='A_UPDATE'
        mid=len(temporary)
        temporary.append(dict(id=mid,engine=engine,tile=tile,label=label,operations=operations,
                              body=body,dependencies=set()))
        owner.update({n:mid for n in body})
    producers={r:i['id'] for i in p.instructions for r in i['destinations']}
    for i in p.instructions:
        deps=set(i['deps'])
        for refs in i['sources']:
            for r in refs:
                if r not in producers: continue
                upstream=producers[r]; deps.add(upstream)
                if owner[upstream]!=owner[i['id']] and not p.values[r[0]].get('small_state'):
                    raise ValueError(f"跨宏大数据缺少 L1 交付：{r} → {i['label']}")
        temporary[owner[i['id']]]['dependencies'].update(owner[n] for n in deps if owner[n]!=owner[i['id']])
    # 稳定拓扑排序也检查非连续融合导致的依赖环；不通过隐式同步消除环。
    ordered=[]; remaining=set(range(len(temporary)))
    def priority(n):
        m=temporary[n]
        phase=vector_phase(m['operations'][0]) if m['engine']=='Vector' else ''
        rank=0 if m['engine']=='DMA' else 1 if m['label']=='QK' else 3 if m['label']=='PV' else {'P_PRODUCER':2,'DENOMINATOR':4,'A_UPDATE':5}[phase]
        return m['tile'],rank,min(m['body'])
    while remaining:
        ready=[n for n in remaining if not (temporary[n]['dependencies'] & remaining)]
        if not ready: raise ValueError('融合区域产生宏依赖环；请拆分融合区域（本阶段不支持宏内部外部同步）')
        n=min(ready,key=priority); ordered.append(n); remaining.remove(n)
    mapping={old:new for new,old in enumerate(ordered)}
    p.macros=[]; p.macro_of={}
    for old in ordered:
        m=dict(temporary[old]); m['id']=mapping[old]
        m['dependencies']=sorted(mapping[x] for x in m['dependencies'])
        if m['engine']=='Vector':
            body=[]
            for pos,n in enumerate(m['body']):
                body.append(n)
                row=p.instructions[n]['row']
                if pos+1==len(m['body']) or p.instructions[m['body'][pos+1]]['row']!=row:
                    p.current_row=row
                    body.append(p.emit('BR','macro row loop',tile=m['tile'],owner_op='BR'))
            m['body']=body
        p.macros.append(m)
        for n in m['body']: p.macro_of[n]=m['id']
    return p


class MacroSimulator(DetailSimulator):
    """单宏串行 engine + 有界待发宏队列；Vector 内部仍使用细粒度后端。"""
    ENGINES=('Tensor','Vector','DMA')

    def __init__(self,program):
        super().__init__(program)
        self.settings=self.c['macro_scheduler']
        self.mqueues={e:deque() for e in self.ENGINES}
        self.active={e:None for e in self.ENGINES}
        self.finished={}; self.macro_pc=0
        self.mrecords=[dict(id=m['id'],engine=m['engine'],tile=m['tile'],label=m['label'],
            operations=m['operations'],dependencies=m['dependencies'],instruction_count=len(m['body']),
            enqueue=None,start=None,end=None) for m in program.macros]
        self.queue_history={e:[] for e in self.ENGINES}
        self.queue_events=[]; self.wait_reasons={e:Counter() for e in self.ENGINES}
        self.l1_pool_high=Counter()
        self.last_activity=0
        self.rf_admission_cache={}

    def l1_accept(self,i,t):
        # 每类独立槽池保留已达到的容量高水位，避免不同类型峰值错峰被错误复用。
        new=[x for x in i['l1_allocate'] if x not in self.l1_live]
        current=Counter()
        for x in list(self.l1_live)+new: current[x.split(':')[1]]+=self.p.l1_sizes[x]
        proposed=self.l1_pool_high | current
        if self.l1_resident+sum(proposed.values())>self.c['l1']['size_kib']*1024:
            self.stalls['L1_pool_capacity']+=1; return False
        if not super().l1_accept(i,t): return False
        self.l1_pool_high=proposed
        return True

    def group_available(self,name):
        if name in self.alloc: return True
        limit=self.d['register']['capacity_registers']
        if limit is None: return True
        size=self.p.values[name]['allocation']
        used={r for b,n in self.alloc.values() for r in range(b,b+n)}
        return any(all(r not in used for r in range(b,b+size)) for b in range(0,limit-size+1,size))

    def macro_rf_admissible(self,m):
        """有界容量下先检查顺序执行工作集；保留其他宏的状态，不假设可抢占释放。

        此检查只用来排除不能启动的宏，实际流水的端口/写回/容量仍逐周期检查。
        不将指令级生命周期下界视为流水峰值或最优分配证明。
        """
        limit=self.d['register']['capacity_registers']
        if limit is None: return True
        if m['id'] in self.rf_admission_cache: return self.rf_admission_cache[m['id']]
        allocated=dict(self.alloc); refs=self.references.copy()
        for n in m['body']:
            i=self.p.instructions[n]
            for name in dict.fromkeys(r[0] for r in i['destinations']):
                if name in allocated: continue
                size=self.p.values[name]['allocation']
                occupied={r for b,k in allocated.values() for r in range(b,b+k)}
                base=next((b for b in range(0,limit-size+1,size) if all(r not in occupied for r in range(b,b+size))),None)
                if base is None:
                    self.rf_admission_cache[m['id']]=False; return False
                allocated[name]=(base,size)
            for operands in i['sources']:
                for r in operands: refs[r]-=1
            for name in list(allocated):
                spec=self.p.values[name]
                if not spec.get('retain_final') and all(refs[name,k]==0 for k in range(spec['count'])):
                    del allocated[name]
        self.rf_admission_cache[m['id']]=True
        return True

    def try_start(self,engine,t):
        if self.active[engine] is not None: return
        q=self.mqueues[engine]
        policy=self.settings['queues'][engine]['policy']
        candidates=list(q)[:1] if policy=='fifo' else list(q)
        for mid in candidates:
            m=self.p.macros[mid]
            if any(d not in self.finished for d in m['dependencies']):
                self.wait_reasons[engine]['dependency']+=1; continue
            body=[self.p.instructions[n] for n in m['body']]
            if engine=='Vector':
                if not self.macro_rf_admissible(m):
                    self.wait_reasons[engine]['RF_macro_working_set']+=1; continue
                first=next((i['destinations'][0][0] for i in body if i['destinations']),None)
                if first and not self.group_available(first):
                    self.wait_reasons[engine]['RF_capacity']+=1; continue
                outputs=list(dict.fromkeys(x for i in body for x in i['l1_allocate']))
                if not self.l1_accept({'l1_allocate':outputs},t):
                    self.wait_reasons[engine]['L1_capacity']+=1; continue
                if first: assert self.allocate(first)
                context=dict(mid=mid,pc=0,branch=None,drain=None)
            else:
                assert len(body)==1
                if not self.step_instruction(body[0],t):
                    self.wait_reasons[engine]['resource']+=1; continue
                self.dispatch[body[0]['id']]=t
                context=dict(mid=mid,drain=self.done[body[0]['id']])
            q.remove(mid)
            self.queue_events.append(dict(cycle=t,engine=engine,action='issue',macro=mid))
            self.active[engine]=context
            self.mrecords[mid]['start']=t
            self.last_activity=t
            break

    def vector_cycle(self,t):
        context=self.active['Vector']
        if context is None: return
        m=self.p.macros[context['mid']]
        if context['branch'] is not None and self.done.get(context['branch'],float('inf'))<=t:
            context['branch']=None
        for _ in range(self.d['dispatch_width']):
            if context['branch'] is not None or context['pc']==len(m['body']): break
            i=self.p.instructions[m['body'][context['pc']]]
            q=self.queues[i['unit']]
            if len(q)>=self.d['queue_depth']:
                self.stalls['vector_fifo_full']+=1; break
            q.append(i); self.dispatch[i['id']]=t; context['pc']+=1
            self.last_activity=t
            if i['unit']=='BR': context['branch']=i['id']; break
        for unit,q in self.queues.items():
            slots=int(self.cap[unit]) if unit in ('LD','ST') else 1
            for i in list(q)[:slots]:
                if self.step_instruction(i,t): q.remove(i)
        if context['pc']==len(m['body']) and not any(self.queues.values()) and context['drain'] is None:
            writes=[self.wb[r] for n in m['body'] for r in self.p.instructions[n]['destinations']]
            context['drain']=max([self.done[n] for n in m['body']]+writes)

    def run(self):
        for t in range(self.d['max_cycles']):
            self.release(t)
            for engine,ctx in self.active.items():
                if ctx is not None and ctx['drain'] is not None and ctx['drain']<=t:
                    mid=ctx['mid']; self.finished[mid]=t; self.mrecords[mid]['end']=t
                    self.active[engine]=None; self.last_activity=t
                    if engine=='Vector': self.rf_admission_cache.clear()
            # 已入队宏可以在本拍发射并立即释放 entry；本拍新入队也允许被空闲引擎接受。
            engines=self.ENGINES[t%3:]+self.ENGINES[:t%3]
            for engine in engines: self.try_start(engine,t)
            if self.macro_pc<len(self.p.macros):
                m=self.p.macros[self.macro_pc]; engine=m['engine']; q=self.mqueues[engine]
                if len(q)<self.settings['queues'][engine]['entries']:
                    q.append(m['id']); self.mrecords[m['id']]['enqueue']=t; self.macro_pc+=1
                    self.queue_events.append(dict(cycle=t,engine=engine,action='enqueue',macro=m['id']))
                    self.last_activity=t
                else: self.stalls['macro_frontend_queue_full']+=1
            for engine in engines: self.try_start(engine,t)
            previous=len(self.events)
            self.vector_cycle(t)
            if len(self.events)!=previous: self.last_activity=t
            self.allocated_series.append(sum(n for _,n in self.alloc.values()))
            for engine in self.ENGINES:
                self.queue_history[engine].append(dict(queued=len(self.mqueues[engine]),active=int(self.active[engine] is not None)))
            if len(self.finished)==len(self.p.macros):
                report=self.report(t+1)
                verify_macro_report(report,self.p)
                return report
            if t-self.last_activity>64 and max([0]+list(self.done.values())+list(self.wb.values()))<=t:
                raise ValueError(f"宏调度无进展：active={self.active}；queues={dict(self.mqueues)}；stalls={dict(self.stalls)}。当前策略下未找到可行排程，非硬件不可行性证明。")
        raise ValueError('宏调度超过 max_cycles；请检查容量、依赖和队列策略')

    def report(self,end):
        report=super().report(end)
        for event in self.events: event['macro_id']=self.p.macro_of[event['instruction']]
        tile_ends=[max(m['end'] for m in self.mrecords if m['tile']==tile) for tile in range(self.d['tiles'])]
        intervals=np.diff(tile_ends).tolist()
        engine_sets={e:set() for e in self.ENGINES}
        for m in self.mrecords: engine_sets[m['engine']].update(range(m['start'],m['end']))
        tensor={t for t,v in self.used['TENSOR'].items() if v}
        vector=set().union(*({t for t,v in self.used[u].items() if v} for u in VECTOR_UNITS))
        dma={t for t,v in self.used['DMA_RD'].items() if v}
        overlap=dict(Tensor_Vector_compute=len(tensor&vector),DMA_Vector_compute=len(dma&vector),DMA_Tensor_compute=len(dma&tensor),
                     Tensor_Vector_macro=len(engine_sets['Tensor']&engine_sets['Vector']))
        report.update(macros=self.mrecords,macro_queue_events=self.queue_events,macro_queue_history=self.queue_history,
            macro_wait_reasons={e:dict(v) for e,v in self.wait_reasons.items()},overlap_cycles=overlap,
            engine_macro_busy_cycles={e:len(s) for e,s in engine_sets.items()},
            tile_completion_cycles=tile_ends,tile_intervals=intervals,
            observed_steady_II=intervals[-1] if len(intervals)>=3 and len(set(intervals[-3:]))==1 else None,
            register_state_layout='每行一个 FP32 分母；当前每行状态各占一个物理 vreg，未紧凑打包',
            persistent_state_values={name:spec for name,spec in self.p.values.items() if spec.get('small_state') or spec.get('initial')},
            todo=['宏内外部同步与中途发布输出：未实现，须显式请求。','同一 Vector 引擎多个宏并发：未实现，须显式请求。'])
        period=next((k for k in range(1,len(intervals)//2+1) if intervals[-k:]==intervals[-2*k:-k]),None)
        report['observed_period_tiles']=period
        report['observed_mean_II']=sum(intervals[-period:])/period if period else None
        report['resource_capacities']=self.cap
        report['scope']+=['全局每 cycle 入队一条宏；队满反压前端；每引擎至多执行一个宏。',
                          '宏完成包括 RF 写回与 L1 输出可见；任何消费者均等待生产宏整体完成。',
                          '默认 EXP_P 与 ROWSUM 融合；仅 FP16 P 出 L1，z、m、alpha、ell 以 FP32 vreg 跨宏传递。']
        return report


def verify_macro_report(report,program):
    """基于输出事件独立核查生命周期、发射次序和依赖，不调用调度选择器。"""
    records=report['macros']; by_id={m['id']:m for m in records}
    enqueue=Counter(m['enqueue'] for m in records)
    if max(enqueue.values(),default=0)>1: raise AssertionError('宏前端超过一条/cycle')
    for engine in MacroSimulator.ENGINES:
        subset=sorted((m for m in records if m['engine']==engine),key=lambda m:m['start'])
        for a,b in zip(subset,subset[1:]):
            if a['end']>b['start']: raise AssertionError('同引擎宏重叠')
        cfg=report['config']['macro_scheduler']['queues'][engine]
        if cfg['policy']=='fifo' and [m['id'] for m in subset]!=[m['id'] for m in records if m['engine']==engine]:
            raise AssertionError('FIFO 发射乱序')
        pending=set()
        for event in report['macro_queue_events']:
            if event['engine']!=engine: continue
            mid=event['macro']
            if event['action']=='enqueue': pending.add(mid)
            else:
                if mid not in pending: raise AssertionError('宏未入队即发射')
                pending.remove(mid)
            if len(pending)>cfg['entries']: raise AssertionError('宏队列超深度')
    for m in records:
        if m['enqueue']>m['start'] or m['start']>=m['end']: raise AssertionError('宏时间非法')
        for dep in m['dependencies']:
            if by_id[dep]['end']>m['start']: raise AssertionError('宏依赖越过完成边界')
    for ev in report['events']:
        m=by_id[ev['macro_id']]
        if ev['start']<m['start'] or max(ev['end'],ev['result'],ev['writeback'] or 0)>m['end']:
            raise AssertionError('宏在内部指令/写回完成之前结束')


def draw_detail(report, output, page=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    configure_plot_fonts()
    events=report["events"]; end=report["cycles"]
    # 细图采用末 tile 的首 180 cycle，保持单元格可辨；完整轨迹另图呈现。
    tile=max(0,max(e["tile"] for e in events)-1)
    origin=min(e["start"] for e in events if e["tile"]==tile and e["unit"]=="LD")
    period=report.get('observed_period_tiles') or 1
    window=sum(report['tile_intervals'][-period:]) if report['tile_intervals'] else end-origin
    window=min(window,end-origin)
    begin=origin+page*180
    stop=min(end,origin+window,begin+180)
    visible=[e for e in events if e["start"]<stop and e["end"]>begin and e["unit"] not in ("DMA_RD","TENSOR")]
    ids=list(dict.fromkeys((e["unit"],e["label"]) for e in visible))
    fig,axes=plt.subplots(6,1,figsize=(24,max(12,len(ids)*.24+6)),sharex=True,
        gridspec_kw={"height_ratios":[max(4,len(ids)*.24),1,1,1,1,1]})
    positions={v:k for k,v in enumerate(ids)}
    for ev in visible:
        y=positions[ev['unit'],ev['label']]; color=plt.cm.tab20(ev.get('row',ev['tile'])%20)
        axes[0].barh(y,ev["end"]-ev["start"],left=ev["start"],height=.7,color=color,edgecolor="black",linewidth=.4)
        if ev.get("register") is not None:
            axes[0].text(ev["start"],y,f"v{ev['register']}",fontsize=7,va="center")
        axes[0].plot(ev["result"],y,"v",ms=3,color="green")
        if ev["writeback"] is not None: axes[0].plot(ev["writeback"],y,"|",ms=6,color="red")
    ticklabels=[]
    for k in ids:
        label=k[1]
        equation=ATTENTION_EQUATIONS.get(label)
        ticklabels.append(k[0]+": "+(f"${equation}$" if equation else label))
    axes[0].set_yticks(range(len(ids)),ticklabels,fontsize=8)
    axes[0].invert_yaxis(); axes[0].set_title("Vector 细粒度轨迹：方块=输入占用；绿三角=结果可转发；红线=RF 写回")
    series=[report["allocated_registers"],report["source_operand_requests"],report["resources"].get("RF_R",{}),report["resources"].get("RF_W",{}),report["forwarding"]]
    labels=["已分配 vreg","源操作数需求","实际 RF 读口","RF 写口","forward 次数"]
    for ax,values,label in zip(axes[1:],series,labels):
        x=np.arange(begin,stop); y=[values[t] if isinstance(values,list) else values.get(t,0) for t in x]
        ax.step(x,y,where="post"); ax.set_ylabel(label); ax.grid(alpha=.25)
    reg=report['config']['vector_detail']['register']
    for ax,cap in ((axes[1],reg['capacity_registers']),(axes[3],reg['read_ports']),(axes[4],reg['write_ports'])):
        if cap is not None: ax.axhline(cap,color='red',linestyle='--')
    axes[1].set_title(f"每 vreg {reg['elements']*4} B；全程分配峰值 {report['peak_registers']} vreg / {report['peak_registers']*reg['elements']*4/1024:g} KiB",fontsize=9)
    axes[-1].set_xlabel("cycle"); axes[-1].set_xlim(begin,stop)
    fig.tight_layout(); fig.savefig(output/("vector_detail.png" if page==0 else f"vector_detail_{page+1:02d}.png"),dpi=160); plt.close(fig)
    if page: return
    for next_page in range(1,math.ceil(window/180)): draw_detail(report,output,next_page)
    lanes=["DMA_RD","TENSOR","LD",*VECTOR_UNITS,"ST","BR"]
    fig,axes=plt.subplots(5,1,figsize=(28,14),sharex=True,gridspec_kw={"height_ratios":[8,1,1,1,1]})
    for r in lanes:
        vals=report["resources"].get(r,{})
        active=sorted(t for t,v in vals.items() if v)
        if active:
            starts=[active[0]]; ends=[]
            for a,b in zip(active,active[1:]):
                if b!=a+1: ends.append(a+1); starts.append(b)
            ends.append(active[-1]+1)
            axes[0].broken_barh([(a,b-a) for a,b in zip(starts,ends)],(lanes.index(r)-.35,.7))
    axes[0].set_yticks(range(len(lanes)),lanes); axes[0].invert_yaxis()
    axes[0].set_title(f"完整执行轨迹；观测 tile 间隔={report['tile_intervals']}；包含填充/排空")
    for ax,r,cap in zip(axes[1:],["L1_R","L1_W","L1_R_BUS","L1_W_BUS"],[None,None,report['config']['l1']['read_gbps']/report['config']['clock_ghz'],report['config']['l1']['write_gbps']/report['config']['clock_ghz']]):
        vals=report["resources"].get(r,{})
        ax.plot([vals.get(t,0)*(100/cap if cap else 1) for t in range(end)],linewidth=.5)
        ax.set_ylabel(r+(" %" if cap else " B/c")); ax.grid(alpha=.2)
    axes[-1].set_xlabel("cycle"); fig.tight_layout(); fig.savefig(output/"pipeline.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5)); names=list(report["engine_busy_cycles"])
    bars=ax.bar(names,[report["engine_busy_cycles"][n] for n in names]); ax.bar_label(bars)
    ax.set_ylabel("输入占用区间并集 / cycle"); ax.set_title("DMA / Tensor / Vector（含控制与访存）")
    fig.tight_layout(); fig.savefig(output/"engine_statistics.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(14,8))
    units=[r for r in lanes if r in report['resources']]
    counts=[sum(v>0 for v in report['resources'][r].values()) for r in units]
    bars=ax.barh(units,counts); ax.bar_label(bars); ax.invert_yaxis()
    ax.set_xlabel('实际接受输入或总线活动周期（流水延迟单独计入结果事件）')
    ax.set_title('逐执行单元时间；不同单元可重叠')
    fig.tight_layout(); fig.savefig(output/'unit_statistics.png',dpi=160); plt.close(fig)


def draw_macro_report(report,output):
    """宏占用、真实活动和队列容量分开画，避免用宏跨度冒充计算忙碌。"""
    import matplotlib.pyplot as plt
    configure_plot_fonts()
    fig,axes=plt.subplots(6,1,figsize=(28,22),sharex=True,
        gridspec_kw={'height_ratios':[7,2,1,1,1,1.5]})
    engines=MacroSimulator.ENGINES
    steps={'QK':1,'SCALE':2,'MASK':3,'ROWMAX':4,'MAX':5,'DELTA':6,'SUB_S':6,
           'ALPHA':7,'EXP_P':7,'ROWSUM':8,'PV':8,'SCALE_L':8,'SCALE_A':8,'ADD_L':9,'ADD_A':9}
    labels=[]
    for m in report['macros']:
        row=engines.index(m['engine']); color=plt.cm.tab20(m['tile']%20)
        axes[0].barh(row,m['end']-m['start'],left=m['start'],height=.65,color=color,edgecolor='black',linewidth=.5)
        operations=sorted(dict.fromkeys(m['operations']),key=lambda op:steps.get(op,99))
        lines=[f"tile {m['tile']}"]
        for op in operations:
            if op in ATTENTION_EQUATIONS:
                prefix=f"[{steps[op]}]\\ " if op in steps else ''
                lines.append('$'+prefix+ATTENTION_EQUATIONS[op]+'$')
            else:
                lines.append(m['label'] if m['engine']=='DMA' else op)
        text=axes[0].text((m['start']+m['end'])/2,row,'\n'.join(lines),ha='center',va='center',fontsize=9,linespacing=1.3,clip_on=True)
        labels.append((m,text))
    axes[0].set_yticks(range(3),engines); axes[0].invert_yaxis()
    axes[0].set_title('宏指令占用：接受 → 所有指令、RF 写回和 L1 输出完成；同引擎宏不重叠；DMA K/V/mask 均表示 load')
    compute={'Tensor':['TENSOR'],'Vector':list(VECTOR_UNITS),'DMA':['DMA_RD','DMA_WR']}
    for row,engine in enumerate(engines):
        active=sorted(set(t for u in compute[engine] for t,v in report['resources'].get(u,{}).items() if v))
        if active:
            starts=[active[0]]; ends=[]
            for a,b in zip(active,active[1:]):
                if b!=a+1: ends.append(a+1); starts.append(b)
            ends.append(active[-1]+1)
            axes[1].broken_barh([(a,b-a) for a,b in zip(starts,ends)],(row-.3,.6))
    axes[1].set_yticks(range(3),engines); axes[1].invert_yaxis()
    axes[1].set_title('真实计算输入活动并集（DMA 为传输）；Vector 此行不含 load/store/BR')
    for ax,engine in zip(axes[2:5],engines):
        hist=report['macro_queue_history'][engine]; cfg=report['config']['macro_scheduler']['queues'][engine]
        ax.step(range(len(hist)),[h['queued'] for h in hist],where='post',label='等待 entry')
        ax.step(range(len(hist)),[h['active'] for h in hist],where='post',label='执行宏',alpha=.7)
        ax.axhline(cfg['entries'],color='red',linestyle='--',label=f"队列上限 {cfg['entries']}")
        ax.set_ylabel(f"{engine}\n{cfg['policy']}"); ax.set_ylim(-.15,cfg['entries']+1)
        ax.legend(loc='upper right',ncol=3,fontsize=8); ax.grid(alpha=.2)
    axes[5].plot(np.asarray(report['l1_live_bytes'])/1024,label='L1 活跃对象')
    axes[5].axhline(report['l1_pool_reserved_bytes']/1024,color='orange',label='逐类槽池容量之和')
    axes[5].axhline(report['config']['l1']['size_kib'],color='red',linestyle='--',label='配置容量')
    axes[5].set_ylabel('L1 / KiB'); axes[5].legend(ncol=3,fontsize=9)
    axes[-1].set_xlabel('cycle'); axes[-1].set_xlim(0,report['cycles'])
    fig.suptitle(f"FlashAttention 两层调度 | 每 cycle 入队 1 宏 | 观测 tile 间隔 {report['tile_intervals']}",fontsize=17)
    fig.tight_layout()
    # 用实际公式渲染边界确定全尺寸图宽；不缩短公式，不改变执行方块时间比例。
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    factor=max([1.0]+[(text.get_window_extent(renderer).width+16)/
        max(1,axes[0].transData.transform((m['end'],0))[0]-axes[0].transData.transform((m['start'],0))[0])
        for m,text in labels])
    if factor>1:
        fig.set_size_inches(fig.get_figwidth()*factor*1.03,fig.get_figheight())
        fig.tight_layout()
    fig.savefig(output/'macro_pipeline.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,6))
    positions=np.arange(3)
    a=ax.bar(positions-.2,[report['engine_macro_busy_cycles'][e] for e in engines],.4,label='宏占用（含内部等待与排空）')
    b=ax.bar(positions+.2,[report['engine_busy_cycles'][e] for e in engines],.4,label='计算/访存/控制活动并集')
    ax.bar_label(a); ax.bar_label(b); ax.set_xticks(positions,engines); ax.set_ylabel('cycle')
    ax.legend(); ax.set_title('同一统计窗口，两种执行时间口径')
    fig.tight_layout(); fig.savefig(output/'macro_statistics.png',dpi=160); plt.close(fig)


def validate_detail(config):
    import copy
    c=copy.deepcopy(config); d=c.pop("vector_detail",copy.deepcopy(DETAIL_DEFAULTS))
    macro=c.pop('macro_scheduler',copy.deepcopy(MACRO_DEFAULTS))
    validate_config(c)
    if set(macro)!={'frontend_dispatch_width','queues'} or macro['frontend_dispatch_width']!=1:
        raise ValueError('当前宏前端固定每 cycle 入队一条；不支持多在途执行上下文配置')
    if set(macro['queues'])!=set(MacroSimulator.ENGINES): raise ValueError('必须配置 Tensor/Vector/DMA 三个宏队列')
    for engine,q in macro['queues'].items():
        if set(q)!={'entries','policy'} or not isinstance(q['entries'],int) or isinstance(q['entries'],bool) or q['entries']<1:
            raise ValueError(engine+' 宏队列 entries 必须为正整数')
        if q['policy'] not in ('fifo','ooo'): raise ValueError(engine+' 宏队列 policy 仅支持 fifo/ooo')
    if d["group_size"] not in (1,2,4,8): raise ValueError("group_size 仅支持 1/2/4/8")
    for key in ("queue_depth","dispatch_width","branch_latency_cycles","tiles","max_cycles","writeback_delay_cycles"):
        if not isinstance(d[key],int) or isinstance(d[key],bool) or d[key]<1: raise ValueError(key+" 必须为正整数")
    for key in ("capacity_registers","read_ports","write_ports"):
        v=d["register"][key]
        if v is not None and (not isinstance(v,int) or isinstance(v,bool) or v<1): raise ValueError(key+" 必须为正整数或 null")
    if d["register"]["dtype"]!="fp32" or d["register"]["elements"]<1: raise ValueError("Vector 寄存器要求 FP32、正元素数")
    if any(c["numeric"][key]!=value for key,value in [("tensor_input_dtype","fp16"),("tensor_output_dtype","fp16"),("vector_dtype","fp32")]) or c["numeric"]["conversion"]["mode"]!="inline":
        raise ValueError("新后端要求 Tensor FP16、Vector FP32、inline conversion")
    flattened=[op for region in d["fusion_regions"] for op in region]
    if len(flattened)!=len(set(flattened)): raise ValueError("融合区域不可重复包含同一操作")
    allowed=set(DETAIL_DEFAULTS['fusion_regions'][0]+DETAIL_DEFAULTS['fusion_regions'][1]+DETAIL_DEFAULTS['fusion_regions'][2])
    if set(flattened)-allowed: raise ValueError('融合区域含未知操作')
    c["vector_detail"]=d
    c['macro_scheduler']=macro
    return c


def main():
    parser=argparse.ArgumentParser(description="FlashAttention Vector 寄存器级指令流时序模型")
    parser.add_argument("--config",type=Path,default=Path(__file__).with_name("flashattention_pipeline.vector.detail.json"))
    parser.add_argument("--output-dir",type=Path,default=Path("artifacts/flashattention_vector_detail"))
    parser.add_argument("--compare-forwarding",action="store_true",help="额外重新调度相反 forwarding 模式并输出独立图表")
    parser.add_argument('--compare-queues',action='store_true',help='额外运行全部引擎的另一种 FIFO/OOO 策略')
    parser.add_argument('--no-plots',action='store_true',help='只输出真实事件和统计，跳过 PNG 渲染')
    parser.add_argument('--flat-detail',action='store_true',help='复现之前的平铺指令后端，不使用宏队列')
    args=parser.parse_args()
    config=validate_detail(json.loads(args.config.read_text(encoding="utf-8-sig")))
    compiler,simulator=(detail_program,DetailSimulator) if args.flat_detail else (macro_program,MacroSimulator)
    p=compiler(config); sim=simulator(p); report=sim.run()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if not args.no_plots:
        draw_detail(report,args.output_dir)
        if 'macros' in report: draw_macro_report(report,args.output_dir)
    (args.output_dir/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("cycles","tile_intervals","observed_steady_II","peak_registers","peak_RF_reads","peak_RF_writes","engine_busy_cycles")},ensure_ascii=False))
    print(args.output_dir.resolve())
    if args.compare_forwarding:
        import copy
        alternative=copy.deepcopy(config)
        alternative['vector_detail']['forwarding']=not config['vector_detail']['forwarding']
        other=simulator(compiler(alternative)).run()
        target=args.output_dir/('with_forwarding' if alternative['vector_detail']['forwarding'] else 'without_forwarding')
        target.mkdir(exist_ok=True)
        if not args.no_plots:
            draw_detail(other,target)
            if 'macros' in other: draw_macro_report(other,target)
        (target/'report.json').write_text(json.dumps(other,ensure_ascii=False,indent=2),encoding='utf-8')
        print('对照：'+json.dumps({k:other[k] for k in ('tile_intervals','peak_registers','peak_RF_reads','peak_RF_writes')},ensure_ascii=False))
    if args.compare_queues:
        import copy
        alternative=copy.deepcopy(config)
        policy='ooo' if config['macro_scheduler']['queues']['Vector']['policy']=='fifo' else 'fifo'
        for q in alternative['macro_scheduler']['queues'].values(): q['policy']=policy
        other=MacroSimulator(macro_program(alternative)).run()
        target=args.output_dir/policy; target.mkdir(exist_ok=True)
        if not args.no_plots: draw_detail(other,target); draw_macro_report(other,target)
        (target/'report.json').write_text(json.dumps(other,ensure_ascii=False,indent=2),encoding='utf-8')
        (target/'input.json').write_text(json.dumps(alternative,ensure_ascii=False,indent=2),encoding='utf-8')
        print('宏队列对照：'+json.dumps({k:other[k] for k in ('tile_intervals','overlap_cycles','peak_registers')},ensure_ascii=False))


if __name__ == "__main__":
    main()
