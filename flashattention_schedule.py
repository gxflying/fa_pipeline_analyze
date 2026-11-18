"""按 tile 依赖、物理资源与缓冲槽重放 FlashAttention 前向时序。"""

import argparse
import heapq
from collections import defaultdict
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Node:
    unit: str
    work: str
    read: str
    write: str


def node(unit, work, read, write):
    return Node(unit, work, read, write)


nodes = {
    'Q': node('IN', '0', 'q*Y', 'q*Y'), 'K': node('IN', '0', 'k*U', 'k*U'),
    'WQ': node('join', '0', '0', '0'), 'QK': node('T', 'G', 'q*Y+k*U', 'a*X'),
    'S1': node('MUL', 'X', 'a*X+s', 'a*X'), 'S2': node('ADD', 'X', 'a*X+b', 'a*X'),
    'R': node('RM', 'J', 'a*X', 'a*H'), 'M0': node('view', '0', 'a*H', 'a*H'),
    'M1': node('MAX', 'H', '2*a*H', 'a*H'), 'D1': node('ADD', 'H', '2*a*H', 'a*H'),
    'E1': node('EXP', 'H', 'a*H', 'a*H'), 'L0': node('view', '0', 'a*H', 'a*H'),
    'A0': node('view', '0', 'a*Y', 'a*Y'), 'RL': node('MUL', 'H', '2*a*H', 'a*H'),
    'RA': node('MUL', 'Y', 'a*(Y+H)', 'a*Y'), 'P1': node('ADD', 'X', 'a*(X+H)', 'a*X'),
    'P2': node('EXP', 'X', 'a*X', 'a*X'), 'Z': node('RS', 'J', 'a*X', 'a*H'),
    'L1': node('ADD', 'H', '2*a*H', 'a*H'), 'CV': node('CVT', 'X', 'a*X', 'h*X'),
    'V': node('IN', '0', 'v*U', 'v*U'), 'WV': node('join', '0', '0', '0'),
    'PV': node('T', 'G', 'h*X+v*U', 'a*Y'), 'A1': node('ADD', 'Y', '2*a*Y', 'a*Y'),
    'INV': node('RCP', 'H', 'a*H', 'a*H'), 'OUT': node('MUL', 'Y', 'a*(Y+H)', 'a*Y'),
    'OC': node('CVT', 'Y', 'a*Y', 'o*Y'), 'OS': node('ST', '0', 'o*Y', 'o*Y'),
    'LOG': node('LOG', 'H', 'a*H', 'a*H'), 'LSE': node('ADD', 'H', '2*a*H', 'a*H'),
    'LS': node('ST', '0', 'a*H', 'a*H'),
}


def parameters(rows=64, cols=128, dim=128):
    return dict(X=rows*cols, Y=rows*dim, U=dim*cols, H=rows, G=rows*cols*dim,
                J=rows*(cols-1), a=4, q=2, k=2, v=2, h=2, o=2, s=0, b=4*rows*cols)


rates = {
    'IN': (1, 128, 1024, 4), 'ST': (1, 1024, 128, 4), 'T': (4096, 1024, 512, 4),
    'MUL': (128, 1024, 512, 2), 'ADD': (128, 1024, 512, 2), 'MAX': (128, 1024, 512, 2),
    'EXP': (64, 1024, 512, 4), 'RCP': (32, 1024, 512, 5), 'LOG': (32, 1024, 512, 6),
    'RM': (128, 1024, 512, 7), 'RS': (128, 1024, 512, 7), 'CVT': (256, 1024, 512, 2),
}
units = {'IN': 'DMA', 'ST': 'DMA', 'EXP': 'SFU', 'RCP': 'SFU', 'LOG': 'SFU'}


def counts(tag, params):
    spec = nodes[tag]
    return tuple(eval(expr, {'__builtins__': {}}, params) for expr in (spec.work, spec.read, spec.write))


def duration(tag, params, hardware=None):
    hardware = hardware or rates
    spec = nodes[tag]
    if spec.unit not in hardware:
        return 0
    work, read, write = counts(tag, params)
    speed, read_bw, write_bw, latency = hardware[spec.unit]
    return latency + max((work+speed-1)//speed, (read+read_bw-1)//read_bw, (write+write_bw-1)//write_bw)


@dataclass
class Event:
    tag: str
    tile: int
    unit: str
    length: int
    deps: set
    start: int = -1
    end: int = -1


def schedule(tiles, policy, params=None, hardware=None):
    params = params or parameters()
    events = {}
    state = {}

    def insert(tag, tile, *dependencies):
        key = (tile, tag)
        unit = nodes[tag].unit
        events[key] = Event(tag, tile, units.get(unit, unit), duration(tag, params, hardware), set(dependencies))
        return key

    query = insert('Q', -1)
    for tile in range(tiles):
        current = {}

        def stage(tag, *dependencies):
            current[tag] = insert(tag, tile, *dependencies)
            return current[tag]

        previous = state.get(tile-1, {})
        earlier = state.get(tile-2, {})
        previous_done = [previous[tag] for tag in ('den', 'num')] if previous else []
        load_gate = previous_done if policy == 'barrier' else []
        stage('K', *load_gate, *([earlier['QK']] if earlier else []))
        stage('V', *load_gate, *([earlier['PV']] if earlier else []))
        contexts = {'barrier': 1, 'local': 1, 'global': 2, 'deep': 3}[policy]
        reused = state.get(tile-contexts, {})
        compute_gate = [reused[tag] for tag in ('den', 'num')] if reused else []
        stage('QK', query, current['K'], *compute_gate)
        stage('S1', current['QK'])
        stage('S2', current['S1'])
        stage('R', current['S2'])
        rowmax = current['R']
        if previous:
            old_bank_readers = [earlier[tag] for tag in ('D1', 'P1') if tag in earlier]
            if 'D1' in previous:
                old_bank_readers.append(previous['D1'])
            rowmax = stage('M1', rowmax, previous['max'], *old_bank_readers)
            stage('D1', rowmax, previous['max'])
            stage('E1', current['D1'])
            stage('RL', current['E1'], previous['den'])
            stage('RA', current['E1'], previous['num'])
        stage('P1', rowmax, current['S2'])
        stage('P2', current['P1'])
        stage('Z', current['P2'])
        stage('CV', current['P2'])
        stage('PV', current['CV'], current['V'])
        current['den'] = stage('L1', current['RL'], current['Z']) if previous else current['Z']
        current['num'] = stage('A1', current['RA'], current['PV']) if previous else current['PV']
        state[tile] = dict(current, max=rowmax)
    final = state[tiles-1]
    inverse = insert('INV', tiles, final['den'])
    out = insert('OUT', tiles, final['num'], inverse)
    cast = insert('OC', tiles, out)
    insert('OS', tiles, cast)
    log = insert('LOG', tiles, final['den'])
    lse = insert('LSE', tiles, log, final['max'])
    insert('LS', tiles, lse)
    order = 'Q K V QK S1 S2 R M1 D1 E1 RL RA P1 P2 Z CV PV L1 A1 INV OUT OC OS LOG LSE LS'.split()
    rank = {tag: index for index, tag in enumerate(order)}
    waiting = {key: len(event.deps) for key, event in events.items()}
    followers = defaultdict(list)
    ready = defaultdict(list)

    def offer(key):
        event = events[key]
        heapq.heappush(ready[event.unit], (event.tile, rank[event.tag], key))

    for key, event in events.items():
        for dep in event.deps:
            followers[dep].append(key)
        if not event.deps:
            offer(key)
    running = []
    occupied = set()
    now = 0
    completed = 0
    while completed < len(events):
        for unit, queue in ready.items():
            if queue and unit not in occupied:
                _, _, key = heapq.heappop(queue)
                event = events[key]
                event.start, event.end = now, now + event.length
                occupied.add(unit)
                heapq.heappush(running, (event.end, key))
        if not running:
            raise RuntimeError('存在未满足的依赖，无法继续排程')
        now = running[0][0]
        while running and running[0][0] == now:
            _, key = heapq.heappop(running)
            occupied.remove(events[key].unit)
            completed += 1
            for child in followers[key]:
                waiting[child] -= 1
                if not waiting[child]:
                    offer(child)
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [('rows', 64), ('cols', 128), ('dim', 128), ('tiles', 4)]:
        parser.add_argument('--'+name, type=int, default=default)
    parser.add_argument('--clock-ghz', type=float, default=1.0)
    parser.add_argument('--policy', choices=['barrier','local','global','deep','all'], default='all')
    parser.add_argument('--unit', action='append', default=[], metavar='UNIT=P,BR,BW,L',
                        help='覆盖指定单元的吞吐、读带宽、写带宽、额外延迟。DMA 的 P 占位填 1')
    parser.add_argument('--param', action='append', default=[], metavar='NAME=BYTES',
                        help='覆盖元素字节数 a/q/k/v/h/o 或每块常量与掩码字节数 s/b')
    parser.add_argument('--trace', action='store_true', help='输出各节点排程与端口字节数')
    args = parser.parse_args()
    if min(args.rows, args.cols, args.dim, args.tiles) < 1 or not isfinite(args.clock_ghz) or args.clock_ghz <= 0:
        parser.error('形状、块数、时钟频率必须为正')
    params = parameters(args.rows, args.cols, args.dim)
    hardware = dict(rates)
    try:
        for entry in args.param:
            name, value = entry.split('=')
            if name not in ('a','q','k','v','h','o','s','b') or int(value) < int(name not in ('s','b')):
                raise ValueError(entry)
            params[name] = int(value)
        if not any(entry.startswith('b=') for entry in args.param):
            params['b'] = params['a'] * params['X']
        for entry in args.unit:
            name, value = entry.split('=')
            speed, read_bw, write_bw, latency = map(int, value.split(','))
            if name not in rates or min(speed, read_bw, write_bw) < 1 or latency < 0:
                raise ValueError(entry)
            hardware[name] = (speed, read_bw, write_bw, latency)
    except ValueError as exc:
        parser.error(f'参数格式或取值无效 {exc}')
    policies = ('barrier', 'local', 'global', 'deep') if args.policy == 'all' else [args.policy]
    for policy in policies:
        events = schedule(args.tiles, policy, params, hardware)
        cycles = max(event.end for event in events.values())
        print(f'{policy:8}  {cycles:8} cycle   {cycles/(1000*args.clock_ghz):.3f} µs')
        if args.trace:
            print('tile  node  unit   start    end     cycles   work    read_B   write_B')
            for event in sorted(events.values(), key=lambda item: (item.start, item.tile, item.tag)):
                work, read, write = counts(event.tag, params)
                print(f'{event.tile:4}  {event.tag:4}  {event.unit:4}  {event.start:6} {event.end:6}'
                      f'  {event.length:6}  {work:7}  {read:7}  {write:7}')


if __name__ == '__main__':
    main()
