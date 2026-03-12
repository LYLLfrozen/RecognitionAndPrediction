#!/usr/bin/env python3
"""
lo0 接口真实流量特征采集工具
按时间顺序输出每个包的特征，用于分析 ML 误判问题。

用法:
    sudo python3 capture_features.py              # 采集100包，输出到控制台
    sudo python3 capture_features.py -n 200       # 采集200包
    sudo python3 capture_features.py -o out.csv   # 保存为CSV
    sudo python3 capture_features.py -i en0       # 指定其他接口
"""

import argparse
import csv
import sys
import time
from collections import defaultdict, deque

import numpy as np

try:
    from scapy.all import IP, TCP, UDP, ICMP, sniff
except ImportError:
    print("❌ 需要 scapy：pip install scapy")
    sys.exit(1)

# ──────────────────────────────────────────────
# 轻量级流量统计（与 FlowTracker 逻辑一致）
# ──────────────────────────────────────────────
class MiniFlowTracker:
    def __init__(self, window_time=2.0, window_count=200):
        self.window_time = window_time
        self.connections = defaultdict(lambda: {
            'packets': [], 'bytes': [], 'flags': [],
            'first_time': None, 'last_time': None, 'count': 0
        })
        self.recent_conns = deque(maxlen=window_count)

    def _flow_key(self, pinfo):
        proto = pinfo['protocol']
        if proto in (6, 17):
            return tuple(sorted([
                (pinfo['src_ip'], pinfo.get('src_port', 0)),
                (pinfo['dst_ip'], pinfo.get('dst_port', 0))
            ]))
        return (pinfo['src_ip'], pinfo['dst_ip'], proto)

    def update(self, pinfo):
        ts = pinfo['timestamp']
        key = self._flow_key(pinfo)
        flow = self.connections[key]
        if flow['first_time'] is None:
            flow['first_time'] = ts
        flow['last_time'] = ts
        flow['count'] += 1
        flow['packets'].append(ts)
        flow['bytes'].append(pinfo.get('packet_size', 0))
        if 'tcp_flags' in pinfo:
            flow['flags'].append(pinfo['tcp_flags'])

        # 时间窗口内同一目标的连接
        recent_same_dst = [c for c in self.recent_conns
                           if ts - c['time'] <= self.window_time
                           and c['dst_ip'] == pinfo['dst_ip']]
        same_dst_count = len(recent_same_dst)

        dst_port = pinfo.get('dst_port', 0)
        same_srv_count = sum(1 for c in recent_same_dst if c['dst_port'] == dst_port)

        if recent_same_dst:
            syn_only = sum(1 for c in recent_same_dst
                           if c['flow_syn'] > 0 and c['flow_psh'] == 0 and c['flow_fin'] == 0)
            rst_count = sum(1 for c in recent_same_dst if c['flow_rst'] > 0)
            serror_rate = syn_only / len(recent_same_dst)
            rerror_rate = rst_count / len(recent_same_dst)
            same_srv_rate = same_srv_count / len(recent_same_dst)
        else:
            serror_rate = rerror_rate = 0.0
            same_srv_rate = 1.0

        diff_srv_rate = 1.0 - same_srv_rate

        syn_c  = sum(1 for f in flow['flags'] if f & 0x02)
        fin_c  = sum(1 for f in flow['flags'] if f & 0x01)
        rst_c  = sum(1 for f in flow['flags'] if f & 0x04)
        psh_c  = sum(1 for f in flow['flags'] if f & 0x08)

        self.recent_conns.append({
            'key': key, 'time': ts,
            'dst_ip': pinfo['dst_ip'],
            'dst_port': dst_port,
            'flow_syn': syn_c, 'flow_fin': fin_c,
            'flow_rst': rst_c, 'flow_psh': psh_c,
        })

        return {
            'duration':       flow['last_time'] - flow['first_time'],
            'src_bytes':      sum(flow['bytes']),
            'flow_count':     flow['count'],
            'same_dst_count': same_dst_count,
            'same_srv_count': same_srv_count,
            'serror_rate':    round(serror_rate, 4),
            'rerror_rate':    round(rerror_rate, 4),
            'same_srv_rate':  round(same_srv_rate, 4),
            'diff_srv_rate':  round(diff_srv_rate, 4),
            'syn_count':      syn_c,
            'fin_count':      fin_c,
            'rst_count':      rst_c,
            'psh_count':      psh_c,
        }


# ──────────────────────────────────────────────
# 包解析
# ──────────────────────────────────────────────
def parse_packet(pkt):
    """从 scapy 包提取 packet_info 字典，失败返回 None。"""
    if IP not in pkt:
        return None
    pinfo = {
        'src_ip':      pkt[IP].src,
        'dst_ip':      pkt[IP].dst,
        'ttl':         pkt[IP].ttl,
        'ip_len':      pkt[IP].len,
        'packet_size': len(pkt),
        'timestamp':   time.time(),
        'tcp_flags':   0,
        'src_port':    0,
        'dst_port':    0,
    }
    if TCP in pkt:
        pinfo['protocol'] = 6
        pinfo['src_port'] = pkt[TCP].sport
        pinfo['dst_port'] = pkt[TCP].dport
        flags = pkt[TCP].flags
        fval = flags.value if hasattr(flags, 'value') else int(flags)
        pinfo['tcp_flags'] = fval
        pinfo['tcp_window'] = pkt[TCP].window
        pinfo['tcp_seq']    = pkt[TCP].seq
        pinfo['tcp_ack']    = pkt[TCP].ack
    elif UDP in pkt:
        pinfo['protocol'] = 17
        pinfo['src_port'] = pkt[UDP].sport
        pinfo['dst_port'] = pkt[UDP].dport
        pinfo['tcp_flags'] = 0
    elif ICMP in pkt:
        pinfo['protocol'] = 1
        pinfo['icmp_type'] = pkt[ICMP].type
        pinfo['icmp_code'] = pkt[ICMP].code
    else:
        pinfo['protocol'] = pkt[IP].proto
    return pinfo


# ──────────────────────────────────────────────
# 特征列定义（用于 CSV 表头）
# ──────────────────────────────────────────────
COLUMNS = [
    # 时间戳
    'abs_time', 'rel_time',
    # 包基础
    'src_ip', 'dst_ip', 'protocol',
    'src_port', 'dst_port',
    'packet_size', 'ip_len', 'ttl',
    # TCP
    'tcp_flags', 'tcp_flags_str',
    'tcp_window', 'tcp_seq', 'tcp_ack',
    # ICMP
    'icmp_type', 'icmp_code',
    # 流统计（与 FlowTracker / KDD 对齐）
    'duration', 'src_bytes', 'flow_count',
    'same_dst_count', 'same_srv_count',
    'serror_rate', 'rerror_rate',
    'same_srv_rate', 'diff_srv_rate',
    'syn_count', 'fin_count', 'rst_count', 'psh_count',
]

TCP_FLAG_NAMES = {
    0x01: 'FIN', 0x02: 'SYN', 0x04: 'RST',
    0x08: 'PSH', 0x10: 'ACK', 0x20: 'URG',
}

def flags_to_str(fval):
    parts = [name for bit, name in TCP_FLAG_NAMES.items() if fval & bit]
    return '|'.join(parts) if parts else '-'


# ──────────────────────────────────────────────
# 主逻辑
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='lo0 流量特征采集工具')
    parser.add_argument('-i', '--interface', default='lo0', help='网络接口（默认 lo0）')
    parser.add_argument('-n', '--count',     type=int, default=100,
                        help='采集包数量（默认 100，0=持续采集直到 Ctrl-C）')
    parser.add_argument('-o', '--output',    default=None,
                        help='输出 CSV 文件路径（默认只输出控制台）')
    parser.add_argument('--window',          type=float, default=2.0,
                        help='流量统计时间窗口（秒，默认 2.0）')
    parser.add_argument('--no-table',        action='store_true',
                        help='禁用控制台表格输出（仅保存 CSV）')
    args = parser.parse_args()

    tracker = MiniFlowTracker(window_time=args.window)
    rows = []
    start_time = None
    pkt_idx = [0]

    # CSV writer（可选）
    csv_file   = None
    csv_writer = None
    if args.output:
        csv_file   = open(args.output, 'w', newline='', encoding='utf-8')
        csv_writer = csv.DictWriter(csv_file, fieldnames=COLUMNS)
        csv_writer.writeheader()
        print(f"[✓] 结果将保存到 {args.output}")

    # 控制台表头
    if not args.no_table:
        print(f"\n{'#':>4}  {'rel_t':>7}  {'src_ip':>15}  {'dst_ip':>15}  "
              f"{'proto':>5}  {'sport':>6}  {'dport':>6}  "
              f"{'size':>5}  {'flags':>14}  "
              f"{'same_dst':>8}  {'serr':>5}  {'diff_srv':>8}  "
              f"{'syn':>3}{'fin':>4}{'psh':>4}{'rst':>4}")
        print('-' * 130)

    def handle(pkt):
        nonlocal start_time
        pinfo = parse_packet(pkt)
        if pinfo is None:
            return

        if start_time is None:
            start_time = pinfo['timestamp']

        stats = tracker.update(pinfo)
        rel_t = round(pinfo['timestamp'] - start_time, 4)

        row = {col: '' for col in COLUMNS}
        row.update({
            'abs_time':       round(pinfo['timestamp'], 4),
            'rel_time':       rel_t,
            'src_ip':         pinfo['src_ip'],
            'dst_ip':         pinfo['dst_ip'],
            'protocol':       pinfo['protocol'],
            'src_port':       pinfo.get('src_port', ''),
            'dst_port':       pinfo.get('dst_port', ''),
            'packet_size':    pinfo['packet_size'],
            'ip_len':         pinfo.get('ip_len', ''),
            'ttl':            pinfo.get('ttl', ''),
            'tcp_flags':      pinfo.get('tcp_flags', ''),
            'tcp_flags_str':  flags_to_str(pinfo.get('tcp_flags', 0)),
            'tcp_window':     pinfo.get('tcp_window', ''),
            'tcp_seq':        pinfo.get('tcp_seq', ''),
            'tcp_ack':        pinfo.get('tcp_ack', ''),
            'icmp_type':      pinfo.get('icmp_type', ''),
            'icmp_code':      pinfo.get('icmp_code', ''),
        })
        row.update(stats)

        pkt_idx[0] += 1
        rows.append(row)

        if csv_writer:
            csv_writer.writerow(row)
            csv_file.flush()

        if not args.no_table:
            proto_name = {6: 'TCP', 17: 'UDP', 1: 'ICMP'}.get(pinfo['protocol'], str(pinfo['protocol']))
            print(f"{pkt_idx[0]:>4}  {rel_t:>7.3f}  "
                  f"{pinfo['src_ip']:>15}  {pinfo['dst_ip']:>15}  "
                  f"{proto_name:>5}  {pinfo.get('src_port', ''):>6}  {pinfo.get('dst_port', ''):>6}  "
                  f"{pinfo['packet_size']:>5}  {flags_to_str(pinfo.get('tcp_flags', 0)):>14}  "
                  f"{stats['same_dst_count']:>8}  "
                  f"{stats['serror_rate']:>5.2f}  "
                  f"{stats['diff_srv_rate']:>8.2f}  "
                  f"{stats['syn_count']:>3}{stats['fin_count']:>4}"
                  f"{stats['psh_count']:>4}{stats['rst_count']:>4}")

        # 达到数量上限时停止（count=0 表示持续采集）
        if args.count > 0 and pkt_idx[0] >= args.count:
            return True  # scapy: returning True stops sniff

    print(f"[*] 正在 {args.interface} 上捕获流量（需要 sudo）... Ctrl-C 停止")
    print(f"[*] 时间窗口={args.window}s，目标包数={args.count if args.count > 0 else '∞'}\n")

    try:
        sniff(
            iface=args.interface,
            filter='ip',
            prn=handle,
            count=args.count if args.count > 0 else 0,
            store=False,
            stop_filter=lambda p: pkt_idx[0] >= args.count > 0,
        )
    except KeyboardInterrupt:
        print("\n[!] 用户中断")
    except PermissionError:
        print("❌ 权限不足，请使用 sudo 运行")
        sys.exit(1)
    finally:
        if csv_file:
            csv_file.close()

    # 最终统计摘要
    if rows:
        print(f"\n{'='*60}")
        print(f"[摘要] 共采集 {len(rows)} 个包")
        protos = {}
        for r in rows:
            p = {6:'TCP',17:'UDP',1:'ICMP'}.get(r['protocol'], str(r['protocol']))
            protos[p] = protos.get(p, 0) + 1
        for p, cnt in sorted(protos.items(), key=lambda x: -x[1]):
            print(f"  {p:5s}: {cnt}")

        # 打印最后10条完整行（方便粘贴给分析）
        print(f"\n[最后{min(10,len(rows))}条完整特征]")
        show = rows[-min(10, len(rows)):]
        header_keys = ['rel_time','src_ip','dst_ip','protocol','src_port','dst_port',
                       'packet_size','tcp_flags_str',
                       'same_dst_count','serror_rate','diff_srv_rate',
                       'syn_count','fin_count','psh_count','rst_count',
                       'flow_count','src_bytes']
        w = max(len(k) for k in header_keys) + 2
        for row in show:
            print(f"\n  --- 包 (rel_time={row['rel_time']}) ---")
            for k in header_keys:
                print(f"    {k:<{w}}: {row[k]}")

        if args.output:
            print(f"\n[✓] 完整数据已保存：{args.output}")


if __name__ == '__main__':
    main()
