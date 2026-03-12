#!/usr/bin/env python3
"""验证 flow_tracker 修复效果"""
import time
import sys
sys.path.insert(0, '/Users/lyll/Documents/class/毕设/RecognitionAndPrediction/VFL')
from flow_tracker import FlowTracker

def run_test(name, packets_fn):
    tracker = FlowTracker()
    last_f = None
    for pkt in packets_fn():
        last_f = tracker.update(pkt)
    print(f"\n=== {name} ===")
    print(f"  same_dst_count : {last_f['same_dst_count']}")
    print(f"  diff_srv_rate  : {last_f['diff_srv_rate']:.3f}")
    print(f"  serror_rate    : {last_f['serror_rate']:.3f}")
    print(f"  rerror_rate    : {last_f['rerror_rate']:.3f}")
    return last_f

# ── 测试1：正常 lo0 流量（client 60000+i <-> server 10808，有实际数据 PSH） ──
def normal_lo0():
    pkts = []
    for i in range(25):
        ts = time.time() + i * 0.01
        pkts.append({'src_ip':'127.0.0.1','dst_ip':'127.0.0.1',
                     'src_port':60000+i,'dst_port':10808,
                     'protocol':6,'packet_size':200,'tcp_flags':0x18,  # PSH|ACK
                     'timestamp':ts})
        pkts.append({'src_ip':'127.0.0.1','dst_ip':'127.0.0.1',
                     'src_port':10808,'dst_port':60000+i,
                     'protocol':6,'packet_size':150,'tcp_flags':0x10,  # ACK
                     'timestamp':ts+0.001})
    return pkts

f1 = run_test("正常 lo0 流量（PSH|ACK, 期望 diff_srv_rate≈0, serror_rate≈0）", normal_lo0)
assert f1['diff_srv_rate'] < 0.05,  f"FAIL: diff_srv_rate={f1['diff_srv_rate']:.3f} 应<0.05"
assert f1['serror_rate'] < 0.05,    f"FAIL: serror_rate={f1['serror_rate']:.3f} 应<0.05"
print("  ✓ PASS")

# ── 测试2：正常 lo0 流量（握手中，还没 PSH，不应被误判为扫描） ──
def normal_handshake():
    pkts = []
    for i in range(25):
        ts = time.time() + i * 0.01
        pkts.append({'src_ip':'127.0.0.1','dst_ip':'127.0.0.1',
                     'src_port':60000+i,'dst_port':10808,
                     'protocol':6,'packet_size':60,'tcp_flags':0x02,'timestamp':ts})   # SYN
        pkts.append({'src_ip':'127.0.0.1','dst_ip':'127.0.0.1',
                     'src_port':10808,'dst_port':60000+i,
                     'protocol':6,'packet_size':60,'tcp_flags':0x12,'timestamp':ts+0.001})  # SYN|ACK
        pkts.append({'src_ip':'127.0.0.1','dst_ip':'127.0.0.1',
                     'src_port':60000+i,'dst_port':10808,
                     'protocol':6,'packet_size':56,'tcp_flags':0x10,'timestamp':ts+0.002})  # ACK only
    return pkts

f2 = run_test("正常握手中（SYN→SYN|ACK→ACK, 无PSH, 期望 serror_rate≈0）", normal_handshake)
assert f2['diff_srv_rate'] < 0.05,  f"FAIL: diff_srv_rate={f2['diff_srv_rate']:.3f} 应<0.05"
assert f2['serror_rate'] < 0.05,    f"FAIL: serror_rate={f2['serror_rate']:.3f} 应<0.05"
print("  ✓ PASS")

# ── 测试3：SYN Flood（全部被 RST，从 SYN 包视角） ──
def syn_flood():
    pkts = []
    ts_base = time.time()
    for i in range(30):
        ts = ts_base + i * 0.001
        pkts.append({'src_ip':f'10.0.0.{i}','dst_ip':'192.168.1.1',
                     'src_port':10000+i,'dst_port':80,
                     'protocol':6,'packet_size':44,'tcp_flags':0x02,'timestamp':ts})   # SYN
        pkts.append({'src_ip':'192.168.1.1','dst_ip':f'10.0.0.{i}',
                     'src_port':80,'dst_port':10000+i,
                     'protocol':6,'packet_size':44,'tcp_flags':0x14,'timestamp':ts+0.0001})  # RST|ACK
    # 返回最后一个 SYN 包（而非 RST 回包）— 从攻击者视角
    return pkts

tracker3 = FlowTracker()
pkts3 = syn_flood()
# 只取奇数包（SYN包）作为最后特征
last_f3 = None
for i, p in enumerate(pkts3):
    f = tracker3.update(p)
    if p['tcp_flags'] == 0x02:  # SYN
        last_f3 = f
print(f"\n=== SYN Flood（从SYN包视角, 期望 serror_rate>0.5） ===")
print(f"  same_dst_count : {last_f3['same_dst_count']}")
print(f"  diff_srv_rate  : {last_f3['diff_srv_rate']:.3f}")
print(f"  serror_rate    : {last_f3['serror_rate']:.3f}")
assert last_f3['serror_rate'] > 0.5, f"FAIL: serror_rate={last_f3['serror_rate']:.3f} 应>0.5"
print("  ✓ PASS")

# ── 测试4：端口扫描（同一目标不同端口，SYN→RST） ──
def port_scan():
    pkts = []
    ts_base = time.time()
    ports = [21, 22, 23, 25, 53, 80, 110, 135, 443, 3306, 5432, 8080, 3389, 139, 445, 1433, 1521, 5900, 6379, 8000]
    for i, port in enumerate(ports):
        ts = ts_base + i * 0.05
        pkts.append({'src_ip':'10.0.0.1','dst_ip':'192.168.1.1',
                     'src_port':55000+i,'dst_port':port,
                     'protocol':6,'packet_size':60,'tcp_flags':0x02,'timestamp':ts})
        pkts.append({'src_ip':'192.168.1.1','dst_ip':'10.0.0.1',
                     'src_port':port,'dst_port':55000+i,
                     'protocol':6,'packet_size':44,'tcp_flags':0x14,'timestamp':ts+0.002})
    return pkts

tracker4 = FlowTracker()
pkts4 = port_scan()
last_f4 = None
for p in pkts4:
    f = tracker4.update(p)
    if p['tcp_flags'] == 0x02:
        last_f4 = f
print(f"\n=== 端口扫描（期望 diff_srv_rate>0.5, serror_rate>0.5） ===")
print(f"  same_dst_count : {last_f4['same_dst_count']}")
print(f"  diff_srv_rate  : {last_f4['diff_srv_rate']:.3f}")
print(f"  serror_rate    : {last_f4['serror_rate']:.3f}")
assert last_f4['diff_srv_rate'] > 0.5, f"FAIL: diff_srv_rate={last_f4['diff_srv_rate']:.3f} 应>0.5"
assert last_f4['serror_rate'] > 0.5,   f"FAIL: serror_rate={last_f4['serror_rate']:.3f} 应>0.5"
print("  ✓ PASS")

print("\n\n所有测试通过 ✓")
