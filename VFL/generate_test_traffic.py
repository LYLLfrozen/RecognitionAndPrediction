#!/usr/bin/env python3
"""
生成测试流量来验证实时监控系统
用于在真实流量模式下测试系统是否能正确检测
"""

import socket
import time
import sys
import random
from datetime import datetime

def print_banner():
    print("=" * 70)
    print("          测试流量生成器 - 用于验证VFL监控系统")
    print("=" * 70)
    print()

def generate_normal_traffic():
    """生成正常流量 - HTTP请求"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 生成正常流量: HTTP请求")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect(("www.baidu.com", 80))
        sock.send(b"GET / HTTP/1.1\r\nHost: www.baidu.com\r\n\r\n")
        sock.recv(1024)
        sock.close()
        print("  ✓ 正常HTTP流量已发送")
    except Exception as e:
        print(f"  ⚠️  连接失败 (这是正常的): {e}")

def generate_port_scan():
    """生成端口扫描 - Probe攻击"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 生成探测流量: 端口扫描")
    ports = [21, 22, 23, 25, 80, 110, 143, 443, 3306, 8080]
    target = "127.0.0.1"
    
    for port in ports:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.1)
            result = sock.connect_ex((target, port))
            sock.close()
        except:
            pass
    print(f"  ✓ 端口扫描完成 (扫描了 {len(ports)} 个端口)")

def generate_syn_flood_simulation():
    """模拟SYN flood - DOS攻击（安全版本）"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 模拟DOS攻击: 快速连接")
    target = "127.0.0.1"
    port = 8888  # 使用不存在的端口
    
    for i in range(20):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.01)
            sock.connect_ex((target, port))
            sock.close()
        except:
            pass
    print("  ✓ DOS模拟流量已发送 (20个快速连接)")

def generate_udp_flood():
    """生成UDP洪水 - DOS攻击"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 生成DOS流量: UDP洪水")
    target = "127.0.0.1"
    port = 9999
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    message = b"X" * 1024
    
    for i in range(30):
        try:
            sock.sendto(message, (target, port))
        except:
            pass
    sock.close()
    print("  ✓ UDP洪水已发送 (30个包)")

def generate_multiple_connections():
    """生成多次连接 - 可能被识别为攻击"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 生成可疑流量: 频繁连接")
    target = "www.google.com"
    port = 80
    
    for i in range(10):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.connect((target, port))
            sock.close()
        except:
            pass
        time.sleep(0.1)
    print("  ✓ 频繁连接已完成 (10次)")

def main():
    print_banner()
    
    print("⚠️  注意事项:")
    print("  1. 请确保VFL监控系统正在运行 (真实流量模式)")
    print("  2. 这个脚本生成各种测试流量用于验证检测能力")
    print("  3. 所有流量都是安全的，不会造成实际危害")
    print("  4. 建议在监控系统启动后运行此脚本")
    print()
    
    input("按Enter键开始生成测试流量...")
    print()
    
    scenarios = [
        ("正常流量", generate_normal_traffic),
        ("端口扫描 (Probe)", generate_port_scan),
        ("SYN Flood模拟 (DOS)", generate_syn_flood_simulation),
        ("UDP洪水 (DOS)", generate_udp_flood),
        ("频繁连接", generate_multiple_connections),
    ]
    
    try:
        for i in range(3):  # 运行3轮
            print(f"\n{'=' * 70}")
            print(f"第 {i+1} 轮测试")
            print('=' * 70)
            
            for name, func in scenarios:
                print()
                func()
                time.sleep(2)  # 每种流量之间间隔2秒
            
            if i < 2:
                print(f"\n等待5秒后开始下一轮...")
                time.sleep(5)
        
        print("\n" + "=" * 70)
        print("✓ 测试流量生成完成！")
        print("=" * 70)
        print()
        print("请检查监控系统的输出，验证是否检测到不同类型的流量")
        print()
        
    except KeyboardInterrupt:
        print("\n\n流量生成已停止")
        sys.exit(0)

if __name__ == "__main__":
    main()
