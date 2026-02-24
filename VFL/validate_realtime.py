#!/usr/bin/env python3
"""
真实流量验证工具
用于生成已知类型的流量并验证实时监控的准确性
"""

import time
import subprocess
import sys

def generate_normal_traffic():
    """生成正常流量"""
    print("生成正常 HTTP 流量...")
    urls = [
        "http://example.com",
        "http://httpbin.org/get",
        "http://www.google.com",
    ]
    for url in urls:
        try:
            subprocess.run(['curl', '-s', '-o', '/dev/null', url], timeout=2)
            time.sleep(0.5)
        except:
            pass

def generate_probe_traffic():
    """生成探测流量（端口扫描）"""
    print("生成探测流量（端口扫描）...")
    # 扫描本地常见端口
    ports = [21, 22, 23, 25, 80, 443, 3306, 3389, 8080]
    for port in ports:
        try:
            subprocess.run(['nc', '-z', '-w', '1', '127.0.0.1', str(port)], 
                         capture_output=True, timeout=2)
            time.sleep(0.1)
        except:
            pass

def generate_dns_traffic():
    """生成 DNS 查询流量"""
    print("生成 DNS 流量...")
    domains = ['google.com', 'github.com', 'stackoverflow.com', 'wikipedia.org']
    for domain in domains:
        try:
            subprocess.run(['nslookup', domain], capture_output=True, timeout=2)
            time.sleep(0.3)
        except:
            pass

def main():
    print("=" * 80)
    print("真实流量验证工具")
    print("=" * 80)
    print("\n请在另一个终端运行实时监控:")
    print("  sudo python3 realtime_monitor.py --real --interface lo0")
    print("\n按 Enter 开始生成测试流量...")
    input()
    
    try:
        # 循环生成不同类型的流量
        for i in range(3):
            print(f"\n第 {i+1} 轮:")
            
            print("\n1. 正常流量")
            generate_normal_traffic()
            time.sleep(2)
            
            print("\n2. DNS 流量")
            generate_dns_traffic()
            time.sleep(2)
            
            print("\n3. 端口扫描")
            generate_probe_traffic()
            time.sleep(2)
            
        print("\n流量生成完成！请查看监控窗口的预测结果。")
        print("\n预期结果:")
        print("  - 正常 HTTP/DNS 流量 → normal")
        print("  - 端口扫描 → probe (或 normal，因为特征不完整)")
        
    except KeyboardInterrupt:
        print("\n\n已中断")

if __name__ == '__main__':
    main()
