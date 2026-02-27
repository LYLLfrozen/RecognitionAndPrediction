#!/usr/bin/env python3
"""
获取本机IP地址的辅助工具
用于攻击模拟测试
"""
import socket
import sys
import os

def get_local_ip():
    """获取本机IP地址"""
    try:
        # 方法1: 通过连接外部地址获取本机IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return None

def get_all_ips():
    """获取所有网卡的IP地址"""
    import platform
    
    if platform.system() == "Windows":
        # Windows特定方法
        try:
            import subprocess
            result = subprocess.run(['ipconfig'], capture_output=True, text=True, encoding='gbk')
            output = result.stdout
            
            ips = []
            current_adapter = None
            
            for line in output.split('\n'):
                line = line.strip()
                
                # 识别适配器名称
                if '适配器' in line or 'adapter' in line.lower():
                    current_adapter = line
                
                # 提取IPv4地址
                if 'IPv4' in line and '.' in line:
                    parts = line.split(':')
                    if len(parts) > 1:
                        ip = parts[1].strip()
                        # 过滤掉loopback
                        if not ip.startswith('127.'):
                            ips.append({
                                'adapter': current_adapter,
                                'ip': ip
                            })
            
            return ips
        except Exception as e:
            print(f"获取网卡信息失败: {e}")
            return []
    else:
        # Linux/Mac方法
        import socket
        try:
            hostname = socket.gethostname()
            addrs = socket.getaddrinfo(hostname, None)
            ips = []
            for addr in addrs:
                if addr[0] == socket.AF_INET:  # IPv4
                    ip = addr[4][0]
                    if not ip.startswith('127.'):
                        ips.append({
                            'adapter': 'default',
                            'ip': ip
                        })
            return ips
        except:
            return []

def print_usage_examples(ip):
    """打印使用示例"""
    print("\n" + "=" * 80)
    print("攻击模拟命令示例")
    print("=" * 80)
    
    examples = [
        {
            'name': 'DoS攻击（SYN Flood）',
            'cmd': f'python simulate_attacks.py dos --target {ip} --port 80 --count 1000',
            'desc': '发送1000个SYN包，模拟DoS攻击'
        },
        {
            'name': '端口扫描（Probe）',
            'cmd': f'python simulate_attacks.py probe --target {ip}',
            'desc': '扫描20个常用端口'
        },
        {
            'name': 'R2L攻击（FTP暴力破解）',
            'cmd': f'python simulate_attacks.py r2l --target {ip} --port 21 --count 10 --interval 1.0',
            'desc': '模拟FTP暴力破解攻击'
        },
        {
            'name': 'U2R攻击（提权尝试）',
            'cmd': f'python simulate_attacks.py u2r --target {ip} --port 80 --count 50 --interval 2.0',
            'desc': '模拟提权攻击'
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}")
        print(f"   {example['desc']}")
        print(f"   {example['cmd']}")
    
    print("\n" + "=" * 80)

def main():
    print("=" * 80)
    print("本机IP地址查询工具")
    print("=" * 80)
    
    # 获取主要IP地址
    primary_ip = get_local_ip()
    
    if primary_ip:
        print(f"\n✓ 本机主要IP地址: {primary_ip}")
    else:
        print("\n⚠️  无法自动获取主要IP地址")
    
    # 获取所有IP地址
    all_ips = get_all_ips()
    
    if all_ips:
        print(f"\n所有可用网卡:")
        for i, info in enumerate(all_ips, 1):
            adapter = info.get('adapter', 'Unknown')
            ip = info['ip']
            
            # 简化适配器名称显示
            if adapter:
                # 提取关键信息
                if '以太网' in adapter:
                    adapter_short = '以太网'
                elif 'WLAN' in adapter or '无线' in adapter:
                    adapter_short = 'WLAN'
                elif 'VPN' in adapter or 'TAP' in adapter:
                    adapter_short = 'VPN'
                else:
                    adapter_short = adapter[:30] + '...' if len(adapter) > 30 else adapter
            else:
                adapter_short = 'Unknown'
            
            marker = "⭐" if ip == primary_ip else "  "
            print(f"{marker} [{i}] {adapter_short}")
            print(f"      IP: {ip}")
    
    # 推荐使用的IP
    recommended_ip = primary_ip if primary_ip else (all_ips[0]['ip'] if all_ips else None)
    
    if recommended_ip:
        print(f"\n{'=' * 80}")
        print(f"推荐使用的IP地址: {recommended_ip}")
        print(f"{'=' * 80}")
        
        print_usage_examples(recommended_ip)
        
        print("\n💡 使用步骤:")
        print("1. 在一个终端启动监控（以管理员身份）:")
        print(f"   python realtime_monitor.py --interface \"以太网\"")
        print("\n2. 在另一个终端运行攻击模拟（以管理员身份）:")
        print(f"   python simulate_attacks.py dos --target {recommended_ip} --port 80 --count 1000")
        print("\n3. 观察监控终端的检测结果")
    else:
        print("\n⚠️  错误: 无法获取本机IP地址")
        print("请手动运行 ipconfig 查看网络配置")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已取消")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
