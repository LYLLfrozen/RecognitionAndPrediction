#!/usr/bin/env python3
"""
增强版实时流量监控
添加流级统计特征，提高攻击检测准确率
"""

import numpy as np
import time
from collections import defaultdict, deque

class FlowTracker:
    """流量连接跟踪器 - 计算KDD风格的统计特征"""
    
    def __init__(self, window_time=2.0, window_count=100):
        """
        Args:
            window_time: 时间窗口大小(秒)
            window_count: 统计窗口大小(最近N个连接)
        """
        self.window_time = window_time
        self.window_count = window_count
        
        # 连接历史 - key: (src_ip, dst_ip, dst_port)
        self.connections = defaultdict(lambda: {
            'packets': [],  # 包时间戳列表
            'bytes': [],    # 每个包的字节数
            'flags': [],    # TCP flags
            'first_time': None,
            'last_time': None,
            'count': 0
        })
        
        # 全局统计 - 用于计算时间窗口特征
        self.recent_conns = deque(maxlen=window_count)
    
    def get_flow_key(self, src_ip, dst_ip, src_port, dst_port, protocol):
        """生成流的唯一标识"""
        # 对于同一连接，不考虑方向
        if protocol in [6, 17]:  # TCP/UDP
            return tuple(sorted([(src_ip, src_port), (dst_ip, dst_port)]))
        else:
            return (src_ip, dst_ip, protocol)
    
    def update(self, packet_info):
        """
        更新流统计
        
        Args:
            packet_info: 包含以下字段的字典
                - src_ip, dst_ip, src_port, dst_port
                - protocol, packet_size, tcp_flags
                - timestamp
        
        Returns:
            完整的KDD风格特征向量
        """
        timestamp = packet_info.get('timestamp', time.time())
        
        flow_key = self.get_flow_key(
            packet_info['src_ip'],
            packet_info['dst_ip'],
            packet_info.get('src_port', 0),
            packet_info.get('dst_port', 0),
            packet_info['protocol']
        )
        
        flow = self.connections[flow_key]
        
        # 更新连接统计
        if flow['first_time'] is None:
            flow['first_time'] = timestamp
        flow['last_time'] = timestamp
        flow['count'] += 1
        flow['packets'].append(timestamp)
        flow['bytes'].append(packet_info.get('packet_size', 0))
        if 'tcp_flags' in packet_info:
            flow['flags'].append(packet_info['tcp_flags'])
        
        # 计算基于连接的特征
        features = self._compute_flow_features(flow, packet_info, timestamp)
        
        # 记录到全局历史
        # 使用归一化服务端口（取非临时端口一侧），避免双向流量导致diff_srv_rate虚高
        service_port = self._get_service_port(
            packet_info.get('src_port', 0),
            packet_info.get('dst_port', 0)
        )
        # 使用归一化服务端 IP：请求包和回包都指向同一个服务端 IP（修复 Problem 1/2）
        server_ip = self._get_server_ip(
            packet_info['src_ip'], packet_info['dst_ip'],
            packet_info.get('src_port', 0), packet_info.get('dst_port', 0)
        )
        self.recent_conns.append({
            'key': flow_key,
            'time': timestamp,
            'dst_ip': packet_info['dst_ip'],
            'server_ip': server_ip,
            'dst_port': packet_info.get('dst_port', 0),
            'service_port': service_port,
            'service': self._identify_service(service_port),
            'features': features
        })
        
        # 清理过期连接
        self._cleanup_old_connections(timestamp)
        
        return features
    
    def _compute_flow_features(self, flow, packet_info, timestamp):
        """计算KDD Cup 99风格的特征"""
        features = {}
        
        # 基础连接特征
        features['duration'] = flow['last_time'] - flow['first_time'] if flow['first_time'] else 0
        features['protocol_type'] = packet_info['protocol']  # 1=ICMP, 6=TCP, 17=UDP
        features['src_bytes'] = sum(flow['bytes'])
        features['dst_bytes'] = 0  # 简化：单向统计
        features['count'] = flow['count']  # 当前连接的包数
        
        # TCP特性
        if packet_info['protocol'] == 6 and flow['flags']:
            # 统计各种TCP flags
            features['syn_count'] = sum(1 for f in flow['flags'] if f & 0x02)
            features['fin_count'] = sum(1 for f in flow['flags'] if f & 0x01)
            features['rst_count'] = sum(1 for f in flow['flags'] if f & 0x04)
            features['psh_count'] = sum(1 for f in flow['flags'] if f & 0x08)
            features['ack_count'] = sum(1 for f in flow['flags'] if f & 0x10)
        else:
            features['syn_count'] = 0
            features['fin_count'] = 0
            features['rst_count'] = 0
            features['psh_count'] = 0
            features['ack_count'] = 0
        
        # 时间窗口统计 (最近2秒)
        # 修复 Problem 1/2：用归一化的 server_ip 过滤，而非原始 dst_ip。
        # 回包时 dst_ip=本机，原先会把所有来自不同服务的回包混在一起，
        # 导致 diff_srv_rate 虚高（Problem 1）及 serror_rate 虚高（Problem 2）。
        cur_server_ip = self._get_server_ip(
            packet_info['src_ip'], packet_info['dst_ip'],
            packet_info.get('src_port', 0), packet_info.get('dst_port', 0)
        )
        recent_same_dst = [c for c in self.recent_conns
                          if timestamp - c['time'] <= self.window_time
                          and c.get('server_ip', c['dst_ip']) == cur_server_ip]
        
        features['same_dst_count'] = len(recent_same_dst)
        
        # 服务统计：使用归一化服务端口（取非临时端口一侧）
        # 这样 60000→10808 和 10808→60000 两个方向都算作访问 service_port=10808，
        # 避免回包的临时端口让 diff_srv_rate 虚高
        cur_service_port = self._get_service_port(
            packet_info.get('src_port', 0),
            packet_info.get('dst_port', 0)
        )
        recent_same_srv = [c for c in recent_same_dst 
                          if c.get('service_port', c['dst_port']) == cur_service_port]
        features['same_srv_count'] = len(recent_same_srv)
        
        # 错误率统计
        # serror_rate: 失败SYN连接占比
        # 关键修复：查询连接的"当前最新"flags状态而非记录时的快照，
        # 避免将"握手中但尚未传数据的正常连接"（syn=2, rst=0, psh=0）误判为失败连接。
        # 判定规则：当前流有 SYN + RST 且无 PSH/FIN → 真正的失败/被拒绝连接
        if recent_same_dst:
            syn_only = 0
            rst_total = 0
            for c in recent_same_dst:
                fkey = c['key']
                if fkey in self.connections:
                    cur_flags = self.connections[fkey]['flags']
                    cur_syn = sum(1 for f in cur_flags if f & 0x02)
                    cur_rst = sum(1 for f in cur_flags if f & 0x04)
                    cur_psh = sum(1 for f in cur_flags if f & 0x08)
                    cur_fin = sum(1 for f in cur_flags if f & 0x01)
                    cur_ack = sum(1 for f in cur_flags if f & 0x10)
                else:
                    # 连接已被清理，回退到快照
                    cur_syn = c['features'].get('syn_count', 0)
                    cur_rst = c['features'].get('rst_count', 0)
                    cur_psh = c['features'].get('psh_count', 0)
                    cur_fin = c['features'].get('fin_count', 0)
                    cur_ack = c['features'].get('ack_count', 0)
                # 失败连接判定（与KDD99 S0/REJ 对齐）：
                # 条件：有SYN + 无数据(PSH=0) + 无正常关闭(FIN=0) +
                #       且 (从未收到ACK响应=S0型) 或 (收到RST=被拒绝型)
                # 这能正确检测SYN洪泛：伪造源IP时SYN+ACK回到不存在地址，
                # 监测机只看到发出的SYN包，cur_ack=0，因此被正确计为失败连接。
                # 正常连接：SYN+ACK被监测到 → cur_ack>0 → 不计为失败。
                if cur_syn > 0 and cur_psh == 0 and cur_fin == 0 and (cur_ack == 0 or cur_rst > 0):
                    syn_only += 1
                if cur_rst > 0:
                    rst_total += 1
            features['serror_rate'] = syn_only / len(recent_same_dst)
            features['rerror_rate'] = rst_total / len(recent_same_dst)
        else:
            features['serror_rate'] = 0
            features['rerror_rate'] = 0
        
        # 最近N个连接统计
        if len(self.recent_conns) > 1:
            features['same_srv_rate'] = features['same_srv_count'] / len(recent_same_dst) if recent_same_dst else 0
            features['diff_srv_rate'] = 1.0 - features['same_srv_rate']
        else:
            features['same_srv_rate'] = 1.0
            features['diff_srv_rate'] = 0.0
        
        return features
    
    def _get_service_port(self, src_port, dst_port):
        """
        返回连接的"服务端"端口（非临时端口一侧）。
        临时端口定义为 >= 32768（覆盖 Linux/macOS 的随机端口范围）。
        用于 same_srv 统计，避免双向流量导致 diff_srv_rate 虚高。
        """
        EPHEMERAL = 32768
        # 双低端口场景（如 2000 <-> 80）：使用更小端口作为服务端口，
        # 使请求/回包统一映射到同一服务，避免同一DoS被统计为多服务扫描。
        if src_port < EPHEMERAL and dst_port < EPHEMERAL:
            return min(src_port, dst_port)
        if dst_port < EPHEMERAL:
            return dst_port      # 目标是固定服务端口（客户端→服务端）
        if src_port < EPHEMERAL:
            return src_port      # 源是固定服务端口（服务端→客户端的回包）
        return dst_port          # 两端都是高端口，保持 dst_port

    def _get_server_ip(self, src_ip, dst_ip, src_port, dst_port):
        """
        返回连接的"服务端" IP（非临时端口所在侧）。
        用于 same_dst 统计：请求包(dst_ip=服务端)和回包(dst_ip=本机)
        都映射到同一个服务端 IP，避免 diff_srv_rate / serror_rate 虚高。
        """
        EPHEMERAL = 32768
        # 双低端口场景下，通过更小端口判定服务端方向，保证双向一致。
        if src_port < EPHEMERAL and dst_port < EPHEMERAL:
            if src_port <= dst_port:
                return src_ip
            return dst_ip
        if dst_port < EPHEMERAL:
            return dst_ip   # client→server：服务端是 dst
        if src_port < EPHEMERAL:
            return src_ip   # server→client 回包：服务端是 src
        return dst_ip       # 两端都是高端口，默认取 dst

    def _identify_service(self, port):
        """识别服务类型"""
        services = {
            20: 'ftp_data', 21: 'ftp', 22: 'ssh', 23: 'telnet',
            25: 'smtp', 53: 'domain', 80: 'http', 110: 'pop3',
            143: 'imap', 443: 'https', 3306: 'mysql', 5432: 'postgres'
        }
        return services.get(port, 'other')
    
    def _cleanup_old_connections(self, current_time):
        """清理超过时间窗口的连接"""
        cutoff_time = current_time - self.window_time * 10  # 保留10个时间窗口
        
        to_delete = []
        for key, flow in self.connections.items():
            if flow['last_time'] and flow['last_time'] < cutoff_time:
                to_delete.append(key)
        
        for key in to_delete:
            del self.connections[key]
    
    def features_to_vector(self, features_dict, base_features):
        """
        将统计特征合并到基础包特征中
        
        Args:
            features_dict: 从update返回的特征字典
            base_features: 41维基础特征向量
        
        Returns:
            增强的特征向量
        """
        # 创建扩展特征向量
        enhanced = np.copy(base_features)
        
        # 添加流统计到相应位置
        # 索引13-25是流量统计特征区域
        if len(enhanced) >= 26:
            enhanced[13] = features_dict.get('duration', 0)
            enhanced[14] = features_dict.get('src_bytes', 0) / 10000.0  # 归一化
            enhanced[15] = features_dict.get('count', 0) / 100.0  # 归一化
            enhanced[16] = features_dict.get('same_dst_count', 0) / 100.0
            enhanced[17] = features_dict.get('same_srv_count', 0) / 100.0
            enhanced[18] = features_dict.get('serror_rate', 0)
            enhanced[19] = features_dict.get('rerror_rate', 0)
            enhanced[20] = features_dict.get('same_srv_rate', 0)
            enhanced[21] = features_dict.get('diff_srv_rate', 0)
            enhanced[22] = features_dict.get('syn_count', 0) / 10.0
            enhanced[23] = features_dict.get('fin_count', 0) / 10.0
            enhanced[24] = features_dict.get('rst_count', 0) / 10.0
            enhanced[25] = features_dict.get('psh_count', 0) / 10.0
            enhanced[26] = features_dict.get('ack_count', 0) / 10.0  # ACK计数（用于流级flag判断）
        
        return enhanced


# 测试代码
if __name__ == '__main__':
    print("="*80)
    print("流量跟踪器测试")
    print("="*80)
    
    tracker = FlowTracker()
    
    # 模拟正常流量
    print("\n【测试1】正常HTTP流量")
    for i in range(5):
        packet = {
            'src_ip': '192.168.1.100',
            'dst_ip': '8.8.8.8',
            'src_port': 50000 + i,
            'dst_port': 80,
            'protocol': 6,
            'packet_size': 1000,
            'tcp_flags': 0x18,  # PSH+ACK
            'timestamp': time.time()
        }
        features = tracker.update(packet)
        print(f"  包{i+1}: count={features['count']}, same_dst={features['same_dst_count']}")
    
    # 模拟SYN Flood攻击
    print("\n【测试2】SYN Flood攻击")
    for i in range(50):
        packet = {
            'src_ip': f'10.0.0.{i}',  # 不同源IP
            'dst_ip': '192.168.1.1',
            'src_port': 10000 + i,
            'dst_port': 80,
            'protocol': 6,
            'packet_size': 60,
            'tcp_flags': 0x02,  # SYN
            'timestamp': time.time()
        }
        features = tracker.update(packet)
        if i % 10 == 0:
            print(f"  包{i+1}: same_dst={features['same_dst_count']}, serror_rate={features['serror_rate']:.2f}")
        time.sleep(0.01)
    
    # 模拟端口扫描
    print("\n【测试3】端口扫描")
    for port in [21, 22, 23, 25, 80, 443, 3306, 8080]:
        packet = {
            'src_ip': '10.0.0.1',
            'dst_ip': '192.168.1.1',
            'src_port': 55000,
            'dst_port': port,
            'protocol': 6,
            'packet_size': 60,
            'tcp_flags': 0x02,
            'timestamp': time.time()
        }
        features = tracker.update(packet)
        print(f"  端口{port}: same_dst={features['same_dst_count']}, same_srv_rate={features['same_srv_rate']:.2f}")
        time.sleep(0.1)
    
    print("\n" + "="*80)
    print("测试完成")
