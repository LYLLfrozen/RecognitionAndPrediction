"""
实时网络流量捕获模块
使用scapy捕获网络数据包
"""
import time
from collections import defaultdict
from datetime import datetime
from scapy.all import sniff, IP, TCP, UDP, ICMP  # type: ignore
from threading import Thread, Event
import queue


class TrafficCapture:
    """网络流量捕获器"""
    
    def __init__(self, interface=None, timeout=10):
        """
        Args:
            interface: 网络接口名称，None表示所有接口
            timeout: 连接超时时间（秒）
        """
        self.interface = interface
        self.timeout = timeout
        self.packet_queue = queue.Queue()
        self.stop_event = Event()
        self.raw_packet_count = 0
        
        # 连接状态跟踪
        from typing import Dict, List, Set, Any, Optional
        def default_conn() -> Dict[str, Any]:
            return {
                'packets': [],  # type: List[Any]
                'start_time': None,  # type: Optional[float]
                'last_time': None,  # type: Optional[float]
                'src_bytes': 0,  # type: int
                'dst_bytes': 0,  # type: int
                'flags': set(),  # type: Set[str]
                'protocol': None  # type: Optional[str]
            }
        self.connections = defaultdict(default_conn)  # type: ignore
        
    def _packet_handler(self, packet):
        """处理捕获的数据包"""
        if IP in packet:
            self.raw_packet_count += 1
            self.packet_queue.put(packet)
            
    def start_capture(self, packet_count=None):
        """
        开始捕获数据包
        
        Args:
            packet_count: 捕获数据包数量，None表示持续捕获
        """
        # 确保 packet_count 是整数 (0 表示无限)
        if packet_count is None:
            packet_count = 0
            
        def capture_thread():
            print(f"开始捕获网络流量...")
            if self.interface:
                print(f"监听接口: {self.interface}")
            else:
                print(f"监听所有接口")
            
            try:
                sniff(
                    iface=self.interface,
                    prn=self._packet_handler,
                    count=packet_count,
                    store=0, # 不存储数据包，节省内存
                    stop_filter=lambda x: self.stop_event.is_set()
                )
            except Exception as e:
                print(f"捕获线程异常: {e}")
        
        thread = Thread(target=capture_thread, daemon=True)
        thread.start()
        return thread
    
    def stop_capture(self):
        """停止捕获"""
        self.stop_event.set()
        print("停止捕获")
        
    def get_packet(self, timeout=1):
        """
        获取一个数据包
        
        Returns:
            packet或None
        """
        try:
            return self.packet_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def build_connection(self, packet):
        """
        构建连接记录
        
        Returns:
            connection_id, connection_info
        """
        if IP not in packet:
            return None, None
        
        ip_layer = packet[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        
        # 确定协议
        protocol = None
        src_port = 0
        dst_port = 0
        flags = []
        
        if TCP in packet:
            protocol = 'tcp'
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
            # 获取TCP标志
            tcp_flags = packet[TCP].flags
            if tcp_flags:
                if 'S' in str(tcp_flags): flags.append('S')
                if 'A' in str(tcp_flags): flags.append('A')
                if 'F' in str(tcp_flags): flags.append('F')
                if 'R' in str(tcp_flags): flags.append('R')
                if 'P' in str(tcp_flags): flags.append('P')
                if 'U' in str(tcp_flags): flags.append('U')
        elif UDP in packet:
            protocol = 'udp'
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport
        elif ICMP in packet:
            protocol = 'icmp'
        else:
            protocol = 'other'
        
        # 连接ID（双向）
        conn_id = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
        
        # 获取数据包大小
        packet_size = len(packet)
        
        # 更新连接信息
        conn = self.connections[conn_id]
        current_time = time.time()
        
        if conn['start_time'] is None:
            conn['start_time'] = current_time
        
        conn['last_time'] = current_time
        conn['packets'].append(packet)
        conn['src_bytes'] += packet_size
        conn['protocol'] = protocol
        conn['flags'].update(flags)
        
        return conn_id, conn
    
    def get_connection_duration(self, conn_id):
        """获取连接持续时间"""
        conn = self.connections[conn_id]
        if conn['start_time'] and conn['last_time']:
            return conn['last_time'] - conn['start_time']
        return 0
    
    def cleanup_old_connections(self):
        """清理过期连接"""
        current_time = time.time()
        to_remove = []
        
        for conn_id, conn in self.connections.items():
            if conn['last_time'] and (current_time - conn['last_time']) > self.timeout:
                to_remove.append(conn_id)
        
        for conn_id in to_remove:
            del self.connections[conn_id]
        
        if to_remove:
            print(f"清理了 {len(to_remove)} 个过期连接")


class ConnectionAggregator:
    """连接聚合器 - 将数据包聚合为连接级统计"""
    
    def __init__(self, window_size=2):
        """
        Args:
            window_size: 时间窗口（秒）
        """
        self.window_size = window_size
        self.connections = {}
        
    def add_packet(self, packet):
        """添加数据包到连接"""
        if IP not in packet:
            return None
        
        # 提取连接信息
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        
        protocol = 'other'
        src_port = 0
        dst_port = 0
        
        if TCP in packet:
            protocol = 'tcp'
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
        elif UDP in packet:
            protocol = 'udp'
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport
        elif ICMP in packet:
            protocol = 'icmp'
        
        # 确定连接方向和Key
        # 默认规则：(src_ip, src_port) < (dst_ip, dst_port)
        # 但对于TCP，如果看到SYN且没ACK，那肯定是发起方
        is_syn = False
        if TCP in packet:
            flags = packet[TCP].flags
            if 'S' in str(flags) and 'A' not in str(flags):
                is_syn = True
        
        if is_syn:
            # 强制当前包为 Forward
            conn_key = (src_ip, src_port, dst_ip, dst_port, protocol)
            direction = 'forward'
        else:
            # 尝试查找现有连接
            key1 = (src_ip, src_port, dst_ip, dst_port, protocol)
            key2 = (dst_ip, dst_port, src_ip, src_port, protocol)
            
            if key1 in self.connections:
                conn_key = key1
                direction = 'forward'
            elif key2 in self.connections:
                conn_key = key2
                direction = 'backward'
            else:
                # 新连接，且不是SYN包，尝试通过端口推断方向
                is_likely_server_response = (src_port < 1024 and dst_port > 1024)
                is_likely_client_request = (src_port > 1024 and dst_port < 1024)
                
                if is_likely_server_response:
                    # 看起来是服务器响应，所以 dst 是发起者 (client)
                    conn_key = (dst_ip, dst_port, src_ip, src_port, protocol)
                    direction = 'backward'
                elif is_likely_client_request:
                    # 看起来是客户端请求
                    conn_key = (src_ip, src_port, dst_ip, dst_port, protocol)
                    direction = 'forward'
                # 否则（都大或都小），回退到地址比较
                elif (src_ip, src_port) < (dst_ip, dst_port):
                    conn_key = (src_ip, src_port, dst_ip, dst_port, protocol)
                    direction = 'forward'
                else:
                    conn_key = (dst_ip, dst_port, src_ip, src_port, protocol)
                    direction = 'backward'
        
        # 初始化或更新连接
        if conn_key not in self.connections:
            self.connections[conn_key] = {
                'start_time': time.time(),
                'last_time': time.time(),
                'packet_count': 0,
                'byte_count': 0,
                'src_bytes': 0,
                'dst_bytes': 0,
                'flags': set(),
                'packets': [],
                'protocol': protocol,
                'src_ip': conn_key[0],
                'src_port': conn_key[1],
                'dst_ip': conn_key[2],
                'dst_port': conn_key[3],
                'finished': False
            }
        
        conn = self.connections[conn_key]
        conn['last_time'] = time.time()
        conn['packet_count'] += 1
        conn['byte_count'] += len(packet)
        
        # 更新流量统计
        if direction == 'forward':
            conn['src_bytes'] += len(packet)
        else:
            conn['dst_bytes'] += len(packet)
            
        # 更新TCP标志
        if TCP in packet:
            # 确保获取到的是字符串格式的 flags
            # scapy 的 flags 可能是 FlagValue 对象
            flags_val = packet[TCP].flags
            flags_str = str(flags_val)
            
            # 某些版本可能返回整数，强制转换
            if isinstance(flags_val, int):
                # 这里简化处理，如果无法转字符串，至少记录 S/R/F/A
                # 实际上 scapy 通常处理得很好
                pass
                
            for f in flags_str:
                conn['flags'].add(f)
            
            # 检查是否结束连接 (FIN or RST)
            if 'F' in flags_str or 'R' in flags_str:
                conn['finished'] = True
        
        conn['packets'].append((time.time(), packet, direction))
        
        return conn_key
    
    def get_ready_connections(self):
        """获取准备好的连接（超过时间窗口 或 已结束）"""
        current_time = time.time()
        ready = []
        
        for conn_key, conn in list(self.connections.items()):
            # 如果超过时间窗口 OR 连接已结束
            if (current_time - conn['start_time'] >= self.window_size) or conn.get('finished', False):
                # 计算持续时间
                conn['duration'] = conn['last_time'] - conn['start_time']
                ready.append((conn_key, conn))
                del self.connections[conn_key]
        
        return ready
