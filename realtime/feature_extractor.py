"""
实时特征提取模块
负责将捕获的网络连接转换为KDD Cup 99数据集格式的41个特征
"""
import numpy as np
from collections import deque
import time

class FeatureExtractor:
    """
    从网络连接中提取KDD Cup 99特征
    """
    def __init__(self):
        # 历史连接记录
        # history_2s: 用于计算 count, srv_count 等 (2秒窗口)
        self.history_2s = deque()
        # history_100: 用于计算 dst_host_* 等 (最后100个连接)
        self.history_100 = deque(maxlen=100)
        
        # 服务映射 (常见端口到服务名)
        self.port_service_map = {
            80: 'http', 443: 'http', 8080: 'http',
            21: 'ftp', 20: 'ftp_data',
            22: 'ssh', 23: 'telnet',
            25: 'smtp', 53: 'domain_u',
            110: 'pop_3', 143: 'imap4',
            123: 'ntp', 445: 'smb', 3306: 'mysql',
            6379: 'redis', 27017: 'mongodb'
        }
        
    def extract(self, connection):
        """
        提取单条连接的特征
        
        Args:
            connection: 字典，包含连接信息
                - start_time: 开始时间戳
                - protocol: 协议 (tcp, udp, icmp)
                - src_ip, src_port, dst_ip, dst_port
                - src_bytes, dst_bytes
                - flags: TCP标志集合 (list or set)
                - duration: 持续时间
                - wrong_fragment, urgent (可选)
        
        Returns:
            dict: 包含41个特征的字典
        """
        # 1. 基础特征 (Intrinsic Features)
        features = {}
        
        features['duration'] = int(connection.get('duration', 0))
        features['protocol_type'] = connection.get('protocol', 'tcp')
        features['service'] = self._get_service(connection)
        features['flag'] = self._get_flag(connection)
        features['src_bytes'] = int(connection.get('src_bytes', 0))
        features['dst_bytes'] = int(connection.get('dst_bytes', 0))
        features['land'] = 1 if connection['src_ip'] == connection['dst_ip'] and \
                              connection['src_port'] == connection['dst_port'] else 0
        features['wrong_fragment'] = int(connection.get('wrong_fragment', 0))
        features['urgent'] = int(connection.get('urgent', 0))
        
        # 2. 内容特征 (Content Features) - 简化处理，默认为0
        # 在真实环境中，这些需要深度包检测(DPI)
        features['hot'] = 0
        features['num_failed_logins'] = 0
        features['logged_in'] = 0 
        features['num_compromised'] = 0
        features['root_shell'] = 0
        features['su_attempted'] = 0
        features['num_root'] = 0
        features['num_file_creations'] = 0
        features['num_shells'] = 0
        features['num_access_files'] = 0
        features['num_outbound_cmds'] = 0
        features['is_host_login'] = 0
        features['is_guest_login'] = 0
        
        # 3. 基于时间的流量特征 (Time-based Traffic Features)
        # 更新历史记录
        self._update_history(connection)
        
        # 计算统计量
        count_stats = self._calc_time_based_stats(connection)
        features.update(count_stats)
        
        # 4. 基于主机的流量特征 (Host-based Traffic Features)
        host_stats = self._calc_host_based_stats(connection)
        features.update(host_stats)
        
        return features

    def _get_service(self, conn):
        dst_port = conn.get('dst_port', 0)
        if dst_port in self.port_service_map:
            return self.port_service_map[dst_port]
        # 简单启发式
        if dst_port > 1024:
            return 'private'
        return 'other'
    
    def _get_flag(self, conn):
        """
        根据TCP标志推断连接状态
        """
        flags = set(conn.get('flags', []))
        protocol = conn.get('protocol', 'tcp')
        
        if protocol != 'tcp':
            return 'SF' # UDP/ICMP 通常视为正常
            
        # 调试：打印flags
        # print(f"DEBUG: flags={flags}")
            
        # REJ: Connection attempt rejected (RST seen)
        if 'R' in flags:
            return 'REJ'
            
        # SF: Normal establishment and termination (SYN, FIN, ACK)
        if 'S' in flags and 'F' in flags:
            return 'SF'
            
        # S0: Connection attempt seen, no reply (Only SYN)
        # 放宽条件：只要有 S，没有 F，没有 A (或者只有 S)
        if 'S' in flags and 'F' not in flags and 'A' not in flags:
            return 'S0'
            
        # S1: Connection established, not terminated (SYN, ACK, no FIN)
        if 'S' in flags and 'A' in flags and 'F' not in flags:
            return 'S1'
            
        # 如果只有 S (防止上面的条件漏掉)
        if flags == {'S'}:
            return 'S0'
        
        return 'SF' # 默认
        
    def _update_history(self, current_conn):
        current_time = current_conn['start_time']
        
        # 确保 history_2s 按时间排序 (理论上应该是，但为了保险)
        # 如果乱序，popleft 可能移除错误的
        
        # 清理2秒前的记录
        # 注意：如果 current_time 比 history 中最早的还早（乱序），可能导致问题
        # 这里假设时间是单调递增的
        
        while self.history_2s and (current_time - self.history_2s[0]['start_time'] > 2.0):
            self.history_2s.popleft()
            
        # 添加当前连接
        self.history_2s.append(current_conn)
        self.history_100.append(current_conn)
        
        # 调试：打印历史记录大小
        # if len(self.history_2s) > 10:
        #    print(f"DEBUG: History size: {len(self.history_2s)}")
        
    def _calc_time_based_stats(self, curr):
        """计算过去2秒内的统计特征"""
        stats = {}
        
        # count: 相同目标主机的连接数
        same_dst = [c for c in self.history_2s if c['dst_ip'] == curr['dst_ip']]
        stats['count'] = len(same_dst)
        
        # srv_count: 相同服务的连接数
        curr_service = self._get_service(curr)
        same_srv = [c for c in self.history_2s if self._get_service(c) == curr_service]
        stats['srv_count'] = len(same_srv)
        
        # serror_rate: 相同目标主机的连接中出现SYN错误的比例 (S0, S1, S2, S3)
        # rerror_rate: 相同目标主机的连接中出现REJ错误的比例 (REJ)
        serror_count = sum(1 for c in same_dst if self._get_flag(c) in ['S0', 'S1', 'S2', 'S3'])
        rerror_count = sum(1 for c in same_dst if self._get_flag(c) == 'REJ')
        
        stats['serror_rate'] = serror_count / len(same_dst) if same_dst else 0.0
        stats['rerror_rate'] = rerror_count / len(same_dst) if same_dst else 0.0
        
        # srv_serror_rate: 相同服务的连接中出现SYN错误的比例
        # srv_rerror_rate: 相同服务的连接中出现REJ错误的比例
        srv_serror_count = sum(1 for c in same_srv if self._get_flag(c) in ['S0', 'S1', 'S2', 'S3'])
        srv_rerror_count = sum(1 for c in same_srv if self._get_flag(c) == 'REJ')
        
        stats['srv_serror_rate'] = srv_serror_count / len(same_srv) if same_srv else 0.0
        stats['srv_rerror_rate'] = srv_rerror_count / len(same_srv) if same_srv else 0.0
        
        # same_srv_rate: 相同目标主机的连接中，服务相同的比例
        same_srv_dst = sum(1 for c in same_dst if self._get_service(c) == curr_service)
        stats['same_srv_rate'] = same_srv_dst / len(same_dst) if same_dst else 0.0
        
        # diff_srv_rate: 相同目标主机的连接中，服务不同的比例
        stats['diff_srv_rate'] = 1.0 - stats['same_srv_rate']
        
        # srv_diff_host_rate: 相同服务的连接中，目标主机不同的比例
        diff_host_srv = sum(1 for c in same_srv if c['dst_ip'] != curr['dst_ip'])
        stats['srv_diff_host_rate'] = diff_host_srv / len(same_srv) if same_srv else 0.0
        
        return stats
        
    def _calc_host_based_stats(self, curr):
        """计算过去100个连接内的统计特征"""
        stats = {}
        
        # dst_host_count: 具有相同目标主机的连接数
        same_dst = [c for c in self.history_100 if c['dst_ip'] == curr['dst_ip']]
        # 限制最大值为255 (KDD数据集特征)
        stats['dst_host_count'] = min(len(same_dst), 255)
        
        # dst_host_srv_count: 具有相同目标主机和相同服务的连接数
        curr_service = self._get_service(curr)
        same_dst_srv = [c for c in same_dst if self._get_service(c) == curr_service]
        stats['dst_host_srv_count'] = min(len(same_dst_srv), 255)
        
        # dst_host_same_srv_rate
        stats['dst_host_same_srv_rate'] = len(same_dst_srv) / len(same_dst) if same_dst else 0.0
        
        # dst_host_diff_srv_rate
        diff_srv = len(same_dst) - len(same_dst_srv)
        stats['dst_host_diff_srv_rate'] = diff_srv / len(same_dst) if same_dst else 0.0
        
        # dst_host_same_src_port_rate: 相同目标主机的连接中，源端口相同的比例
        same_src_port = sum(1 for c in same_dst if c['src_port'] == curr['src_port'])
        stats['dst_host_same_src_port_rate'] = same_src_port / len(same_dst) if same_dst else 0.0
        
        # dst_host_srv_diff_host_rate: 相同目标主机相同服务的连接中，源主机不同的比例
        # 注意：KDD定义可能略有不同，这里按字面理解
        diff_src_host = sum(1 for c in same_dst_srv if c['src_ip'] != curr['src_ip'])
        stats['dst_host_srv_diff_host_rate'] = diff_src_host / len(same_dst_srv) if same_dst_srv else 0.0
        
        # dst_host_serror_rate
        serror_count = sum(1 for c in same_dst if self._get_flag(c) in ['S0', 'S1', 'S2', 'S3'])
        stats['dst_host_serror_rate'] = serror_count / len(same_dst) if same_dst else 0.0
        
        # dst_host_srv_serror_rate
        srv_serror_count = sum(1 for c in same_dst_srv if self._get_flag(c) in ['S0', 'S1', 'S2', 'S3'])
        stats['dst_host_srv_serror_rate'] = srv_serror_count / len(same_dst_srv) if same_dst_srv else 0.0
        
        # dst_host_rerror_rate
        rerror_count = sum(1 for c in same_dst if self._get_flag(c) == 'REJ')
        stats['dst_host_rerror_rate'] = rerror_count / len(same_dst) if same_dst else 0.0
        
        # dst_host_srv_rerror_rate
        srv_rerror_count = sum(1 for c in same_dst_srv if self._get_flag(c) == 'REJ')
        stats['dst_host_srv_rerror_rate'] = srv_rerror_count / len(same_dst_srv) if same_dst_srv else 0.0
        
        return stats
