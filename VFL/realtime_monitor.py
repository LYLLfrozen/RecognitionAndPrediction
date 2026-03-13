#!/usr/bin/env python3
"""
实时网络流量监测和识别系统（GUI版本）
使用训练好的VFL模型实时识别网络流量类型
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import sys
import time
from datetime import datetime
import threading
import queue
from collections import deque, Counter
from typing import TYPE_CHECKING
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 设置中文字体支持
import platform
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'STHeiti', 'SimHei']
elif platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 条件导入 scapy
try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP  # type: ignore
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("警告: scapy未安装，将使用模拟数据")
    # 为类型检查定义占位符
    if TYPE_CHECKING:
        from typing import Any
        IP = TCP = UDP = ICMP = Any  # type: ignore
        def sniff(*args, **kwargs): pass  # type: ignore

# 导入VFL模块
from federated_learning.vfl_server import VFLServer
from federated_learning.vfl_client import create_vfl_parties
from federated_learning.vfl_utils import create_vfl_model_split, split_features_for_cnn

# 导入流量跟踪器和混合检测器
from flow_tracker import FlowTracker
from hybrid_detector import HybridAttackDetector

# 设置
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else "cpu")

# 全局配置
MODEL_DIR = 'models/vfl_network'
DATA_DIR = 'data/processed_data'
MONITOR_WINDOW = 100  # 监控窗口大小（最近N个样本）
UPDATE_INTERVAL = 2   # 更新间隔（秒）
CAPTURE_INTERFACE = None  # None表示捕获所有接口
CAPTURE_COUNT = 1  # 每次捕获的包数量


class VFLFlowClassifier:
    """VFL流量分类器"""
    
    def __init__(self, model_dir, device):
        """初始化分类器"""
        self.device = device
        self.model_dir = model_dir
        
        # 加载配置
        with open(os.path.join(model_dir, 'config.pkl'), 'rb') as f:
            self.config = pickle.load(f)
        
        # 加载数据处理器
        with open(os.path.join(DATA_DIR, 'processor.pkl'), 'rb') as f:
            self.processor = pickle.load(f)
        
        self.class_names = self.config['class_names']
        self.num_parties = self.config['num_parties']
        self.shapes = self.config['shapes']
        
        # 创建并加载模型
        self._load_models()
        
        print(f"✓ VFL分类器已加载")
        print(f"  参与方数: {self.num_parties}")
        print(f"  类别: {self.class_names}")
    
    def _load_models(self):
        """加载VFL模型"""
        # 创建模型架构
        bottom_models, top_model = create_vfl_model_split(
            self.num_parties, self.shapes, num_classes=len(self.class_names)
        )
        
        # 加载权重
        top_model.load_state_dict(
            torch.load(os.path.join(self.model_dir, 'top_model.pth'),
                      map_location=self.device)
        )
        
        for i, model in enumerate(bottom_models):
            model.load_state_dict(
                torch.load(os.path.join(self.model_dir, f'bottom_model_party{i+1}.pth'),
                          map_location=self.device)
            )
        
        # 设置为评估模式
        top_model.eval()
        for model in bottom_models:
            model.eval()
        
        self.bottom_models = [m.to(self.device) for m in bottom_models]
        self.top_model = top_model.to(self.device)
    
    def preprocess_flow(self, flow_data):
        """
        预处理流量数据（优化版 - 支持增强特征）
        
        Args:
            flow_data: 原始流量特征（41维增强包特征 或 121维训练格式）
        
        Returns:
            预处理后的数据（1, 1, 11, 11）
        """
        # 确保是numpy数组
        if not isinstance(flow_data, np.ndarray):
            flow_data = np.array(flow_data)
        
        # 检查输入维度
        original_dim = flow_data.shape[-1]
        
        # 如果是41维（增强的真实包特征），映射到 KDD Cup 99 的115维格式再用训练scaler归一化
        if original_dim == 41:
            # ── 解析41维特征向量中的各字段 ──
            protocol   = int(flow_data[1])
            src_port   = int(flow_data[2])
            dst_port   = int(flow_data[3])
            tcp_flags  = int(flow_data[4])
            # flow_tracker 写入的流统计（features_to_vector 中已归一化，此处还原）
            duration       = float(flow_data[13])
            src_bytes      = float(flow_data[14]) * 10000.0
            same_dst_count = float(flow_data[16]) * 100.0
            same_srv_count = float(flow_data[17]) * 100.0
            serror_rate    = float(flow_data[18])
            rerror_rate    = float(flow_data[19])
            same_srv_rate  = float(flow_data[20])
            diff_srv_rate  = float(flow_data[21])

            # TCP flags 各位
            syn = bool(tcp_flags & 0x02)
            ack = bool(tcp_flags & 0x10)
            fin = bool(tcp_flags & 0x01)
            rst = bool(tcp_flags & 0x04)
            psh = bool(tcp_flags & 0x08)

            kdd = np.zeros(115)

            # ── 基础连接特征（KDD 0-18）──
            kdd[0] = duration    # duration
            kdd[1] = src_bytes   # src_bytes
            # kdd[2] dst_bytes = 0（反向流量未跟踪）
            # kdd[3:8] = 0（land / wrong_fragment / urgent / hot / num_failed_logins）
            # logged_in：已建立连接（ACK且非纯SYN握手）
            login_ports = {21, 22, 23, 513, 514}
            service_port_for_login = dst_port if dst_port < 32768 else src_port
            kdd[8] = 1 if (ack and not syn and service_port_for_login in login_ports) else 0
            # kdd[9:19] = 0（高层语义特征无法从原始包获得）

            # ── 时间窗口统计特征（KDD 19-37）──
            kdd[19] = same_dst_count   # count
            kdd[20] = same_srv_count   # srv_count
            kdd[21] = serror_rate      # serror_rate
            kdd[22] = serror_rate      # srv_serror_rate（近似）
            kdd[23] = rerror_rate      # rerror_rate
            kdd[24] = rerror_rate      # srv_rerror_rate（近似）
            kdd[25] = same_srv_rate    # same_srv_rate
            kdd[26] = diff_srv_rate    # diff_srv_rate
            # kdd[27] = 0  srv_diff_host_rate（未跟踪）
            kdd[28] = same_dst_count   # dst_host_count
            kdd[29] = same_srv_count   # dst_host_srv_count
            kdd[30] = same_srv_rate    # dst_host_same_srv_rate
            kdd[31] = diff_srv_rate    # dst_host_diff_srv_rate
            # kdd[32:34] = 0
            kdd[34] = serror_rate      # dst_host_serror_rate
            kdd[35] = serror_rate      # dst_host_srv_serror_rate
            kdd[36] = rerror_rate      # dst_host_rerror_rate
            kdd[37] = rerror_rate      # dst_host_srv_rerror_rate

            # ── 协议类型 one-hot（38-39，icmp=两者均0）──
            kdd[38] = 1 if protocol == 6  else 0  # protocol_type_tcp
            kdd[39] = 1 if protocol == 17 else 0  # protocol_type_udp

            # ── 服务类型 one-hot（40-104）──
            # 端口→KDD服务索引映射（仅常见端口）
            _SVC_MAP = {
                20: 58, 21: 57, 22: 91, 23: 95, 25: 89,
                37: 98, 42: 71, 43: 104, 53: 49, 70: 59,
                79: 56, 80: 61, 101: 60, 105: 45, 109: 81,
                110: 82, 111: 92, 113: 42, 119: 77, 143: 63,
                179: 43, 389: 67, 433: 76, 443: 62, 512: 55,
                513: 69, 514: 88, 515: 83, 530: 44, 540: 101,
                543: 65, 544: 66, 6000: 40,
            }
            if protocol == 1:  # ICMP：type 储存在 src_port 位置
                kdd[52 if src_port == 8 else 53] = 1  # eco_i(8) 或 ecr_i(0)
            else:
                # 使用非临时端口一侧作为服务端口
                svc_port = dst_port if dst_port < 32768 else src_port
                if protocol == 17 and svc_port == 53:
                    kdd[50] = 1  # domain_u（UDP DNS）
                else:
                    svc_idx = _SVC_MAP.get(svc_port, 84 if svc_port > 1024 else 79)
                    kdd[svc_idx] = 1

            # ── 连接flag one-hot（105-114）──
            # 优先使用流级统计（positions 22-26），这比单包tcp_flags更准确：
            #   - SYN flood：每条流仅有SYN(ack_c=0)  → S0，与KDD99 neptune特征一致
            #   - 正常连接：SYN+ACK到达后ack_c>0，PSH传数据后psh_c>0 → SF
            #   - 单包per-packet判断会把每个正常SYN都错误映射为S0
            flow_syn_c = round(float(flow_data[22]) * 10.0) if len(flow_data) > 22 else 0
            flow_fin_c = round(float(flow_data[23]) * 10.0) if len(flow_data) > 23 else 0
            flow_rst_c = round(float(flow_data[24]) * 10.0) if len(flow_data) > 24 else 0
            flow_psh_c = round(float(flow_data[25]) * 10.0) if len(flow_data) > 25 else 0
            flow_ack_c = round(float(flow_data[26]) * 10.0) if len(flow_data) > 26 else 0

            # 判断是否有有效的流级统计（非零数据则使用流级判断）
            has_flow_stats = (flow_syn_c + flow_fin_c + flow_rst_c + flow_psh_c + flow_ack_c) > 0

            if has_flow_stats:
                # 流级 KDD flag 映射（与KDD99定义完全对齐）
                if flow_rst_c > 0:
                    if flow_syn_c > 0 and flow_ack_c == 0:
                        kdd[107] = 1  # RSTOS0：SYN已发但收到RST且无ACK
                    elif (flow_syn_c > 0 and flow_ack_c > 0 and
                          same_dst_count >= 20 and serror_rate >= 0.6 and diff_srv_rate <= 0.3):
                        # 本地回环DoS：TCP栈回RST|ACK导致ack_c>0，但统计特征符合SYN flood
                        # （同一目标/服务的高失败率），与KDD99 neptune(S0)特征对齐
                        kdd[109] = 1  # S0：DoS洪泛
                    else:
                        kdd[105] = 1  # REJ：真实端口扫描/连接拒绝
                elif flow_syn_c > 0 and flow_ack_c == 0 and flow_psh_c == 0 and flow_fin_c == 0:
                    kdd[109] = 1   # S0：SYN已发，从未收到ACK响应（SYN flood典型特征）
                elif flow_psh_c > 0 or flow_fin_c > 0:
                    kdd[113] = 1   # SF：有数据传输或正常关闭
                elif flow_syn_c > 0 and flow_ack_c > 0:
                    kdd[110] = 1   # S1：握手中（收到SYN+ACK但无数据）
                else:
                    kdd[113] = 1   # SF：默认（纯ACK流量）
            else:
                # 回退：使用单包tcp_flags（仅在flow_tracker统计不可用时）
                if rst:
                    kdd[107 if (syn and not ack) else 105] = 1  # RSTOS0 或 REJ
                elif syn and not ack and not fin:
                    kdd[109] = 1   # S0
                elif syn and ack and not fin and not rst:
                    kdd[110] = 1   # S1
                else:
                    kdd[113] = 1   # SF

            flow_data = kdd

            # 应用训练时的 StandardScaler，使特征分布与训练数据一致
            scaler = (self.processor.get('scaler')
                      if isinstance(self.processor, dict)
                      else getattr(self.processor, 'scaler', None))
            if scaler is not None:
                try:
                    flow_data = scaler.transform(flow_data.reshape(1, -1))[0]
                except Exception:
                    pass

            # 填充到121维
            padding = np.zeros(121 - 115)
            flow_data = np.concatenate([flow_data, padding])
        
        # 如果是115维（训练格式），使用 scaler 并填充到121
        elif original_dim == 115:
            scaler = None
            if isinstance(self.processor, dict):
                scaler = self.processor.get('scaler', None)
            else:
                scaler = getattr(self.processor, 'scaler', None)
            
            if scaler is not None:
                try:
                    flow_data = scaler.transform(flow_data.reshape(1, -1))[0]
                except Exception as e:
                    print(f"⚠️ scaler.transform 失败: {e}")
            
            # 填充到121维
            padding = np.zeros(121 - 115)
            flow_data = np.concatenate([flow_data, padding])
        
        # 如果已经是121维或更大，直接截取
        elif original_dim >= 121:
            flow_data = flow_data[:121]
        
        # 其他维度，填充到121
        else:
            padding = np.zeros(121 - original_dim)
            flow_data = np.concatenate([flow_data, padding])
        
        # 重塑为图像格式 (1, 1, 11, 11)
        flow_data = flow_data.reshape(1, 1, 11, 11)
        
        return flow_data.astype(np.float32)
    
    def classify(self, flow_data):
        """
        分类单个流量
        
        Args:
            flow_data: 预处理后的流量数据
        
        Returns:
            (预测类别, 置信度, 所有概率)
        """
        with torch.no_grad():
            # 确保维度正确 (batch, channel, height, width)
            if len(flow_data.shape) == 3:
                flow_data = flow_data[np.newaxis, :]  # 添加batch维度
            
            # 垂直分割数据
            X_parties, _ = split_features_for_cnn(flow_data, self.num_parties)
            
            # 各方计算嵌入
            embeddings = []
            for i, model in enumerate(self.bottom_models):
                X_tensor = torch.FloatTensor(X_parties[i]).to(self.device)
                emb = model(X_tensor)
                embeddings.append(emb)
            
            # 聚合（不使用隐私保护以提高速度）
            combined = torch.cat(embeddings, dim=-1)
            
            # 顶层预测
            outputs = self.top_model(combined)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = probs.max(1)
            
            pred_class = self.class_names[predicted.item()]
            conf_value = confidence.item()
            all_probs = probs.cpu().numpy()[0]
            
            return pred_class, conf_value, all_probs


class RealTimeFlowMonitor:
    """实时流量监控器"""
    
    def __init__(self, classifier, use_real_traffic=True, interface=None):
        self.classifier = classifier
        self.flow_queue = queue.Queue()
        self.recent_predictions = deque(maxlen=MONITOR_WINDOW)
        self.running = False
        self.use_real_traffic = use_real_traffic and SCAPY_AVAILABLE
        self.interface = interface  # 网络接口
        
        # 统计数据
        self.total_flows = 0
        self.class_counts = Counter()
        self.start_time = time.time()
        self.captured_packets = 0
        
        # 流量跟踪器 - 用于计算统计特征
        self.flow_tracker = FlowTracker(window_time=2.0, window_count=100)
        
        # 混合检测器 - 结合规则和ML
        self.hybrid_detector = HybridAttackDetector(classifier, self.flow_tracker)
    
    def extract_flow_features(self, packet):
        """
        从网络包提取特征（增强版 - 包含流统计）
        返回 (41维基础特征向量, packet_info字典用于流跟踪)
        
        特征说明：
        0: 包长度
        1: 协议类型 (6=TCP, 17=UDP, 1=ICMP)
        2: 源端口
        3: 目标端口
        4: TCP flags (仅TCP)
        5: TTL
        6: IP总长度
        7-10: 连接状态相关
        11-40: 流量统计特征（通过FlowTracker填充）
        """
        features = np.zeros(41)
        packet_info = {}  # 用于流跟踪
        
        try:
            if IP in packet:
                # 提取IP地址
                packet_info['src_ip'] = packet[IP].src
                packet_info['dst_ip'] = packet[IP].dst
                packet_info['timestamp'] = time.time()
                
                # 基本网络层特征
                ip_len = len(packet)
                features[0] = min(ip_len, 65535)  # 包长度，限制最大值
                features[5] = packet[IP].ttl  # TTL
                features[6] = packet[IP].len  # IP头中的总长度
                packet_info['packet_size'] = ip_len
                
                # 传输层特征
                if TCP in packet:
                    features[1] = 6  # TCP协议
                    features[2] = packet[TCP].sport % 65536  # 源端口
                    features[3] = packet[TCP].dport % 65536  # 目标端口
                    packet_info['protocol'] = 6
                    packet_info['src_port'] = features[2]
                    packet_info['dst_port'] = features[3]
                    
                    # TCP flags (转换为数值)
                    flags = packet[TCP].flags
                    if hasattr(flags, 'value'):
                        features[4] = flags.value
                        packet_info['tcp_flags'] = flags.value
                    else:
                        # 手动计算flags值
                        flag_val = 0
                        if 'F' in str(flags): flag_val |= 0x01  # FIN
                        if 'S' in str(flags): flag_val |= 0x02  # SYN
                        if 'R' in str(flags): flag_val |= 0x04  # RST
                        if 'P' in str(flags): flag_val |= 0x08  # PSH
                        if 'A' in str(flags): flag_val |= 0x10  # ACK
                        if 'U' in str(flags): flag_val |= 0x20  # URG
                        features[4] = flag_val
                        packet_info['tcp_flags'] = flag_val
                    
                    # 序列号和确认号（归一化）
                    features[7] = (packet[TCP].seq % 100000) / 100000.0
                    features[8] = (packet[TCP].ack % 100000) / 100000.0
                    
                    # 窗口大小
                    features[9] = packet[TCP].window / 65535.0
                    
                elif UDP in packet:
                    features[1] = 17  # UDP协议
                    features[2] = packet[UDP].sport % 65536
                    features[3] = packet[UDP].dport % 65536
                    features[7] = packet[UDP].len / 65535.0  # UDP长度
                    packet_info['protocol'] = 17
                    packet_info['src_port'] = features[2]
                    packet_info['dst_port'] = features[3]
                    
                elif ICMP in packet:
                    features[1] = 1  # ICMP协议
                    features[2] = packet[ICMP].type if hasattr(packet[ICMP], 'type') else 0
                    features[3] = packet[ICMP].code if hasattr(packet[ICMP], 'code') else 0
                    packet_info['protocol'] = 1
                
                # IP层其他特征
                features[10] = packet[IP].tos  # Type of Service
                features[11] = packet[IP].id % 65536  # IP标识
                
                # 负载大小
                if hasattr(packet, 'payload'):
                    payload_len = len(bytes(packet.payload))
                    features[12] = min(payload_len, 65535)
                
        except Exception as e:
            # 如果提取失败，返回零特征向量
            # 在生产环境中可以记录日志
            pass
        
        return features, packet_info
    
    
    def real_flow_capture(self):
        """
        捕获真实网络流量
        """
        print("\n✓ 开始捕获真实网络流量...")
        interface = self.interface if self.interface else CAPTURE_INTERFACE
        print(f"  接口: {'所有接口' if interface is None else interface}")
        print("  提示: 需要root权限才能捕获网络包")
        print("  建议: 在另一个终端生成流量以测试检测功能")
        print("       python3 generate_test_traffic.py")
        print("-" * 80)
        
        # 统计原始捕获的包数
        raw_packet_count = 0
        error_count = 0
        # 用于无包超时回退
        no_packet_seconds = 0
        last_captured = self.captured_packets
        
        def packet_handler(packet):
            nonlocal raw_packet_count, error_count
            
            if not self.running:
                return
            
            raw_packet_count += 1
            
            # 每100个包显示一次进度
            if raw_packet_count % 100 == 0:
                print(f"  已捕获 {raw_packet_count} 个原始包，成功处理 {self.captured_packets} 个")
            
            try:
                # 提取基础特征和包信息
                features, packet_info = self.extract_flow_features(packet)

                # 只在完全无法提取特征时跳过包
                if features is None or not packet_info:
                    return
                
                # 更新流跟踪器并获取统计特征
                flow_stats = self.flow_tracker.update(packet_info)
                
                # 将流统计特征合并到基础特征
                enhanced_features = self.flow_tracker.features_to_vector(flow_stats, features)

                # 预处理为模型输入格式
                flow_data = self.classifier.preprocess_flow(enhanced_features)
                
                # 放入队列（真实流量没有真实标签）
                # 在捕获最初几个包时打印诊断信息
                if self.captured_packets < 5:
                    try:
                        # 打印原始特征摘要
                        print(f"[诊断] 基础特征: min={features.min():.3f}, max={features.max():.3f}")
                        print(f"[诊断] 流统计: same_dst={flow_stats.get('same_dst_count', 0)}, serror_rate={flow_stats.get('serror_rate', 0):.3f}")
                        print(f"[诊断] 增强特征: min={enhanced_features.min():.3f}, max={enhanced_features.max():.3f}")
                        # 直接计算模型输出
                        pred_class, confidence, all_probs = self.classifier.classify(flow_data)
                        print(f"[诊断] 模型预测: {pred_class}, confidence={confidence:.4f}")
                        print(f"[诊断] 所有概率: {', '.join([f'{self.classifier.class_names[i]}={all_probs[i]:.3f}' for i in range(len(all_probs))])}")
                    except Exception as e:
                        print(f"[诊断] 快速预测失败: {e}")

                self.flow_queue.put((flow_data, -1, self.captured_packets, enhanced_features, packet_info, flow_stats))
                self.captured_packets += 1
                
            except Exception as e:
                error_count += 1
                if error_count <= 5:  # 只显示前5个错误
                    print(f"  ⚠️  处理包时出错: {e}")
        
        try:
            # 修正 Windows 下常见接口别名
            if os.name == 'nt':
                if interface == 'lo0' or interface == 'lo':
                    print("  正在查找 Windows Loopback 适配器...")
                    try:
                        from scapy.arch.windows import get_windows_if_list
                        win_if_list = get_windows_if_list()
                        for iface in win_if_list:
                            if 'loopback' in iface['name'].lower() or \
                               'loopback' in iface['description'].lower():
                                interface = iface['name']
                                print(f"  >>> 自动映射 lo0 -> {interface}")
                                break
                    except:
                        pass
            print(f"\n🔍 正在监听接口 {interface or '所有接口'}...")
            print("   等待网络流量中...")

            # 显示更多诊断信息
            try:
                from scapy.all import get_if_list, conf
                if os.name == 'nt':
                    print("  正在获取Windows网络接口列表...")
                    # 在Windows上尝试显示更友好的名称
                    try:
                        from scapy.arch.windows import get_windows_if_list
                        win_if_list = get_windows_if_list()
                        print("\n  可用网络接口:")
                        target_interface = interface
                        for i, iface in enumerate(win_if_list):
                            desc = f"{iface['name']} - {iface['description']}"
                            print(f"  [{i}] {desc}")
                            # 尝试匹配用户输入的接口名（如果只给了部分名称），仅作提示，不修改interface变量以免影响后续逻辑
                            if target_interface and (target_interface.lower() in iface['name'].lower() or 
                                            target_interface.lower() in iface['description'].lower()):
                                print(f"  >>> (提示) 匹配到接口: {iface['name']}")

                    except ImportError:
                        if_list = get_if_list()
                        print(f"  可用网络接口(GUID): {', '.join(if_list)}")
                else:
                    if_list = get_if_list()
                    print(f"  可用网络接口: {', '.join(if_list)}")
            except Exception as e:
                print(f"  获取接口列表失败: {e}")

            # 在Windows上尝试更智能的接口匹配
            if os.name == 'nt' and interface:
                try:
                    from scapy.arch.windows import get_windows_if_list
                    win_if_list = get_windows_if_list()
                    matched = False
                    
                    # 1. 优先匹配非虚拟接口
                    # 先按照精确名称查找
                    candidates = []
                    for iface in win_if_list:
                        if interface.lower() == iface['name'].lower():
                            candidates = [iface]
                            break
                    
                    # 模糊匹配
                    if not candidates:
                        temp_candidates = []
                        for iface in win_if_list:
                            # 排除明显的虚拟接口/过滤器/Loopback，除非用户明确指定
                            desc_lower = iface['description'].lower()
                            is_virtual = 'loopback' in desc_lower or \
                                         'tap-' in desc_lower or \
                                         'virtual' in desc_lower or \
                                         'wfp' in desc_lower or \
                                         'packet driver' in desc_lower
                            
                            # 名字、描述或GUID匹配
                            match = (interface.lower() in iface['name'].lower() or \
                                     interface.lower() in iface['description'].lower() or \
                                     interface.lower() in iface['guid'].lower())
                            
                            if match:
                                temp_candidates.append((iface, is_virtual))
                        
                        # 选择最佳匹配
                        if temp_candidates:
                            # 优先选择非虚拟接口
                            real_ifaces = [c[0] for c in temp_candidates if not c[1]]
                            if real_ifaces:
                                best_iface = real_ifaces[0]
                            else:
                                # 只有虚拟接口匹配
                                best_iface = temp_candidates[0][0]
                            candidates = [best_iface]

                    if candidates:
                        best_iface = candidates[0]
                        print(f"\n  >>> 自动匹配到Windows接口: {best_iface['name']} ({best_iface['description']})")
                        interface = best_iface['name']
                        matched = True
                    
                    if not matched:
                        print(f"\n  ⚠️ 未找到包含 '{interface}' 的接口，将尝试默认接口")
                        # 列出可用接口供用户参考
                        print("  可用接口列表:")
                        for i, iface in enumerate(win_if_list):
                            print(f"    {i}. {iface['name']} ({iface['description']})")
                except ImportError:
                    pass

            # 循环调用 sniff，设置短超时以便检查是否长时间无包
            sniff_timeout = 5
            max_no_packet = 10
            while self.running:
                # 在Windows上如果没有WinPcap/Npcap，可能无法进行L2捕获
                # 尝试使用L3捕获
                try:
                    sniff(iface=interface,
                          prn=packet_handler,
                          filter=None,  # 移除过滤器，捕获所有包
                          store=False,
                          timeout=sniff_timeout)
                except (OSError, RuntimeError) as e:
                    # Catch both OSError (file not found/permission) and RuntimeError (scapy layer 2 unavailable)
                    err_msg = str(e).lower()
                    if "winpcap" in err_msg or "layer 2" in err_msg or "pcap" in err_msg:
                        print("\n⚠️  WinPcap未安装或L2不可用，尝试使用L3 Socket...")
                        from scapy.all import conf
                        conf.L3socket = conf.L3socket
                        
                        # L3捕获通常不需要指定复杂接口名，尝试留空让其自动选择或使用简单名称
                        # 或者尝试传入 None (监听所有)
                        l3_interface = interface
                        if os.name == 'nt' and interface and "filter" in interface.lower():
                             # Windows下WFP过滤器接口通常不支持L3 Socket绑定
                             print(f"  提示: 接口 '{interface}' 可能是WFP过滤器，L3模式下将尝试自动选择最佳接口")
                             l3_interface = None
                             
                        try:
                            sniff(iface=l3_interface,
                                  prn=packet_handler,
                                  filter=None, # 移除过滤器
                                  store=False,
                                  timeout=sniff_timeout,
                                  L2socket=conf.L3socket)
                        except Exception as l3_err:
                            print(f"\n❌ L3捕获也失败: {l3_err}")
                            print("  提示: 请尝试以管理员身份运行，或安装 Npcap (https://npcap.com/)")
                            raise l3_err
                    else:
                        raise e

                # 检查是否有包到达
                if self.captured_packets == last_captured:
                    no_packet_seconds += sniff_timeout
                    if no_packet_seconds >= max_no_packet:
                        print("\n⚠️  长时间未捕获到包，启用模拟回退模式（测试集）...")
                        # 切换到模拟捕获（在当前线程中运行）
                        self.simulate_flow_capture()
                        return
                else:
                    no_packet_seconds = 0
                    last_captured = self.captured_packets
        except PermissionError:
            print("\n❌ 错误: 需要root权限捕获网络流量")
            print("   请使用: sudo python3 realtime_monitor.py")
            print("   或切换到模拟模式")
            self.running = False
        except Exception as e:
            print(f"\n❌ 流量捕获错误: {e}")
            import traceback
            traceback.print_exc()
            self.running = False
    
    def simulate_flow_capture(self):
        """
        模拟流量捕获（使用测试集）
        """
        try:
            # 加载测试数据作为模拟流量
            X_test = np.load(os.path.join(DATA_DIR, 'test_images.npy'))
            y_test = np.load(os.path.join(DATA_DIR, 'test_labels.npy'))
            
            print("\n✓ 开始捕获网络流量（模拟模式）...")
            print(f"  数据源: 测试集 ({len(X_test)} 个样本)")
            print("-" * 80)
            
            idx = 0
            while self.running:
                # 模拟捕获一个流量包
                if idx < len(X_test):
                    flow = X_test[idx]
                    true_label = y_test[idx]
                    # 创建假的raw_features, packet_info, flow_stats用于测试集
                    fake_raw = np.zeros(41)
                    fake_packet_info = {}
                    fake_flow_stats = {}
                    self.flow_queue.put((flow, true_label, idx, fake_raw, fake_packet_info, fake_flow_stats))
                    idx += 1
                else:
                    # 循环使用测试集
                    idx = 0
                
                # 控制捕获速度
                time.sleep(0.05)  # 每秒捕获20个包
        except Exception as e:
            print(f"\n❌ 流量捕获错误: {e}")
            import traceback
            traceback.print_exc()
    
    def process_flows(self):
        """处理捕获的流量（使用混合检测器）"""
        print("✓ 流量处理线程已启动（混合检测模式）")
        while self.running:
            try:
                # 解包数据（新增packet_info和flow_stats）
                flow, true_label, idx, raw_features, packet_info, flow_stats = self.flow_queue.get(timeout=1)
                
                # 使用混合检测器
                base_features = raw_features[:41] if len(raw_features) >= 41 else raw_features
                pred_class, confidence, method = self.hybrid_detector.detect(
                    base_features, packet_info, flow_stats
                )
                
                # 更新统计
                self.total_flows += 1
                self.class_counts[pred_class] += 1
                
                # 保存预测结果（添加method信息）
                if true_label >= 0:  # 有真实标签（模拟模式）
                    true_class = self.classifier.class_names[true_label]
                    is_correct = (pred_class == true_class)
                else:  # 真实流量没有标签
                    true_class = 'unknown'
                    is_correct = None
                
                self.recent_predictions.append({
                    'idx': idx,
                    'predicted': pred_class,
                    'true': true_class,
                    'confidence': confidence,
                    'correct': is_correct,
                    'method': method,  # 'rule' 或 'ml'
                    'timestamp': time.time()
                })
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"\n处理错误: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def display_stats(self):
        """显示实时统计（终端模式，GUI模式不使用）"""
        while self.running:
            time.sleep(UPDATE_INTERVAL)

            # 清空屏幕（仅在终端中有效）
            os.system('clear' if os.name == 'posix' else 'cls')

            # 计算运行时间
            elapsed = time.time() - self.start_time

            # 计算准确率（仅在模拟模式下）
            if self.recent_predictions:
                has_labels = any(p['correct'] is not None for p in self.recent_predictions)
                if has_labels:
                    recent_correct = sum(1 for p in self.recent_predictions if p['correct'])
                    recent_accuracy = recent_correct / len(self.recent_predictions) * 100
                else:
                    recent_accuracy = None  # 真实流量无法计算准确率

                # 计算每个类别的数量
                recent_pred_dist = Counter(p['predicted'] for p in self.recent_predictions)
                recent_true_dist = Counter(p['true'] for p in self.recent_predictions)
            else:
                recent_accuracy = None
                recent_pred_dist = Counter()
                recent_true_dist = Counter()

            # 显示标题
            print("=" * 80)
            print(f"{'实时网络流量监控':^80}")
            print("=" * 80)
            print(f"设备: {device} | 运行时间: {elapsed:.1f}秒 | 更新间隔: {UPDATE_INTERVAL}秒")
            print("-" * 80)

            # 显示总体统计
            print(f"\n【总体统计】")
            print(f"  总流量包: {self.total_flows}")
            print(f"  处理速度: {self.total_flows / elapsed:.2f} 包/秒")
            print(f"  队列长度: {self.flow_queue.qsize()}")

            # 显示最近窗口准确率
            print(f"\n【最近 {len(self.recent_predictions)} 个样本】")
            if recent_accuracy is not None:
                print(f"  准确率: {recent_accuracy:.2f}%")
            else:
                print(f"  准确率: N/A (真实流量无标签)")
                # 显示置信度统计（真实流量无标签时）
                if self.recent_predictions:
                    confs = [p['confidence'] for p in self.recent_predictions]
                    print(f"  平均置信度: {np.mean(confs):.3f} (min={min(confs):.3f}, max={max(confs):.3f})")
                    # 低置信度样本数
                    low_conf = sum(1 for c in confs if c < 0.8)
                    if low_conf > 0:
                        print(f"  ⚠️  低置信度样本(<0.8): {low_conf} ({low_conf/len(confs)*100:.1f}%)")

            # 显示类别分布
            print(f"\n【流量识别统计】")
            if not self.class_counts:
                print("  (暂无数据)")
            else:
                for cls in sorted(self.class_counts.keys()):
                    count = self.class_counts[cls]
                    pct = count / self.total_flows * 100 if self.total_flows > 0 else 0
                    bar = '█' * int(pct / 2)

                    # 添加类别说明
                    cls_desc = {
                        'normal': '正常流量',
                        'dos': 'DoS攻击',
                        'probe': '探测扫描',
                        'r2l': '远程登录攻击',
                        'u2r': '提权攻击'
                    }.get(cls, '')

                    print(f"  {cls:8s} ({cls_desc:10s}): {count:5d} ({pct:5.1f}%) {bar}")

            # 显示最近5个预测
            print(f"\n【最近识别】")
            if not any(p['correct'] is not None for p in self.recent_predictions):
                # 真实流量模式：突出显示识别结果
                print(f"  {'时间':8s} {'识别类型':10s} {'置信度':8s} {'说明':20s}")
                print("  " + "-" * 55)

                for p in list(self.recent_predictions)[-5:]:
                    ts = datetime.fromtimestamp(p['timestamp']).strftime('%H:%M:%S')
                    conf_str = f"{p['confidence']:.3f}"

                    # 根据置信度添加说明
                    if p['confidence'] >= 0.9:
                        desc = "高度确信"
                    elif p['confidence'] >= 0.7:
                        desc = "较为确定"
                    else:
                        desc = "不太确定"

                    print(f"  {ts:8s} {p['predicted']:10s} {conf_str:8s} {desc:20s}")

                print("\n💡 说明:")
                print("   '识别类型' = 模型识别出的流量类型（这就是识别结果！）")
                print("   真实流量没有预先标注，无法显示参考答案")
                print("   要验证模型准确率，请运行: python3 realtime_monitor.py")
            else:
                # 测试集模式：显示完整对比
                print(f"  {'时间':8s} {'预测':8s} {'真实':8s} {'置信度':8s} {'结果':4s}")
                print("  " + "-" * 50)

                for p in list(self.recent_predictions)[-5:]:
                    ts = datetime.fromtimestamp(p['timestamp']).strftime('%H:%M:%S')
                    result = '✓' if p['correct'] else '✗' if p['correct'] is not None else '-'
                    print(f"  {ts:8s} {p['predicted']:8s} {p['true']:8s} "
                          f"{p['confidence']:.3f}    {result}")

            print("\n" + "=" * 80)
            print("按 Ctrl+C 退出监控")

    def get_stats(self):
        """获取当前统计数据（供GUI使用）"""
        elapsed = time.time() - self.start_time

        stats = {
            'total_flows': self.total_flows,
            'elapsed_time': elapsed,
            'speed': self.total_flows / elapsed if elapsed > 0 else 0,
            'queue_size': self.flow_queue.qsize(),
            'class_counts': dict(self.class_counts),
            'recent_predictions': list(self.recent_predictions)
        }

        # 计算准确率
        if self.recent_predictions:
            has_labels = any(p['correct'] is not None for p in self.recent_predictions)
            if has_labels:
                recent_correct = sum(1 for p in self.recent_predictions if p['correct'])
                stats['accuracy'] = recent_correct / len(self.recent_predictions) * 100
            else:
                stats['accuracy'] = None
                # 计算置信度统计
                confs = [p['confidence'] for p in self.recent_predictions]
                stats['avg_confidence'] = np.mean(confs)
                stats['min_confidence'] = min(confs)
                stats['max_confidence'] = max(confs)
        else:
            stats['accuracy'] = None

        return stats
    
    def start(self, duration=None):
        """
        启动监控
        
        Args:
            duration: 监控持续时间（秒），None表示无限运行
        """
        print("=" * 80)
        print(f"{'VFL 实时流量监控系统':^80}")
        print("=" * 80)
        print(f"\n正在初始化...")
        print(f"  设备: {device}")
        print(f"  监控窗口: {MONITOR_WINDOW} 个样本")
        if self.use_real_traffic and self.interface:
            print(f"  网络接口: {self.interface}")
        print(f"  更新间隔: {UPDATE_INTERVAL} 秒")
        print(f"  流量模式: {'真实网络流量' if self.use_real_traffic else '模拟数据'}")
        
        # 设置运行标志
        self.running = True
        
        # 启动线程
        if self.use_real_traffic:
            capture_thread = threading.Thread(target=self.real_flow_capture, daemon=True)
        else:
            capture_thread = threading.Thread(target=self.simulate_flow_capture, daemon=True)
        process_thread = threading.Thread(target=self.process_flows, daemon=True)
        display_thread = threading.Thread(target=self.display_stats, daemon=True)
        
        capture_thread.start()
        process_thread.start()
        display_thread.start()
        
        try:
            if duration:
                time.sleep(duration)
            else:
                # 无限运行直到用户中断
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n正在停止监控...")
        finally:
            self.running = False
            capture_thread.join(timeout=2)
            process_thread.join(timeout=2)
            display_thread.join(timeout=2)
            
            # 显示最终统计
            print("\n" + "=" * 80)
            print(f"{'监控已停止':^80}")
            print("=" * 80)
            print(f"\n最终统计:")
            print(f"  总处理流量: {self.total_flows}")
            print(f"  运行时间: {time.time() - self.start_time:.1f} 秒")
            
            if self.recent_predictions:
                has_labels = any(p['correct'] is not None for p in self.recent_predictions)
                if has_labels:
                    correct = sum(1 for p in self.recent_predictions if p['correct'])
                    accuracy = correct / len(self.recent_predictions) * 100
                    print(f"  最终准确率: {accuracy:.2f}%")
                else:
                    print(f"  准确率: N/A (真实流量无标签)")
            
            print("\n各类别统计:")
            for cls in sorted(self.class_counts.keys()):
                count = self.class_counts[cls]
                pct = count / self.total_flows * 100 if self.total_flows > 0 else 0
                print(f"  {cls:8s}: {count:5d} ({pct:5.1f}%)")
            
            # 如果使用真实流量模式，显示捕获统计
            if self.use_real_traffic:
                print(f"\n💡 提示:")
                if self.total_flows == 0:
                    print(f"  未检测到流量。建议:")
                    print(f"  1. 确认网络接口活跃: ifconfig {self.interface or 'en0'}")
                    print(f"  2. 在另一个终端生成流量:")
                    print(f"     python3 generate_test_traffic.py")
                    print(f"  3. 或尝试其他接口:")
                    print(f"     ./list_interfaces.sh")
                else:
                    print(f"  成功检测到 {self.total_flows} 个流量包")


class MonitorGUI:
    """网络流量监控GUI"""

    def __init__(self, classifier, use_real_traffic=True, interface=None):
        self.classifier = classifier
        self.use_real_traffic = use_real_traffic
        self.interface = interface
        self.monitor = None

        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("VFL 实时网络流量监控系统")
        self.root.geometry("1200x800")

        # 设置样式
        self.setup_styles()

        # 创建UI组件
        self.create_widgets()

        # 数据存储
        self.traffic_history = deque(maxlen=100)  # 存储流量速率历史
        self.class_history = {cls: deque(maxlen=100) for cls in classifier.class_names}

        # 更新标志
        self.is_monitoring = False

    def setup_styles(self):
        """设置样式"""
        style = ttk.Style()
        style.theme_use('clam')

    def create_widgets(self):
        """创建UI组件"""
        # 顶部控制面板
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)

        # 标题
        title_label = ttk.Label(control_frame, text="VFL 实时网络流量监控系统",
                                font=('Arial', 16, 'bold'))
        title_label.pack(side=tk.LEFT, padx=10)

        # 控制按钮
        self.start_btn = ttk.Button(control_frame, text="开始监测",
                                     command=self.start_monitoring, width=15)
        self.start_btn.pack(side=tk.RIGHT, padx=5)

        self.stop_btn = ttk.Button(control_frame, text="停止监测",
                                    command=self.stop_monitoring, width=15, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.RIGHT, padx=5)

        # 模式选择
        mode_frame = ttk.Frame(control_frame)
        mode_frame.pack(side=tk.RIGHT, padx=20)
        ttk.Label(mode_frame, text="模式:").pack(side=tk.LEFT, padx=5)
        self.mode_var = tk.StringVar(value="real" if self.use_real_traffic else "sim")
        ttk.Radiobutton(mode_frame, text="真实流量", variable=self.mode_var,
                       value="real").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="模拟数据", variable=self.mode_var,
                       value="sim").pack(side=tk.LEFT)

        # 中间内容区域
        content_frame = ttk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 左侧：统计信息
        left_frame = ttk.LabelFrame(content_frame, text="实时统计", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # 统计标签
        stats_frame = ttk.Frame(left_frame)
        stats_frame.pack(fill=tk.X, pady=5)

        self.stats_labels = {}
        stats_items = [
            ('status', '状态:', '未启动'),
            ('total', '总流量包:', '0'),
            ('speed', '处理速度:', '0.0 包/秒'),
            ('elapsed', '运行时间:', '0 秒'),
            ('accuracy', '准确率:', 'N/A')
        ]

        for key, label, default in stats_items:
            row = ttk.Frame(stats_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, font=('Arial', 10, 'bold'), width=12).pack(side=tk.LEFT)
            value_label = ttk.Label(row, text=default, font=('Arial', 10))
            value_label.pack(side=tk.LEFT)
            self.stats_labels[key] = value_label

        # 流量分类统计
        class_frame = ttk.LabelFrame(left_frame, text="流量分类统计", padding="10")
        class_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.class_labels = {}
        class_desc = {
            'normal': '正常流量',
            'dos': 'DoS攻击',
            'probe': '探测扫描',
            'r2l': '远程登录攻击',
            'u2r': '提权攻击'
        }

        for cls in self.classifier.class_names:
            row = ttk.Frame(class_frame)
            row.pack(fill=tk.X, pady=3)
            desc = class_desc.get(cls, '')
            ttk.Label(row, text=f"{cls} ({desc}):", width=25).pack(side=tk.LEFT)

            # 进度条
            progress = ttk.Progressbar(row, mode='determinate', length=200)
            progress.pack(side=tk.LEFT, padx=5)

            # 数量标签
            count_label = ttk.Label(row, text="0 (0.0%)", width=15)
            count_label.pack(side=tk.LEFT)

            self.class_labels[cls] = {'progress': progress, 'label': count_label}

        # 最近检测记录
        recent_frame = ttk.LabelFrame(left_frame, text="最近检测记录", padding="10")
        recent_frame.pack(fill=tk.BOTH, expand=True)

        # 创建表格
        columns = ('时间', '预测类型', '置信度', '方法')
        self.tree = ttk.Treeview(recent_frame, columns=columns, show='headings', height=8)

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)

        scrollbar = ttk.Scrollbar(recent_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 右侧：可视化图表
        right_frame = ttk.LabelFrame(content_frame, text="流量可视化", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # 创建matplotlib图表
        self.fig = Figure(figsize=(6, 8), dpi=100)

        # 流量速率图
        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_title('流量处理速率')
        self.ax1.set_xlabel('时间 (秒)')
        self.ax1.set_ylabel('包/秒')
        self.ax1.grid(True, alpha=0.3)
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=2)

        # 流量分类饼图
        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_title('流量分类分布')

        self.fig.tight_layout()

        # 嵌入到tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 底部日志区域
        log_frame = ttk.LabelFrame(self.root, text="系统日志", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.log_text = scrolledtext.ScrolledText(log_frame, height=6,
                                                   font=('Courier', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def log(self, message):
        """添加日志"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    def start_monitoring(self):
        """开始监测"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        # 更新模式
        use_real = (self.mode_var.get() == "real")

        # 创建监控器
        self.monitor = RealTimeFlowMonitor(
            self.classifier,
            use_real_traffic=use_real and SCAPY_AVAILABLE,
            interface=self.interface
        )

        self.log(f"开始监测 - 模式: {'真实流量' if use_real else '模拟数据'}")
        self.stats_labels['status'].config(text='监测中', foreground='green')

        # 启动监控线程
        if use_real and SCAPY_AVAILABLE:
            capture_thread = threading.Thread(target=self.monitor.real_flow_capture, daemon=True)
        else:
            capture_thread = threading.Thread(target=self.monitor.simulate_flow_capture, daemon=True)

        process_thread = threading.Thread(target=self.monitor.process_flows, daemon=True)

        self.monitor.running = True
        capture_thread.start()
        process_thread.start()

        # 启动GUI更新
        self.update_gui()

    def stop_monitoring(self):
        """停止监测"""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

        if self.monitor:
            self.monitor.running = False

        self.log("监测已停止")
        self.stats_labels['status'].config(text='已停止', foreground='red')

    def update_gui(self):
        """更新GUI显示"""
        if not self.is_monitoring or not self.monitor:
            return

        try:
            stats = self.monitor.get_stats()

            # 更新统计信息
            self.stats_labels['total'].config(text=str(stats['total_flows']))
            self.stats_labels['speed'].config(text=f"{stats['speed']:.2f} 包/秒")
            self.stats_labels['elapsed'].config(text=f"{stats['elapsed_time']:.1f} 秒")

            if stats['accuracy'] is not None:
                self.stats_labels['accuracy'].config(text=f"{stats['accuracy']:.2f}%")
            else:
                if 'avg_confidence' in stats:
                    self.stats_labels['accuracy'].config(
                        text=f"置信度: {stats['avg_confidence']:.3f}"
                    )
                else:
                    self.stats_labels['accuracy'].config(text="N/A")

            # 更新类别统计
            total = stats['total_flows']
            for cls, data in self.class_labels.items():
                count = stats['class_counts'].get(cls, 0)
                pct = (count / total * 100) if total > 0 else 0
                data['progress']['value'] = pct
                data['label'].config(text=f"{count} ({pct:.1f}%)")

            # 更新最近记录
            recent = stats['recent_predictions'][-10:]  # 最近10条

            # 清空树
            for item in self.tree.get_children():
                self.tree.delete(item)

            # 添加新记录
            for p in reversed(recent):
                ts = datetime.fromtimestamp(p['timestamp']).strftime('%H:%M:%S')
                method = p.get('method', 'ml')
                values = (ts, p['predicted'], f"{p['confidence']:.3f}", method)
                self.tree.insert('', 0, values=values)

            # 更新图表
            self.update_charts(stats)

        except Exception as e:
            self.log(f"更新错误: {e}")

        # 继续更新
        if self.is_monitoring:
            self.root.after(1000, self.update_gui)  # 每秒更新

    def update_charts(self, stats):
        """更新图表"""
        # 更新流量速率
        self.traffic_history.append(stats['speed'])

        x_data = list(range(len(self.traffic_history)))
        y_data = list(self.traffic_history)

        self.line1.set_data(x_data, y_data)
        self.ax1.relim()
        self.ax1.autoscale_view()

        # 更新饼图
        self.ax2.clear()
        self.ax2.set_title('流量分类分布')

        class_counts = stats['class_counts']
        if class_counts and sum(class_counts.values()) > 0:
            labels = []
            sizes = []
            pie_colors = []
            
            # 固定类别颜色映射：normal=绿色, dos=红色
            color_map = {
                'normal': '#2ecc71',  # 绿色
                'dos': '#e74c3c',     # 红色
                'probe': '#f39c12',   # 橙色
                'r2l': '#9b59b6',     # 紫色
                'u2r': '#3498db'      # 蓝色
            }

            for cls in sorted(class_counts.keys()):
                if class_counts[cls] > 0:
                    labels.append(cls)
                    sizes.append(class_counts[cls])
                    pie_colors.append(color_map.get(cls, '#95a5a6'))  # 默认灰色

            if sizes:
                self.ax2.pie(sizes, labels=labels, autopct='%1.1f%%',
                            colors=pie_colors, startangle=90)
                self.ax2.axis('equal')

        self.canvas.draw()

    def run(self):
        """运行GUI"""
        self.log("VFL 网络流量监控系统已启动")
        self.log(f"设备: {device}")
        self.log(f"模型: {len(self.classifier.class_names)} 个类别")

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        """关闭窗口"""
        if self.is_monitoring:
            if messagebox.askokcancel("退出", "监测正在进行，确定要退出吗？"):
                self.stop_monitoring()
                self.root.destroy()
        else:
            self.root.destroy()


def main():
    """主函数"""
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='VFL实时网络流量监控系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # GUI模式（默认）
  python3 realtime_monitor.py --gui

  # 终端模式 - 监测本地真实流量
  sudo python3 realtime_monitor.py

  # 终端模式 - 模拟模式（使用测试集验证准确率）
  python3 realtime_monitor.py --sim

  # 检测本地回环接口（lo0）
  sudo python3 realtime_monitor.py --interface lo0

  # 检测指定WiFi接口
  sudo python3 realtime_monitor.py --interface en0

  # 查看可用网络接口
  ifconfig  # macOS/Linux
  ipconfig  # Windows
        """
    )

    parser.add_argument(
        '-g', '--gui',
        action='store_true',
        help='使用GUI模式'
    )

    parser.add_argument(
        '-s', '--sim',
        action='store_true',
        help='使用模拟数据（测试集）'
    )

    parser.add_argument(
        '-i', '--interface',
        type=str,
        default=None,
        help='指定网络接口（如: lo0, en0, eth0等），默认捕获所有接口'
    )

    parser.add_argument(
        '-d', '--duration',
        type=int,
        default=None,
        help='运行时长（秒），默认无限运行（仅终端模式）'
    )

    args = parser.parse_args()

    # 默认使用真实流量，除非指定了 --sim
    use_real = not args.sim
    
    # 检查模型文件
    if not os.path.exists(MODEL_DIR):
        print(f"❌ 错误: 模型目录不存在: {MODEL_DIR}")
        print("请先运行 train_vfl_network.py 训练模型")
        return
    
    required_files = ['config.pkl', 'top_model.pth', 
                     'bottom_model_party1.pth', 
                     'bottom_model_party2.pth', 
                     'bottom_model_party3.pth']
    
    for fname in required_files:
        fpath = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(fpath):
            print(f"❌ 错误: 缺少模型文件: {fname}")
            return
    
    # 创建分类器
    try:
        classifier = VFLFlowClassifier(MODEL_DIR, device)
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 检查权限和依赖
    if use_real and not SCAPY_AVAILABLE and not args.gui:
        print("\n❌ 错误: scapy未安装，无法捕获真实流量")
        print("   安装: pip install scapy")
        print("   或使用测试集模式: python3 realtime_monitor.py --sim\n")
        return

    if args.interface and not use_real:
        print("\n⚠️  警告: --interface 参数需要配合真实流量模式使用")
        print("   忽略 --interface 参数\n")

    # GUI模式
    if args.gui:
        gui = MonitorGUI(
            classifier,
            use_real_traffic=use_real,
            interface=args.interface
        )
        gui.run()
    # 终端模式
    else:
        monitor = RealTimeFlowMonitor(
            classifier,
            use_real_traffic=use_real,
            interface=args.interface
        )
        monitor.start(duration=args.duration)


if __name__ == '__main__':
    main()
