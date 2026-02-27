#!/usr/bin/env python3
"""
实时网络流量监测和识别系统
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
        
        # 如果是41维（增强的真实包特征），使用改进的归一化
        if original_dim == 41:
            # 手动归一化：根据实际网络特征的合理范围
            normalized = np.zeros(41)
            
            # 包长度和IP长度：归一化到[0,1]，假设最大1500
            normalized[0] = min(flow_data[0] / 1500.0, 1.0)
            normalized[6] = min(flow_data[6] / 1500.0, 1.0)
            
            # 协议类型：已经是离散值(1/6/17)，除以20归一化
            normalized[1] = flow_data[1] / 20.0
            
            # 端口号：归一化到[0,1]
            normalized[2] = flow_data[2] / 65535.0
            normalized[3] = flow_data[3] / 65535.0
            
            # TCP flags：归一化到[0,1]
            normalized[4] = flow_data[4] / 255.0
            
            # TTL：归一化到[0,1]，假设最大255
            normalized[5] = flow_data[5] / 255.0
            
            # 其他基础特征：直接复制（已经是归一化范围）
            normalized[7:13] = flow_data[7:13]
            
            # 流统计特征（索引13+）：这些已经在flow_tracker中归一化
            if flow_data.shape[0] > 13:
                normalized[13:] = flow_data[13:]
            
            flow_data = normalized
            
            # 扩展到121维（填充零）
            padding = np.zeros(121 - 41)
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
                        print(f"[诊断] 所有概率: {', '.join([f'{classifier.class_names[i]}={all_probs[i]:.3f}' for i in range(len(all_probs))])}")
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
        """显示实时统计"""
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


def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='VFL实时网络流量监控系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 默认模式（监测本地真实流量）
  sudo python3 realtime_monitor.py
  
  # 模拟模式（使用测试集验证准确率）
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
        help='运行时长（秒），默认无限运行'
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
    if use_real and not SCAPY_AVAILABLE:
        print("\n❌ 错误: scapy未安装，无法捕获真实流量")
        print("   安装: pip install scapy")
        print("   或使用测试集模式: python3 realtime_monitor.py --sim\n")
        return
    
    if args.interface and not use_real:
        print("\n⚠️  警告: --interface 参数需要配合真实流量模式使用")
        print("   忽略 --interface 参数\n")
    
    # 创建并启动监控器
    monitor = RealTimeFlowMonitor(
        classifier, 
        use_real_traffic=use_real,
        interface=args.interface
    )
    
    # 运行监控
    monitor.start(duration=args.duration)


if __name__ == '__main__':
    main()
