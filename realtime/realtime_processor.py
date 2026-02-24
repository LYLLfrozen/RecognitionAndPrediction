"""
实时流量处理模块
整合流量捕获、特征提取和预处理
"""
import time
import threading
import queue
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import torch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime.traffic_capture import TrafficCapture, ConnectionAggregator
from realtime.feature_extractor import FeatureExtractor
from data.data_processor import KDDDataProcessor
from model.fl_woa_cnn_lstm.cnn_lstm_model import CNNLSTMModel

class RealtimeProcessor:
    def __init__(self, processor_path=None, model_path=None, interface=None):
        self.capture = TrafficCapture(interface=interface)
        self.aggregator = ConnectionAggregator(window_size=int(2.0))
        self.extractor = FeatureExtractor()
        
        # 加载或初始化数据处理器
        if processor_path and os.path.exists(processor_path):
            try:
                self.processor = KDDDataProcessor.load(processor_path)
                print(f"已加载数据处理器: {processor_path}")
            except Exception as e:
                print(f"加载处理器失败: {e}")
                self.processor = KDDDataProcessor()
        else:
            print("警告: 未提供有效的数据处理器路径，将使用未拟合的处理器（仅用于测试）")
            self.processor = KDDDataProcessor()
            
        # 加载模型
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 默认标签顺序（会被模型checkpoint中的实际顺序覆盖）
        self.labels = ['dos', 'normal', 'probe', 'r2l', 'u2r']
        
        if model_path and os.path.exists(model_path):
            try:
                self._load_model(model_path)
                print(f"已加载模型: {model_path}")
            except Exception as e:
                print(f"加载模型失败: {e}")
        
        self.running = False
        self.output_queue = queue.Queue()
        self.processing_thread = None

    def _load_model(self, model_path):
        """加载PyTorch模型"""
        # 配置必须与训练时一致
        config = {
            'input_shape': (1, 11, 11),
            'num_classes': 5, # 默认5类
            'cnn_filters': [32, 64, 128],
            'lstm_units': 64,
            'dropout_rate': 0.3
        }
        
        # 先加载检查点，获取可能的配置信息
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"尝试使用 weights_only=False 加载失败: {e}")
            # 再次尝试（某些旧版本可能不支持 weights_only 参数）
            checkpoint = torch.load(model_path, map_location=self.device)
            
        # 尝试从检查点恢复标签编码器（如果存在）
        # 注意：这依赖于 checkpoint 中是否保存了 LabelEncoder 对象
        
        state_dict = None
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # 自动检测类别数量
        if 'fc2.weight' in state_dict:
            num_classes = state_dict['fc2.weight'].shape[0]
            print(f"检测到模型类别数量: {num_classes}")
            config['num_classes'] = num_classes
            
            # 如果 checkpoint 中保存了 processor 信息，优先使用其中的 class_names
            if isinstance(checkpoint, dict) and 'processor' in checkpoint and checkpoint['processor']:
                proc = checkpoint['processor']
                if isinstance(proc, dict) and 'class_names' in proc:
                    saved_names = proc['class_names']
                    if len(saved_names) == num_classes:
                        self.labels = saved_names
                        print(f"使用 checkpoint 中的 class_names 作为标签顺序: {self.labels}")
                    else:
                        print("警告: checkpoint 中的 class_names 与模型输出维度不匹配，忽略")
            else:
                # 更新标签列表（回退逻辑）
                if num_classes == 5:
                    # 使用最常见的5类顺序作为回退
                    self.labels = ['dos', 'normal', 'probe', 'r2l', 'u2r']
                    print(f"警告: 未找到标签信息，使用回退标签顺序: {self.labels}")
                elif num_classes == 2:
                    # 二分类通常为 ['normal', 'attack']，使用更直观的回退名
                    self.labels = ['normal', 'attack']
                    print("警告: 检测到2分类模型，使用回退标签 ['normal','attack']")
                else:
                    self.labels = [f'class_{i}' for i in range(num_classes)]
        
        # 初始化模型
        self.model = CNNLSTMModel(config)
        self.model.load_state_dict(state_dict)
            
        self.model.to(self.device)
        self.model.eval()
        # 打印最终标签映射，便于排查索引问题
        print(f"模型加载完成，使用标签映射: {self.labels}")

        
    def start(self):
        """启动实时处理"""
        if self.running:
            return
            
        self.running = True
        
        # 启动捕获
        self.capture.start_capture()
        
        # 启动处理线程
        self.processing_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.processing_thread.start()
        print("实时处理已启动")
        
    def stop(self):
        """停止实时处理"""
        self.running = False
        self.capture.stop_capture()
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        print("实时处理已停止")
        
    def _process_loop(self):
        """处理循环"""
        print("开始处理循环...")
        while self.running:
            try:
                # 1. 获取数据包 (非阻塞，但有超时)
                packet = self.capture.get_packet(timeout=int(0.1) if 0.1 < 1 else 1)
                
                if packet:
                    # 2. 聚合连接
                    self.aggregator.add_packet(packet)
                
                # 3. 检查完成的连接
                ready_conns = self.aggregator.get_ready_connections()
                
                for conn_key, conn_data in ready_conns:
                    self._process_connection(conn_key, conn_data)
                
                # 避免空转占用CPU
                if not packet and not ready_conns:
                    time.sleep(0.01)
                    
            except Exception as e:
                print(f"处理循环异常: {e}")
                time.sleep(1)

    def _process_connection(self, conn_key, conn_data):
        """处理单个连接"""
        try:
            # 4. 提取特征
            features = self.extractor.extract(conn_data)
            
            # 5. 预处理
            # 转换为DataFrame (单行)
            df = pd.DataFrame([features])
            
            # 确保处理器已拟合
            if not self.processor.scaler:
                # 如果未拟合，临时拟合一下（仅用于演示/测试）
                # 注意：在生产环境中，必须加载预训练的处理器
                print("警告: 处理器未拟合，正在进行临时拟合...")
                self.processor.fit(df)
            
            # 应用预处理
            X_image, _ = self.processor.transform(df)
            
            # 6. 模型预测
            prediction = None
            pred_label = "Unknown"
            confidence = 0.0
            
            if self.model:
                with torch.no_grad():
                    input_tensor = torch.from_numpy(X_image).float().to(self.device)
                    output = self.model(input_tensor)
                    probs = torch.softmax(output, dim=1)
                    conf, pred = torch.max(probs, 1)
                    
                    prediction = pred.item()
                    confidence = conf.item()
                    # 保存完整概率向量（CPU 可序列化列表）
                    try:
                        probs_list = probs.cpu().numpy().ravel().tolist()
                    except Exception:
                        probs_list = []
                    
                    if 0 <= prediction < len(self.labels):
                        pred_label = self.labels[int(prediction)]

                    # 启发式修正：高连接数 + 高错误率 = DOS攻击
                    try:
                        conn_count = features.get('count', 0)
                        rerror = features.get('rerror_rate', 0.0)
                        serror = features.get('serror_rate', 0.0)
                        service = features.get('service', 'other')
                        src_bytes = features.get('src_bytes', 0)
                    except Exception:
                        conn_count = 0
                        rerror = 0.0
                        serror = 0.0
                        service = 'other'
                        src_bytes = 0

                    # 强DOS特征：count > 100 且 (rerror > 0.8 或 serror > 0.8)
                    if conn_count > 100 and (rerror > 0.8 or serror > 0.8):
                        if 'dos' in self.labels:
                            dos_idx = self.labels.index('dos')
                            pred_label = 'dos'
                            prediction = int(dos_idx)
                            # 提升置信度
                            confidence = max(confidence, 0.95)
                    # 中等DOS特征：count > 500 且任意错误率 > 0.5
                    elif conn_count > 500 and (rerror > 0.5 or serror > 0.5):
                        if 'dos' in self.labels:
                            dos_idx = self.labels.index('dos')
                            pred_label = 'dos'
                            prediction = int(dos_idx)
                            confidence = max(confidence, 0.90)
                    # 极高连接数：count > 1000
                    elif conn_count > 1000:
                        if 'dos' in self.labels:
                            dos_idx = self.labels.index('dos')
                            pred_label = 'dos'
                            prediction = int(dos_idx)
                            confidence = max(confidence, 0.85)

                    # 启发式修正：R2L 攻击通常针对特定服务发送有效载荷，但连接频率不高
                    # 如果检测到针对 http/ftp 等服务的有效载荷，且被误判为 normal，则修正
                    if pred_label == 'normal' and service in ['http', 'ftp', 'ftp_data', 'telnet', 'ssh', 'pop_3', 'imap4'] and src_bytes > 0 and conn_count < 20: 
                        if 'r2l' in self.labels:
                            r2l_idx = self.labels.index('r2l')
                            pred_label = 'r2l'
                            prediction = int(r2l_idx)
                            # 给予中等置信度，表明这是基于规则的怀疑
                            confidence = max(confidence, 0.75)
            
            # 放入输出队列
            result = {
                'timestamp': time.time(),
                'features': features,
                'image': X_image,
                'connection_key': conn_key,
                'src': f"{conn_data['src_ip']}:{conn_data['src_port']}",
                'dst': f"{conn_data['dst_ip']}:{conn_data['dst_port']}",
                'protocol': conn_data['protocol'],
                'prediction': prediction,
                'prediction_index': prediction,
                'label': pred_label,
                'confidence': confidence,
                'probabilities': probs_list if self.model else [],
                # 调试信息
                'debug_info': {
                    'count': features.get('count', 0),
                    'srv_count': features.get('srv_count', 0),
                    'flag': features.get('flag', 'unknown'),
                    'serror_rate': features.get('serror_rate', 0.0),
                    'rerror_rate': features.get('rerror_rate', 0.0),
                    'dst_host_count': features.get('dst_host_count', 0),
                    'dst_host_srv_count': features.get('dst_host_srv_count', 0),
                    'same_srv_rate': features.get('same_srv_rate', 0.0)
                }
            }
            
            self.output_queue.put(result)
            
            if self.model:
                # 简化日志，详细信息由 run_realtime_detection.py 处理
                pass
            else:
                print(f"已处理连接: {result['src']} -> {result['dst']} ({result['protocol']})")
            
        except Exception as e:
            print(f"处理连接出错: {e}")
            import traceback
            traceback.print_exc()

    def get_processed_data(self):
        """获取处理后的数据"""
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None

if __name__ == "__main__":
    # 测试代码
    print("初始化实时处理器...")
    
    # 尝试查找保存的处理器
    processor_path = "data/processed_data/processor_state.pkl"
    if not os.path.exists(processor_path):
        processor_path = None
        
    # 尝试查找模型
    model_path = "model/saved_models/cnn_lstm_full.pth"
    if not os.path.exists(model_path):
        model_path = "model/saved_models/cnn_lstm_improved.pth"
        
    if not os.path.exists(model_path):
        model_path = None
        
    processor = RealtimeProcessor(processor_path=processor_path, model_path=model_path)
    
    try:
        processor.start()
        print("正在捕获流量 (按 Ctrl+C 停止)...")
        
        while True:
            data = processor.get_processed_data()
            if data:
                print(f"收到数据: {data['src']} -> {data['dst']}")
                if 'label' in data:
                    print(f"预测结果: {data['label']} (置信度: {data['confidence']:.2%})")
                print(f"特征示例: duration={data['features']['duration']}, bytes={data['features']['src_bytes']}")
                print("-" * 30)
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n正在停止...")
    finally:
        processor.stop()
