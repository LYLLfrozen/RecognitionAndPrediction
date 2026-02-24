"""
联邦学习客户端
负责本地模型训练
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Tuple
import copy
import numpy as np


class FederatedClient:
    """联邦学习客户端"""
    
    def __init__(self, client_id: int, X_train: np.ndarray, y_train: np.ndarray,
                 device: torch.device, batch_size: int = 64):
        """
        初始化联邦学习客户端
        
        Args:
            client_id: 客户端ID
            X_train: 训练数据
            y_train: 训练标签
            device: 计算设备
            batch_size: 批次大小
        """
        self.client_id = client_id
        self.device = device
        self.batch_size = batch_size
        
        # 创建数据加载器
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.LongTensor(y_train)
        dataset = TensorDataset(X_tensor, y_tensor)
        self.train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.num_samples = len(y_train)
        
        # 本地模型（将在训练时设置）
        self.local_model = None
    
    def get_num_samples(self) -> int:
        """获取客户端数据量"""
        return self.num_samples
    
    def set_weights(self, weights: Dict[str, torch.Tensor]):
        """
        设置本地模型权重
        
        Args:
            weights: 模型权重字典
        """
        if self.local_model is None:
            raise ValueError("本地模型未初始化")
        self.local_model.load_state_dict(copy.deepcopy(weights))
    
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """
        获取本地模型权重
        
        Returns:
            模型权重字典
        """
        if self.local_model is None:
            raise ValueError("本地模型未初始化")
        return copy.deepcopy(self.local_model.state_dict())
    
    def train(self, model_template: nn.Module, global_weights: Dict[str, torch.Tensor],
             epochs: int, learning_rate: float, criterion) -> Tuple[Dict[str, torch.Tensor], float, float]:
        """
        本地训练
        
        Args:
            model_template: 模型模板（用于创建本地模型）
            global_weights: 全局模型权重
            epochs: 本地训练轮数
            learning_rate: 学习率
            criterion: 损失函数
            
        Returns:
            (更新后的模型权重, 平均损失, 准确率)
        """
        # 创建本地模型副本
        self.local_model = copy.deepcopy(model_template).to(self.device)
        self.local_model.load_state_dict(global_weights)
        
        # 优化器
        optimizer = optim.Adam(self.local_model.parameters(), lr=learning_rate)
        
        # 训练模式
        self.local_model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = self.local_model(inputs)
                loss = criterion(outputs, labels)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 统计
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += labels.size(0)
                epoch_correct += (predicted == labels).sum().item()
            
            total_loss += epoch_loss / len(self.train_loader)
            correct += epoch_correct
            total += epoch_total
        
        # 计算平均指标
        avg_loss = total_loss / epochs
        avg_accuracy = correct / total
        
        return self.get_weights(), avg_loss, avg_accuracy
    
    def evaluate(self, test_loader, criterion) -> Tuple[float, float]:
        """
        评估本地模型
        
        Args:
            test_loader: 测试数据加载器
            criterion: 损失函数
            
        Returns:
            (loss, accuracy)
        """
        if self.local_model is None:
            raise ValueError("本地模型未初始化")
        
        self.local_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.local_model(inputs)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
