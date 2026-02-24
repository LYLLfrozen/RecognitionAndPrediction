"""
联邦学习服务器
负责协调各客户端训练，聚合模型参数
"""
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import copy
import numpy as np


class FederatedServer:
    """联邦学习服务器"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        初始化联邦学习服务器
        
        Args:
            model: 全局模型
            device: 计算设备
        """
        self.global_model = model
        self.device = device
        self.round_history = []
        
    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """
        获取全局模型权重
        
        Returns:
            全局模型权重字典
        """
        return copy.deepcopy(self.global_model.state_dict())
    
    def set_global_weights(self, weights: Dict[str, torch.Tensor]):
        """
        设置全局模型权重
        
        Args:
            weights: 模型权重字典
        """
        self.global_model.load_state_dict(weights)
    
    def aggregate(self, client_weights_list: List[Dict[str, torch.Tensor]],
                 client_sizes: List[int]) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        聚合客户端模型（FedAvg算法）
        
        Args:
            client_weights_list: 客户端模型权重列表
            client_sizes: 各客户端的数据量
            
        Returns:
            (聚合后的全局模型权重, 模型变化量)
        """
        # 保存旧的全局权重用于对比
        old_weights = self.get_global_weights()
        
        # 执行加权平均
        total_size = sum(client_sizes)
        new_weights = copy.deepcopy(old_weights)
        
        # 将所有参数初始化为0
        for key in new_weights.keys():
            new_weights[key] = torch.zeros_like(new_weights[key])
        
        # 加权求和
        for client_weights, client_size in zip(client_weights_list, client_sizes):
            weight = client_size / total_size
            for key in new_weights.keys():
                new_weights[key] += client_weights[key] * weight
        
        # 计算模型变化量
        model_diff = 0.0
        for key in old_weights.keys():
            model_diff += torch.norm(old_weights[key] - new_weights[key]).item() ** 2
        model_diff = np.sqrt(model_diff)
        
        # 更新全局模型
        self.set_global_weights(new_weights)
        
        return new_weights, model_diff
    
    def evaluate(self, test_loader, criterion) -> Tuple[float, float]:
        """
        在测试集上评估全局模型
        
        Args:
            test_loader: 测试数据加载器
            criterion: 损失函数
            
        Returns:
            (loss, accuracy)
        """
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.global_model(inputs)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def save_model(self, path: str):
        """
        保存全局模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'model_state_dict': self.global_model.state_dict(),
            'round_history': self.round_history
        }, path)
        print(f"全局模型已保存至: {path}")
    
    def load_model(self, path: str):
        """
        加载全局模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        self.round_history = checkpoint.get('round_history', [])
        print(f"全局模型已从 {path} 加载")
