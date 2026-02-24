"""
垂直联邦学习服务器（协调器）
负责协调各参与方的训练，聚合嵌入和梯度
"""
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
import copy
from .privbox import PrivBoxProtocol


class VFLServer:
    """
    垂直联邦学习服务器/协调器
    
    职责：
    1. 协调各参与方的训练流程
    2. 使用PrivBox协议聚合嵌入向量
    3. 计算顶层模型的梯度
    4. 分发梯度给各参与方
    """
    
    def __init__(self, top_model: nn.Module, device: torch.device, 
                 num_parties: int, use_privbox: bool = True):
        """
        初始化VFL服务器
        
        Args:
            top_model: 顶层聚合模型
            device: 计算设备
            num_parties: 参与方数量
            use_privbox: 是否使用PrivBox隐私保护
        """
        self.top_model = top_model.to(device)
        self.device = device
        self.num_parties = num_parties
        self.use_privbox = use_privbox
        
        if use_privbox:
            self.privbox: Optional[PrivBoxProtocol] = PrivBoxProtocol(num_parties, use_encryption=False)
        else:
            self.privbox: Optional[PrivBoxProtocol] = None
        
        self.round_history = []
        self.communication_cost = 0.0
    
    def aggregate_embeddings(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        聚合各参与方的嵌入向量
        
        Args:
            embeddings: 各参与方的嵌入向量列表
            
        Returns:
            聚合后的嵌入向量
        """
        if self.use_privbox and self.privbox is not None:
            # 使用PrivBox协议安全聚合（添加差分隐私噪声）
            combined = self.privbox.secure_compute_embedding(
                embeddings, 
                noise_scale=0.001  # 较小的噪声以保持性能
            )
        else:
            # 直接拼接
            combined = torch.cat(embeddings, dim=-1)
        
        return combined
    
    def forward_top_model(self, combined_embedding: torch.Tensor) -> torch.Tensor:
        """
        通过顶层模型前向传播
        
        Args:
            combined_embedding: 聚合后的嵌入向量
            
        Returns:
            模型输出
        """
        self.top_model.train()
        return self.top_model(combined_embedding)
    
    def backward_top_model(self, loss: torch.Tensor) -> None:
        """
        顶层模型反向传播
        
        Args:
            loss: 损失值
        """
        loss.backward()
        # 梯度已存储在计算图中
    
    def split_gradients(self, combined_embedding: torch.Tensor, 
                       embedding_sizes: List[int]) -> List[torch.Tensor]:
        """
        分割梯度给各参与方
        
        Args:
            combined_embedding: 聚合后的嵌入向量（带梯度）
            embedding_sizes: 各参与方的嵌入大小
            
        Returns:
            各参与方的梯度列表
        """
        # 使用retain_grad确保中间张量保留梯度
        if not combined_embedding.is_leaf and combined_embedding.grad is None:
            # 如果是中间张量且没有梯度，说明需要使用backward时保存的梯度
            # 我们从输入的嵌入向量中获取梯度
            raise ValueError("嵌入向量没有梯度，请确保在backward前调用retain_grad()")
        
        grad = combined_embedding.grad if combined_embedding.grad is not None else combined_embedding
        
        # 按维度分割梯度
        gradients = []
        start_idx = 0
        for size in embedding_sizes:
            end_idx = start_idx + size
            grad_slice = grad[:, start_idx:end_idx].clone()
            
            # 使用PrivBox添加梯度噪声（隐私保护）
            if self.use_privbox and self.privbox is not None:
                grad_slice = self.privbox.add_dp_noise(
                    grad_slice, 
                    epsilon=1.0,  # 隐私预算
                    sensitivity=1.0
                )
            
            gradients.append(grad_slice)
            start_idx = end_idx
        
        return gradients
    
    def update_top_model(self, optimizer: torch.optim.Optimizer):
        """
        更新顶层模型参数
        
        Args:
            optimizer: 优化器
        """
        optimizer.step()
        optimizer.zero_grad()
    
    def evaluate(self, test_embeddings_list: List[List[torch.Tensor]], 
                test_labels: torch.Tensor, criterion) -> Tuple[float, float]:
        """
        评估模型
        
        Args:
            test_embeddings_list: 测试集的嵌入向量列表 [batch1, batch2, ...]
                                 每个batch是各参与方嵌入的列表
            test_labels: 测试标签
            criterion: 损失函数
            
        Returns:
            (loss, accuracy)
        """
        self.top_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for embeddings, labels in zip(test_embeddings_list, test_labels):
                labels = labels.to(self.device)
                
                # 聚合嵌入
                combined = self.aggregate_embeddings(embeddings)
                
                # 前向传播
                outputs = self.top_model(combined)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(test_embeddings_list)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def save_model(self, path: str):
        """
        保存顶层模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'top_model_state_dict': self.top_model.state_dict(),
            'round_history': self.round_history,
            'communication_cost': self.communication_cost
        }, path)
        print(f"VFL顶层模型已保存至: {path}")
    
    def load_model(self, path: str):
        """
        加载顶层模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.top_model.load_state_dict(checkpoint['top_model_state_dict'])
        self.round_history = checkpoint.get('round_history', [])
        self.communication_cost = checkpoint.get('communication_cost', 0.0)
        print(f"VFL顶层模型已从 {path} 加载")
    
    def get_communication_cost(self) -> float:
        """获取累计通信成本（MB）"""
        return self.communication_cost
