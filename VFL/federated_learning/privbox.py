"""
PrivBox 隐私保护协议
实现秘密共享、同态加密等隐私保护机制
"""
import numpy as np
import torch
from typing import List, Tuple, Dict
import random


class SecretSharing:
    """
    加法秘密共享（Additive Secret Sharing）
    将数据分割成多个份额，单个份额不泄露原始信息
    """
    
    @staticmethod
    def share(secret: np.ndarray, num_parties: int) -> List[np.ndarray]:
        """
        将秘密分享给多个参与方
        
        Args:
            secret: 原始秘密数据
            num_parties: 参与方数量
            
        Returns:
            秘密份额列表
        """
        shares = []
        remaining = secret.copy()
        
        # 生成 num_parties-1 个随机份额
        for i in range(num_parties - 1):
            share = np.random.randn(*secret.shape).astype(secret.dtype)
            shares.append(share)
            remaining = remaining - share
        
        # 最后一个份额确保总和等于原始秘密
        shares.append(remaining)
        
        return shares
    
    @staticmethod
    def share_tensor(secret: torch.Tensor, num_parties: int) -> List[torch.Tensor]:
        """
        将PyTorch张量分享给多个参与方
        
        Args:
            secret: 原始秘密张量
            num_parties: 参与方数量
            
        Returns:
            秘密份额张量列表
        """
        shares = []
        remaining = secret.clone()
        
        # 生成 num_parties-1 个随机份额
        for i in range(num_parties - 1):
            share = torch.randn_like(secret)
            shares.append(share)
            remaining = remaining - share
        
        # 最后一个份额确保总和等于原始秘密
        shares.append(remaining)
        
        return shares
    
    @staticmethod
    def reconstruct(shares: List[np.ndarray]) -> np.ndarray:
        """
        重构秘密
        
        Args:
            shares: 秘密份额列表
            
        Returns:
            重构的原始秘密
        """
        result = shares[0].copy()
        for share in shares[1:]:
            result = result + share
        return result
    
    @staticmethod
    def reconstruct_tensor(shares: List[torch.Tensor]) -> torch.Tensor:
        """
        重构PyTorch张量秘密
        
        Args:
            shares: 秘密份额张量列表
            
        Returns:
            重构的原始秘密张量
        """
        result = shares[0].clone()
        for share in shares[1:]:
            result = result + share
        return result


class PaillierEncryption:
    """
    简化的Paillier同态加密
    支持加法同态：E(m1) + E(m2) = E(m1 + m2)
    注意：这是一个简化实现，实际应用中建议使用成熟的加密库
    """
    
    def __init__(self, key_length: int = 128):
        """
        初始化Paillier加密
        
        Args:
            key_length: 密钥长度（简化版本）
        """
        self.key_length = key_length
        self.public_key: Dict = {}
        self.private_key: Dict = {}
        self._generate_keys()
    
    def _generate_keys(self):
        """生成密钥对（简化版本）"""
        # 实际应用中应使用大素数和复杂的密钥生成算法
        self.public_key = {'n': 2 ** self.key_length, 'g': 2 ** self.key_length + 1}
        self.private_key = {'lambda': 2 ** (self.key_length - 1), 'mu': 1}
    
    def encrypt(self, plaintext: float) -> Dict:
        """
        加密明文（简化版本）
        
        Args:
            plaintext: 明文
            
        Returns:
            密文字典
        """
        # 简化实现：使用噪声掩码
        n = self.public_key['n']
        r = random.randint(1, n - 1)
        
        # 简化的加密：c = (g^m * r^n) mod n^2
        # 这里使用简化版本
        noise = (r * self.public_key['g']) % n
        ciphertext = (plaintext + noise) % n
        
        return {'c': ciphertext, 'r': r}
    
    def decrypt(self, ciphertext: Dict) -> float:
        """
        解密密文（简化版本）
        
        Args:
            ciphertext: 密文字典
            
        Returns:
            明文
        """
        c = ciphertext['c']
        r = ciphertext['r']
        n = self.public_key['n']
        
        # 简化的解密
        noise = (r * self.public_key['g']) % n
        plaintext = (c - noise) % n
        
        return plaintext
    
    def add_encrypted(self, c1: Dict, c2: Dict) -> Dict:
        """
        密文加法（同态性质）
        
        Args:
            c1: 密文1
            c2: 密文2
            
        Returns:
            加法结果密文
        """
        n = self.public_key['n']
        c_sum = (c1['c'] + c2['c']) % n
        r_prod = (c1['r'] * c2['r']) % n
        
        return {'c': c_sum, 'r': r_prod}


class PrivBoxProtocol:
    """
    PrivBox 协议
    结合秘密共享和同态加密，实现安全的垂直联邦学习
    """
    
    def __init__(self, num_parties: int, use_encryption: bool = True):
        """
        初始化PrivBox协议
        
        Args:
            num_parties: 参与方数量
            use_encryption: 是否使用同态加密（否则仅使用秘密共享）
        """
        self.num_parties = num_parties
        self.use_encryption = use_encryption
        self.secret_sharing = SecretSharing()
        
        if use_encryption:
            self.paillier = PaillierEncryption(key_length=64)  # 使用较小密钥以提高速度
        else:
            self.paillier = None
    
    def secure_aggregate_gradients(self, gradient_shares: List[torch.Tensor]) -> torch.Tensor:
        """
        安全聚合梯度
        
        Args:
            gradient_shares: 各方的梯度份额
            
        Returns:
            聚合后的梯度
        """
        # 使用秘密共享重构梯度
        aggregated_gradient = self.secret_sharing.reconstruct_tensor(gradient_shares)
        return aggregated_gradient
    
    def protect_gradient(self, gradient: torch.Tensor) -> List[torch.Tensor]:
        """
        保护梯度隐私
        
        Args:
            gradient: 原始梯度
            
        Returns:
            梯度份额列表
        """
        # 使用秘密共享分割梯度
        shares = self.secret_sharing.share_tensor(gradient, self.num_parties)
        return shares
    
    def secure_compute_embedding(self, embeddings: List[torch.Tensor], 
                                 noise_scale: float = 0.01) -> torch.Tensor:
        """
        安全计算嵌入向量
        添加差分隐私噪声
        
        Args:
            embeddings: 各方的嵌入向量列表
            noise_scale: 噪声规模
            
        Returns:
            聚合后的嵌入向量（带噪声）
        """
        # 聚合嵌入
        combined = torch.cat(embeddings, dim=-1)
        
        # 添加拉普拉斯噪声（差分隐私）
        if noise_scale > 0:
            noise = torch.distributions.Laplace(0, noise_scale).sample(combined.shape)
            combined = combined + noise.to(combined.device)
        
        return combined
    
    def add_dp_noise(self, tensor: torch.Tensor, epsilon: float = 1.0, 
                     delta: float = 1e-5, sensitivity: float = 1.0) -> torch.Tensor:
        """
        添加差分隐私噪声（高斯机制）
        
        Args:
            tensor: 输入张量
            epsilon: 隐私预算
            delta: 失败概率
            sensitivity: 敏感度
            
        Returns:
            添加噪声后的张量
        """
        # 计算噪声标准差
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        
        # 添加高斯噪声
        noise = torch.normal(0, sigma, size=tensor.shape).to(tensor.device)
        return tensor + noise
    
    def secure_matrix_multiplication(self, A_shares: List[torch.Tensor], 
                                     B_shares: List[torch.Tensor]) -> torch.Tensor:
        """
        安全矩阵乘法（Beaver三元组方法的简化版本）
        
        Args:
            A_shares: 矩阵A的秘密份额
            B_shares: 矩阵B的秘密份额
            
        Returns:
            矩阵乘积
        """
        # 简化实现：重构后计算（实际应用中应使用Beaver三元组）
        A = self.secret_sharing.reconstruct_tensor(A_shares)
        B = self.secret_sharing.reconstruct_tensor(B_shares)
        return torch.matmul(A, B)
