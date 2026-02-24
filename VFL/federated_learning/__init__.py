"""
垂直联邦学习模块（Vertical Federated Learning with PrivBox）
实现多进程模拟的垂直联邦学习训练，使用PrivBox协议保护隐私
"""

from .vfl_server import VFLServer
from .vfl_client import VFLClient, VFLActiveParty, VFLPassiveParty
from .privbox import PrivBoxProtocol, SecretSharing, PaillierEncryption
from .vfl_utils import split_features_vertical, create_vfl_model_split

__all__ = [
    'VFLServer',
    'VFLClient',
    'VFLActiveParty',
    'VFLPassiveParty',
    'PrivBoxProtocol',
    'SecretSharing',
    'PaillierEncryption',
    'split_features_vertical',
    'create_vfl_model_split'
]
