"""
垂直联邦学习工具函数（重命名以避免冲突）
"""
from .utils import (
    split_features_vertical,
    split_features_for_cnn,
    create_vfl_model_split,
    print_vfl_data_distribution,
    calculate_communication_cost
)

__all__ = [
    'split_features_vertical',
    'split_features_for_cnn',
    'create_vfl_model_split',
    'print_vfl_data_distribution',
    'calculate_communication_cost'
]
