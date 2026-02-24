#!/usr/bin/env python3
"""
KDD Cup 99数据集预处理脚本
将网络流量数据转换为适合VFL的格式
"""

import pandas as pd
import numpy as np
import os
import pickle
import gzip
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter

# 特征名称（基于kddcup.names）
FEATURE_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
]

# 攻击类型分类（5大类）
ATTACK_CATEGORIES = {
    'normal': 'normal',
    # DoS attacks
    'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos',
    'smurf': 'dos', 'teardrop': 'dos', 'mailbomb': 'dos', 'apache2': 'dos',
    'processtable': 'dos', 'udpstorm': 'dos',
    # Probe attacks
    'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe', 'satan': 'probe',
    'mscan': 'probe', 'saint': 'probe',
    # R2L attacks (Remote to Local)
    'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l', 'multihop': 'r2l',
    'phf': 'r2l', 'spy': 'r2l', 'warezclient': 'r2l', 'warezmaster': 'r2l',
    'sendmail': 'r2l', 'named': 'r2l', 'snmpgetattack': 'r2l', 'snmpguess': 'r2l',
    'xlock': 'r2l', 'xsnoop': 'r2l', 'worm': 'r2l',
    # U2R attacks (User to Root)
    'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r', 'rootkit': 'u2r',
    'httptunnel': 'u2r', 'ps': 'u2r', 'sqlattack': 'u2r', 'xterm': 'u2r'
}


def load_kddcup_data(data_path, sample_size=None):
    """
    加载KDD Cup数据
    
    Args:
        data_path: 数据文件路径（.gz格式）
        sample_size: 采样大小（用于快速测试）
    
    Returns:
        DataFrame
    """
    print(f"加载数据: {data_path}")
    
    # 读取gzip压缩文件
    if data_path.endswith('.gz'):
        with gzip.open(data_path, 'rt') as f:
            df = pd.read_csv(f, names=FEATURE_NAMES, header=None)
    else:
        df = pd.read_csv(data_path, names=FEATURE_NAMES, header=None)
    
    print(f"原始数据: {df.shape}")
    
    # 如果指定了采样大小
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"采样后: {df.shape}")
    
    return df


def preprocess_data(df, use_binary=False):
    """
    预处理数据
    
    Args:
        df: 原始数据框
        use_binary: 是否使用二分类（正常/攻击）
    
    Returns:
        X, y, processor_info
    """
    print("\n【数据预处理】")
    
    # 1. 清理标签（移除末尾的点）
    df['label'] = df['label'].str.rstrip('.')
    
    # 2. 将攻击类型映射到大类
    df['attack_category'] = df['label'].map(ATTACK_CATEGORIES)
    
    # 处理未知攻击类型（可能在测试集中出现）
    df['attack_category'].fillna('unknown', inplace=True)
    
    print(f"\n标签分布:")
    print(df['attack_category'].value_counts())
    
    # 3. 选择分类方式
    if use_binary:
        # 二分类：正常 vs 攻击
        y = (df['attack_category'] != 'normal').astype(int)
        class_names = ['normal', 'attack']
        print("\n使用二分类：正常 vs 攻击")
    else:
        # 多分类：5类
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df['attack_category'])
        class_names = list(label_encoder.classes_)
        print(f"\n使用多分类：{class_names}")
    
    # 4. 特征工程
    # 分离数值特征和类别特征
    categorical_features = ['protocol_type', 'service', 'flag']
    
    # 对类别特征进行独热编码
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # 移除标签列
    feature_cols = [col for col in df_encoded.columns 
                   if col not in ['label', 'attack_category']]
    X = df_encoded[feature_cols].values
    
    print(f"\n特征数: {X.shape[1]}")
    print(f"样本数: {X.shape[0]}")
    
    # 5. 标准化数值特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 6. 保存处理器信息
    processor_info = {
        'scaler': scaler,
        'feature_names': feature_cols,
        'class_names': class_names,
        'n_classes': len(class_names),
        'use_binary': use_binary,
        'n_features': X.shape[1]
    }
    
    return X, y, processor_info


def reshape_for_vfl(X, target_shape=(11, 11)):
    """
    将特征重塑为适合VFL的格式
    
    Args:
        X: 特征矩阵 (n_samples, n_features)
        target_shape: 目标形状 (height, width)
    
    Returns:
        重塑后的数据 (n_samples, 1, height, width)
    """
    n_samples, n_features = X.shape
    target_size = target_shape[0] * target_shape[1]
    
    print(f"\n【特征重塑】")
    print(f"原始特征数: {n_features}")
    print(f"目标大小: {target_size} ({target_shape[0]}x{target_shape[1]})")
    
    # 如果特征数不足，填充0
    if n_features < target_size:
        padding = np.zeros((n_samples, target_size - n_features))
        X_padded = np.concatenate([X, padding], axis=1)
        print(f"填充 {target_size - n_features} 个特征")
    # 如果特征数过多，截断
    elif n_features > target_size:
        X_padded = X[:, :target_size]
        print(f"截断为 {target_size} 个特征")
    else:
        X_padded = X
    
    # 重塑为图像格式: (n_samples, 1, height, width)
    X_reshaped = X_padded.reshape(n_samples, 1, target_shape[0], target_shape[1])
    
    print(f"最终形状: {X_reshaped.shape}")
    
    return X_reshaped


def main():
    """主函数"""
    print("="*80)
    print("KDD Cup 99 数据集预处理")
    print("="*80)
    
    # 配置
    RAW_DATA_DIR = 'raw_data'
    OUTPUT_DIR = 'data/processed_data'
    
    # 使用10%数据集进行快速训练（可改为kddcup.data.gz使用完整数据）
    DATA_FILE = 'kddcup.data_10_percent.gz'
    USE_BINARY = False  # False=5类分类, True=二分类
    SAMPLE_SIZE = None  # None=使用全部数据，或设置具体数量如50000
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载数据
    data_path = os.path.join(RAW_DATA_DIR, DATA_FILE)
    df = load_kddcup_data(data_path, sample_size=SAMPLE_SIZE)
    
    # 2. 预处理
    X, y, processor_info = preprocess_data(df, use_binary=USE_BINARY)
    
    # 3. 划分训练集和测试集
    print("\n【划分数据集】")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    # 4. 重塑为VFL格式
    X_train_reshaped = reshape_for_vfl(X_train)
    X_test_reshaped = reshape_for_vfl(X_test)
    
    # 5. 保存处理后的数据
    print("\n【保存数据】")
    np.save(os.path.join(OUTPUT_DIR, 'train_images.npy'), X_train_reshaped)
    np.save(os.path.join(OUTPUT_DIR, 'train_labels.npy'), y_train)
    np.save(os.path.join(OUTPUT_DIR, 'test_images.npy'), X_test_reshaped)
    np.save(os.path.join(OUTPUT_DIR, 'test_labels.npy'), y_test)
    
    with open(os.path.join(OUTPUT_DIR, 'processor.pkl'), 'wb') as f:
        pickle.dump(processor_info, f)
    
    print(f"\n数据已保存到: {OUTPUT_DIR}")
    print(f"  - train_images.npy: {X_train_reshaped.shape}")
    print(f"  - train_labels.npy: {y_train.shape}")
    print(f"  - test_images.npy: {X_test_reshaped.shape}")
    print(f"  - test_labels.npy: {y_test.shape}")
    print(f"  - processor.pkl")
    
    # 6. 数据统计
    print("\n【数据统计】")
    print(f"类别数: {processor_info['n_classes']}")
    print(f"类别名称: {processor_info['class_names']}")
    print(f"\n训练集标签分布:")
    for i, class_name in enumerate(processor_info['class_names']):
        count = np.sum(y_train == i)
        print(f"  {class_name}: {count} ({count/len(y_train)*100:.1f}%)")
    
    print("\n" + "="*80)
    print("预处理完成! ✓")
    print("="*80)
    print("\n下一步：运行 python train_vfl_network.py 开始VFL训练")


if __name__ == '__main__':
    main()
