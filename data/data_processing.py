"""
KDD Cup 99 数据集加载、清洗、归一化和特征重构脚本
包含：load_essential_data, verify_data_counts, create_image_dataset
"""
from pathlib import Path
from typing import Optional, Tuple, Literal, cast
import gzip
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# 数据文件定义
ESSENTIAL_FILES = {
    'train': 'kddcup.data_10_percent.gz',
    'test': 'corrected.gz',
    'features': 'kddcup.names',
    'attack_types': 'training_attack_types'
}


def load_essential_data(data_dir: Optional[str] = None):
    """
    加载论文使用的KDD Cup 99数据集
    
    Args:
        data_dir: 数据目录路径，默认为 'data/raw_data'
        
    Returns:
        tuple: (train_data, test_data, feature_names, attack_map)
    """
    # 统一使用 Path 类型，避免字符串路径的运算符错误
    if data_dir is None:
        data_path = Path(__file__).parent / 'raw_data'
    else:
        data_path = Path(data_dir)
    
    print("开始加载KDD Cup 99数据集...")
    
    # 1. 加载特征名
    features_path = data_path / ESSENTIAL_FILES['features']
    with open(features_path, 'r') as f:
        feature_info = f.read()
    
    # 提取特征名（跳过第一行）
    lines = feature_info.strip().split('\n')[1:]
    feature_names = []
    for line in lines:
        if ':' in line:
            feature_names.append(line.split(':')[0])
    
    # 添加标签列名
    feature_names.append('label')
    
    print(f"特征数量: {len(feature_names) - 1}")  # 41个特征 + 1个标签
    
    # 2. 加载训练数据（10%子集）
    print("加载训练数据...")
    train_path = data_path / ESSENTIAL_FILES['train']
    # 直接传递文件路径，并显式声明压缩类型，满足类型检查器的期望
    train_data = pd.read_csv(train_path, header=None, names=feature_names, compression='gzip')
    
    print(f"训练集大小: {len(train_data)} 条记录")
    
    # 3. 加载测试数据（corrected数据集）
    print("加载测试数据...")
    test_path = data_path / ESSENTIAL_FILES['test']
    test_data = pd.read_csv(test_path, header=None, names=feature_names, compression='gzip')
    
    print(f"测试集大小: {len(test_data)} 条记录")
    
    # 4. 加载攻击类型映射
    attack_types_path = data_path / ESSENTIAL_FILES['attack_types']
    attack_map = {}
    with open(attack_types_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                attack, attack_type = parts[0], parts[1]
                # 添加带点号的版本（数据中的标签格式）
                attack_map[attack + '.'] = attack_type
    
    # 添加normal类型
    attack_map['normal.'] = 'normal'
    
    print(f"攻击类型映射: {len(attack_map)} 种攻击")
    
    return train_data, test_data, feature_names[:-1], attack_map


def verify_data_counts(train_df, test_df, attack_map):
    """
    验证数据量是否符合论文表1
    
    Args:
        train_df: 训练数据
        test_df: 测试数据
        attack_map: 攻击类型映射
        
    Returns:
        tuple: (train_mapped, test_mapped) - 映射到5大类后的数据
    """
    # 论文表1的预期数据
    paper_counts = {
        'train_10_percent': {
            'normal': 97278,
            'dos': 391458,
            'probe': 4107,
            'u2r': 52,
            'r2l': 1126,
            'Total': 494021
        },
        'test_corrected': {
            'normal': 60593,
            'dos': 229853,
            'probe': 4166,
            'u2r': 228,
            'r2l': 16189,
            'Total': 311029
        }
    }
    
    # 将攻击类型映射到5大类
    def map_to_5_classes(df, attack_map):
        df_mapped = df.copy()
        df_mapped['attack_class'] = df_mapped['label'].apply(
            lambda x: attack_map.get(x, 'normal')
        )
        return df_mapped
    
    train_mapped = map_to_5_classes(train_df, attack_map)
    test_mapped = map_to_5_classes(test_df, attack_map)
    
    print("\n" + "="*50)
    print("数据验证（对照论文表1）")
    print("="*50)
    
    print("\n训练集分布（按5大类）：")
    train_dist = train_mapped['attack_class'].value_counts()
    print(train_dist)
    print(f"训练集总数: {len(train_mapped)}")
    
    print("\n测试集分布（按5大类）：")
    test_dist = test_mapped['attack_class'].value_counts()
    print(test_dist)
    print(f"测试集总数: {len(test_mapped)}")
    
    print("\n论文预期数据（训练集）：")
    for key, val in paper_counts['train_10_percent'].items():
        print(f"  {key}: {val}")
    
    print("\n论文预期数据（测试集）：")
    for key, val in paper_counts['test_corrected'].items():
        print(f"  {key}: {val}")
    
    return train_mapped, test_mapped


def create_image_dataset(
    df: pd.DataFrame,
    image_size: int = 11,
    scaler: Optional[MinMaxScaler] = None,
    zero_cols: Optional[np.ndarray] = None,
    return_scaler: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, MinMaxScaler, np.ndarray]:
    """
    将数据转换为图像格式（按论文方法）
    
    Args:
        df: 输入数据
        image_size: 图像尺寸（默认11x11）
        scaler: 已有的归一化器（用于测试集）
        zero_cols: 需要删除的列索引（用于测试集）
        return_scaler: 是否返回归一化器和零列索引
        
    Returns:
        tuple: (X_images, y, scaler, zero_cols) 或 (X_images, y)
    """
    print("\n开始转换数据为图像格式...")
    
    # 1. 分离特征和标签
    X = df.drop(['label'], axis=1, errors='ignore')
    if 'attack_class' in df.columns:
        X = X.drop('attack_class', axis=1)
    
    y = np.asarray(df['label'].values)
    
    # 2. 数据预处理（数值化类别特征）
    # protocol_type (列1), service (列2), flag (列3)
    X_processed = X.copy()
    
    categorical_cols = ['protocol_type', 'service', 'flag']
    for col in categorical_cols:
        if col in X_processed.columns:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
    
    # 3. 转换为float
    X_array = X_processed.values.astype(np.float32)
    
    # 4. 删除全零列（仅在训练时确定）
    if zero_cols is None:
        zero_cols = np.where(np.all(X_array == 0, axis=0))[0]
        if len(zero_cols) > 0:
            X_array = np.delete(X_array, zero_cols, axis=1)
            print(f"删除 {len(zero_cols)} 个全零特征后，维度：{X_array.shape[1]}")
    else:
        # 测试集使用相同的零列
        X_array = np.delete(X_array, zero_cols, axis=1)
    
    # 5. 归一化
    if scaler is None:
        scaler = MinMaxScaler()
        X_array = scaler.fit_transform(X_array)
    else:
        X_array = scaler.transform(X_array)
    
    # 6. 转换为图像（121维 → 11×11）
    n_features = image_size * image_size
    if X_array.shape[1] > n_features:
        X_array = X_array[:, :n_features]  # 截断
        print(f"特征维度截断到 {n_features}")
    elif X_array.shape[1] < n_features:
        # 补零
        padding = np.zeros((X_array.shape[0], n_features - X_array.shape[1]))
        X_array = np.hstack([X_array, padding])
        print(f"特征维度补零到 {n_features}")
    
    # 重塑为图像 (samples, 1, height, width)
    X_images = X_array.reshape(-1, 1, image_size, image_size)
    
    print(f"图像数据形状: {X_images.shape}")
    print(f"标签形状: {y.shape}")
    
    if return_scaler:
        return X_images, y, scaler, zero_cols
    else:
        return X_images, y


def load_raw(path: str) -> pd.DataFrame:
    """加载原始数据，返回 DataFrame（向后兼容）"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Raw data not found: {path}")
    return pd.read_csv(p)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """数据清洗：去除空值、异常值检测等"""
    return df.dropna().copy()


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """归一化（示例使用 min-max）"""
    numeric = df.select_dtypes(include="number")
    norm = (numeric - numeric.min()) / (numeric.max() - numeric.min())
    df[numeric.columns] = norm
    return df


def save_processed(df: pd.DataFrame, out_path: str) -> None:
    """保存处理后的数据"""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


if __name__ == "__main__":
    # 快速验证数据
    print("="*60)
    print("KDD Cup 99 数据集加载测试")
    print("="*60)
    
    train_df, test_df, features, attack_map = load_essential_data()
    
    print("\n数据统计：")
    print(f"训练集攻击类型分布（前10）：")
    print(train_df['label'].value_counts().head(10))
    print(f"\n测试集攻击类型分布（前10）：")
    print(test_df['label'].value_counts().head(10))
    
    # 验证数据量
    train_mapped, test_mapped = verify_data_counts(train_df, test_df, attack_map)
    
    # 创建图像数据集（示例：只处理前1000条）
    print("\n" + "="*60)
    print("图像数据集转换测试（前1000条）")
    print("="*60)
    sample_train = train_mapped.head(1000)
    X_images, y, scaler, zero_cols = cast(
        Tuple[np.ndarray, np.ndarray, MinMaxScaler, np.ndarray],
        create_image_dataset(sample_train, return_scaler=True)
    )
    
    print("\n测试完成！")
