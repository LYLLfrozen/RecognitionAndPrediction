"""
改进的数据处理器，支持训练集和测试集的一致性处理
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from typing import Tuple, Optional, Dict


class KDDDataProcessor:
    """KDD Cup 99数据集处理器"""
    
    def __init__(self, image_size: int = 11):
        """
        初始化数据处理器
        
        Args:
            image_size: 图像尺寸（默认11x11）
        """
        self.image_size = image_size
        self.n_features = image_size * image_size
        
        # 类别特征编码器
        self.encoders = {}
        self.categorical_cols = ['protocol_type', 'service', 'flag']
        
        # 数值归一化器
        self.scaler = None
        
        # 需要删除的全零列
        self.zero_cols = None
        
        # 特征名
        self.feature_names = None
        
    def fit(self, df: pd.DataFrame) -> 'KDDDataProcessor':
        """
        在训练集上拟合处理器
        
        Args:
            df: 训练数据
            
        Returns:
            self
        """
        print("拟合数据处理器...")
        
        # 分离特征
        X = self._get_features(df)
        
        # 1. 拟合类别编码器
        for col in self.categorical_cols:
            if col in X.columns:
                self.encoders[col] = LabelEncoder()
                self.encoders[col].fit(X[col].astype(str))
        
        # 2. 编码类别特征
        X_encoded = self._encode_categorical(X, fit=False)
        
        # 3. 转换为数组
        X_array = X_encoded.values.astype(np.float32)
        
        # 4. 找出全零列
        self.zero_cols = np.where(np.all(X_array == 0, axis=0))[0]
        if len(self.zero_cols) > 0:
            X_array = np.delete(X_array, self.zero_cols, axis=1)
            print(f"识别出 {len(self.zero_cols)} 个全零特征")
        
        # 5. 拟合归一化器
        self.scaler = MinMaxScaler()
        self.scaler.fit(X_array)
        
        print(f"处理器拟合完成，特征维度：{X_array.shape[1]}")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        转换数据为图像格式
        
        Args:
            df: 输入数据
            
        Returns:
            tuple: (X_images, y)
        """
        if self.scaler is None:
            raise ValueError("处理器未拟合，请先调用 fit()")
        
        # 分离特征和标签
        X = self._get_features(df)
        # 优先使用attack_class（5大类），如果不存在则使用label
        if 'attack_class' in df.columns:
            y = df['attack_class'].values
        elif 'label' in df.columns:
            y = df['label'].values
        else:
            y = None
        
        # 1. 编码类别特征
        X_encoded = self._encode_categorical(X, fit=False)
        
        # 2. 转换为数组
        X_array = X_encoded.values.astype(np.float32)
        
        # 3. 删除全零列
        if self.zero_cols is not None and len(self.zero_cols) > 0:
            X_array = np.delete(X_array, self.zero_cols, axis=1)
        
        # 4. 归一化
        X_array = self.scaler.transform(X_array)
        
        # 5. 裁剪到[0, 1]范围（处理测试集中的异常值）
        X_array = np.clip(X_array, 0, 1)
        
        # 5. 调整维度到 n_features
        X_array = self._adjust_dimensions(X_array)
        
        # 6. 重塑为图像
        X_images = X_array.reshape(-1, 1, self.image_size, self.image_size)
        
        # 确保 y 是 numpy 数组而不是 None
        if y is None:
            y = np.array([], dtype=np.int64)
        elif not isinstance(y, np.ndarray):
            y = np.asarray(y)
        return X_images, y
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        拟合并转换训练数据
        
        Args:
            df: 训练数据
            
        Returns:
            tuple: (X_images, y)
        """
        self.fit(df)
        return self.transform(df)
    
    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取特征（去除标签列）"""
        # 删除所有标签相关的列
        cols_to_drop = ['label', 'attack_class']
        X = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
        return X
    
    def _encode_categorical(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        编码类别特征
        
        Args:
            X: 特征数据
            fit: 是否拟合编码器
            
        Returns:
            编码后的数据
        """
        X_encoded = X.copy()
        
        for col in self.categorical_cols:
            if col in X_encoded.columns:
                if fit:
                    self.encoders[col] = LabelEncoder()
                    X_encoded[col] = self.encoders[col].fit_transform(
                        X_encoded[col].astype(str)
                    )
                else:
                    # 处理未见过的类别 - 映射到最大编码值+1
                    le = self.encoders[col]
                    max_code = len(le.classes_)
                    
                    def safe_transform(x):
                        if x in le.classes_:
                            return le.transform([x])[0]
                        else:
                            return max_code  # 未知类别编码
                    
                    X_encoded[col] = X_encoded[col].astype(str).apply(safe_transform)
        
        return X_encoded
    
    def _adjust_dimensions(self, X: np.ndarray) -> np.ndarray:
        """
        调整特征维度到 n_features
        
        Args:
            X: 输入数组
            
        Returns:
            调整后的数组
        """
        if X.shape[1] > self.n_features:
            # 截断
            X = X[:, :self.n_features]
        elif X.shape[1] < self.n_features:
            # 补零
            padding = np.zeros((X.shape[0], self.n_features - X.shape[1]))
            X = np.hstack([X, padding])
        
        return X
    
    def save(self, filepath: str):
        """保存处理器状态"""
        import pickle
        state = {
            'image_size': self.image_size,
            'n_features': self.n_features,
            'encoders': self.encoders,
            'scaler': self.scaler,
            'zero_cols': self.zero_cols,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"处理器状态已保存至 {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'KDDDataProcessor':
        """加载处理器状态"""
        import pickle
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        processor = cls(image_size=state['image_size'])
        processor.n_features = state['n_features']
        processor.encoders = state['encoders']
        processor.scaler = state['scaler']
        processor.zero_cols = state['zero_cols']
        processor.feature_names = state.get('feature_names')
        
        return processor

    def get_stats(self) -> Dict:
        """获取处理器统计信息"""
        return {
            'image_size': self.image_size,
            'n_features': self.n_features,
            'zero_cols_count': len(self.zero_cols) if self.zero_cols is not None else 0,
            'categorical_encoders': {
                col: len(enc.classes_) 
                for col, enc in self.encoders.items()
            },
            'is_fitted': self.scaler is not None
        }


# 便捷函数
def process_kdd_data(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                     image_size: int = 11) -> Tuple:
    """
    处理KDD训练集和测试集
    
    Args:
        train_df: 训练数据
        test_df: 测试数据
        image_size: 图像尺寸
        
    Returns:
        tuple: (X_train, y_train, X_test, y_test, processor)
    """
    print("\n" + "="*60)
    print("使用 KDDDataProcessor 处理数据")
    print("="*60)
    
    # 创建处理器
    processor = KDDDataProcessor(image_size=image_size)
    
    # 处理训练集
    print("\n处理训练集...")
    X_train, y_train = processor.fit_transform(train_df)
    print(f"训练集图像形状: {X_train.shape}")
    print(f"训练集值范围: [{X_train.min():.4f}, {X_train.max():.4f}]")
    
    # 处理测试集
    print("\n处理测试集...")
    X_test, y_test = processor.transform(test_df)
    print(f"测试集图像形状: {X_test.shape}")
    print(f"测试集值范围: [{X_test.min():.4f}, {X_test.max():.4f}]")
    
    # 显示统计信息
    print("\n处理器统计信息:")
    stats = processor.get_stats()
    for key, val in stats.items():
        print(f"  {key}: {val}")
    
    return X_train, y_train, X_test, y_test, processor


if __name__ == "__main__":
    # 测试代码
    from data_processing import load_essential_data, verify_data_counts
    
    print("加载数据...")
    train_df, test_df, features, attack_map = load_essential_data()
    train_mapped, test_mapped = verify_data_counts(train_df, test_df, attack_map)
    
    # 使用前10000条训练数据和5000条测试数据
    sample_train = train_mapped.head(10000)
    sample_test = test_mapped.head(5000)
    
    # 处理数据
    X_train, y_train, X_test, y_test, processor = process_kdd_data(
        sample_train, sample_test, image_size=11
    )
    
    print("\n✓ 测试完成！")
