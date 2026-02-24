"""
保存和加载处理后数据的示例
"""

import sys
from pathlib import Path
import numpy as np
import pickle

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.data_processing import load_essential_data, verify_data_counts
from data.data_processor import process_kdd_data


def save_processed_data(output_dir='data/processed_data', sample_size=None):
    """
    处理并保存数据
    
    Args:
        output_dir: 输出目录
        sample_size: 样本大小（None表示使用全部数据）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("保存处理后的KDD Cup 99数据")
    print("="*70)
    
    # 1. 加载数据
    print("\n【1/4】加载原始数据...")
    train_df, test_df, features, attack_map = load_essential_data()
    train_mapped, test_mapped = verify_data_counts(train_df, test_df, attack_map)
    
    # 2. 采样（如果指定）
    if sample_size:
        print(f"\n使用采样数据: 训练集{sample_size}条, 测试集{sample_size//5}条")
        train_mapped = train_mapped.head(sample_size)
        test_mapped = test_mapped.head(sample_size // 5)
    else:
        print("\n使用全部数据（可能需要较长时间）")
    
    # 3. 处理数据
    print("\n【2/4】转换为图像格式...")
    X_train, y_train, X_test, y_test, processor = process_kdd_data(
        train_mapped, test_mapped, image_size=11
    )
    
    # 4. 保存数据
    print("\n【3/4】保存数据...")
    
    files = {
        'train_images.npy': X_train,
        'train_labels.npy': y_train,
        'test_images.npy': X_test,
        'test_labels.npy': y_test,
    }
    
    for filename, data in files.items():
        filepath = output_dir / filename
        np.save(filepath, data)
        print(f"  ✓ {filepath}")
    
    # 5. 保存处理器
    processor_path = output_dir / 'processor_state.pkl'
    processor.save(str(processor_path))
    print(f"  ✓ {processor_path}")
    
    # 保存attack_map
    attack_map_path = output_dir / 'attack_map.pkl'
    with open(attack_map_path, 'wb') as f:
        pickle.dump(attack_map, f)
    print(f"  ✓ {attack_map_path}")
    
    # 6. 保存元数据
    # 统计标签分布（使用5大类）
    unique_train_labels = set(y_train)
    unique_test_labels = set(y_test)
    train_label_dist = {str(label): int(np.sum(y_train == label)) for label in unique_train_labels}
    test_label_dist = {str(label): int(np.sum(y_test == label)) for label in unique_test_labels}
    
    metadata = {
        'train_shape': X_train.shape,
        'test_shape': X_test.shape,
        'image_size': 11,
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'n_train_labels': len(unique_train_labels),
        'n_test_labels': len(unique_test_labels),
        'train_label_distribution': train_label_dist,
        'test_label_distribution': test_label_dist,
        'value_range': [0.0, 1.0],
        'processor_stats': processor.get_stats()
    }
    
    metadata_path = output_dir / 'metadata.txt'
    with open(metadata_path, 'w') as f:
        for key, val in metadata.items():
            f.write(f"{key}: {val}\n")
    print(f"  ✓ {metadata_path}")
    
    print("\n【4/4】完成！")
    print(f"\n数据已保存到: {output_dir.absolute()}")
    print(f"总文件大小: {sum(f.stat().st_size for f in output_dir.glob('*.npy')) / 1024 / 1024:.2f} MB")
    
    return output_dir


def load_processed_data(input_dir='data/processed_data'):
    """
    加载处理后的数据
    
    Args:
        input_dir: 输入目录
        
    Returns:
        tuple: (X_train, y_train, X_test, y_test, processor)
    """
    input_dir = Path(input_dir)
    
    print("="*70)
    print("加载处理后的数据")
    print("="*70)
    
    # 检查文件是否存在
    required_files = [
        'train_images.npy',
        'train_labels.npy', 
        'test_images.npy',
        'test_labels.npy',
        'processor_state.pkl'
    ]
    
    for filename in required_files:
        filepath = input_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")
    
    # 加载数据
    print("\n加载数据文件...")
    X_train = np.load(input_dir / 'train_images.npy')
    y_train = np.load(input_dir / 'train_labels.npy', allow_pickle=True)
    X_test = np.load(input_dir / 'test_images.npy')
    y_test = np.load(input_dir / 'test_labels.npy', allow_pickle=True)
    
    print(f"  ✓ X_train: {X_train.shape}")
    print(f"  ✓ y_train: {y_train.shape}")
    print(f"  ✓ X_test: {X_test.shape}")
    print(f"  ✓ y_test: {y_test.shape}")
    
    # 加载处理器（与保存时的文件名保持一致）
    with open(input_dir / 'processor_state.pkl', 'rb') as f:
        processor = pickle.load(f)
    print(f"  ✓ processor")
    
    # 加载attack_map（如果存在）
    attack_map = None
    attack_map_path = input_dir / 'attack_map.pkl'
    if attack_map_path.exists():
        with open(attack_map_path, 'rb') as f:
            attack_map = pickle.load(f)
        print(f"  ✓ attack_map")
    
    # 读取元数据
    metadata_path = input_dir / 'metadata.txt'
    if metadata_path.exists():
        print("\n元数据信息:")
        with open(metadata_path, 'r') as f:
            print("  " + "  ".join(f.readlines()[:6]))
    
    print("\n✓ 数据加载完成！")
    
    return X_train, y_train, X_test, y_test, processor, attack_map


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='保存/加载处理后的KDD数据')
    parser.add_argument('--action', choices=['save', 'load', 'both'], default='both',
                        help='操作: save(保存), load(加载), both(两者都做)')
    parser.add_argument('--sample-size', type=int, default=50000,
                        help='采样大小（默认50000，设为0使用全部数据）')
    parser.add_argument('--output-dir', default='data/processed_data',
                        help='输出目录')
    
    args = parser.parse_args()
    
    if args.action in ['save', 'both']:
        sample_size = None if args.sample_size == 0 else args.sample_size
        output_dir = save_processed_data(
            output_dir=args.output_dir,
            sample_size=sample_size
        )
        print("\n" + "="*70)
    
    if args.action in ['load', 'both']:
        if args.action == 'both':
            print()  # 空行分隔
        result = load_processed_data(input_dir=args.output_dir)
        X_train, y_train, X_test, y_test, processor = result[:5]
        attack_map = result[5] if len(result) > 5 else None
        print("\n" + "="*70)
        print("数据已加载，可以开始训练模型！")
        if attack_map:
            print(f"攻击类型映射: {len(attack_map)} 种")
        print("="*70)
