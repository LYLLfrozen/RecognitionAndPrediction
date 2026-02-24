"""
训练进度可视化
实时显示训练过程
"""
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm


def print_banner():
    """打印欢迎横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║          🚀 网络流量分类 AI 模型训练系统 🚀                      ║
║                                                                  ║
║              CNN-LSTM 深度学习模型                               ║
║              KDD Cup 99 数据集                                   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_model_summary():
    """打印模型架构摘要"""
    print("\n" + "="*70)
    print("模型架构: CNN-LSTM 混合神经网络")
    print("="*70)
    
    architecture = """
    输入层 (1×11×11)
        ↓
    ┌─────────────────┐
    │  CNN 特征提取   │
    │  - Conv2D(32)   │  ← 第一卷积层
    │  - Conv2D(64)   │  ← 第二卷积层
    │  - Conv2D(128)  │  ← 第三卷积层
    └─────────────────┘
        ↓
    ┌─────────────────┐
    │  LSTM 序列建模  │
    │  - LSTM(64)     │  ← 第一LSTM层
    │  - LSTM(32)     │  ← 第二LSTM层
    └─────────────────┘
        ↓
    ┌─────────────────┐
    │  全连接层       │
    │  - Dense(128)   │
    │  - Dense(6)     │  ← 输出层
    └─────────────────┘
        ↓
    输出 (6类分类)
    """
    print(architecture)
    print("="*70)


def print_training_config(config):
    """打印训练配置"""
    print("\n" + "="*70)
    print("训练配置")
    print("="*70)
    print(f"  训练轮数 (Epochs):      {config['epochs']}")
    print(f"  批次大小 (Batch Size):  {config['batch_size']}")
    print(f"  学习率 (Learning Rate): {config['learning_rate']}")
    print(f"  优化器 (Optimizer):     Adam")
    print(f"  损失函数:               Sparse Categorical Crossentropy")
    print("="*70)


def print_data_info(X_train, y_train, X_test, y_test):
    """打印数据信息"""
    print("\n" + "="*70)
    print("数据信息")
    print("="*70)
    print(f"  训练集形状: {X_train.shape}")
    print(f"  训练集样本: {len(X_train):,} 个")
    print(f"  测试集形状: {X_test.shape}")
    print(f"  测试集样本: {len(X_test):,} 个")
    print(f"  类别数量:   {len(np.unique(y_train))} 类")
    print(f"  数值范围:   [{X_train.min():.3f}, {X_train.max():.3f}]")
    print("="*70)


def print_progress_bar(epoch, total_epochs, metrics):
    """打印训练进度"""
    progress = (epoch + 1) / total_epochs
    bar_length = 40
    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    print(f"\n进度: [{bar}] {progress*100:.1f}%")
    print(f"Epoch {epoch + 1}/{total_epochs}")
    print(f"  训练损失: {metrics['loss']:.4f} | 训练准确率: {metrics['acc']:.4f}")
    print(f"  验证损失: {metrics['val_loss']:.4f} | 验证准确率: {metrics['val_acc']:.4f}")


def print_final_results(test_acc, test_loss):
    """打印最终结果"""
    print("\n" + "="*70)
    print("训练完成！")
    print("="*70)
    
    # ASCII艺术
    if test_acc >= 0.90:
        status = "🎉 优秀!"
    elif test_acc >= 0.80:
        status = "✅ 良好"
    else:
        status = "⚠️  可以改进"
    
    print(f"\n  测试准确率: {test_acc:.4f} ({test_acc*100:.2f}%)  {status}")
    print(f"  测试损失:   {test_loss:.4f}")
    print("\n" + "="*70)


def print_checklist():
    """打印检查清单"""
    print("\n" + "="*70)
    print("✅ 训练检查清单")
    print("="*70)
    print("  [✓] 数据已加载")
    print("  [✓] 模型已构建")
    print("  [✓] 开始训练...")
    print("="*70 + "\n")


def simulate_training_display():
    """模拟训练显示（用于测试）"""
    print_banner()
    print_model_summary()
    
    config = {
        'epochs': 50,
        'batch_size': 64,
        'learning_rate': 0.001
    }
    print_training_config(config)
    
    # 模拟数据信息
    X_train = np.random.rand(10000, 1, 11, 11)
    y_train = np.random.randint(0, 6, 10000)
    X_test = np.random.rand(2000, 1, 11, 11)
    y_test = np.random.randint(0, 6, 2000)
    
    print_data_info(X_train, y_train, X_test, y_test)
    print_checklist()
    
    # 模拟几个epoch
    print("开始训练...")
    for epoch in range(5):
        metrics = {
            'loss': 1.0 - epoch*0.15,
            'acc': 0.5 + epoch*0.08,
            'val_loss': 1.1 - epoch*0.13,
            'val_acc': 0.48 + epoch*0.07
        }
        print_progress_bar(epoch, 50, metrics)
    
    print("\n...")
    print_final_results(0.92, 0.35)


if __name__ == "__main__":
    simulate_training_display()
