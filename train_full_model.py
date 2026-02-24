#!/usr/bin/env python3
"""
完整版CNN-LSTM模型训练脚本
支持5种攻击类别的识别
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 切换到项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

# 导入模型
from model.fl_woa_cnn_lstm.cnn_lstm_model import CNNLSTMModel

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else "cpu")

print("="*70)
print("完整版CNN-LSTM模型训练 - 5类攻击识别")
print("="*70)
print(f"使用设备: {device}")

# 超参数
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
EPOCHS = 50
PATIENCE = 10

def load_data():
    """加载处理后的数据"""
    print("\n【1/6】加载数据...")
    
    data_dir = 'data/processed_data'
    
    X_train = np.load(os.path.join(data_dir, 'train_images.npy'))
    y_train = np.load(os.path.join(data_dir, 'train_labels.npy'), allow_pickle=True)
    X_test = np.load(os.path.join(data_dir, 'test_images.npy'))
    y_test = np.load(os.path.join(data_dir, 'test_labels.npy'), allow_pickle=True)
    
    # 构造元数据字典，五类攻击分类
    processor = {
        'class_names': ['normal', 'dos', 'probe', 'u2r', 'r2l'],
        'n_classes': 5,
        'image_size': 11
    }
    
    print(f"  训练集: {X_train.shape}, 标签: {y_train.shape}")
    print(f"  测试集: {X_test.shape}, 标签: {y_test.shape}")
    print(f"  类别: {processor['class_names']}")
    print(f"  类别数: {processor['n_classes']}")
    
    # 标签处理：如果是字符串标签（5大类），则编码为数字
    from sklearn.preprocessing import LabelEncoder
    
    if not np.issubdtype(np.asarray(y_train).dtype, np.number):
        # 字符串标签，需要编码
        le = LabelEncoder()
        # 合并训练和测试标签以确保一致的编码
        all_labels = np.concatenate([y_train, y_test])
        le.fit(all_labels)
        
        y_train = np.asarray(le.transform(y_train), dtype=np.int64)
        y_test = np.asarray(le.transform(y_test), dtype=np.int64)
        
        # 更新类别名称为实际顺序
        processor['class_names'] = le.classes_.tolist()
        processor['n_classes'] = len(le.classes_)
        
        print(f"\n  标签编码映射:")
        for idx, name in enumerate(le.classes_):
            count = np.sum(y_train == idx)
            print(f"    {idx}: {name} (训练集: {count} 样本)")
    else:
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)
    
    # 检查标签范围
    unique_train = np.unique(y_train)
    unique_test = np.unique(y_test)
    print(f"\n  训练集标签范围: {unique_train}")
    print(f"  测试集标签范围: {unique_test}")
    
    return X_train, y_train, X_test, y_test, processor

def create_data_loaders(X_train, y_train, X_test, y_test):
    """创建数据加载器"""
    print("\n【2/6】创建数据加载器...")
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # 创建数据集
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"  训练批次: {len(train_loader)}")
    print(f"  测试批次: {len(test_loader)}")
    
    return train_loader, test_loader

def compute_class_weights(y_train, n_classes):
    """计算类别权重"""
    from sklearn.utils.class_weight import compute_class_weight
    
    # 只计算训练集中实际存在的类别的权重
    unique_classes = np.unique(y_train)
    
    # 计算存在的类别的权重
    weights_dict = {}
    if len(unique_classes) > 1:
        class_weights_arr = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=y_train
        )
        weights_dict = dict(zip(unique_classes, class_weights_arr))
    
    # 为所有类别分配权重（不存在的类别权重为1.0）
    class_weights = np.ones(n_classes, dtype=np.float32)
    for cls, weight in weights_dict.items():
        if cls < n_classes:
            class_weights[cls] = weight
    
    print(f"  实际存在的类别: {unique_classes}")
    
    return torch.FloatTensor(class_weights).to(device)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='训练', leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def validate(model, test_loader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def plot_training_history(history, save_path='training_history_full.png'):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n训练曲线已保存: {save_path}")

def evaluate_model(model, test_loader, class_names, device):
    """详细评估模型"""
    print("\n【6/6】详细评估模型...")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='评估'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # 分类报告
    print("\n分类报告:")
    # 计算实际出现的标签并构建对应的 target_names 子集，避免 sklearn 报错
    import numpy as _np
    present_labels = _np.unique(_np.concatenate([_np.asarray(all_labels), _np.asarray(all_preds)]))
    present_labels = present_labels.astype(int).tolist()
    # 构造与 present_labels 对应的 target_names 子集
    try:
        present_names = [class_names[i] for i in present_labels]
    except Exception:
        present_names = [str(i) for i in present_labels]

    print(classification_report(all_labels, all_preds, labels=present_labels, target_names=present_names, digits=4))

    # 混淆矩阵（使用相同的 labels 顺序）
    cm = confusion_matrix(all_labels, all_preds, labels=present_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=present_names, yticklabels=present_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_full.png', dpi=300, bbox_inches='tight')
    print("\n混淆矩阵已保存: confusion_matrix_full.png")

def main():
    # 加载数据
    X_train, y_train, X_test, y_test, processor = load_data()
    
    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test)
    
    # 计算类别权重
    print("\n【3/6】计算类别权重...")
    class_weights = compute_class_weights(y_train, processor['n_classes'])
    print(f"  类别权重: {class_weights.cpu().numpy()}")
    
    # 创建模型
    print("\n【4/6】创建模型...")
    model_config = {
        'input_shape': (1, 11, 11),
        'num_classes': processor['n_classes'],
        'cnn_filters': [32, 64, 128],
        'lstm_units': 64,
        'dropout_rate': 0.5
    }
    model = CNNLSTMModel(config=model_config).to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # 训练
    print("\n【5/6】开始训练...")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  训练轮数: {EPOCHS}")
    print(f"  早停耐心: {PATIENCE}")
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(EPOCHS):
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 调整学习率
        scheduler.step(val_acc)
        
        # 打印进度
        print(f"Epoch {epoch+1}/{EPOCHS}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 早停和保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            
            # 保存最佳模型
            os.makedirs('model/saved_models', exist_ok=True)
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'processor': processor,
                'class_weights': class_weights.cpu()
            }
            torch.save(checkpoint, 'model/saved_models/cnn_lstm_full.pth')
            print(f"  ✓ 保存最佳模型 (验证准确率: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n早停触发! 最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")
                break
    
    # 绘制训练历史
    plot_training_history(history)
    
    # 加载最佳模型进行最终评估
    checkpoint = torch.load('model/saved_models/cnn_lstm_full.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 详细评估
    evaluate_model(model, test_loader, processor['class_names'], device)
    
    print("\n" + "="*70)
    print("训练完成！")
    print("="*70)
    print(f"最佳模型: model/saved_models/cnn_lstm_full.pth")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"最佳epoch: {best_epoch}")

if __name__ == '__main__':
    main()
