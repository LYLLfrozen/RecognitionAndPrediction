# KDD Cup 99 数据集处理模块

本模块提供了完整的KDD Cup 99数据集加载、验证和图像化处理功能。

## 文件说明

### 1. `data_processing.py` - 基础数据处理
提供基本的数据加载和转换函数。

**主要功能：**
- `load_essential_data()` - 加载KDD Cup 99数据集
- `verify_data_counts()` - 验证数据分布是否符合论文
- `create_image_dataset()` - 将数据转换为11×11图像格式

**使用示例：**
```python
from data.data_processing import load_essential_data, verify_data_counts

# 加载数据
train_df, test_df, features, attack_map = load_essential_data()

# 验证数据分布
train_mapped, test_mapped = verify_data_counts(train_df, test_df, attack_map)
```

### 2. `data_processor.py` - 面向对象的数据处理器（推荐）
提供更健壮的数据处理类，自动处理训练集和测试集的一致性问题。

**主要特性：**
- ✅ 自动处理类别特征编码的一致性
- ✅ 处理测试集中的未知类别
- ✅ 裁剪异常值到[0, 1]范围
- ✅ 保持训练集和测试集的预处理一致性

**使用示例：**
```python
from data.data_processor import KDDDataProcessor, process_kdd_data

# 方法1：使用便捷函数
X_train, y_train, X_test, y_test, processor = process_kdd_data(
    train_df, test_df, image_size=11
)

# 方法2：使用类
processor = KDDDataProcessor(image_size=11)
X_train, y_train = processor.fit_transform(train_df)
X_test, y_test = processor.transform(test_df)
```

### 3. `quick_start.py` - 快速入门脚本
快速开始使用数据处理模块，展示完整的数据加载和处理流程。

**运行：**
```bash
cd /Users/lyll/Documents/class/毕设/RecognitionAndPrediction
python3 data/quick_start.py
```

### 4. `save_load_data.py` - 数据保存和加载工具
处理并保存数据，或加载已处理的数据用于模型训练。

**使用：**
```bash
# 保存处理后的数据
python3 data/save_load_data.py --action save --sample-size 50000

# 加载数据
python3 data/save_load_data.py --action load
```

## 数据集文件结构

```
data/
├── raw_data/
│   ├── kddcup.data_10_percent.gz  # 训练集（10%子集，494,021条）
│   ├── corrected.gz                # 测试集（311,029条）
│   ├── kddcup.names                # 特征名称
│   └── training_attack_types       # 攻击类型映射
├── processed_data/                 # 处理后的数据保存位置
├── data_processing.py              # 基础处理函数
├── data_processor.py               # 数据处理器类（推荐）
├── quick_start.py                  # 快速入门脚本
└── save_load_data.py               # 保存/加载工具
```

## 数据集说明

### 特征信息
- **特征数量**: 41个特征 + 1个标签
- **类别特征**: protocol_type, service, flag
- **数值特征**: 38个（duration, src_bytes, dst_bytes等）

### 攻击类型（5大类）
1. **Normal** - 正常流量
2. **DoS** - 拒绝服务攻击（smurf, neptune, back等）
3. **Probe** - 探测扫描（portsweep, ipsweep, nmap, satan）
4. **R2L** - 远程到本地攻击（guess_passwd, ftp_write, warezclient等）
5. **U2R** - 用户到根权限攻击（buffer_overflow, loadmodule, rootkit等）

### 数据分布（论文表1）

**训练集（kddcup.data_10_percent）:**
- Normal: 97,278
- DoS: 391,458
- Probe: 4,107
- U2R: 52
- R2L: 1,126
- **总计: 494,021**

**测试集（corrected）:**
- Normal: 60,593
- DoS: 229,853
- Probe: 4,166
- U2R: 228
- R2L: 16,189
- **总计: 311,029**

## 数据处理流程

### 1. 加载数据
```python
from data.data_processing import load_essential_data

train_df, test_df, features, attack_map = load_essential_data()
```

### 2. 验证和映射攻击类型
```python
from data.data_processing import verify_data_counts

train_mapped, test_mapped = verify_data_counts(train_df, test_df, attack_map)
```

### 3. 转换为图像格式（推荐使用KDDDataProcessor）
```python
from data.data_processor import process_kdd_data

X_train, y_train, X_test, y_test, processor = process_kdd_data(
    train_mapped, test_mapped, image_size=11
)

# 数据形状：(samples, 1, 11, 11)
# 值范围：[0.0, 1.0]
print(f"训练集: {X_train.shape}")
print(f"测试集: {X_test.shape}")
```

### 4. 保存处理后的数据（可选）
```python
import numpy as np

np.save('data/processed_data/train_images.npy', X_train)
np.save('data/processed_data/train_labels.npy', y_train)
np.save('data/processed_data/test_images.npy', X_test)
np.save('data/processed_data/test_labels.npy', y_test)
```

## 技术细节

### 图像转换方法
1. **类别特征编码**: 使用LabelEncoder将protocol_type, service, flag转为数值
2. **删除零列**: 删除所有样本都为0的特征列（通常删除7列）
3. **归一化**: 使用MinMaxScaler归一化到[0,1]范围
4. **异常值处理**: clip超出范围的值（处理测试集中的新攻击模式）
5. **维度调整**: 调整到121维（11×11）
6. **重塑**: reshape为(N, 1, 11, 11)图像格式

### 为什么选择11×11图像？
- KDD Cup 99原始特征：41个
- 处理类别特征后：约34-38个数值特征
- 删除全零列后：约27-34个有效特征
- 补零到121维：11 × 11 = 121
- 适合CNN处理的图像尺寸

## 常见问题

### Q1: 测试集分布与论文不符？
**A:** 这是正常的。测试集（corrected.gz）包含一些训练集中没有的新攻击类型（如snmpgetattack, mailbomb等）。这些新攻击会被归类到对应的5大类中，但某些攻击可能无法映射，导致分布略有差异。

### Q2: 为什么需要裁剪值到[0,1]？
**A:** 测试集中有些特征值远超训练集范围（如src_bytes），MinMaxScaler会产生>1的值。裁剪可以确保所有值都在合理范围内。

### Q3: 如何处理未知的类别特征？
**A:** `KDDDataProcessor`会自动将测试集中未见过的service类型等编码为新的数值，确保不会出错。

## 下一步

数据准备完成后，可以：
1. 使用图像数据训练CNN-LSTM模型 (`model/woa_cnn_lstm/`)
2. 实现WOA优化器进行超参数优化
3. 构建垂直联邦学习框架 (`model/vfl_model/`)
4. 在GUI中展示实时检测结果 (`gui/`)

## 依赖包

```bash
pip install pandas numpy scikit-learn
```

或使用项目的requirements.txt：
```bash
pip install -r requirements.txt
```
