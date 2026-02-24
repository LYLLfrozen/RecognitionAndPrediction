# 实时监控诊断报告

## 问题根本原因

### 1. 训练集 vs 真实捕获的特征差异

**训练集（KDD Cup 99）:**
- 维度：115维高级统计特征（连接时长、流量统计、内容特征等）
- 数据范围：已经过 StandardScaler 归一化，均值0，标准差1
- 特征工程：包含窗口聚合、连接状态、协议分析等复杂特征

**真实网络捕获:**
- 维度：41维基础包特征（包长、协议、端口、TTL等）
- 数据范围：原始网络层数据（端口0-65535，包长0-1500等）
- 特征工程：仅从单个包提取，**缺少连接级统计和时序信息**

### 2. 为什么全部预测为 DOS

模型在训练时学习的是 KDD Cup 特征模式：
- DOS 攻击：大量短连接、高包速率、特定端口模式
- Normal：正常连接时长、合理流量统计
- Probe：端口扫描特征、连续探测模式

但真实捕获只提取了**单包瞬时特征**，缺少：
- 连接持续时间
- 累计包数/字节数
- 同目标连接数
- 错误率统计
- 内容特征（登录失败次数等）

**结论**：手动归一化的 41 维特征与训练集特征分布完全不同，模型无法正确分类。

## 解决方案

### 方案 1：完整特征工程（推荐但复杂）

实现连接追踪和窗口统计，生成与 KDD Cup 一致的 115 维特征：

```python
class ConnectionTracker:
    """追踪连接状态并生成统计特征"""
    
    def __init__(self, window_size=2):
        self.connections = {}  # {(src_ip, dst_ip, src_port, dst_port): stats}
        self.window_size = window_size
    
    def update(self, packet):
        # 提取5元组
        key = (src_ip, dst_ip, src_port, dst_port, proto)
        
        if key not in self.connections:
            self.connections[key] = {
                'start_time': time.time(),
                'duration': 0,
                'num_packets': 0,
                'total_bytes': 0,
                'src_bytes': 0,
                'dst_bytes': 0,
                # ... 更多统计
            }
        
        # 更新统计
        conn = self.connections[key]
        conn['num_packets'] += 1
        conn['total_bytes'] += packet_len
        conn['duration'] = time.time() - conn['start_time']
        
        # 生成115维特征向量
        return self.generate_kdd_features(conn)
```

**优点**：准确率高，与训练集对齐  
**缺点**：实现复杂，需要状态管理

### 方案 2：重新训练（当前最佳）

使用真实网络包特征重新训练模型：

1. 从 pcap 文件提取 41 维包特征作为训练集
2. 标注或使用已标注的攻击流量数据集（如 CIC-IDS2017）
3. 用 41 维特征训练新的 VFL 模型

**文件**: `train_vfl_with_packet_features.py`（需创建）

### 方案 3：混合模式（临时）

为真实流量添加置信度阈值和默认分类：

```python
# 在 classify() 中
if confidence < 0.95:  # 低置信度
    return "suspicious", confidence, probs
```

### 方案 4：启用规则引擎（快速验证）

对常见模式使用简单规则：

```python
def rule_based_classify(features):
    """基于规则的快速分类"""
    # 端口扫描检测
    if features[3] in [21, 22, 23, 3389]:  # 常见扫描端口
        return "probe"
    
    # SYN flood检测
    if features[1] == 6 and features[4] & 0x02:  # TCP SYN
        return "dos"
    
    # ICMP flood
    if features[1] == 1 and features[0] > 1000:
        return "dos"
    
    # 默认正常
    return "normal"
```

## 当前配置

- 模型：VFL (3方)，训练于 KDD Cup 99（115维）
- 测试集准确率：99.67%
- 真实流量准确率：**无法准确评估**（特征不匹配）

## 测试验证

### 测试集（121维，正常工作）
```
准确率: 99.67% (299/300)
平均置信度: 0.9964
预测分布正常
```

### 真实包（41维，特征不匹配）
```
预处理后范围: [0.0, 0.85]  ✓ 正常
预测: 全部 DOS  ✗ 不准确
原因: 特征分布与训练集不同
```

## 建议操作

### 立即可行（演示用）

1. **启用模拟模式**（默认）：
   ```bash
   python3 realtime_monitor.py
   ```
   使用测试集，准确率 99%+

2. **添加规则引擎**：修改 `preprocess_flow` 添加启发式规则

### 长期方案

1. **收集真实数据**：
   ```bash
   # 捕获流量到 pcap
   sudo tcpdump -i en0 -w real_traffic.pcap
   
   # 标注并提取特征
   python3 extract_packet_features.py real_traffic.pcap
   ```

2. **重新训练**：使用 41 维包特征训练新模型

3. **实现连接追踪**：生成完整的 KDD Cup 特征

## 文件说明

- `realtime_monitor.py`: 实时监控主程序（已修复维度问题）
- `preprocess_flow()`: 现支持 41/115/121 维输入
- `extract_flow_features()`: 从网络包提取 41 维特征

## 结论

**"全是 DOS" 不是因为模型训练差**（测试集准确率99.67%），而是因为：

1. 训练特征（KDD 115维）vs 真实特征（41维单包）**不匹配**
2. 缺少连接级统计和时序信息
3. 手动归一化无法弥补特征语义差异

**解决路径**：要么重新训练用包特征，要么实现完整特征工程。模拟模式（测试集）可以展示模型确实有效。
