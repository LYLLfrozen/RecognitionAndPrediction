# RecognitionAndPrediction

本仓库是一个面向网络入侵检测的完整实验工程，包含：
- 垂直联邦学习（VFL）训练与推理
- PrivBox 隐私保护（秘密共享 + 差分隐私噪声）
- 实时流量捕获、特征提取、在线分类与可视化监控
- CNN-LSTM 单机模型（用于集中式基线）

## 目录结构（按真实代码）

- `VFL/`
  - `train_vfl.py`：VFL 训练主入口
  - `train_vfl_network.py`：另一套 VFL 训练流程
  - `realtime_monitor.py`：实时监控主程序（GUI/终端）
  - `run_monitor_gui.py`：GUI 启动器
  - `hybrid_detector.py`：ML 优先 + 规则兜底检测
  - `flow_tracker.py`：连接级统计和特征跟踪
  - `simulate_attacks.py`：DoS/Probe/R2L/U2R 攻击流量模拟
  - `federated_learning/`
    - `vfl_server.py`：服务器/协调器
    - `vfl_client.py`：主动方/被动方
    - `vfl_utils.py`：特征切分、模型切分、通信成本估计
    - `privbox.py`：隐私保护组件
- `data/`
  - `data_processing.py`：KDD Cup 99 加载与基础处理
  - `data_processor.py`：数据预处理与图像化
  - `save_load_data.py`：处理后数据保存/加载
- `model/`
  - `fl_woa_cnn_lstm/cnn_lstm_model.py`：CNN-LSTM 模型定义
  - `model_utils.py`：模型保存、加载、评估工具
- `realtime/`
  - `traffic_capture.py`：流量捕获
  - `feature_extractor.py`：特征提取
  - `realtime_processor.py`：实时处理流水线
- `test/`
  - `performance_test.py`、`scene_test.py`、`test_report.py`

## 当前可运行能力

### 1. VFL 训练

- 支持按特征维度垂直切分样本
- 支持主动方/被动方协同训练
- 支持顶层聚合模型训练
- 支持通信成本统计
- 支持 PrivBox 隐私保护开关

启动方式：

```bash
cd VFL
python3 train_vfl.py
```

### 2. 实时监控与检测

- 真实流量捕获（依赖 scapy）
- 模拟数据回放验证
- 混合检测策略：ML 主导，规则兜底
- GUI 实时展示：吞吐、分类分布、日志、最近检测记录

启动方式：

```bash
cd VFL
python3 run_monitor_gui.py
# 或
python3 realtime_monitor.py --gui
```

终端模式：

```bash
cd VFL
python3 realtime_monitor.py
# 或模拟模式
python3 realtime_monitor.py --sim
```

### 3. 数据处理

保存处理后数据：

```bash
python3 data/save_load_data.py --action save --sample-size 50000
```

加载处理后数据：

```bash
python3 data/save_load_data.py --action load
```

### 4. 攻击流量模拟（用于联调）

```bash
cd VFL
python3 simulate_attacks.py dos --target 127.0.0.1 --port 80 --count 500
python3 simulate_attacks.py probe --target 127.0.0.1
python3 simulate_attacks.py r2l --target 127.0.0.1 --port 21 --count 10 --interval 1.0
python3 simulate_attacks.py u2r --target 127.0.0.1 --port 80 --count 20 --interval 2.0
```

## 核心算法（仅保留代码已实现部分）

### 垂直联邦学习（VFL）

设第 k 方持有特征子集 $x_i^{(k)}$，底层嵌入为：

$$
h_i^{(k)} = f_k(x_i^{(k)};\theta_k)
$$

服务器拼接嵌入：

$$
z_i = \operatorname{Concat}(h_i^{(1)},\dots,h_i^{(K)})
$$

顶层分类：

$$
\hat{y}_i = g(z_i;\phi)
$$

损失最小化并反向切片梯度到各参与方，这是 `VFL/federated_learning/vfl_server.py` 与 `VFL/federated_learning/vfl_client.py` 的核心训练逻辑。

### CNN + LSTM（单机模型）

模型定义在 `model/fl_woa_cnn_lstm/cnn_lstm_model.py`：
- CNN 负责局部空间特征提取
- LSTM 负责时序依赖建模
- 全连接层输出最终类别

### PrivBox（已实现）

- 秘密共享：

$$
s = \sum_{k=1}^{K}s_k
$$

- 差分隐私噪声（高斯机制）：

$$
\tilde{g} = g + \mathcal{N}(0,\sigma^2I),\quad
\sigma = \frac{\sqrt{2\ln(1.25/\delta)}\,\Delta}{\epsilon}
$$

对应实现位于 `VFL/federated_learning/privbox.py`。

## 说明

README 仅保留当前仓库可直接对应到代码文件的能力与入口。若后续新增训练脚本或优化器，可再扩展文档章节。
