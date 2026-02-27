# RecognitionAndPrediction

此仓库包含网络入侵检测的完整解决方案：

- ✨ **垂直联邦学习 (VFL) with PrivBox**: 使用多进程模拟多参与方，结合PrivBox协议提供隐私保护。位于 [federated_learning/](federated_learning/)
- **WOA + CNN + LSTM**: 使用鲸鱼优化算法优化的混合模型进行网络流量预测。位于 [model/woa_cnn_lstm](model/woa_cnn_lstm/model_train.py)

## 🚀 垂直联邦学习 (VFL) - 最新功能！

### 特性
- ✅ **垂直联邦学习**: 不同参与方拥有相同样本的不同特征
- ✅ **PrivBox隐私保护**: 秘密共享 + 差分隐私 + 安全聚合
- ✅ **多进程模拟**: 支持2-4个参与方的协同训练
- ✅ **通信成本追踪**: 记录每轮训练的通信开销
- ✅ **完整测试套件**: 验证所有隐私保护机制

### 快速开始
```bash
# 1. 测试VFL系统
(已移除测试脚本 `test_vfl.py`) 

# 2. 训练VFL模型
python3 train_vfl.py

# 或使用一键脚本
./run_vfl.sh
```

### 详细文档
- 📖 [VFL完整指南](VFL_README.md) - 架构、使用方法、技术细节
-- 🧪 项目中的测试脚本已移除
- 🎯 [训练脚本](train_vfl.py) - 开始训练

### VFL系统架构
```
服务器/协调器 (顶层模型 + 隐私保护)
    ↕️ 加密嵌入向量和梯度
参与方1(主动) | 参与方2(被动) | ... | 参与方N(被动)
特征子集1     | 特征子集2     |     | 特征子集N
底层模型1     | 底层模型2     |     | 底层模型N
```

## 📦 数据集处理

**KDD Cup 99数据集已完全实现！** 🎉

### 快速开始
```bash
# 最简单：快速入门
python3 data/quick_start.py

# 保存处理后的数据（用于模型训练）
python3 data/save_load_data.py --action save --sample-size 50000

# 加载已处理的数据
python3 data/save_load_data.py --action load
```

### 数据处理功能
- ✅ 自动加载KDD Cup 99数据集（训练集494,021条 + 测试集311,029条）
- ✅ 验证数据分布（对照论文表1）
- ✅ 将数据转换为11×11图像格式
- ✅ 处理未知类别和异常值
- ✅ 保持训练集和测试集的一致性
- ✅ 支持数据保存和快速加载

### 详细文档
- 📖 [使用指南](data/USAGE_GUIDE.md) - 快速入门和使用示例
- 📖 [详细文档](data/README.md) - 完整API文档
- 📖 [实现总结](data/IMPLEMENTATION_SUMMARY.md) - 技术细节

### 数据格式
```python
# 输出格式
X_train.shape = (N, 1, 11, 11)  # 图像格式
X_train.dtype = float64
Value range: [0.0, 1.0]

# 直接用于CNN/LSTM训练
model.fit(X_train, y_train)
```

**环境依赖**

- Python 3.14
- 主要依赖：`numpy`, `pandas`, `scikit-learn`, `torch`, `tqdm`，`matplotlib`。
- 安装示例：

```bash
python -m pip install -r requirements.txt
```

（如果仓库里没有 `requirements.txt`，请创建并包含上述包及其版本。）

**代码位置指引**

- **数据处理（新增）**： [data/quick_start.py](data/quick_start.py) - 快速入门
- **数据处理器（推荐）**： [data/data_processor.py](data/data_processor.py)
- **数据加载**： [data/data_processing.py](data/data_processing.py)
- 联邦学习（训练入口）： [model/vfl_model/vfl_train.py](model/vfl_model/vfl_train.py)
- 联邦学习聚合器/代理： [model/vfl_model/vfl_aggregator.py](model/vfl_model/vfl_aggregator.py) 和 [model/vfl_model/vfl_agent.py](model/vfl_model/vfl_agent.py)
- 预测模型训练入口： [model/woa_cnn_lstm/model_train.py](model/woa_cnn_lstm/model_train.py)
- 预测模型定义： [model/woa_cnn_lstm/cnn_lstm_model.py](model/woa_cnn_lstm/cnn_lstm_model.py)
- 优化器/算法： [model/woa_cnn_lstm/woa_optimizer.py](model/woa_cnn_lstm/woa_optimizer.py)
- 全局与模型参数配置： [config/model_config.py](config/model_config.py)

**联邦学习（多进程）训练说明 — 网络流量识别**

方案概述：使用多进程在一台机器上模拟多个客户（clients/agents）。每个进程加载本地数据分片，执行本地训练若干轮（local epochs），随后将模型权重/梯度或必要信息发送给聚合进程（aggregator），聚合后分发全局模型并继续下一轮。该流程在 `model/vfl_model` 中已有基础实现。

运行要点：

- 使用 Python 的 `multiprocessing` 或 `torch.multiprocessing` 启动多个训练进程。
- 每个客户端进程应读取不同的数据分片（或使用 `data_processing.py` 进行分割），避免数据竞争。
- 聚合采用安全加权平均（如 FedAvg）或论文中指定的聚合策略。

示例（多进程模拟 4 个客户端）：

```bash
# 在项目根目录执行
python -u model/vfl_model/vfl_train.py \
	--mode multiprocess \
	--num_clients 4 \
	--rounds 50 \
	--local_epochs 5 \
	--batch_size 64 \
	--lr 0.001 \
	--aggregate_method fedavg
```

示例参数说明（建议按论文1替换或微调）：

- `num_clients`: 客户数量（多进程数）
- `rounds`（全局轮数）：50
- `local_epochs`（本地训练轮数/每轮）：5
- `batch_size`: 64
- `lr`（学习率）: 0.001
- `aggregate_method`: `fedavg`（或论文中指定的聚合方式）
- `workers`（可选）: 用于数据加载的线程数（每个进程内部）

在 `config/model_config.py` 中集中设置上述默认值：

- 打开并编辑 [config/model_config.py](config/model_config.py)
- 将 `federated` 或 `vfl` 部分的参数替换为论文1里推荐的值

注意事项：

- 若模型代码使用 GPU，请在每个进程中分配不同 GPU（或用 CPU 进行模拟）。
- 为避免多进程间的数据读取冲突，建议将每个客户端的数据单独放在 `data/processed_data/client_i/` 下或在启动时让每个进程读取不同的索引切片。

**VOA/WOA + CNN + LSTM 训练说明 — 网络流量预测**

方案概述：采用鲸鱼优化算法（VOA/WOA）搜索 CNN+LSTM 模型的超参数或权重初始化（参考论文2），随后训练 CNN-LSTM 组合模型以进行时间序列流量预测。实现细节在 `model/woa_cnn_lstm` 下。

运行要点：

- 首先运行进化/优化阶段（VOA/WOA），以搜索最佳超参数（如卷积核数、LSTM 隐层大小、学习率、批量大小、训练轮数等）。
- 使用搜索出的超参数训练最终的 CNN+LSTM 模型。

示例（单机运行，示例参数基于常见论文设置，请按论文2替换）：

```bash
# 先运行 VOA/WOA 搜索（示例）
python -u model/woa_cnn_lstm/model_train.py \
	--mode optimize \
	--optimizer voa \
	--population 30 \
	--generations 50 \
	--search_space config/woa_search_space.json

# 使用搜索结果训练最终模型
python -u model/woa_cnn_lstm/model_train.py \
	--mode train \
	--epochs 100 \
	--batch_size 64 \
	--lr 0.001 \
	--best_config outputs/best_voa_config.json
```

示例参数说明：

- `population`（种群大小）: 30
- `generations`（迭代代数）: 50
- `epochs`（训练轮数）: 100
- `batch_size`: 64
- `lr`: 0.001

在 `config/model_config.py` 或 `config/woa_search_space.json` 中定义需要搜索的超参数域（例如：卷积层数量、每层卷积核数量、LSTM 隐层维度、dropout、学习率范围等）。

**数据准备**

- 使用 `data/data_processing.py` 对 raw CSV（路径：`data/raw_data/`）做预处理、归一化与时间序列切片，输出到 `data/processed_data/`。
- 为联邦学习准备按客户端分割的数据集（例如 `data/processed_data/client_0/` ...）。

**日志与结果**

- 训练期间建议将每轮/每代的损失、准确率（识别任务）或 RMSE/MAPE（预测任务）记录到 `outputs/` 下的文件或 tensorboard 日志，以便横向对比。

## macOS：长期允许普通用户抓包（ChmodBPF / Wireshark）

在 macOS 上抓包需要访问 `/dev/bpf*` 设备。临时使用 `sudo chmod o+rw /dev/bpf*` 可以立即生效，但在重启后会失效。推荐使用 Wireshark 提供的 ChmodBPF 方案来长期允许非 root 用户抓包。

推荐步骤（长期、安全）：

1. 安装 Wireshark（Homebrew cask，会安装 ChmodBPF 支持）：
```bash
brew install --cask wireshark
```

2. 将当前用户加入 `access_bpf` 组（Wireshark 安装会创建该组）：
```bash
sudo dseditgroup -o edit -a $(whoami) -t user access_bpf
# 注：添加后需要登出并重新登录或重启，使组成员身份生效
```

3. 验证：
```bash
groups $(whoami)
ls -l /dev/bpf*
```

说明：加入 `access_bpf` 后，你无需再以 `sudo` 运行程序或每次 `chmod /dev/bpf*`，系统会在启动时维护设备权限，推荐用于长期开发与使用。

## 🛡️ 实时入侵检测系统

### 快速开始
```bash
# 1. 获取本机IP地址（Windows上不要使用127.0.0.1）
cd VFL
python get_local_ip.py

# 2. 启动实时监控（需要管理员权限）
python realtime_monitor.py --interface "以太网"

# 3. 在另一个终端模拟攻击（测试检测效果）
# Windows需要以管理员身份运行PowerShell
# 使用步骤1获取的真实IP地址（例如192.168.1.100）

# 模拟DoS攻击（SYN Flood）
python simulate_attacks.py dos --target 192.168.1.100 --port 80 --count 1000

# 模拟端口扫描（Probe）
python simulate_attacks.py probe --target 192.168.1.100

# 模拟R2L攻击（暴力破解）
python simulate_attacks.py r2l --target 192.168.1.100 --port 21 --count 10 --interval 1.0

# 模拟U2R攻击（提权尝试）
python simulate_attacks.py u2r --target 192.168.1.100 --port 80 --count 50 --interval 2.0
```

### 重要说明
1. **Windows用户**：发送原始网络包需要以**管理员身份运行PowerShell**
2. **目标IP选择**：在Windows上**不要使用127.0.0.1**，使用 `python get_local_ip.py` 获取本机真实IP
3. **网卡选择**：使用 `ipconfig` 查看可用网卡，选择实际使用的网卡（如"以太网"、"WLAN"等）
4. **正常流量识别**：系统已优化，正常浏览网页等行为会被正确识别为 `normal`
5. **攻击检测**：只有满足严格条件的流量才会被识别为攻击（避免误报）
6. **U2R攻击**：由于训练数据较少，可能识别准确率较低

### 检测阈值说明
- **DoS检测**：需要40+个快速连接 + 85%以上错误率
- **Probe检测**：需要100+个连接 + 访问15+个不同端口 + 90%服务多样性 + 70%以上错误率
- **R2L检测**：针对SSH/FTP/Telnet等登录服务的多次尝试
- **正常流量**：访问常见端口（80/443等）+ 连接数<80 + 低错误率



