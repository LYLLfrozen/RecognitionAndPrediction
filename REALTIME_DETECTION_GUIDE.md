# 实时流量入侵检测系统使用指南

本模块实现了基于 CNN-LSTM 模型的实时网络流量入侵检测系统。它能够捕获真实的网络数据包，提取 KDD Cup 99 标准特征，并实时识别流量类型（正常或攻击）。

## 1. 系统架构

- **流量捕获 (`realtime/traffic_capture.py`)**: 使用 `scapy` 监听网络接口，聚合数据包为连接记录。
- **特征提取 (`realtime/feature_extractor.py`)**: 将原始连接转换为 KDD Cup 99 数据集的 41 维特征。
- **实时处理 (`realtime/realtime_processor.py`)**: 核心控制器，协调捕获、提取、预处理和模型推理。
- **启动脚本 (`run_realtime_detection.py`)**: 用户入口，负责初始化和展示结果。

## 2. 准备工作

在运行实时检测之前，请确保满足以下条件：

### 2.1 安装依赖
需要安装 `scapy` 用于抓包：
```bash
pip install scapy
```

### 2.2 准备数据处理器状态
系统需要加载与训练时一致的归一化参数。如果 `data/processed_data/processor_state.pkl` 不存在，请运行：

```bash
# 生成处理器状态文件
python3 data/save_load_data.py --action save --sample-size 50000
```

### 2.3 准备模型
确保 `model/saved_models/` 目录下有训练好的模型文件：
- `cnn_lstm_full.pth` (优先使用，5分类模型)
- 或 `cnn_lstm_improved.pth`

## 3. 启动实时检测

使用 `sudo` 权限运行启动脚本（抓包需要 root 权限）：

```bash
sudo python3 run_realtime_detection.py
```

**运行效果：**
系统将持续运行，并在终端实时打印检测到的连接：

```text
[10:30:01] TCP 192.168.1.5:54321 -> 192.168.1.1:80 | normal (99.8%)
[10:30:02] TCP 192.168.1.5:54322 -> 192.168.1.1:80 | !!! DOS (85.2%) !!!
```

- **正常流量**：显示为白色/默认颜色。
- **攻击流量**：显示为 `!!! 攻击类型 (置信度) !!!`，便于快速识别。
- **状态检查**：每 5 秒打印一次原始数据包计数，证明系统正在运行。

按 `Ctrl+C` 可停止检测并查看统计信息。

## 4. 模拟攻击测试

为了验证系统的检测能力，可以使用提供的攻击模拟脚本 `simulate_attacks.py`。

请打开一个新的终端窗口（保持检测系统在另一个窗口运行）：

### 4.1 模拟 DoS 攻击 (SYN Flood)
向目标发送大量 TCP SYN 包，模拟拒绝服务攻击。

```bash
# 向本机 80 端口发送 500 个伪造包
sudo python3 simulate_attacks.py dos --target 127.0.0.1 --port 80 --count 500
```
**预期结果**：检测系统应识别出大量 `dos` 或 `neptune` 类型的连接。

### 4.2 模拟端口扫描 (Probe)
扫描目标的常用端口。

```bash
# 扫描本机常用端口
sudo python3 simulate_attacks.py probe --target 127.0.0.1
```
**预期结果**：检测系统应识别出 `probe`、`portsweep` 或 `satan` 类型的连接。

### 4.3 产生正常流量
您可以通过正常的网络活动产生 `normal` 流量：
- 打开浏览器访问网页
- 在终端执行 `ping www.baidu.com`
- 使用 `curl` 命令

## 5. 常见问题

**Q: 为什么一直显示 "已捕获原始数据包 0"？**
A: 可能是监听的网卡接口不正确。`scapy` 默认监听默认网卡。如果需要指定网卡（如 `eth0` 或 `wlan0`），请修改 `run_realtime_detection.py` 中的初始化代码：
```python
processor = RealtimeProcessor(..., interface="eth0")
```

**Q: 报错 "Permission denied"**
A: 网络抓包和发送伪造数据包都需要系统底层权限，请务必使用 `sudo` 运行脚本。

**Q: 报错 "Socket ... failed with '<' not supported"**
A: 这是 `scapy` 在某些环境下的已知问题，已在最新代码中修复（通过传递 `count=0` 而非 `None`）。如果仍遇到问题，请确保代码已更新。

启动项目
python3 run_realtime_detection.py --interface lo0

模拟dos攻击
python3 simulate_attacks.py dos --target 127.0.0.1 --port 80 --count 5000

模拟正常请求

