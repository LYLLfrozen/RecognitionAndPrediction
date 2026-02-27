# 攻击模拟与检测系统使用指南

## 目录
- [环境准备](#环境准备)
- [启动监控系统](#启动监控系统)
- [模拟攻击](#模拟攻击)
- [常见问题](#常见问题)

## 环境准备

### Windows 系统

1. **安装 Npcap**（必需）
   - 下载地址：https://npcap.com/#download
   - 安装时勾选 "Install Npcap in WinPcap API-compatible Mode"
   - 安装后重启计算机

2. **以管理员身份运行 PowerShell**
   - 右键点击 PowerShell 图标
   - 选择"以管理员身份运行"

3. **查看可用网卡**
   ```powershell
   ipconfig
   # 记下你正在使用的网卡名称，如"以太网"、"WLAN"等
   ```

## 启动监控系统

### 基本用法

```powershell
# 进入VFL目录
cd VFL

# 启动监控（自动选择网卡）
python realtime_monitor.py

# 或手动指定网卡
python realtime_monitor.py --interface "以太网"

# 使用模拟模式（测试集验证）
python realtime_monitor.py --sim
```

### 监控界面说明

```
实时网络流量监控
================================================================================
设备: cuda | 运行时间: 10.5秒 | 更新间隔: 2秒
--------------------------------------------------------------------------------

【总体统计】
  总流量包: 150
  处理速度: 14.29 包/秒
  队列长度: 0

【流量识别统计】
  normal  (正常流量    ): 145 ( 96.7%) ████████████████████████████████████████████████
  dos     (DoS攻击     ):   3 (  2.0%) █
  probe   (探测扫描    ):   2 (  1.3%) █

【最近识别】
  时间       识别类型       置信度      说明
  -------------------------------------------------------
  14:23:45  normal    0.900    高度确信
  14:23:46  normal    0.850    较为确定
```

## 模拟攻击

### ⚠️ 重要提示：目标IP地址选择

在 Windows 上，向 `127.0.0.1`（本地回环地址）发送原始网络包可能**无法被监控系统捕获**。

**推荐做法**：使用本机的真实IP地址

```powershell
# 1. 查看本机IP地址
ipconfig

# 查找 "以太网适配器" 或 "无线局域网适配器" 的 IPv4 地址
# 例如：192.168.1.100

# 2. 使用真实IP地址进行测试
python simulate_attacks.py dos --target 192.168.1.100 --port 80 --count 1000
```

**为什么不能用 127.0.0.1？**
- Windows 的网络栈对 loopback 地址有特殊处理
- 原始包不会经过真实的网卡驱动
- 监控系统无法在网卡层面捕获这些包

### 1. DoS 攻击（SYN Flood）

**特征**：大量快速连接，高错误率

```powershell
# 基本用法（使用本机真实IP）
python simulate_attacks.py dos --target 192.168.1.100 --port 80 --count 1000

# 参数说明：
# --target: 目标IP地址（使用本机IP，不要用127.0.0.1）
# --port: 目标端口
# --count: 发送包数量
```

**预期检测结果**：
- 识别类型：`dos`
- 置信度：> 0.90
- 检测条件：40+连接 + 85%错误率

### 2. 端口扫描（Probe）

**特征**：扫描多个不同端口

```powershell
# 扫描常用端口
python simulate_attacks.py probe --target 192.168.1.100
```

扫描端口列表：21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 445, 1433, 1521, 3306, 3389, 5432, 5900, 6379, 8080

**预期检测结果**：
- 识别类型：`probe`
- 置信度：> 0.88
- 检测条件：100+连接 + 访问15+不同端口 + 90%服务多样性

### 3. R2L 攻击（远程登录暴力破解）

**特征**：针对登录服务的多次尝试

```powershell
# FTP暴力破解
python simulate_attacks.py r2l --target 192.168.1.100 --port 21 --count 10 --interval 1.0

# SSH暴力破解
python simulate_attacks.py r2l --target 192.168.1.100 --port 22 --count 10 --interval 1.0

# 参数说明：
# --interval: 每次尝试的间隔（秒），推荐1.0以上
```

**预期检测结果**：
- 识别类型：`r2l`
- 置信度：> 0.75
- 检测条件：针对21/22/23/3389端口 + 多次尝试 + 有效载荷

### 4. U2R 攻击（提权尝试）

**特征**：异常的大型载荷

```powershell
python simulate_attacks.py u2r --target 192.168.1.100 --port 80 --count 50 --interval 2.0
```

**注意**：由于U2R训练数据较少，识别准确率可能较低，更多会被识别为 `normal` 或交由ML模型判断。

## 完整测试流程

### 步骤1：启动监控

```powershell
# 终端1（管理员）
cd VFL
python realtime_monitor.py --interface "以太网"
```

### 步骤2：生成正常流量

在浏览器中访问几个网站，观察监控界面，应该显示：
- 识别类型：`normal`
- 置信度：0.85-0.90

### 步骤3：模拟DoS攻击

```powershell
# 终端2（管理员）
cd VFL

# 先获取本机IP
python get_local_ip.py
# 假设获取到的IP是 192.168.1.100

python simulate_attacks.py dos --target 192.168.1.100 --port 80 --count 1000
```

观察监控界面，应该显示：
- 识别类型：`dos`
- 统计中 `dos` 数量增加

### 步骤4：模拟端口扫描

```powershell
python simulate_attacks.py probe --target 192.168.1.100
```

观察监控界面，应该显示：
- 识别类型：`probe`
- 统计中 `probe` 数量增加

### 步骤5：停止监控

按 `Ctrl+C` 停止监控，查看最终统计。

## 常见问题

### Q0: 为什么不能使用 127.0.0.1？（重要！）

**问题**：在Windows上向 `127.0.0.1`（本地回环地址）发送攻击流量时，提示错误或监控系统捕获不到。

**原因**：
- Windows的网络栈对loopback地址有特殊处理
- 发往127.0.0.1的原始网络包不经过物理网卡
- 监控系统在网卡层面进行抓包，无法捕获loopback流量

**解决方法**：
```powershell
# 1. 运行辅助工具获取本机IP
python get_local_ip.py

# 会显示类似输出：
# ✓ 本机主要IP地址: 192.168.1.100
# 推荐使用的IP地址: 192.168.1.100

# 2. 使用显示的真实IP地址
python simulate_attacks.py dos --target 192.168.1.100 --port 80 --count 1000
```

**⚠️ 重要**：后续所有攻击模拟命令都使用真实IP，不要使用127.0.0.1！

### Q1: 提示"需要root权限"

**Windows**：
- 必须以**管理员身份**运行 PowerShell
- 右键点击 PowerShell → "以管理员身份运行"

### Q2: 提示"No libpcap provider available"

**解决方法**：
1. 安装 Npcap：https://npcap.com/
2. 安装时勾选 "WinPcap API-compatible Mode"
3. 重启计算机

### Q3: 正常流量被识别为 probe

**原因**：浏览器可能并发打开很多连接

**解决方法**：
- 系统已优化，访问常见端口（80/443）的流量会被识别为 `normal`
- 如果仍有误判，检查是否真的是正常流量（看目标端口）

### Q4: DoS攻击没有被检测到

**可能原因**：
1. 包数量太少（需要1000+）
2. 发送速度太慢
3. 目标IP/端口不可达
4. **使用了127.0.0.1导致包无法被监控系统捕获**

**解决方法**：
```powershell
# 1. 先获取本机真实IP
python get_local_ip.py

# 2. 使用真实IP进行测试（假设是192.168.1.100）
python simulate_attacks.py dos --target 192.168.1.100 --port 80 --count 5000
```

### Q5: 无法捕获到任何流量

**检查清单**：
1. ✅ 是否以管理员身份运行
2. ✅ 是否安装了 Npcap
3. ✅ 网卡名称是否正确（使用 `ipconfig` 查看）
4. ✅ 防火墙是否阻止了流量

**测试方法**：
```powershell
# 使用模拟模式验证系统本身是否正常
python realtime_monitor.py --sim
```

### Q6: U2R攻击识别率低

**正常现象**：
- U2R在KDD数据集中样本很少（<200个）
- 模型对U2R的识别能力有限
- 大部分U2R会被识别为 `normal` 或其他类型

**建议**：
- 主要测试 DoS、Probe、R2L
- U2R仅作为参考

## 技术细节

### 检测算法

系统使用**混合检测器**（规则引擎 + 机器学习）：

1. **规则引擎**（快速检测）：
   - 正常流量：常见端口 + 低错误率
   - DoS：高频连接 + 高错误率
   - Probe：多端口 + 高服务多样性
   - R2L：登录端口 + 多次尝试

2. **机器学习**（复杂模式）：
   - VFL模型（3方联邦学习）
   - CNN特征提取 + 全连接分类

### 检测阈值

| 攻击类型 | 主要条件 | 次要条件 |
|---------|---------|---------|
| DoS | same_dst_count ≥ 40 | serror_rate ≥ 0.85 |
| Probe | same_dst_count ≥ 100 | diff_srv_rate ≥ 0.9 |
| R2L | 登录端口 (21/22/23/3389) | 多次有效载荷 |
| Normal | 常见端口 (80/443) | serror_rate < 0.6 |

### 流量特征

系统提取41维特征，包括：
- 基础特征：包长度、协议、端口、TTL、Flags
- 流统计：连接数、错误率、服务多样性
- 时间特征：2秒窗口内的统计

## 性能指标

在测试集上的表现：
- **准确率**：> 95%
- **DoS检测率**：> 98%
- **Probe检测率**：> 92%
- **R2L检测率**：> 85%
- **U2R检测率**：< 60%（样本太少）
- **误报率**：< 3%

## 下一步

1. 收集更多真实流量数据
2. 优化U2R检测算法
3. 添加更多攻击类型（如DDoS、SQL注入等）
4. 改进实时性能（目前约20包/秒）

## 参考资料

- KDD Cup 99数据集：http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
- 垂直联邦学习：[VFL_README.md](VFL_README.md)
- 混合检测器：[hybrid_detector.py](hybrid_detector.py)
