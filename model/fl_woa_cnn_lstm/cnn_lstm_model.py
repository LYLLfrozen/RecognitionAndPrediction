"""
CNN-LSTM 网络流量分类模型 (PyTorch版本)
用于KDD Cup 99数据集的入侵检测
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM混合模型用于网络流量分类
    - CNN: 提取空间特征
    - LSTM: 捕获时序特征
    """
    
    def __init__(self, config=None):
        """
        Args:
            config: 模型配置字典，包含:
                - input_shape: 输入形状 (channels, height, width)
                - num_classes: 分类数量
                - cnn_filters: CNN滤波器数量列表
                - lstm_units: LSTM单元数量
                - dropout_rate: Dropout比率
        """
        super(CNNLSTMModel, self).__init__()
        
        self.config = config or self._default_config()
        
        input_channels = self.config['input_shape'][0]
        num_classes = self.config['num_classes']
        cnn_filters = self.config['cnn_filters']
        lstm_units = self.config['lstm_units']
        dropout_rate = self.config['dropout_rate']
        
        # CNN层
        self.conv1 = nn.Conv2d(input_channels, cnn_filters[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(cnn_filters[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(cnn_filters[0], cnn_filters[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(cnn_filters[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(cnn_filters[1], cnn_filters[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(cnn_filters[2])
        
        self.dropout_cnn = nn.Dropout(dropout_rate)
        
        # 计算CNN输出后的特征维度
        # 输入: 11x11 -> pool1: 5x5 -> pool2: 2x2
        self.cnn_output_size = 2 * 2 * cnn_filters[2]  # 2x2x128
        
        # LSTM层
        self.lstm1 = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        self.lstm2 = nn.LSTM(
            input_size=lstm_units,
            hidden_size=lstm_units // 2,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        self.dropout_lstm = nn.Dropout(dropout_rate)
        
        # 全连接层
        self.fc1 = nn.Linear(lstm_units // 2, 128)
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
        
    def _default_config(self):
        """默认配置"""
        return {
            'input_shape': (1, 11, 11),  # (channels, height, width)
            'num_classes': 6,
            'cnn_filters': [32, 64, 128],
            'lstm_units': 64,
            'dropout_rate': 0.3,
            'learning_rate': 0.001
        }
    
    def forward(self, x):
        """前向传播"""
        batch_size = x.size(0)
        
        # CNN部分
        x = self.pool1(self.dropout_cnn(self.bn1(F.relu(self.conv1(x)))))
        x = self.pool2(self.dropout_cnn(self.bn2(F.relu(self.conv2(x)))))
        x = self.dropout_cnn(self.bn3(F.relu(self.conv3(x))))
        
        # 重塑为序列: (batch, seq_len=1, features)
        x = x.view(batch_size, 1, -1)
        
        # LSTM部分
        x, _ = self.lstm1(x)
        x = self.dropout_lstm(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout_lstm(x)
        
        # 取最后时间步的输出
        x = x[:, -1, :]
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x
    
    def summary(self):
        """打印模型结构"""
        print(self)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
