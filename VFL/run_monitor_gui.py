#!/usr/bin/env python3
"""
VFL 网络流量监控系统 - GUI启动器
"""

import sys
import os

# 确保可以导入相关模块
sys.path.insert(0, os.path.dirname(__file__))

if __name__ == '__main__':
    # 添加 --gui 参数并启动
    sys.argv.append('--gui')

    from realtime_monitor import main
    main()
