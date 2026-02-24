import sys
import os
import time
import threading

# 添加当前目录到路径
sys.path.append(os.getcwd())

from realtime.realtime_processor import RealtimeProcessor

def test_realtime_prediction():
    print("开始测试实时预测模块...")
    
    # 检查处理器状态文件
    processor_path = "data/processed_data/processor_state.pkl"
    if not os.path.exists(processor_path):
        print(f"错误: 找不到处理器状态文件 {processor_path}")
        print("请先运行: python3 data/save_load_data.py --action save --sample-size 1000")
        return

    # 检查模型文件
    # 优先尝试 full model
    model_path = "model/saved_models/cnn_lstm_full.pth"
    if not os.path.exists(model_path):
        model_path = "model/saved_models/cnn_lstm_improved.pth"
        
    if not os.path.exists(model_path):
        print(f"警告: 找不到模型文件")
        print("将只进行特征提取，不进行预测")
        model_path = None
    else:
        print(f"找到模型文件: {model_path}")

    try:
        # 初始化处理器
        processor = RealtimeProcessor(
            processor_path=processor_path,
            model_path=model_path
        )
        
        print("处理器初始化成功")
        
        # 启动处理器
        print("正在启动捕获 (运行 10 秒)...")
        processor.start()
        
        # 运行 10 秒
        start_time = time.time()
        packet_count = 0
        
        while time.time() - start_time < 10:
            data = processor.get_processed_data()
            if data:
                packet_count += 1
                print(f"[{packet_count}] {data['src']} -> {data['dst']}")
                if 'label' in data:
                    print(f"    预测: {data['label']} (置信度: {data['confidence']:.2%})")
                print("-" * 30)
            time.sleep(0.1)
            
        print(f"测试结束，共处理 {packet_count} 个连接")
        
        # 停止
        processor.stop()
        print("处理器已停止")
        
    except PermissionError:
        print("\n错误: 权限不足。捕获网络流量通常需要 root 权限。")
        print("请尝试使用 sudo 运行此脚本")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_realtime_prediction()
