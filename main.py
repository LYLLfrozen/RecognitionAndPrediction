"""
网络流量分类系统主程序
提供数据处理、模型训练、预测等功能
"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from model.fl_woa_cnn_lstm.model_train import train_cnn_lstm


def main():
    """主函数"""
    print("="*70)
    print("网络流量分类系统 - RecognitionAndPrediction")
    print("="*70)
    
    print("\n可用功能：")
    print("1. 训练CNN-LSTM模型")
    print("2. 测试模型")
    print("3. 实时预测")
    
    choice = input("\n请选择功能 (1-3): ")
    
    if choice == '1':
        print("\n启动模型训练...")
        model, history = train_cnn_lstm(
            data_dir='data/processed_data',
            epochs=50,
            batch_size=64,
            learning_rate=0.001,
            save_model_path='model/saved_models/cnn_lstm_traffic_classifier.keras'
        )
        print("\n训练完成！")
        
    elif choice == '2':
        print("\n模型测试功能开发中...")
        
    elif choice == '3':
        print("\n实时预测功能开发中...")
        
    else:
        print("\n无效选择！")


if __name__ == "__main__":
    main()
