import sys
import os
import time
import signal
import argparse

# 添加当前目录到路径
sys.path.append(os.getcwd())

from realtime.realtime_processor import RealtimeProcessor

def main():
    parser = argparse.ArgumentParser(description="实时网络流量入侵检测系统")
    parser.add_argument("-i", "--interface", help="监听的网络接口 (例如: lo0, en0, eth0)", default=None)
    args = parser.parse_args()

    print("="*60)
    print("实时网络流量入侵检测系统")
    print("="*60)
    
    # 检查文件
    processor_path = "data/processed_data/processor_state.pkl"
    if not os.path.exists(processor_path):
        print(f"错误: 找不到处理器状态文件 {processor_path}")
        print("请先运行数据处理脚本生成状态文件。")
        return

    # 优先使用 full model
    model_path = "model/saved_models/cnn_lstm_full.pth"
    if not os.path.exists(model_path):
        model_path = "model/saved_models/cnn_lstm_improved.pth"
    
    if not os.path.exists(model_path):
        print("警告: 未找到模型文件，将仅进行特征提取。")
        model_path = None
    else:
        print(f"使用模型: {model_path}")

    # 初始化
    processor = None
    packet_count = 0
    start_time = time.time()
    
    try:
        # 如果未指定接口，提示用户
        if args.interface is None:
            print("\n提示: 未指定网络接口，将监听默认接口。")
            print("      如果是本地测试 (simulate_attacks.py)，请使用 --interface lo0")
        else:
            print(f"\n指定监听接口: {args.interface}")

        processor = RealtimeProcessor(
            processor_path=processor_path,
            model_path=model_path,
            interface=args.interface
        )
        
        print("\n正在启动捕获引擎...")
        print("按 Ctrl+C 停止检测")
        print("-" * 60)
        
        processor.start()
        
        last_status_time = time.time()
        
        while True:
            # 每5秒打印一次状态，证明程序在运行
            if time.time() - last_status_time > 5:
                raw_count = processor.capture.raw_packet_count
                print(f"--- 状态检查: 已捕获原始数据包 {raw_count} 个，已处理连接 {packet_count} 个 ---")
                last_status_time = time.time()
                
            data = processor.get_processed_data()
            if data:
                packet_count += 1
                timestamp = time.strftime("%H:%M:%S", time.localtime(data['timestamp']))
                
                # 格式化输出
                src = data['src']
                dst = data['dst']
                proto = data['protocol'].upper()
                
                output = f"[{timestamp}] {proto} {src} -> {dst}"
                
                if 'label' in data:
                    label = data['label']
                    conf = data['confidence']
                    
                    # 颜色高亮 (如果支持)
                    if label != 'normal' and label != 'class_0':
                        # 红色显示攻击
                        # 若有概率向量，显示 prediction 索引与前3个概率，便于调试
                        pred_idx = data.get('prediction_index', None)
                        probs = data.get('probabilities', [])
                        probs_snip = ''
                        if probs:
                            topk = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:3]
                            probs_snip = ' | top: ' + ','.join([f"{i}:{p:.2%}" for i,p in topk])
                        status = f"!!! {label.upper()} ({conf:.1%}) !!!"
                        if pred_idx is not None:
                            status += f" [idx:{pred_idx}]"
                        status += probs_snip
                    else:
                        status = f"{label} ({conf:.1%})"
                        
                    output += f" | {status}"
                    
                    # 如果是攻击或置信度较低，打印调试信息
                    if label != 'normal' or conf < 0.9 or True: # 强制显示调试信息以便排查
                        debug = data.get('debug_info', {})
                        output += f" [Flag: {debug.get('flag')} | Count: {debug.get('count')} | SError: {debug.get('serror_rate'):.2f} | RError: {debug.get('rerror_rate'):.2f}]"
                
                print(output)
                
            else:
                time.sleep(0.05)
                
    except KeyboardInterrupt:
        print("\n\n正在停止检测系统...")
    except PermissionError:
        print("\n错误: 权限不足，请使用 sudo 运行。")
    except Exception as e:
        print(f"\n发生错误: {e}")
    finally:
        if processor is not None:
            processor.stop()
        
        duration = time.time() - start_time
        print("-" * 60)
        print(f"检测结束。运行时长: {duration:.1f}秒, 处理连接数: {packet_count}")

if __name__ == "__main__":
    main()
