#!/usr/bin/env python3
"""
快速测试脚本 - 验证检测系统是否正常工作
无需管理员权限，使用测试集验证准确率
"""

import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from realtime_monitor import VFLFlowClassifier
from hybrid_detector import HybridAttackDetector
from flow_tracker import FlowTracker

def test_classifier():
    """测试分类器基本功能"""
    print("=" * 80)
    print("测试1: VFL分类器加载")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        classifier = VFLFlowClassifier('models/vfl_network', device)
        print("✓ 分类器加载成功")
        print(f"  类别: {classifier.class_names}")
        return classifier
    except Exception as e:
        print(f"✗ 分类器加载失败: {e}")
        return None

def test_detection_accuracy(classifier):
    """测试检测准确率"""
    print("\n" + "=" * 80)
    print("测试2: 检测准确率（测试集）")
    print("=" * 80)
    
    try:
        # 加载测试数据
        DATA_DIR = 'data/processed_data'
        X_test = np.load(os.path.join(DATA_DIR, 'test_images.npy'))
        y_test = np.load(os.path.join(DATA_DIR, 'test_labels.npy'))
        
        print(f"✓ 测试集加载成功: {len(X_test)} 个样本")
        
        # 测试前100个样本
        test_size = min(100, len(X_test))
        correct = 0
        predictions = []
        
        print(f"\n正在测试前 {test_size} 个样本...")
        
        for i in range(test_size):
            flow = X_test[i]
            true_label = y_test[i]
            true_class = classifier.class_names[true_label]
            
            pred_class, confidence, all_probs = classifier.classify(flow)
            is_correct = (pred_class == true_class)
            
            if is_correct:
                correct += 1
            
            predictions.append({
                'true': true_class,
                'pred': pred_class,
                'conf': confidence,
                'correct': is_correct
            })
            
            if (i + 1) % 20 == 0:
                print(f"  已测试: {i+1}/{test_size}")
        
        accuracy = correct / test_size * 100
        
        print(f"\n【结果】")
        print(f"  准确率: {accuracy:.2f}% ({correct}/{test_size})")
        
        # 统计各类别
        from collections import Counter
        pred_dist = Counter(p['pred'] for p in predictions)
        true_dist = Counter(p['true'] for p in predictions)
        
        print(f"\n【预测分布】")
        for cls in sorted(pred_dist.keys()):
            count = pred_dist[cls]
            pct = count / test_size * 100
            print(f"  {cls:8s}: {count:3d} ({pct:5.1f}%)")
        
        print(f"\n【真实分布】")
        for cls in sorted(true_dist.keys()):
            count = true_dist[cls]
            pct = count / test_size * 100
            print(f"  {cls:8s}: {count:3d} ({pct:5.1f}%)")
        
        # 显示错误样本
        errors = [p for p in predictions if not p['correct']]
        if errors:
            print(f"\n【错误样本】({len(errors)}个)")
            for i, err in enumerate(errors[:5]):  # 只显示前5个
                print(f"  {i+1}. 真实={err['true']:8s} 预测={err['pred']:8s} 置信度={err['conf']:.3f}")
        
        return accuracy >= 90.0  # 期望准确率 >= 90%
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hybrid_detector(classifier):
    """测试混合检测器"""
    print("\n" + "=" * 80)
    print("测试3: 混合检测器（规则引擎）")
    print("=" * 80)
    
    tracker = FlowTracker()
    detector = HybridAttackDetector(classifier, tracker)
    
    test_cases = [
        {
            'name': '正常HTTP流量',
            'packet': {
                'src_ip': '192.168.1.100',
                'dst_ip': '8.8.8.8',
                'src_port': 50000,
                'dst_port': 80,
                'protocol': 6,
                'tcp_flags': 0x18,
                'packet_size': 1000,
                'timestamp': 0.0
            },
            'expected': 'normal',
            'count': 10
        },
        {
            'name': 'SYN Flood攻击',
            'packet': {
                'src_ip': '10.0.0.1',
                'dst_ip': '192.168.1.1',
                'src_port': 10000,
                'dst_port': 80,
                'protocol': 6,
                'tcp_flags': 0x02,
                'packet_size': 60,
                'timestamp': 0.0
            },
            'expected': 'dos',
            'count': 50
        },
        {
            'name': '端口扫描',
            'packet': {
                'src_ip': '10.0.0.1',
                'dst_ip': '192.168.1.1',
                'src_port': 55000,
                'dst_port': 21,  # 会变化
                'protocol': 6,
                'tcp_flags': 0x02,
                'packet_size': 60,
                'timestamp': 0.0
            },
            'expected': 'probe',
            'count': 100,
            'vary_port': True
        }
    ]
    
    all_passed = True
    
    for test in test_cases:
        print(f"\n测试场景: {test['name']}")
        
        # 重置tracker
        tracker = FlowTracker()
        detector = HybridAttackDetector(classifier, tracker)
        
        last_result = None
        
        for i in range(test['count']):
            packet = test['packet'].copy()
            packet['timestamp'] = i * 0.01
            
            # 变化端口（用于端口扫描测试）
            if test.get('vary_port'):
                ports = [21, 22, 23, 25, 80, 110, 443, 3306, 8080, 5432]
                packet['dst_port'] = ports[i % len(ports)]
                packet['src_port'] += i
            else:
                packet['src_port'] += i
            
            # 提取基础特征
            base_features = np.zeros(41)
            base_features[0] = packet['packet_size']
            base_features[1] = packet['protocol']
            base_features[2] = packet['src_port']
            base_features[3] = packet['dst_port']
            base_features[4] = packet.get('tcp_flags', 0)
            
            # 更新流统计
            flow_stats = tracker.update(packet)
            
            # 检测
            pred_class, confidence, method = detector.detect(base_features, packet, flow_stats)
            last_result = (pred_class, confidence, method)
        
        # 检查最后的检测结果
        pred_class, confidence, method = last_result
        expected = test['expected']
        
        if pred_class == expected:
            print(f"  ✓ 通过: 识别为 {pred_class} (置信度={confidence:.3f}, 方法={method})")
        else:
            print(f"  ✗ 失败: 期望={expected}, 实际={pred_class} (置信度={confidence:.3f})")
            all_passed = False
    
    return all_passed

def main():
    """主测试流程"""
    print("\n" + "=" * 80)
    print("VFL 入侵检测系统 - 快速测试")
    print("=" * 80)
    print("\n此脚本将验证系统的基本功能和准确率")
    print("预计耗时：1-2分钟\n")
    
    results = []
    
    # 测试1: 分类器加载
    classifier = test_classifier()
    if classifier:
        results.append(("分类器加载", True))
    else:
        results.append(("分类器加载", False))
        print("\n✗ 分类器加载失败，无法继续测试")
        return
    
    # 测试2: 检测准确率
    accuracy_ok = test_detection_accuracy(classifier)
    results.append(("检测准确率", accuracy_ok))
    
    # 测试3: 混合检测器
    hybrid_ok = test_hybrid_detector(classifier)
    results.append(("混合检测器", hybrid_ok))
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name:20s}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 所有测试通过！系统运行正常。")
        print("\n下一步：")
        print("  1. 启动实时监控: python realtime_monitor.py")
        print("  2. 模拟攻击测试: python simulate_attacks.py dos --target 127.0.0.1 --port 80 --count 1000")
        print("  3. 查看详细文档: ATTACK_SIMULATION_GUIDE.md")
    else:
        print("\n⚠️  部分测试失败，请检查配置和模型文件。")
        print("\n常见问题：")
        print("  - 确保已训练模型: python train_vfl_network.py")
        print("  - 确保数据已处理: python preprocess_kddcup.py")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
