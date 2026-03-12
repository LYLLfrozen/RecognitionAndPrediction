#!/usr/bin/env python3
import numpy as np, pickle, os, sys
sys.path.insert(0, '.')

data_dir = 'data/processed_data'
with open(os.path.join(data_dir, 'processor.pkl'), 'rb') as f:
    proc = pickle.load(f)

scaler = proc['scaler']
print('feature_names count:', len(proc['feature_names']))
print('feature_names:', proc['feature_names'][:30])
print('n_features:', proc['n_features'])
print('class_names:', proc['class_names'])
print()
print('scaler mean (first 20):', scaler.mean_[:20])
print('scaler scale (first 20):', scaler.scale_[:20])

# 查看VFL训练数据目录
vfl_data = 'data/processed_data'
print('\nVFL data dir contents:', os.listdir(vfl_data) if os.path.exists(vfl_data) else 'not found')

# 看看模型目录
model_dir = 'models/vfl_network'
print('\nmodel dir contents:', os.listdir(model_dir) if os.path.exists(model_dir) else 'not found')
with open(os.path.join(model_dir, 'config.pkl'), 'rb') as f:
    config = pickle.load(f)
print('config keys:', list(config.keys()))
for k, v in config.items():
    print(f'  {k}: {v}')
