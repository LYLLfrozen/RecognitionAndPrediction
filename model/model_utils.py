"""
模型通用工具（评估、保存、加载等）
"""
from pathlib import Path
import pickle

def save_model(obj, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(obj, f)


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def evaluate(y_true, y_pred) -> dict:
    """占位评估函数，返回字典格式指标"""
    # 实际实现请使用 sklearn.metrics
    return {"placeholder_metric": 0.0}
