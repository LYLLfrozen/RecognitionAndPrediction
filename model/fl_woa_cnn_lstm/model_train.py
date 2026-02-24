"""
兼容性包装：在旧接口中提供 `train_cnn_lstm`。

此模块将项目中的改进训练函数 `train_cnn_lstm_improved`
作为 `train_cnn_lstm` 暴露，保持与现有调用处（例如 `main.py`）的兼容性。
"""
from typing import Any, Dict, Tuple

def train_cnn_lstm(*args, **kwargs) -> Tuple[Any, Dict]:
    """兼容性包装。

    原先此模块会调用项目根目录下的 `train_improved.py` 中的
    `train_cnn_lstm_improved`。该脚本已移除（仓库使用完整版训练
    `train_full_model.py`），因此在尝试调用时给出明确的提示。

    若需要训练模型，请使用 `train_full_model.py` 或实现自己的训练
    函数并直接调用它。
    """
    raise ImportError(
        "train_cnn_lstm_improved 已被移除；使用 train_full_model.py 或者\n"
        "在项目中实现并调用适当的训练函数。"
    )
