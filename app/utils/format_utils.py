import json
from typing import Any, Dict


def format_state(state: Dict[str, Any], indent: int = 4) -> str:  # 定义状态格式化函数
    """将工作流状态字典格式化为 JSON 字符串。"""
    return json.dumps(state, indent=indent, ensure_ascii=False)  # 序列化为格式化 JSON


def format_json(data: Any, indent: int = 4, ensure_ascii: bool = False) -> str:  # 定义通用 JSON 格式化函数
    """将可序列化数据格式化为 JSON 字符串。"""
    return json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)  # 序列化为格式化 JSON
