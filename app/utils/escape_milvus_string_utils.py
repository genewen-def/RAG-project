def escape_milvus_string(value: str) -> str:  # 定义 Milvus 字符串转义函数
    """对输入字符串做 Milvus 过滤表达式安全转义。"""
    if value is None:  # 空值直接返回空字符串
        return ""  # 返回空字符串
    s = str(value)  # 将输入转为字符串
    s = s.replace("\\", "\\\\").replace('"', '\\"')  # 转义反斜杠和双引号
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")  # 替换空白控制字符为空格
    return s  # 返回转义后的字符串
