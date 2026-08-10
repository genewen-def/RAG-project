import numpy as np


def normalize_sparse_vector(sparse_vec):  # 定义稀疏向量归一化函数
    """对稀疏向量做 L2 归一化，仅处理非零维度。"""
    if not sparse_vec:  # 空向量直接返回
        return sparse_vec  # 返回原向量

    values = np.array(list(sparse_vec.values()), dtype=np.float64)  # 提取非零值并转为数组
    l2_norm = np.linalg.norm(values)  # 计算 L2 范数
    if l2_norm < 1e-9:  # 范数过小时直接返回原向量
        return sparse_vec  # 返回原向量

    normalized_values = values / l2_norm  # 对数值做归一化
    return dict(zip(sparse_vec.keys(), normalized_values))  # 重建为稀疏向量字典
