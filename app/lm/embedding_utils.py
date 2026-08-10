from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from app.core.logger import logger
from app.conf.embedding_config import embedding_config

_bge_m3_ef = None  # BGE-M3 模型单例缓存


def get_bge_m3_ef():  # 定义 BGE-M3 模型单例获取函数
    """获取 BGE-M3 模型单例，未初始化时自动加载配置并创建。"""
    global _bge_m3_ef  # 声明使用全局单例变量
    if _bge_m3_ef is not None:  # 单例已存在则直接返回
        logger.debug("BGE-M3模型单例已存在，直接返回实例")
        return _bge_m3_ef  # 返回现有模型单例

    model_name = embedding_config.bge_m3_path or "BAAI/bge-m3"  # 取模型路径，默认使用官方模型名
    device = embedding_config.bge_device or "cpu"  # 取运行设备，默认 CPU
    use_fp16 = embedding_config.bge_fp16 or False  # 取是否使用 FP16，默认 False

    logger.info(  # 打印模型初始化配置
        "开始初始化BGE-M3模型",
        extra={
            "model_name": model_name,  # 模型名称
            "device": device,  # 运行设备
            "use_fp16": use_fp16,  # FP16 开关
            "normalize_embeddings": True  # 启用归一化
        }
    )

    try:  # 捕获模型初始化异常
        _bge_m3_ef = BGEM3EmbeddingFunction(  # 初始化 BGE-M3 嵌入模型
            model_name=model_name,  # 传入模型名称
            device=device,  # 传入运行设备
            use_fp16=use_fp16,  # 传入 FP16 开关
            normalize_embeddings=True  # 启用原生 L2 归一化
        )
        logger.success("BGE-M3模型初始化成功，已开启原生L2归一化")
        return _bge_m3_ef  # 返回初始化后的模型实例
    except Exception as e:  # 捕获模型初始化异常
        logger.error(f"BGE-M3模型初始化失败：{str(e)}", exc_info=True)  # 记录错误日志
        raise  # 向上抛出异常


def generate_embeddings(texts):  # 定义文本向量生成函数
    """为文本列表生成稠密和稀疏混合向量嵌入。"""
    if not isinstance(texts, list) or len(texts) == 0:  # 校验入参为非空列表
        logger.warning("生成向量入参不合法，texts必须为非空列表")  # 记录入参不合法警告
        raise ValueError("参数texts必须是包含文本的非空列表")  # 抛出参数异常

    logger.info(f"开始为{len(texts)}条文本生成混合向量嵌入")  # 记录开始生成向量日志
    try:  # 捕获向量生成异常
        model = get_bge_m3_ef()  # 获取 BGE-M3 模型单例
        embeddings = model.encode_documents(texts)  # 编码生成稠密和稀疏向量
        logger.debug(f"模型编码完成，开始解析稀疏向量格式，共{len(texts)}条")  # 记录编码完成日志

        processed_sparse = []  # 初始化稀疏向量解析结果列表
        for i in range(len(texts)):  # 遍历每个文本
            sparse_indices = embeddings["sparse"].indices[  # 提取当前文本稀疏向量索引
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]  # 取当前文本切片范围
            ].tolist()  # 转为 Python 列表
            sparse_data = embeddings["sparse"].data[  # 提取当前文本稀疏向量权重
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]  # 取当前文本切片范围
            ].tolist()  # 转为 Python 列表
            sparse_dict = {k: v for k, v in zip(sparse_indices, sparse_data)}  # 组合为索引到权重的字典
            processed_sparse.append(sparse_dict)  # 加入解析结果列表

        result = {  # 构造最终返回结果
            "dense": [emb.tolist() for emb in embeddings["dense"]],  # 稠密向量转为嵌套列表
            "sparse": processed_sparse  # 稀疏向量使用字典列表
        }
        logger.success(f"{len(texts)}条文本向量生成完成，格式已适配工业级使用")  # 记录生成完成日志
        return result  # 返回向量结果字典

    except Exception as e:  # 捕获向量生成异常
        logger.error(f"文本向量生成失败：{str(e)}", exc_info=True)  # 记录错误日志
        raise  # 向上抛出异常
