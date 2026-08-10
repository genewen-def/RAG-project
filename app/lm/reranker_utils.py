from FlagEmbedding import FlagReranker
from app.conf.reranker_config import reranker_config

_reranker_model = None  # 重排序模型单例缓存


def get_reranker_model():  # 定义重排序模型单例获取函数
    """获取 BGE 重排序模型单例，未初始化时根据配置创建。"""
    global _reranker_model  # 声明使用全局单例变量
    if _reranker_model is None:  # 单例未初始化时创建
        _reranker_model = FlagReranker(  # 初始化重排序模型
            model_name_or_path=reranker_config.bge_reranker_large,  # 传入模型路径
            device=reranker_config.bge_reranker_device,  # 传入运行设备
            use_fp16=reranker_config.bge_reranker_fp16  # 传入 FP16 开关
        )
    return _reranker_model  # 返回模型单例
