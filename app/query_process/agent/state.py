from typing_extensions import TypedDict
from typing import List


class QueryGraphState(TypedDict):  # 定义查询流程状态字典类型
    """定义查询流程中传递的数据结构。"""
    session_id: str  # 会话唯一标识
    original_query: str  # 用户原始问题

    embedding_chunks: list  # 向量检索得到的切片
    hyde_embedding_chunks: list  # HyDE 检索得到的切片
    kg_chunks: list  # 图谱检索得到的切片
    web_search_docs: list  # 网络搜索得到的文档

    rrf_chunks: list  # RRF 融合排序后的切片
    reranked_docs: list  # 重排序后的最终文档

    prompt: str  # 组装好的提示词
    answer: str  # 最终生成的答案

    item_names: List[str]  # 提取的商品名称
    rewritten_query: str  # 改写后的问题
    history: list  # 历史对话记录
    is_stream: bool  # 是否流式输出
