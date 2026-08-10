import sys
import os
from app.utils.task_utils import add_running_task, add_done_task
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import create_hybrid_search_requests, hybrid_search, get_milvus_client
from app.core.logger import logger
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def node_search_embedding(state):
    """基于商品名与改写后的问题执行 Milvus 混合向量检索。"""
    logger.info("---search_milvus 开始处理---")  # 打印节点开始日志
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])  # 标记任务开始

    query = state.get("rewritten_query")  # 获取改写后的查询
    item_names = state.get("item_names")  # 获取已确认的商品名列表

    logger.info(f"核心入参提取: query='{query}', item_names={item_names}")  # 打印核心入参

    logger.info(f"开始为文本获取嵌入值: {query[:50]}..." if len(query) > 50 else f"开始为“{query}”文本获取嵌入值...")  # 打印向量化开始日志
    embeddings = generate_embeddings([query])  # 生成查询文本的稠密与稀疏向量

    dense_vec = embeddings.get("dense")[0]  # 取稠密向量
    sparse_vec = embeddings.get("sparse")[0]  # 取稀疏向量
    logger.debug(f"向量生成成功: dense_dim={len(dense_vec)}, sparse_len={len(sparse_vec)}")  # 打印向量维度日志

    collection_name = os.environ.get("CHUNKS_COLLECTION")  # 从环境变量获取集合名
    logger.info(f"正在连接到 Milvus 并准备集合 '{collection_name}'...")  # 打印连接日志

    if not item_names:  # 若无商品名则跳过检索
        logger.warning("item_names 为空，跳过检索，返回空结果")  # 打印警告日志
        return {"embedding_chunks": []}  # 返回空结果

    quoted = ", ".join(f'"{v}"' for v in item_names)  # 为商品名添加双引号
    expr = f"item_name in [{quoted}]"  # 构造 item_name 过滤表达式
    logger.info(f"创建搜索请求过滤表达式: {expr}")  # 打印过滤表达式

    reqs = create_hybrid_search_requests(  # 构造混合搜索请求
        dense_vector=dense_vec,
        sparse_vector=sparse_vec,
        expr=expr,
        limit=10
    )

    logger.info("开始执行 Milvus 混合检索...")  # 打印检索开始日志
    client = get_milvus_client()  # 获取 Milvus 客户端
    res = hybrid_search(  # 执行混合检索
        client=client,
        collection_name=collection_name,
        reqs=reqs,
        ranker_weights=(0.8, 0.2),
        norm_score=True,
        limit=5,
        output_fields=["chunk_id", "content", "item_name"]
    )

    hit_count = len(res[0]) if res and len(res) > 0 else 0  # 计算命中数量
    logger.info(f"节点 search_embedding 处理成功，检索到 {hit_count} 条相关片段")  # 打印命中数量
    if hit_count > 0:  # 若命中结果大于 0
        logger.debug(f"Top1 检索结果示例: {res[0][0]}")  # 打印 Top1 结果

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))  # 标记任务完成

    return {"embedding_chunks": res[0] if res else []}  # 返回检索结果


if __name__ == "__main__":
    test_state = {  # 构造测试状态
        "session_id": "test_search_embedding_001",
        "rewritten_query": "HAK 180 烫金机使用说明",
        "item_names": ["HAK 180 烫金机"],
        "is_stream": False
    }

    print("\n>>> 开始测试 node_search_embedding 节点...")  # 打印测试开始
    try:
        result = node_search_embedding(test_state)  # 执行节点函数
        logger.info(f"检索结果汇总：{result}")  # 打印结果汇总
        chunks = result.get("embedding_chunks", [])  # 获取检索结果列表
        print(f"\n>>> 测试完成！检索到 {len(chunks)} 条结果")  # 打印结果数量

        if chunks:  # 若有检索结果
            print("\n>>> Top 1 结果详情:")  # 打印 Top1 标题
            top1 = chunks[0]  # 取第一条结果
            print(f"ID: {top1.get('id')}")  # 打印结果 ID
            print(f"Distance: {top1.get('distance')}")  # 打印相似度距离
            entity = top1.get('entity', {})  # 获取业务字段
            print(f"Item Name: {entity.get('item_name')}")  # 打印商品名
            print(f"Content Preview: {entity.get('content', '')[:100]}...")  # 打印内容预览
        else:  # 若无结果
            print("\n>>> 警告：未检索到任何结果，请检查 Milvus 数据或 item_names 是否匹配")  # 打印警告

    except Exception as e:
        logger.error(f"测试运行失败: {e}", exc_info=True)  # 打印测试异常
