# HyDE节点
import sys
from app.utils.task_utils import add_running_task, add_done_task
from app.lm.lm_utils import *
from app.lm.embedding_utils import *
from app.clients.milvus_utils import *
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def step_1_create_hyde_doc(rewritten_query: str) -> str:
    """利用大模型根据用户查询生成假设性文档。"""
    if not rewritten_query:  # 校验查询是否为空
        logger.error("Step 1 Error: rewritten_query 为空")  # 打印错误日志
        raise ValueError("rewritten_query 不能为空")  # 抛出参数错误

    logger.info(f"Step 1: 开始生成假设性文档 (HyDE), Query: {rewritten_query}")  # 打印开始日志

    try:
        llm = get_llm_client()  # 获取 LLM 客户端
        hyde_prompt = load_prompt("hyde_prompt", rewritten_query=rewritten_query)  # 加载 HyDE 提示词
        logger.debug(f"Step 1: Prompt加载成功, 长度: {len(hyde_prompt)}")  # 打印提示词长度

        response = llm.invoke(hyde_prompt)  # 调用 LLM 生成假设文档
        hyde_doc = response.content  # 提取生成内容

        logger.info(f"Step 1: 假设文档生成完成, 长度: {len(hyde_doc)} 字符")  # 打印生成完成日志
        logger.debug(f"Step 1: 文档预览: {hyde_doc[:50]}...")  # 打印文档预览

        return hyde_doc  # 返回假设文档

    except Exception as e:
        logger.error(f"Step 1: 生成假设文档失败: {e}")  # 打印生成失败日志
        raise e  # 抛出异常


def step_2_search_embedding_hyde(
    rewritten_query: str,
    hyde_doc: str,
    item_names=None,
    req_limit: int = 10,
    top_k: int = 5,
    ranker_weights=(0.8, 0.2),
    norm_score: bool = True,
    output_fields=["chunk_id", "content", "item_name"],
):
    """利用重写问题与假设文档生成向量并执行 Milvus 混合检索。"""
    if not rewritten_query:  # 校验重写查询是否为空
        raise ValueError("rewritten_query 不能为空")  # 抛出参数错误
    if not hyde_doc:  # 校验假设文档是否为空
        raise ValueError("hypothetical_doc 不能为空")  # 抛出参数错误

    combined_text = rewritten_query + " " + hyde_doc  # 拼接查询与假设文档
    logger.info(f"Step 2: 拼接 Query + HyDE Doc, 总长度: {len(combined_text)}")  # 打印拼接日志

    logger.info("Step 2: 正在生成混合向量 (Embedding)...")  # 打印向量化日志
    embeddings = generate_embeddings([combined_text])  # 生成稠密与稀疏向量

    collection_name = os.environ.get("CHUNKS_COLLECTION")  # 获取集合名
    if not collection_name:  # 若未配置集合名
        logger.error("Step 2 Error: 环境变量 CHUNKS_COLLECTION 未设置")  # 打印错误日志
        return []  # 返回空结果

    logger.info(f"Step 2: 准备在集合 '{collection_name}' 中执行混合检索")  # 打印准备日志

    expr = None  # 初始化过滤表达式
    if item_names:  # 若指定商品名
        quoted = ", ".join(f'"{v}"' for v in item_names)  # 为商品名加引号
        expr = f"item_name in [{quoted}]"  # 构造过滤表达式
        logger.info(f"Step 2: 应用过滤条件: {expr}")  # 打印过滤条件
    else:  # 未指定商品名
        logger.info("Step 2: 未指定商品名过滤，将全库检索")  # 打印全库检索日志

    try:
        reqs = create_hybrid_search_requests(  # 构造搜索请求
            dense_vector=embeddings.get("dense")[0],
            sparse_vector=embeddings.get("sparse")[0],
            expr=expr,
            limit=req_limit,
        )

        client = get_milvus_client()  # 获取 Milvus 客户端
        if not client:  # 若连接失败
            logger.error("Step 2 Error: 无法连接到 Milvus")  # 打印错误日志
            return []  # 返回空结果

        logger.info(f"Step 2: 执行 Hybrid Search, Weights={ranker_weights}, TopK={top_k}")  # 打印检索参数
        res = hybrid_search(  # 执行混合检索
            client=client,
            collection_name=collection_name,
            reqs=reqs,
            ranker_weights=ranker_weights,
            norm_score=norm_score,
            limit=top_k,
            output_fields=list(output_fields),
        )

        hit_count = len(res[0]) if res and len(res) > 0 else 0  # 计算命中数量
        logger.info(f"Step 2: 检索完成, 找到 {hit_count} 个匹配切片")  # 打印命中数量

        return res  # 返回检索结果

    except Exception as e:
        logger.error(f"Step 2: 检索过程发生异常: {e}")  # 打印检索异常
        return []  # 返回空结果


def node_search_embedding_hyde(state):
    """HyDE 检索节点：生成假设文档并向量化后检索真实切片。"""
    logger.info("---HyDE (假设文档检索) 节点开始处理---")  # 打印节点开始日志
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))  # 标记任务开始

    rewritten_query = state.get("rewritten_query")  # 获取改写后查询
    if not rewritten_query:  # 若无改写查询
        rewritten_query = state.get("original_query")  # 降级使用原始查询

    if not rewritten_query:  # 若仍无有效查询
        logger.error("HyDE节点错误: 未找到有效的用户查询 (rewritten_query/original_query 均为空)")  # 打印错误日志
        return {}  # 返回空字典

    item_names = state.get("item_names")  # 获取商品名列表
    logger.info(f"HyDE检索入参: query='{rewritten_query}', item_names={item_names}")  # 打印入参日志

    hyde_doc = ""  # 初始化假设文档
    try:
        logger.info("Step 1: 开始生成假设性文档 (HyDE Doc)...")  # 打印步骤开始日志
        hyde_doc = step_1_create_hyde_doc(rewritten_query)  # 生成假设文档
        logger.info(f"Step 1: 假设文档生成成功 (长度: {len(hyde_doc)})")  # 打印生成成功日志
        logger.debug(f"假设文档预览: {hyde_doc[:100]}...")  # 打印文档预览
    except Exception as e:
        logger.error(f"Step 1 (生成假设文档) 发生异常: {e}", exc_info=True)  # 打印生成异常
        return {}  # 返回空字典

    try:
        logger.info("Step 2: 基于假设文档执行 Milvus 混合检索...")  # 打印步骤开始日志
        res = step_2_search_embedding_hyde(  # 执行检索
            rewritten_query=rewritten_query,
            hyde_doc=hyde_doc,
            item_names=item_names,
            top_k=5,
        )

        hit_count = len(res[0]) if res and len(res) > 0 else 0  # 计算命中数量
        logger.info(f"Step 2: 检索完成，召回 {hit_count} 条相关切片")  # 打印召回数量

        if hit_count > 0:  # 若命中结果大于 0
            first_hit = res[0][0]  # 取第一条结果
            score = first_hit.get("distance")  # 获取相似度分数
            content_preview = first_hit.get("entity", {}).get("content", "")[:30]  # 获取内容预览
            logger.debug(f"Top1 结果: Score={score}, Content='{content_preview}...'")  # 打印 Top1 信息

        return {  # 返回检索结果与假设文档
            "hyde_embedding_chunks": res[0] if res else [],
            "hyde_doc": hyde_doc,
        }
    except Exception as e:
        logger.error(f"Step 2 (向量生成与检索) 发生异常: {e}", exc_info=True)  # 打印检索异常
        return {}  # 返回空字典
    finally:
        add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))  # 标记任务结束
        logger.info("---HyDE 节点处理结束---")  # 打印节点结束日志


if __name__ == "__main__":
    print("\n" + "="*50)  # 打印分隔线
    print(">>> 启动 node_search_embedding_hyde 本地测试")  # 打印测试标题
    print("="*50)  # 打印分隔线

    mock_state = {  # 构造模拟输入状态
        "session_id": "test_hyde_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤是什么？",
        "item_names": ["HAK 180 烫金机"],
        "is_stream": False
    }

    try:
        result = node_search_embedding_hyde(mock_state)  # 运行节点

        print("\n" + "="*50)  # 打印分隔线
        print(">>> 测试结果摘要:")  # 打印结果标题
        print(f"HyDE Doc Generated: {bool(result.get('hyde_doc'))}")  # 打印是否生成假设文档
        if result.get("hyde_doc"):  # 若生成假设文档
            print(f"Doc Preview: {result.get('hyde_doc')[:50]}...")  # 打印文档预览

        chunks = result.get("hyde_embedding_chunks", [])  # 获取检索结果
        print(f"Chunks Found: {len(chunks)} , chunks内容：{chunks}")  # 打印结果数量与内容
        if chunks:  # 若有检索结果
            print(f"Top Chunk Score: {chunks[0].get('distance')}")  # 打印 Top1 分数
        print("="*50)  # 打印分隔线

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")  # 打印测试异常
