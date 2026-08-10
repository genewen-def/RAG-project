from app.utils.task_utils import *
from app.lm.reranker_utils import get_reranker_model
from app.core.logger import logger
import sys

RERANK_MAX_TOPK: int = 10
RERANK_MIN_TOPK: int = 1
RERANK_GAP_RATIO: float = 0.25
RERANK_GAP_ABS: float = 0.5


def step_1_merge_docs(state):
    """合并本地与联网搜索结果为标准格式。"""
    rrf_docs = state.get("rrf_chunks") or []  # 获取本地 RRF 文档
    web_docs = state.get("web_search_docs") or []  # 获取联网搜索文档

    logger.info(f"Step 1: 开始合并文档 - 本地RRF源: {len(rrf_docs)}条, 联网Web源: {len(web_docs)}条")  # 打印输入统计
    doc_items = []  # 初始化标准化文档列表

    for i, doc in enumerate(rrf_docs):  # 遍历本地文档
        entity = doc.get("entity") if isinstance(doc, dict) and "entity" in doc else doc  # 提取 entity

        if not isinstance(entity, dict):  # 格式异常则跳过
            logger.warning(f"本地文档格式异常 (index={i}): {type(entity)}")
            continue

        content = entity.get("content")  # 获取文档内容
        if not content:  # 无内容则跳过
            logger.debug(f"跳过无内容文档 (index={i}, keys={list(entity.keys())})")
            continue

        doc_id = entity.get("chunk_id") or entity.get("id")  # 获取文档 ID
        title = entity.get("title") or entity.get("item_name") or ""  # 获取标题

        doc_items.append({  # 组装本地标准文档
            "text": content,
            "doc_id": doc_id,
            "chunk_id": doc_id,
            "title": title,
            "url": "",
            "source": "local",
        })

    for i, doc in enumerate(web_docs):  # 遍历联网文档
        text = (doc.get("snippet") or doc.get("content") or "").strip()  # 获取摘要文本
        url = (doc.get("url") or "").strip()  # 获取链接
        title = (doc.get("title") or "").strip()  # 获取标题

        if not text:  # 无内容则跳过
            logger.debug(f"跳过无内容联网结果 (index={i})")
            continue

        doc_items.append({  # 组装联网标准文档
            "text": text,
            "doc_id": None,
            "chunk_id": None,
            "title": title,
            "url": url,
            "source": "web",
        })

    logger.info(f"Step 1: 文档合并完成，共输出 {len(doc_items)} 条标准化文档")  # 打印合并完成日志
    return doc_items  # 返回标准化文档列表


def step_2_rerank_docs(state, doc_items):
    """对文档按问题相关性进行重排序。"""
    question = state.get("rewritten_query") or state.get("original_query") or ""  # 获取排序用问题

    if not doc_items or not question:  # 缺少输入则返回空
        logger.warning("Step 2: 跳过重排序 (无文档或无问题)")
        return []

    logger.info(f"Step 2: 开始重排序 (Rerank), 待排序文档数: {len(doc_items)}")  # 打印开始日志

    texts = [x["text"] for x in doc_items]  # 提取文档文本列表
    try:
        reranker = get_reranker_model()  # 获取重排序模型

        sentence_pairs = [[question, t] for t in texts]  # 构造 (问题, 文档) 配对
        logger.info("Step 2: 正在计算相关性得分...")  # 打印计算日志
        scores = reranker.compute_score(sentence_pairs)  # 计算相关性分数

        scored_docs = []  # 初始化带分文档列表
        for item, text, score in zip(doc_items, texts, scores):  # 组装文档与分数
            score_val = float(score)  # 转为浮点数
            scored_docs.append(
                {
                    "text": text,
                    "score": score_val,
                    "source": item.get("source") or "",
                    "chunk_id": item.get("chunk_id"),
                    "doc_id": item.get("doc_id"),
                    "url": item.get("url") or "",
                    "title": item.get("title") or "",
                }
            )

        scored_docs.sort(key=lambda x: x["score"], reverse=True)  # 按分数降序排序
        return scored_docs  # 返回重排序结果

    except Exception as e:
        logger.error(f"Step 2: 重排序过程发生异常: {e}", exc_info=True)  # 打印异常日志
        fallback_docs = [  # 构造降级结果
            {
                "text": x.get("text"),
                "score": 0.0,
                "source": x.get("source") or "",
                "chunk_id": x.get("chunk_id"),
                "doc_id": x.get("doc_id"),
                "url": x.get("url") or "",
                "title": x.get("title") or "",
            }
            for x in doc_items
        ]
        return fallback_docs  # 返回降级结果


def step_3_topk(scored_docs):
    """基于断崖阈值动态截断 TopK 文档。"""
    max_topk = min(RERANK_MAX_TOPK, len(scored_docs))  # 计算实际硬上限
    min_topk = RERANK_MIN_TOPK  # 硬下限
    gap_ratio = RERANK_GAP_RATIO  # 相对断崖阈值
    gap_abs = RERANK_GAP_ABS  # 绝对断崖阈值

    topk = max_topk  # 默认取满上限
    if topk > min_topk:  # 超过下限时才检测断崖
        for i in range(min_topk - 1, max_topk - 1):  # 遍历相邻分数
            s1 = scored_docs[i].get("score")  # 当前分数
            s2 = scored_docs[i + 1].get("score")  # 下一个分数

            gap = s1 - s2  # 相邻分数差
            rel = gap / (abs(s1) + 1e-6)  # 相对差距

            if gap >= gap_abs or rel >= gap_ratio:  # 触发断崖截断
                logger.info(f"Step 3: 触发断崖截断 @ index={i} (Score {s1:.4f} -> {s2:.4f}, Gap={gap:.4f})")
                topk = i + 1  # 截断到当前位置
                break

    topk_docs = scored_docs[:topk]  # 截取前 topk 条

    logger.info(f"Step 3: 截断完成，保留前 {len(topk_docs)} 条文档 (TopK={topk})")  # 打印截断结果

    if topk_docs:  # 若有结果
        preview = ", ".join([f"{d.get('chunk_id') or 'Web'}({d.get('score'):.3f})" for d in topk_docs[:3]])
        logger.debug(f"Step 3: Top3 文档预览: {preview}")  # 打印 Top3 预览

    return topk_docs  # 返回截断后的文档


def node_rerank(state):
    """Rerank 节点：合并、重排序并截断文档。"""
    logger.info("---Rerank (重排序) 节点开始处理---")  # 打印节点开始日志
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))  # 标记任务开始

    doc_items = step_1_merge_docs(state)  # 合并文档
    scored_docs = step_2_rerank_docs(state, doc_items)  # 重排序
    topk_docs = step_3_topk(scored_docs)  # 动态截断

    logger.info(f"Rerank 节点处理结束, 最终输出 {len(topk_docs)} 条文档")  # 打印节点结束日志

    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))  # 标记任务结束
    return {"reranked_docs": topk_docs}  # 返回最终文档


if __name__ == "__main__":
    print("\n" + "="*50)  # 打印分隔线
    print(">>> 启动 node_rerank 本地测试")  # 打印测试标题
    print("="*50)  # 打印分隔线

    mock_rrf_chunks = [  # 模拟本地 RRF 结果
        {"chunk_id": "local_1", "content": "RRF是一种倒数排名融合算法", "title": "算法介绍", "score": 0.9},
        {"chunk_id": "local_2", "content": "BGE是一个强大的重排序模型", "title": "模型介绍", "score": 0.8},
        {"chunk_id": "local_3", "content": "无关的测试文档内容", "title": "测试文档", "score": 0.1}
    ]

    mock_web_docs = [  # 模拟联网结果
        {"title": "Rerank技术详解", "url": "http://web.com/1", "snippet": "Rerank即重排序，常用于RAG系统的第二阶段"},
        {"title": "无关网页", "url": "http://web.com/2", "snippet": "今天天气不错，适合出去游玩"}
    ]

    mock_state = {  # 模拟输入状态
        "session_id": "test_rerank_session",
        "rewritten_query": "什么是RRF和Rerank？",
        "rrf_chunks": mock_rrf_chunks,
        "web_search_docs": mock_web_docs,
        "is_stream": False
    }

    try:
        result = node_rerank(mock_state)  # 运行节点
        reranked = result.get("reranked_docs", [])  # 获取重排序结果

        print("\n" + "="*50)  # 打印分隔线
        print(">>> 测试结果摘要:")  # 打印结果标题
        print(f"输入文档总数: {len(mock_rrf_chunks) + len(mock_web_docs)}")  # 打印输入总数
        print(f"输出文档总数: {len(reranked)}")  # 打印输出总数
        print("-" * 30)  # 打印分隔线

        print("最终排名:")  # 打印排名标题
        for i, doc in enumerate(reranked, 1):  # 遍历输出结果
            print(f"Rank {i}: Source={doc.get('source')}, Score={doc.get('score'):.4f}, Text={doc.get('text')[:20]}...")  # 打印排名信息

        top1_score = reranked[0].get("score")  # 获取 Top1 分数
        if top1_score > 0:  # 分数正常
            print("\n[PASS] Rerank 打分正常")
        else:  # 分数异常
            print("\n[FAIL] Rerank 打分异常 (均为0或负数)")

        print("="*50)  # 打印分隔线

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")  # 打印测试异常
