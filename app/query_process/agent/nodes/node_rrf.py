import sys
from typing import List, Dict, Any
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger


def _as_entity_list(state_list) -> List[Dict[str, Any]]:
    """将上游节点输出统一规整为 entity dict 列表。"""
    out: List[Dict[str, Any]] = []  # 初始化输出列表
    for doc in (state_list or []):  # 遍历输入列表
        if not doc:  # 跳过空文档
            continue

        final_ent = {}  # 初始化标准化实体字典

        if hasattr(doc, "entity") and hasattr(doc, "id"):  # 处理 Pymilvus 的 Hit 对象
            entity_content = doc.entity  # 获取 entity 内容
            if hasattr(entity_content, "to_dict"):  # 若 entity 可转字典
                 final_ent = entity_content.to_dict()  # 转为字典
            elif isinstance(entity_content, dict):  # 若 entity 已是字典
                 final_ent = entity_content.copy()  # 复制字典
            else:  # 其他类型时尝试转换
                 try:  # 尝试作为字典访问
                     final_ent = dict(entity_content)
                 except:  # 转换失败则保留空字典
                     pass

            if "id" not in final_ent and "chunk_id" not in final_ent:  # 补充外层 id
                final_ent["id"] = doc.id

            if hasattr(doc, "distance"):  # 补充 distance 作为 score
                final_ent["score"] = doc.distance

        elif isinstance(doc, dict):  # 处理字典类型
             if "entity" in doc:  # 嵌套 entity 结构
                 ent = doc["entity"]
                 if isinstance(ent, dict):
                     final_ent = ent.copy()
                 if "id" in doc and "id" not in final_ent:  # 补充外层 id
                     final_ent["id"] = doc["id"]
                 if "distance" in doc:  # 补充外层 distance
                     final_ent["score"] = doc["distance"]
             else:  # 扁平结构直接使用
                 final_ent = doc

        elif hasattr(doc, "get"):  # 处理带 get 方法的对象
             ent = doc.get("entity") or doc  # 获取 entity 或对象本身
             if isinstance(ent, dict):
                 final_ent = ent

        if final_ent and isinstance(final_ent, dict):  # 校验为非空字典后入列
            out.append(final_ent)

    return out  # 返回标准化实体列表


def reciprocal_rank_fusion(
        source_weights: list,
        k: int = 60,
        max_results: int = None,
) -> List[tuple]:
    """通用带权重的 RRF 算法实现。"""
    score_map = {}  # 记录 chunk_id 到 RRF 累加得分的映射
    chunk_map = {}  # 记录 chunk_id 到文档实体的映射

    for docs, weight in source_weights:  # 遍历各来源及其权重
        for rank, item in enumerate(docs, start=1):  # 按排名遍历文档
            chunk_id = item.get("chunk_id") or item.get("id")  # 获取文档唯一标识

            if not chunk_id:  # 无标识则跳过
                logger.warning(
                    f"RRF Warning: item missing chunk_id/id: {list(item.keys()) if isinstance(item, dict) else item}")
                continue

            score_map[chunk_id] = score_map.get(chunk_id, 0.0) + weight * (1.0 / (k + rank))  # 累加 RRF 分数

            chunk_map.setdefault(chunk_id, item)  # 记录首次遇到的文档实体

    merged = []  # 初始化融合结果列表
    for chunk_id, score in score_map.items():  # 遍历分数映射
        doc_item = chunk_map[chunk_id]  # 获取对应文档实体
        merged.append((doc_item, score))  # 组装为 (文档, 分数) 元组

    merged.sort(key=lambda x: x[1], reverse=True)  # 按分数降序排序

    if max_results is not None:  # 若指定最大结果数
        merged = merged[:max_results]  # 截断结果

    return merged  # 返回融合排序结果


def node_rrf(state):
    """RRF 倒数排名融合节点，融合多路检索结果。"""
    logger.info("---RRF (倒数排名融合) 开始处理---")  # 打印节点开始日志
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))  # 标记任务开始

    embedding_chunks = _as_entity_list(state.get("embedding_chunks"))  # 标准化 Embedding 检索结果
    hyde_embedding_chunks = _as_entity_list(state.get("hyde_embedding_chunks"))  # 标准化 HyDE 检索结果

    logger.info(f"RRF 输入统计: Embedding源={len(embedding_chunks)}条, HyDE源={len(hyde_embedding_chunks)}条")  # 打印输入统计

    if embedding_chunks:  # 打印 Embedding 源前 5 个 chunk_id
        logger.debug(f"Embedding源 chunk_ids (前5个): {[c.get('chunk_id') for c in embedding_chunks[:5]]}")
    if hyde_embedding_chunks:  # 打印 HyDE 源前 5 个 chunk_id
        logger.debug(f"HyDE源 chunk_ids (前5个): {[c.get('chunk_id') for c in hyde_embedding_chunks[:5]]}")

    source_weights = [  # 设置各来源权重
        (embedding_chunks, 1.0),
        (hyde_embedding_chunks, 1.0)
    ]

    rrf_res = reciprocal_rank_fusion(source_weights, k=60, max_results=10)  # 执行 RRF 融合

    rrf_chunks = [doc for doc, score in rrf_res]  # 提取融合后的文档列表
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))  # 标记任务结束

    return {"rrf_chunks": rrf_chunks}  # 返回融合结果


if __name__ == "__main__":
    print("\n" + "="*50)  # 打印分隔线
    print(">>> 启动 node_rrf 本地测试")  # 打印测试标题
    print("="*50)  # 打印分隔线

    mock_embedding_chunks = [  # 模拟 Embedding 检索结果
        {
            "id": "doc_1",
            "pk": "pk_1",
            "file_title": "操作手册_v1.pdf",
            "item_name": "HAK 180 烫金机",
            "content": "内容1：打开电源开关...",
            "score": 0.9
        },
        {
            "id": "doc_2",
            "pk": "pk_2",
            "file_title": "维修指南.pdf",
            "item_name": "HAK 180 烫金机",
            "content": "内容2：遇到故障请联系...",
            "score": 0.8
        },
        {
            "id": "doc_3",
            "pk": "pk_3",
            "file_title": "参数表.xlsx",
            "item_name": "HAK 180 烫金机",
            "content": "内容3：电压220V...",
            "score": 0.7
        }
    ]

    mock_hyde_chunks = [  # 模拟 HyDE 检索结果
        {
            "id": "doc_3",
            "pk": "pk_3",
            "file_title": "参数表.xlsx",
            "item_name": "HAK 180 烫金机",
            "content": "内容3：电压220V...",
            "score": 0.85
        },
        {
            "id": "doc_1",
            "pk": "pk_1",
            "file_title": "操作手册_v1.pdf",
            "item_name": "HAK 180 烫金机",
            "content": "内容1：打开电源开关...",
            "score": 0.82
        },
        {
            "id": "doc_4",
            "pk": "pk_4",
            "file_title": "安全须知.docx",
            "item_name": "HAK 180 烫金机",
            "content": "内容4：操作时请佩戴手套...",
            "score": 0.75
        }
    ]

    mock_state = {  # 模拟输入状态
        "session_id": "test_rrf_session",
        "is_stream": False,
        "embedding_chunks": mock_embedding_chunks,
        "hyde_embedding_chunks": mock_hyde_chunks
    }

    try:
        result = node_rrf(mock_state)  # 运行节点

        rrf_chunks = result.get("rrf_chunks", [])  # 获取融合结果
        print("\n" + "="*50)  # 打印分隔线
        print(">>> 测试结果摘要:")  # 打印结果标题
        print(f"输入数量: Embedding={len(mock_embedding_chunks)}, HyDE={len(mock_hyde_chunks)}")  # 打印输入数量
        print(f"输出数量: {len(rrf_chunks)}")  # 打印输出数量
        print("-" * 30)  # 打印分隔线

        print("最终排名:")  # 打印排名标题
        for i, doc in enumerate(rrf_chunks, 1):  # 遍历输出结果
            doc_id = doc.get('chunk_id') or doc.get('id')  # 获取文档 id
            print(f"Rank {i}: ID={doc_id}, Title={doc.get('file_title')}, Content={doc.get('content')[:20]}...")  # 打印排名信息

        ids = [d.get("id") or d.get("chunk_id") for d in rrf_chunks]  # 提取所有结果 id

        if "doc_1" in ids and "doc_3" in ids:  # 验证交叉文档是否保留
            print("\n[PASS] 交叉文档 (doc_1, doc_3) 成功融合保留")
        else:
            print("\n[FAIL] 交叉文档丢失")

        if len(ids) == 4:  # 验证并集数量
            print("[PASS] 并集数量正确 (3+3-2重叠=4)")
        else:
            print(f"[FAIL] 并集数量错误: 期望4, 实际{len(ids)}")

        print("="*50)  # 打印分隔线

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")  # 打印异常日志
