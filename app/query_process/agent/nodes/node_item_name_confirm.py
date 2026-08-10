import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.load_prompt import load_prompt
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.clients.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from dotenv import load_dotenv, find_dotenv
from app.core.logger import logger

load_dotenv(find_dotenv())


def step_3_extract_info(query: str, history: List[Dict]) -> Dict:
    """利用 LLM 从当前问题与历史会话中提取商品名并重写问题。"""
    logger.info("Step 3: 开始提取信息 (LLM)")  # 打印步骤开始日志

    client = get_llm_client(json_mode=True)  # 获取 JSON 模式 LLM 客户端

    history_text = ""  # 初始化历史文本
    for msg in history:  # 遍历历史消息
        history_text += f"{msg.get('role', 'unknown')}: {msg.get('text', '')}\n"  # 拼接角色与文本

    logger.info(f"Step 3: 历史上下文构建完成，长度: {len(history_text)} 字符")  # 打印历史长度

    try:
        prompt = load_prompt("rewritten_query_and_itemnames", history_text=history_text, query=query)  # 加载提示词
        logger.debug(f"Step 3: 提示词加载成功，Prompt长度: {len(prompt)}")  # 打印提示词长度
    except Exception as e:
        logger.error(f"Step 3: 加载提示词失败: {e}")  # 打印加载失败日志
        return {"item_names": [], "rewritten_query": query}  # 返回默认结果

    messages = [  # 构造消息列表
        SystemMessage(content="你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"),
        HumanMessage(content=prompt)
    ]

    try:
        logger.info("Step 3: 正在调用 LLM 进行提取...")  # 打印调用日志
        response = client.invoke(messages)  # 调用 LLM
        content = response.content  # 获取响应内容
        logger.debug(f"Step 3: LLM 原始响应: {content}")  # 打印原始响应

        if content.startswith("```json"):  # 清理 Markdown 代码块标记
            content = content.replace("```json", "").replace("```", "")

        result = json.loads(content)  # 解析 JSON

        if "item_names" not in result:  # 确保 item_names 字段存在
            result["item_names"] = []
        if "rewritten_query" not in result:  # 确保 rewritten_query 字段存在
            result["rewritten_query"] = query

        logger.info(f"Step 3: 提取结果解析成功 - 商品名: {result['item_names']}, 重写问题: {result['rewritten_query']}")  # 打印解析结果
        return result  # 返回提取结果

    except Exception as e:
        logger.error(f"Step 3: LLM 提取或解析失败: {e}")  # 打印解析失败日志
        return {"item_names": [], "rewritten_query": query}  # 返回默认结果


def step_4_vectorize_and_query(item_names: List[str]) -> List[Dict]:
    """对商品名向量化并在 Milvus 中执行混合搜索。"""
    logger.info(f"Step 4: 开始向量化检索，目标商品: {item_names}")  # 打印步骤开始日志
    results = []  # 初始化结果列表

    client = get_milvus_client()  # 获取 Milvus 客户端
    if not client:  # 连接失败
        logger.error("Step 4: 无法连接到 Milvus")  # 打印错误日志
        return results  # 返回空结果

    collection_name = os.environ.get("ITEM_NAME_COLLECTION")  # 获取商品名集合
    if not collection_name:  # 未配置集合名
        logger.error("Step 4: 环境变量中未找到 ITEM_NAME_COLLECTION")  # 打印错误日志
        return results  # 返回空结果

    try:
        logger.info("Step 4: 正在生成 Embedding (Dense + Sparse)...")  # 打印向量化日志
        embeddings = generate_embeddings(item_names)  # 生成商品名向量
        logger.info(f"Step 4: 向量生成完成，开始 Milvus 搜索 (Collection: {collection_name})")  # 打印搜索开始日志

        for i, name in enumerate(item_names):  # 遍历每个商品名
            try:
                dense_vector = embeddings.get("dense")[i]  # 取稠密向量
                sparse_vector = embeddings.get("sparse")[i]  # 取稀疏向量

                reqs = create_hybrid_search_requests(  # 构造混合搜索请求
                    dense_vector=dense_vector,
                    sparse_vector=sparse_vector,
                    limit=5
                )

                search_res = hybrid_search(  # 执行混合搜索
                    client=client,
                    collection_name=collection_name,
                    reqs=reqs,
                    ranker_weights=(0.8, 0.2),
                    limit=5,
                    norm_score=True,
                    output_fields=["item_name"]
                )

                matches = []  # 初始化匹配列表
                if search_res and len(search_res) > 0:  # 若有搜索结果
                    for hit in search_res[0]:  # 遍历命中项
                        entity = hit.get("entity") or {}  # 获取业务字段
                        item_name = entity.get("item_name")  # 获取商品名
                        score = hit.get("distance")  # 获取相似度分数

                        if item_name:  # 商品名有效则加入匹配
                            matches.append({
                                "item_name": item_name,
                                "score": score
                            })
                            logger.debug(f"Step 4: '{name}' 匹配项: {item_name} (Score: {score:.4f})")

                results.append({  # 加入当前商品名搜索结果
                    "extracted_name": name,
                    "matches": matches
                })
                logger.info(f"Step 4: 商品 '{name}' 检索完成，找到 {len(matches)} 个匹配项")

            except Exception as inner_e:
                logger.error(f"Step 4: 处理商品 '{name}' 时出错: {inner_e}")  # 打印单商品异常
                results.append({"extracted_name": name, "matches": []})  # 加入空匹配

    except Exception as e:
        logger.error(f"Step 4: 向量化或搜索过程发生全局错误: {e}")  # 打印全局异常

    return results  # 返回所有搜索结果


def step_5_align_item_names(query_results: List[Dict]) -> Dict:
    """根据搜索评分对齐商品名，输出确认项与候选项。"""
    logger.info("Step 5: 开始对齐商品名 (Score Analysis)")  # 打印步骤开始日志

    confirmed_item_names = []  # 初始化确认商品名列表
    options = []  # 初始化候选商品名列表

    for res in query_results:  # 遍历各商品名搜索结果
        extracted_name = res.get("extracted_name", "").strip()  # 获取提取名
        matches = res.get("matches", []) or []  # 获取匹配列表

        if not matches:  # 无匹配则跳过
            logger.info(f"Step 5: '{extracted_name}' 无匹配结果")
            continue

        matches.sort(key=lambda x: x.get("score", 0), reverse=True)  # 按分数降序排序

        top_matches_log = ", ".join([f"{m['item_name']}({m['score']:.3f})" for m in matches[:3]])  # 构造 Top3 日志
        logger.info(f"Step 5: '{extracted_name}' Top匹配: {top_matches_log}")  # 打印 Top 匹配

        high = [m for m in matches if m.get("score", 0) > 0.85]  # 高置信度匹配
        mid = [m for m in matches if m.get("score", 0) >= 0.6]  # 中置信度匹配

        if len(high) == 1:  # 规则 A：单个高置信度
            confirmed_name = high[0].get("item_name")
            confirmed_item_names.append(confirmed_name)
            logger.info(f"Step 5: 规则A命中 (Single High) -> 确认: {confirmed_name}")
            continue

        if len(high) > 1:  # 规则 B：多个高置信度
            picked = None  # 初始化选中项
            if extracted_name:  # 优先匹配同名
                for m in high:
                    if m.get("item_name") == extracted_name:
                        picked = m
                        logger.info(f"Step 5: 规则B命中 (Exact Match in High) -> 确认: {picked.get('item_name')}")
                        break

            if not picked:  # 否则取最高分
                picked = high[0]
                logger.info(f"Step 5: 规则B命中 (Highest Score) -> 确认: {picked.get('item_name')}")

            confirmed_item_names.append(picked.get("item_name"))
            continue

        if len(mid) > 0:  # 规则 C：中置信度候选
            current_options = [m.get("item_name") for m in mid[:5]]
            options.extend(current_options)
            logger.info(f"Step 5: 规则C命中 (Mid Confidence) -> 添加候选: {current_options}")
            continue

        logger.info(f"Step 5: 规则D命中 (Low Confidence) -> 无匹配")  # 规则 D：低置信度

    result = {  # 组装对齐结果
        "confirmed_item_names": list(set(confirmed_item_names)),
        "options": list(set(options))
    }
    logger.info(f"Step 5: 对齐结果: {result}")  # 打印对齐结果
    return result  # 返回对齐结果


def step_6_check_confirmation(state: Dict, align_result: Dict, session_id: str, history: List[Dict], rewritten_query: str) -> Dict:
    """检查商品名对齐结果并更新 State。"""
    logger.info("Step 6: 检查确认状态并更新 State")  # 打印步骤开始日志

    if align_result is None:  # 对齐结果为空则初始化为空字典
        align_result = {}

    confirmed = align_result.get("confirmed_item_names", [])  # 获取确认商品名
    options = align_result.get("options", [])  # 获取候选商品名

    if confirmed:  # 分支 A：有确认商品名
        logger.info(f"Step 6: [分支A] 存在确认商品名: {confirmed}")

        ids_to_update = []  # 初始化待更新历史消息 ID 列表
        for msg in history:  # 遍历历史消息
            if not msg.get("item_names"):  # 未关联商品名
                mid = msg.get("_id")
                if mid:
                    ids_to_update.append(str(mid))  # 收集消息 ID

        if ids_to_update:  # 有需要更新的消息
            logger.info(f"Step 6: 更新 {len(ids_to_update)} 条历史消息的关联商品名")
            update_message_item_names(ids_to_update, confirmed)  # 更新历史消息商品名

        state["item_names"] = confirmed  # 写入确认商品名
        state["rewritten_query"] = rewritten_query  # 写入重写问题
        if "answer" in state:  # 清除可能存在的 answer
            del state["answer"]
        return state  # 返回更新后的 state

    if options:  # 分支 B：有候选商品名
        logger.info(f"Step 6: [分支B] 存在候选商品名: {options}")
        options_str = "、".join(options[:3])  # 拼接候选商品名字符串
        answer = f"您是想问以下哪个产品：{options_str}？请明确一下型号。"  # 构造澄清回复
        state["answer"] = answer  # 写入澄清回复
        state["item_names"] = []  # 清空商品名
        return state  # 返回更新后的 state

    logger.info("Step 6: [分支C] 无确认也无候选")  # 分支 C：无结果
    state["answer"] = "抱歉，未找到相关产品，请提供准确型号以便我为您查询。"  # 写入未找到回复
    state["item_names"] = []  # 清空商品名
    return state  # 返回更新后的 state


def step_7_write_history(state: Dict, session_id: str, history: List[Dict], rewritten_query: str, message_id: str) -> Dict:
    """写入最终会话历史记录。"""
    logger.info("Step 7: 写入会话历史")  # 打印步骤开始日志

    if state.get("answer"):  # 若有助手回复
        logger.info("Step 7: 保存助手回答")
        save_chat_message(  # 保存助手消息
            session_id=session_id,
            role="assistant",
            text=state["answer"],
            rewritten_query="",
            item_names=[]
        )

    logger.info(f"Step 7: 更新用户消息 (ID: {message_id})")
    save_chat_message(  # 更新用户消息
        session_id=session_id,
        role="user",
        text=state["original_query"],
        rewritten_query=rewritten_query,
        item_names=state.get("item_names", []),
        message_id=message_id
    )

    return state  # 返回 state


def node_item_name_confirm(state: QueryGraphState) -> QueryGraphState:
    """商品名称确认流程主节点。"""
    logger.info(">>> node_item_name_confirm: 开始处理")  # 打印节点开始日志

    session_id = state["session_id"]  # 获取会话 ID
    original_query = state.get("original_query", "")  # 获取原始查询
    is_stream = state.get("is_stream", False)  # 获取是否流式

    add_running_task(session_id, "node_item_name_confirm", is_stream)  # 标记任务开始

    history = get_recent_messages(session_id, limit=10)  # 获取近期历史消息
    logger.info(f"Node: 获取到 {len(history)} 条历史消息")  # 打印历史数量

    message_id = save_chat_message(session_id, "user", original_query, "", state.get("item_names", []))  # 初始保存用户消息
    logger.debug(f"Node: 用户消息已初始保存, ID: {message_id}")  # 打印消息 ID

    extract_res = step_3_extract_info(original_query, history)  # 提取商品名与重写问题
    item_names = extract_res.get("item_names", [])  # 获取提取的商品名
    rewritten_query = extract_res.get("rewritten_query", original_query)  # 获取重写问题

    state["rewritten_query"] = rewritten_query  # 更新 state 中的重写问题

    align_result = {}  # 初始化对齐结果

    if len(item_names) > 0:  # 若提取到商品名
        query_results = step_4_vectorize_and_query(item_names)  # 向量化检索
        align_result = step_5_align_item_names(query_results)  # 对齐商品名
    else:  # 未提取到商品名
        logger.info("Node: 未提取到商品名，跳过向量检索")

    state = step_6_check_confirmation(state, align_result, session_id, history, rewritten_query)  # 检查确认状态

    final_state = step_7_write_history(state, session_id, history, rewritten_query, message_id)  # 写入历史

    final_state["history"] = history  # 将历史存入 state 供下游使用

    add_done_task(session_id, "node_item_name_confirm", is_stream)  # 标记任务完成

    logger.info(f"Node: 处理结束, Final State Item Names: {final_state.get('item_names')}")  # 打印结束日志
    return final_state  # 返回最终 state


if __name__ == "__main__":
    print("\n" + "="*50)  # 打印分隔线
    print(">>> 启动 node_item_name_confirm 本地测试")  # 打印测试标题
    print("="*50)  # 打印分隔线

    mock_state = {  # 模拟输入状态
        "session_id": "test_debug_session_001",
        "original_query": "HAK 180 烫金机多少钱？",
        "is_stream": False,
        "item_names": []
    }

    try:
        result = node_item_name_confirm(mock_state)  # 运行节点

        print("\n" + "="*50)  # 打印分隔线
        print(">>> 测试结果摘要:")  # 打印结果标题
        print(f"Rewritten Query: {result.get('rewritten_query')}")  # 打印重写问题
        print(f"Item Names: {result.get('item_names')}")  # 打印确认商品名
        print(f"Answer: {result.get('answer')}")  # 打印助手回复
        print("="*50)  # 打印分隔线

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")  # 打印测试异常
