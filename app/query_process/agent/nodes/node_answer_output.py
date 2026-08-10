import sys
from app.utils.task_utils import add_running_task, add_done_task, set_task_result
from app.utils.sse_utils import push_to_session, SSEEvent
from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.clients.mongo_history_utils import save_chat_message
import re

_IMAGE_BLOCK_MARKER = "【图片】"
MAX_CONTEXT_CHARS = 12000


def step_1_check_answer(state) -> bool:
    """检查 state 中是否已有 answer，存在则直接推送或设置结果。"""
    answer = state.get("answer", None)  # 获取已有答案
    is_stream = state.get("is_stream")  # 获取是否流式
    if answer:  # 若已有答案
        if is_stream:  # 流式模式下推送增量
            logger.info("---Step 1: 发现已有答案，执行流式推送---")
            push_to_session(state["session_id"], SSEEvent.DELTA, {"delta": answer})
        else:  # 非流式模式下设置任务结果
            set_task_result(state["session_id"], "answer", answer)
        return True  # 返回已存在
    else:
        return False  # 返回不存在


def step_2_construct_prompt(state: QueryGraphState) -> str:
    """根据 state 中的问题、历史、商品名与重排文档构建 Prompt。"""
    original_query = state.get("original_query", "")  # 获取原始查询
    rewritten_query = state.get("rewritten_query", "")  # 获取重写查询
    question = rewritten_query if rewritten_query else original_query  # 优先使用重写查询
    history = state.get("history", [])  # 获取历史对话
    item_names = state.get("item_names", [])  # 获取商品名列表
    reranked_docs = state.get("reranked_docs") or []  # 获取重排文档

    docs = []  # 初始化资料列表
    used = 0  # 初始化已用字符数
    for i, doc in enumerate(reranked_docs, start=1):  # 遍历重排文档
        text = (doc.get("text") or "").strip()  # 获取正文
        if not text:  # 跳过空正文
            continue
        source = doc.get("source") or ""  # 获取来源
        chunk_id = doc.get("chunk_id")  # 获取切片 ID
        url = (doc.get("url") or "").strip()  # 获取链接
        title = (doc.get("title") or "").strip()  # 获取标题
        score = doc.get("score")  # 获取分数

        meta_parts = [f"[{i}]"]  # 初始化元数据片段
        if source:  # 添加来源
            meta_parts.append(f"[{source}]")
        if chunk_id:  # 添加切片 ID
            meta_parts.append(f"[chunk_id={chunk_id}]")
        if url:  # 添加链接
            meta_parts.append(f"[url={url}]")
        if score is not None:  # 添加分数
            meta_parts.append(f"[score={float(score):.4f}]")
        if title:  # 添加标题
            meta_parts.append(f"[title={title}]")
        doc = " ".join(meta_parts) + "\n" + text  # 组装资料字符串
        if used + len(doc) > MAX_CONTEXT_CHARS:  # 超过长度限制则停止
            break
        docs.append(doc)  # 加入资料列表
        used += len(doc) + 2  # 累加已用长度
    context_str = "\n\n".join(docs) if docs else "无参考内容"  # 组装上下文字符串

    history_str = ""  # 初始化历史字符串
    if history:  # 若有历史对话
        for msg in history:  # 遍历历史消息
            role = msg.get("role")  # 获取角色
            text = msg.get("text")  # 获取文本
            if role == "user" and text:  # 用户消息
                history_str += f"用户: {text}\n"
            elif role == "assistant" and text:  # 助手消息
                history_str += f"助手: {text}\n"

            used += len(history_str) + 2  # 累加已用长度
            if used > MAX_CONTEXT_CHARS:  # 超过长度限制则停止
                break
    else:  # 无历史对话
        history_str = "无历史对话"

    item_names_str = ", ".join(item_names) if item_names else "无指定商品"  # 组装商品名字符串

    prompt = load_prompt("answer_out",  # 加载回答生成提示词
        context=context_str,
        history=history_str,
        item_names=item_names_str,
        question=question
    )

    logger.info(f"组装后的提示词为：{prompt}")  # 打印提示词

    return prompt  # 返回 Prompt


def step_3_generate_response(state: QueryGraphState, prompt: str) -> QueryGraphState:
    """调用 LLM 生成回答，支持流式与非流式输出。"""
    logger.info("---Step 3: 开始生成回答 (LLM Generation)---")  # 打印步骤开始日志
    logger.debug(f"最终Prompt内容: {prompt}")  # 打印 Prompt 内容

    llm = get_llm_client()  # 获取 LLM 客户端

    session_id = state.get("session_id")  # 获取会话 ID
    is_stream = state.get("is_stream")  # 获取是否流式

    if is_stream:  # 流式输出
        logger.info(f"模式: 流式输出 (Streaming), Session: {session_id}")
        final_text = ""  # 初始化最终文本
        try:
            for chunk in llm.stream(prompt):  # 流式生成
                delta = getattr(chunk, "content", "") or ""  # 获取增量内容
                if delta:  # 增量非空
                    final_text += delta  # 累加文本
                    push_to_session(session_id, SSEEvent.DELTA, {"delta": delta})  # 推送增量

            logger.info(f"流式输出完成，总长度: {len(final_text)}")  # 打印完成日志

        except Exception as e:  # 流式异常
            logger.error(f"流式生成出错: {e}", exc_info=True)
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})  # 推送错误

        state["answer"] = final_text  # 写入最终答案
    else:  # 非流式输出
        logger.info(f"模式: 非流式输出 (Blocking), Session: {session_id}")
        try:
            response = llm.invoke(prompt)  # 调用 LLM
            content = response.content  # 获取生成内容
            state["answer"] = content  # 写入答案
            set_task_result(session_id, "answer", content)  # 设置任务结果
            logger.info(f"生成回答完成，长度: {len(content)}")  # 打印完成日志
        except Exception as e:  # 生成异常
            logger.error(f"生成回答出错: {e}", exc_info=True)
            state["answer"] = "抱歉，生成回答时出现错误。"  # 写入错误提示

    return state  # 返回 state


def _extract_images_from_docs(docs):
    """从文档列表中提取图片 URL。"""
    images = []  # 初始化图片列表
    seen = set()  # 初始化去重集合
    if not docs:  # 无文档直接返回
        return []

    md_img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')  # 编译 Markdown 图片正则

    logger.info(f"开始提取图片，待处理文档数: {len(docs)}")  # 打印开始日志

    for i, doc in enumerate(docs):  # 遍历文档
        url = (doc.get("url") or "").strip()  # 获取 url 字段
        if url:  # url 非空
            if url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg')):  # 判断是否为图片后缀
                if url not in seen:  # 去重
                    logger.debug(f"文档[{i}] 发现图片 URL (字段): {url}")
                    seen.add(url)  # 加入已见集合
                    images.append(url)  # 加入图片列表

        text = (doc.get("text") or "").strip()  # 获取正文
        if text:  # 正文非空
            matches = md_img_pattern.findall(text)  # 查找 Markdown 图片
            for img_url in matches:  # 遍历匹配到的图片 URL
                img_url = img_url.strip()  # 去除空白
                if img_url and img_url not in seen:  # 去重
                    logger.debug(f"文档[{i}] 正文发现 Markdown 图片: {img_url}")
                    seen.add(img_url)  # 加入已见集合
                    images.append(img_url)  # 加入图片列表

    logger.info(f"图片提取完成，共找到 {len(images)} 张唯一图片: {images}")  # 打印完成日志
    return images  # 返回图片列表


def step_4_write_history(state: QueryGraphState, image_urls=None) -> QueryGraphState:
    """将本轮答案写入 MongoDB 历史记录。"""
    session_id = state.get("session_id", "default")  # 获取会话 ID
    answer = (state.get("answer") or "").strip()  # 获取答案
    item_names = state.get("item_names") or []  # 获取商品名

    try:
        if answer:  # 答案非空则保存
            save_chat_message(
                session_id=session_id,
                role="assistant",
                text=answer,
                rewritten_query="",
                item_names=item_names,
                image_urls=image_urls,
                message_id=None
            )
    except Exception as e:  # 保存历史异常不影响主链路
        logger.error(f"写入Mongo历史记录失败: {e}")

    return state  # 返回 state


def node_answer_output(state: QueryGraphState) -> QueryGraphState:
    """答案生成与输出节点：生成答案、保存历史并推送最终事件。"""
    logger.info("---node_answer_output (答案生成) 节点开始处理---")  # 打印节点开始日志
    add_running_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))  # 标记任务开始

    answer_exists = step_1_check_answer(state)  # 检查是否已有答案

    if not answer_exists:  # 若无答案
        prompt = step_2_construct_prompt(state)  # 构建 Prompt
        state["prompt"] = prompt  # 保存 Prompt

        step_3_generate_response(state, prompt)  # 生成回答

    image_urls = _extract_images_from_docs(state.get("reranked_docs") or [])  # 提取图片 URL

    if state.get("answer"):  # 若已有答案
        logger.info("---写入MongoDB历史记录---")
        step_4_write_history(state, image_urls=image_urls)  # 写入历史

    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))  # 标记任务完成

    logger.info(f"---发送 final 事件---图片为：{image_urls}")  # 打印 final 事件日志
    if state.get("is_stream"):  # 流式模式下推送 final 事件
        push_to_session(
            state['session_id'],
            SSEEvent.FINAL,
            {
                "answer": state["answer"],
                "status": "completed",
                "image_urls": image_urls
            }
        )

    logger.info("---node_answer_output 节点处理结束---")  # 打印节点结束日志
    return state  # 返回 state


if __name__ == "__main__":
    print("\n" + "="*50)  # 打印分隔线
    print(">>> 启动 node_answer_output 本地测试")  # 打印测试标题
    print("="*50)  # 打印分隔线

    mock_reranked_docs = [  # 模拟重排文档
        {
            "chunk_id": "local_101",
            "source": "local",
            "title": "HAK 180 烫金机操作手册_v2.pdf",
            "score": 0.95,
            "text": """
            HAK 180 烫金机的操作面板位于机器正前方。
            开启电源后，您需要先设置温度，默认建议设置在 110℃ 左右。
            具体的操作面板布局请参考下图：
            ![操作面板布局图](http://local-server/images/panel_view.jpg)

            如果是进行局部烫金，请调节侧面的旋钮。
            ![侧面旋钮细节](http://local-server/images/knob_detail.png)
            """
        },
        {
            "chunk_id": None,
            "source": "web",
            "title": "HAK 180 常见故障排除 - 官网",
            "score": 0.88,
            "url": "http://example.com/hak180_troubleshooting.jpeg",
            "text": "如果机器无法加热，请检查保险丝是否熔断..."
        },
        {
            "chunk_id": "local_102",
            "source": "local",
            "title": "安全注意事项",
            "score": 0.82,
            "text": "操作时请务必佩戴隔热手套，避免高温烫伤。"
        }
    ]

    mock_history = [  # 模拟历史对话
        {"role": "user", "text": "你好，这款机器怎么用？"},
        {"role": "assistant", "text": "您好！请问您具体指的是哪一款机器？"},
        {"role": "user", "text": "HAK 180 烫金机"}
    ]

    mock_state = {  # 模拟输入状态
        "session_id": "test_answer_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤和面板设置方法",
        "item_names": ["HAK 180 烫金机"],
        "history": mock_history,
        "reranked_docs": mock_reranked_docs,
        "is_stream": False,
        "answer": None
    }

    try:
        result = node_answer_output(mock_state)  # 运行节点

        print("\n" + "="*50)  # 打印分隔线
        print(">>> 测试结果摘要:")  # 打印结果标题

        if "prompt" in result:  # 验证 Prompt 构建
            print(f"[PASS] Prompt 构建成功 (长度: {len(result['prompt'])})")
        else:
            print("[FAIL] Prompt 未构建")

        answer = result.get("answer")  # 获取答案
        if answer and len(answer) > 10:  # 验证答案生成
            print(f"[PASS] 答案生成成功 (长度: {len(answer)})")
            print(f"答案预览: {answer[:50]}...")
        else:
            print(f"[WARN] 答案生成可能异常 (Content: {answer})")

        print("\n[INFO] 请检查上方日志中是否包含 '图片提取完成' 及以下 URL:")  # 提示检查图片提取
        print(" - http://local-server/images/panel_view.jpg")
        print(" - http://local-server/images/knob_detail.png")
        print(" - http://example.com/hak180_troubleshooting.jpeg")

        print("="*50)  # 打印分隔线

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")  # 打印测试异常
