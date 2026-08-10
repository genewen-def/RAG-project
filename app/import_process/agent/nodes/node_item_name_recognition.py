import os
import sys
from typing import List, Dict, Any, Tuple
from pymilvus import MilvusClient, DataType
from langchain_core.messages import SystemMessage, HumanMessage
from app.import_process.agent.state import ImportGraphState
from app.clients.milvus_utils import get_milvus_client
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
from app.utils.normalize_sparse_vector import normalize_sparse_vector
from app.utils.task_utils import add_running_task
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.utils.escape_milvus_string_utils import escape_milvus_string

DEFAULT_ITEM_NAME_CHUNK_K = 5
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
CONTEXT_TOTAL_MAX_CHARS = 2500


def step_1_get_inputs(state: ImportGraphState) -> Tuple[str, List[Dict]]:
    """从状态中提取文件标题和切片列表，并做基础校验。"""
    file_title = state.get("file_title", "") or state.get("file_name", "")  # 多层兜底获取文件标题
    chunks = state.get("chunks") or []  # 获取切片列表，空值兜底为空列表

    if not file_title:  # 判断文件标题是否为空
        if chunks and isinstance(chunks[0], dict):  # 判断是否存在有效切片
            file_title = chunks[0].get("file_title", "")  # 从第一个切片提取标题兜底
            logger.warning("state中无有效file_title，已从第一个切片中提取兜底标题")  # 记录兜底日志

    if not file_title:  # 二次判断文件标题是否仍为空
        logger.warning("state中缺少file_title和file_name，后续大模型识别可能精度下降")  # 记录缺失警告

    if not isinstance(chunks, list) or not chunks:  # 判断切片是否为有效非空列表
        logger.warning("state中chunks为空或非列表类型，无法进行商品名称识别")  # 记录无效切片警告
        return file_title, []  # 返回空切片列表

    logger.info(f"步骤1：输入校验完成，获取到{len(chunks)}个有效文本切片")  # 记录校验完成日志
    return file_title, chunks  # 返回文件标题和切片列表


def step_2_build_context(chunks: List[Dict], k: int = DEFAULT_ITEM_NAME_CHUNK_K, max_chars: int = CONTEXT_TOTAL_MAX_CHARS) -> str:
    """从前k个切片中构建用于大模型识别商品名称的格式化上下文。"""
    if not chunks:  # 判断切片是否为空
        return ""  # 返回空字符串

    parts: List[str] = []  # 初始化上下文片段列表
    total_chars = 0  # 初始化累计字符数

    for idx, chunk in enumerate(chunks[:k]):  # 遍历前k个切片
        if not isinstance(chunk, dict):  # 跳过非字典类型切片
            logger.debug(f"第{idx+1}个切片非字典类型，已过滤")  # 记录过滤日志
            continue

        chunk_title = chunk.get("title", "").strip()  # 获取切片标题
        chunk_content = chunk.get("content", "").strip()  # 获取切片内容

        if not (chunk_title or chunk_content):  # 判断标题和内容是否均为空
            logger.debug(f"第{idx+1}个切片为空白内容，已过滤")  # 记录过滤日志
            continue

        if len(chunk_content) > SINGLE_CHUNK_CONTENT_MAX_LEN:  # 判断内容是否超过截断长度
            chunk_content = chunk_content[:SINGLE_CHUNK_CONTENT_MAX_LEN]  # 截断内容
            logger.debug(f"第{idx+1}个切片内容过长，已截断至{SINGLE_CHUNK_CONTENT_MAX_LEN}字符")  # 记录截断日志

        piece = f"【切片{idx + 1}】\n标题：{chunk_title} \n内容：{chunk_content}"  # 格式化切片片段
        parts.append(piece)  # 加入片段列表
        total_chars += len(piece)  # 累计字符数

        if total_chars > max_chars:  # 判断总字符数是否超限
            logger.info(f"上下文总字符数即将超限（{max_chars}），已停止拼接后续切片")  # 记录停止日志
            break  # 停止拼接

    context = "\n\n".join(parts).strip()  # 拼接片段为上下文
    final_context = context[:max_chars]  # 最终截断确保不超限
    logger.info(f"步骤2：上下文构建完成，最终长度{len(final_context)}字符")  # 记录构建完成日志
    return final_context  # 返回格式化上下文


def step_3_call_llm(file_title: str, context: str) -> str:
    """调用大模型识别商品名称，异常或空结果时返回file_title兜底。"""
    logger.info("开始执行步骤3：调用大模型识别商品名称")  # 记录步骤开始日志

    if not context:  # 判断上下文是否为空
        logger.warning("上下文为空，跳过大模型调用，直接使用文件标题作为商品名称")  # 记录跳过日志
        return file_title  # 返回文件标题兜底

    try:
        human_prompt = load_prompt("item_name_recognition", file_title=file_title, context=context)  # 加载人类提示词
        system_prompt = load_prompt("product_recognition_system")  # 加载系统提示词
        logger.debug(f"大模型调用提示词构建完成，系统提示词长度{len(system_prompt)}，人类提示词长度{len(human_prompt)}")  # 记录提示词长度

        llm = get_llm_client(json_mode=False)  # 获取大模型客户端
        if not llm:  # 判断客户端是否为空
            logger.error("大模型客户端获取失败，使用文件标题兜底")  # 记录客户端失败日志
            return file_title  # 返回文件标题兜底

        messages = [
            SystemMessage(content=system_prompt),  # 构造系统消息
            HumanMessage(content=human_prompt)  # 构造人类消息
        ]
        resp = llm.invoke(messages)  # 调用大模型

        item_name = getattr(resp, "content", "").strip()  # 提取返回内容
        item_name = item_name.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")  # 清洗无效字符

        if not item_name:  # 判断清洗后结果是否为空
            logger.warning("大模型返回空内容，使用文件标题作为商品名称兜底")  # 记录空结果日志
            return file_title  # 返回文件标题兜底

        logger.info(f"步骤3：大模型识别商品名称成功，结果为：{item_name}")  # 记录识别成功日志
        return item_name  # 返回识别结果

    except Exception as e:
        logger.error(f"步骤3：大模型调用失败，原因：{str(e)}", exc_info=True)  # 记录调用失败日志
        return file_title  # 异常时返回文件标题兜底


def step_4_update_chunks(state: ImportGraphState, chunks: List[Dict], item_name: str):
    """将商品名称回填到全局状态和每个切片中。"""
    state["item_name"] = item_name  # 更新全局状态中的商品名称
    for chunk in chunks:  # 遍历所有切片
        chunk["item_name"] = item_name  # 为每个切片设置商品名称
    state["chunks"] = chunks  # 同步更新状态中的切片列表
    logger.info(f"步骤4：商品名称回填完成，共为{len(chunks)}个切片添加item_name字段，值为：{item_name}")  # 记录回填完成日志


def step_5_generate_vectors(item_name: str) -> Tuple[Any, Any]:
    """为商品名称生成BGE-M3稠密和稀疏双向量。"""
    logger.info(f"开始执行步骤5：为商品名称[{item_name}]生成BGE-M3双向量")  # 记录步骤开始日志

    if not item_name:  # 判断商品名称是否为空
        logger.warning("商品名称为空，跳过向量生成，返回空向量")  # 记录跳过日志
        return None, None  # 返回空向量

    try:
        vector_result = generate_embeddings([item_name])  # 调用向量生成工具

        if vector_result and "dense" in vector_result and "sparse" in vector_result:  # 判断向量结果是否完整
            dense_vector = vector_result["dense"][0]  # 提取稠密向量
            sparse_vector = vector_result["sparse"][0]  # 提取稀疏向量
            logger.info("步骤5：BGE-M3稠密+稀疏向量生成成功")  # 记录生成成功日志
        else:
            logger.warning("步骤5：向量生成工具返回空结果，无法提取双向量")  # 记录空结果警告
            dense_vector, sparse_vector = None, None  # 返回空向量

    except Exception as e:
        logger.error(f"步骤5：向量生成失败，原因：{str(e)}", exc_info=True)  # 记录生成失败日志
        dense_vector, sparse_vector = None, None  # 返回空向量

    return dense_vector, sparse_vector  # 返回双向量


def step_6_save_to_milvus(state: ImportGraphState, file_title: str, item_name: str, dense_vector, sparse_vector):
    """将商品名称及双向量持久化到Milvus向量数据库。"""
    milvus_uri = os.environ.get("MILVUS_URL")  # 获取Milvus连接地址
    collection_name = os.environ.get("ITEM_NAME_COLLECTION")  # 获取集合名称

    if not all([milvus_uri, collection_name]):  # 判断核心配置是否缺失
        logger.warning("Milvus配置缺失（MILVUS_URL/ITEM_NAME_COLLECTION），跳过数据保存")  # 记录配置缺失警告
        return  # 直接返回

    logger.info(f"开始执行步骤6：将商品名称[{item_name}]保存到Milvus集合[{collection_name}]")  # 记录步骤开始日志

    try:
        client = get_milvus_client()  # 获取Milvus客户端
        if not client:  # 判断客户端是否为空
            logger.error("无法获取Milvus客户端（连接失败），跳过数据保存")  # 记录连接失败日志
            return  # 直接返回

        if not client.has_collection(collection_name=collection_name):  # 判断集合是否存在
            logger.info(f"Milvus集合[{collection_name}]不存在，开始创建Schema和索引")  # 记录创建日志
            schema = client.create_schema(auto_id=True, enable_dynamic_field=True)  # 创建Schema
            schema.add_field(
                field_name="pk",  # 设置主键字段名
                datatype=DataType.INT64,  # 设置数据类型
                is_primary=True,  # 设置为主键
                auto_id=True  # 设置自增
            )
            schema.add_field(
                field_name="file_title",  # 设置文件标题字段名
                datatype=DataType.VARCHAR,  # 设置数据类型
                max_length=65535  # 设置最大长度
            )
            schema.add_field(
                field_name="item_name",  # 设置商品名字段名
                datatype=DataType.VARCHAR,  # 设置数据类型
                max_length=65535  # 设置最大长度
            )
            schema.add_field(
                field_name="dense_vector",  # 设置稠密向量字段名
                datatype=DataType.FLOAT_VECTOR,  # 设置数据类型
                dim=1024  # 设置维度
            )
            schema.add_field(
                field_name="sparse_vector",  # 设置稀疏向量字段名
                datatype=DataType.SPARSE_FLOAT_VECTOR  # 设置数据类型
            )

            index_params = client.prepare_index_params()  # 准备索引参数
            index_params.add_index(
                field_name="dense_vector",  # 指定稠密向量字段
                index_name="dense_vector_index",  # 指定索引名称
                index_type="HNSW",  # 使用HNSW索引
                metric_type="COSINE",  # 使用余弦相似度
                params={"M": 16, "efConstruction": 200}  # 设置索引参数
            )

            index_params.add_index(
                field_name="sparse_vector",  # 指定稀疏向量字段
                index_name="sparse_vector_index",  # 指定索引名称
                index_type="SPARSE_INVERTED_INDEX",  # 使用稀疏倒排索引
                metric_type="IP",  # 使用内积相似度
                params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"}  # 设置稀疏索引参数
            )

            client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)  # 创建集合
            logger.info(f"Milvus集合[{collection_name}]创建成功，包含Schema和向量索引")  # 记录创建成功日志

        clean_item_name = (item_name or "").strip()  # 清洗商品名称
        if clean_item_name:  # 判断商品名是否非空
            client.load_collection(collection_name=collection_name)  # 加载集合
            safe_item_name = escape_milvus_string(clean_item_name)  # 转义商品名特殊字符
            filter_expr = f'item_name=="{safe_item_name}"'  # 构造删除过滤表达式
            client.delete(collection_name=collection_name, filter=filter_expr)  # 删除旧数据
            logger.info(f"Milvus幂等性处理完成，已删除集合中[{clean_item_name}]的历史数据")  # 记录清理完成日志

        data = {
            "file_title": file_title,  # 设置文件标题
            "item_name": item_name  # 设置商品名称
        }
        if dense_vector is not None:  # 判断稠密向量是否非空
            data["dense_vector"] = dense_vector  # 添加稠密向量
        if sparse_vector is not None:  # 判断稀疏向量是否非空
            data["sparse_vector"] = normalize_sparse_vector(sparse_vector)  # 归一化后添加稀疏向量

        client.insert(collection_name=collection_name, data=[data])  # 插入数据
        client.load_collection(collection_name=collection_name)  # 加载集合使数据可查

        state["item_name"] = item_name  # 同步商品名称到全局状态
        logger.info(f"步骤6：商品名称[{item_name}]成功存入Milvus集合[{collection_name}]，数据：{list(data.keys())}")  # 记录保存成功日志

    except Exception as e:
        logger.error(f"步骤6：数据存入Milvus失败，原因：{str(e)}", exc_info=True)  # 记录保存失败日志


def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """商品名称识别主节点：构建上下文、调用LLM、回填数据并持久化到Milvus。"""
    node_name = sys._getframe().f_code.co_name  # 获取当前节点名
    logger.info(f">>> 开始执行核心节点：【商品名称识别】{node_name}")  # 记录节点启动日志
    add_running_task(state.get("task_id", ""), node_name)  # 标记当前节点为运行中

    try:
        file_title, chunks = step_1_get_inputs(state)  # 提取并校验输入
        if not chunks:  # 判断是否存在有效切片
            logger.warning(f">>> 节点执行警告：{node_name}（无有效切片数据），跳过识别")  # 记录无切片警告
            return state  # 返回原状态

        context = step_2_build_context(chunks)  # 构建识别上下文
        item_name = step_3_call_llm(file_title, context)  # 调用大模型识别商品名称
        step_4_update_chunks(state, chunks, item_name)  # 回填商品名称
        dense_vector, sparse_vector = step_5_generate_vectors(item_name)  # 生成双向量
        step_6_save_to_milvus(state, file_title, item_name, dense_vector, sparse_vector)  # 持久化到Milvus

        logger.info(f">>> 核心节点执行完成：【商品名称识别】{node_name}，识别结果：{item_name}，已存入Milvus")  # 记录完成日志

    except Exception as e:
        logger.error(f">>> 核心节点执行失败：【商品名称识别】{node_name}，错误信息：{str(e)}", exc_info=True)  # 记录异常日志
        state["item_name"] = "未知商品"  # 异常时设置默认值

    return state  # 返回更新后的状态


def test_node_item_name_recognition():
    """商品名称识别节点本地测试方法。"""
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")  # 记录测试开始日志
    try:
        mock_state = ImportGraphState({  # 构造模拟状态
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
            "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名
            "chunks": [  # 模拟文本切片
                {
                    "title": "产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        })

        result_state = node_item_name_recognition(mock_state)  # 调用核心节点

        logger.info("=== 商品名称识别节点本地测试完成 ===")  # 记录测试完成日志
        logger.info(f"测试任务ID：{result_state.get('task_id')}")  # 打印测试任务ID
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")  # 打印识别结果
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")  # 打印切片数量
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")  # 打印首个切片商品名

        milvus_client = get_milvus_client()  # 获取Milvus客户端
        collection_name = os.environ.get("ITEM_NAME_COLLECTION")  # 获取集合名称
        if milvus_client and collection_name:  # 判断是否可查询Milvus
            milvus_client.load_collection(collection_name)  # 加载集合
            item_name = result_state.get('item_name')  # 获取识别结果
            safe_name = escape_milvus_string(item_name)  # 转义商品名
            res = milvus_client.query(
                collection_name=collection_name,  # 指定集合
                filter=f'item_name=="{safe_name}"',  # 构造查询过滤条件
                output_fields=["file_title", "item_name"]  # 指定输出字段
            )
            logger.info(f"Milvus中检索到的数据：{res}")  # 打印检索结果

    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)  # 记录测试失败日志


if __name__ == "__main__":
    test_node_item_name_recognition()  # 执行本地测试
