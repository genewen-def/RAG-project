import os
import sys
from typing import List, Dict, Any
from pymilvus import DataType
from app.import_process.agent.state import ImportGraphState
from app.clients.milvus_utils import get_milvus_client
from app.utils.task_utils import add_running_task
from app.core.logger import logger
from app.conf.milvus_config import milvus_config
from app.utils.escape_milvus_string_utils import escape_milvus_string

CHUNKS_COLLECTION_NAME = milvus_config.chunks_collection


def node_import_milvus(state: Dict[str, Any]) -> Dict[str, Any]:
    """Milvus切片数据入库节点：校验输入、准备集合、清理旧数据并批量插入。"""
    current_node = sys._getframe().f_code.co_name  # 获取当前函数名
    logger.info(f">>> 开始执行LangGraph节点：{current_node}（Milvus切片数据入库）")  # 记录节点启动日志
    add_running_task(state["task_id"], current_node)  # 标记当前节点为运行中
    logger.info("--- Milvus切片数据入库流程启动 ---")  # 记录流程启动日志

    try:
        chunks_json_data, vector_dimension = step_1_check_input(state)  # 校验输入并提取向量维度
        client = step_2_prepare_collection(vector_dimension)  # 准备Milvus集合
        step_3_clean_old_data(client, chunks_json_data)  # 清理同item_name旧数据
        updated_chunks = step_4_insert_data(client, chunks_json_data)  # 批量插入并回填主键
        state["chunks"] = updated_chunks  # 将回填后的切片写回状态

        logger.info("--- Milvus切片数据入库流程完成 ---")  # 记录流程完成日志
    except Exception as e:
        logger.error(f"Milvus切片数据入库节点执行失败：{str(e)}", exc_info=True)  # 记录节点异常日志
        raise ValueError(f"Milvus 导入过程中发生错误: {e}")  # 抛出异常终止节点

    return state  # 返回更新后的状态


def step_1_check_input(state: Dict[str, Any]) -> tuple[List[Dict[str, Any]], int]:
    """校验state中chunks有效性，返回切片列表和稠密向量维度。"""
    chunks_json_data = state.get("chunks")  # 获取待入库切片数据
    if not chunks_json_data:  # 判断chunks是否为空
        logger.error("Milvus入库校验失败：state中chunks字段为空")  # 记录空数据错误
        raise ValueError("错误: chunks为空，无法执行Milvus入库")  # 抛出异常
    if not isinstance(chunks_json_data, list) or len(chunks_json_data) == 0:  # 判断是否为非空列表
        logger.error("Milvus入库校验失败：chunks非列表类型或为空列表")  # 记录类型错误
        raise ValueError("错误: chunks数据格式不正确，必须为非空列表")  # 抛出异常
    first_chunk = chunks_json_data[0]  # 获取第一个切片
    if 'dense_vector' not in first_chunk:  # 判断是否存在稠密向量字段
        logger.error("Milvus入库校验失败：切片缺失dense_vector字段，上游向量化节点可能执行失败")  # 记录缺失字段错误
        raise ValueError("错误: 数据中缺失dense_vector字段，请检查上游向量化节点执行状态")  # 抛出异常

    vector_dimension = len(first_chunk['dense_vector'])  # 计算向量维度
    item_name = first_chunk.get('item_name', '未知商品名')  # 获取商品名称
    logger.info(
        f"Milvus入库校验通过，待入库切片数：{len(chunks_json_data)} | 向量维度：{vector_dimension} | 商品名称：{item_name}")  # 记录校验通过日志

    return chunks_json_data, vector_dimension  # 返回切片和维度


def create_collection(client, collection_name: str, vector_dimension: int):
    """创建Milvus集合及稠密/稀疏向量索引。"""
    schema = client.create_schema(auto_id=True, enable_dynamic_fields=True)  # 创建Schema

    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)  # 添加自增主键字段
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)  # 添加内容字段
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)  # 添加标题字段
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)  # 添加父标题字段
    schema.add_field(field_name="part", datatype=DataType.INT8)  # 添加分片编号字段
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)  # 添加文件标题字段
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)  # 添加商品名字段
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)  # 添加稀疏向量字段
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dimension)  # 添加稠密向量字段

    index_params = client.prepare_index_params()  # 准备索引参数
    index_params.add_index(
        field_name="dense_vector",  # 指定稠密向量字段
        index_name="dense_vector_index",  # 指定索引名称
        index_type="HNSW",  # 使用HNSW索引类型
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
    logger.info(f"Milvus集合创建成功：{collection_name}，向量维度：{vector_dimension}")  # 记录创建成功日志


def step_2_prepare_collection(vector_dimension: int):
    """连接Milvus并确保目标集合存在，不存在则自动创建。"""
    logger.info(f"开始准备Milvus环境，目标集合：{CHUNKS_COLLECTION_NAME}")  # 记录准备日志
    client = get_milvus_client()  # 获取Milvus客户端
    if client is None:  # 判断客户端是否为空
        logger.error("Milvus客户端获取失败：get_milvus_client()返回空，连接可能异常")  # 记录连接失败错误
        raise ValueError("Milvus 连接失败：get_milvus_client() 返回空")  # 抛出异常
    if not CHUNKS_COLLECTION_NAME:  # 判断集合名称是否为空
        logger.error("Milvus集合名称未配置：CHUNKS_COLLECTION_NAME为空")  # 记录配置缺失错误
        raise ValueError("未配置CHUNKS_COLLECTION集合名称")  # 抛出异常

    if not client.has_collection(collection_name=CHUNKS_COLLECTION_NAME):  # 判断集合是否存在
        logger.info(f"Milvus集合{CHUNKS_COLLECTION_NAME}不存在，开始自动创建Schema和索引")  # 记录自动创建日志
        create_collection(client, CHUNKS_COLLECTION_NAME, vector_dimension)  # 创建集合
    else:
        logger.info(f"Milvus集合{CHUNKS_COLLECTION_NAME}已存在，直接复用")  # 记录复用日志

    return client  # 返回客户端


def step_3_clean_old_data(client, chunks_json_data: List[Dict[str, Any]]):
    """基于item_name清理Milvus中的旧切片数据，实现幂等性。"""
    item_names = sorted(
    {
        name
        for x in chunks_json_data or []
        if (name := str(x.get("item_name", "")).strip())
    })  # 提取并去重非空item_name

    if not item_names:  # 判断是否存在有效item_name
        logger.warning("Milvus幂等性清理跳过：切片中无有效item_name")  # 记录跳过日志
        return  # 直接返回
    if len(item_names) > 1:  # 判断是否存在多个item_name
        logger.warning(f"Milvus幂等性清理：本次检测到多个item_name，将逐个清理：{item_names}")  # 记录多商品名警告

    for i_name in item_names:  # 遍历每个item_name
        _clear_chunks_by_item_name(client, CHUNKS_COLLECTION_NAME, i_name)  # 清理对应旧数据


def _clear_chunks_by_item_name(client, collection_name: str, item_name: str):
    """根据item_name删除指定集合中的旧切片数据。"""
    i_name = (item_name or "").strip()  # 去除首尾空格
    if not i_name:  # 判断item_name是否为空
        logger.warning("Milvus单商品清理跳过：item_name为空")  # 记录空值跳过日志
        return  # 直接返回
    if not collection_name:  # 判断集合名称是否为空
        logger.warning("Milvus单商品清理跳过：集合名称未配置")  # 记录配置缺失跳过日志
        return  # 直接返回

    try:
        if not client.has_collection(collection_name=collection_name):  # 判断集合是否存在
            logger.info(f"Milvus单商品清理跳过：集合{collection_name}不存在")  # 记录不存在跳过日志
            return  # 直接返回

        safe_item_name = escape_milvus_string(i_name)  # 转义item_name特殊字符
        filter_expr = f'item_name == "{safe_item_name}"'  # 构造删除过滤表达式
        logger.info(f"Milvus幂等性清理：开始删除集合{collection_name}中item_name={i_name}的旧数据")  # 记录清理日志

        client.delete(collection_name=collection_name, filter=filter_expr)  # 执行删除操作

        if hasattr(client, "flush"):  # 判断客户端是否支持flush
            try:
                client.flush(collection_name=collection_name)  # 强制刷新保证删除生效
            except Exception as e:
                logger.warning(f"Milvus幂等性清理：flush操作失败，不影响主流程 | 错误：{str(e)}")  # 记录flush警告

        logger.info(f"Milvus幂等性清理完成：成功删除item_name={i_name}的旧数据")  # 记录清理完成日志
    except Exception as e:
        logger.error(f"Milvus幂等性清理失败：item_name={i_name} | 错误：{str(e)}", exc_info=True)  # 记录清理失败日志
        raise ValueError(f"幂等清理失败（item_name={i_name}）: {e}")  # 抛出异常


def step_4_insert_data(client, chunks_json_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """批量插入切片数据到Milvus，并将自增chunk_id回填到切片。"""
    data_to_insert = []  # 初始化待插入数据列表
    for item in chunks_json_data:  # 遍历切片数据
        item_copy = item.copy()  # 复制切片避免修改原数据
        if isinstance(item_copy, dict) and "chunk_id" in item_copy:  # 判断是否包含手动chunk_id
            item_copy.pop("chunk_id", None)  # 移除手动chunk_id避免冲突
        data_to_insert.append(item_copy)  # 加入待插入列表

    logger.info(f"Milvus数据插入：准备{len(data_to_insert)}条切片数据，开始批量插入")  # 记录插入准备日志
    insert_result = client.insert(collection_name=CHUNKS_COLLECTION_NAME, data=data_to_insert)  # 执行批量插入
    insert_count = insert_result.get('insert_count', 0)  # 获取插入数量
    logger.info(f"Milvus数据插入完成：成功插入{insert_count}条数据，插入结果：{insert_result}")  # 记录插入完成日志

    inserted_ids = insert_result.get('ids', [])  # 获取生成的主键列表
    if inserted_ids and len(inserted_ids) == len(chunks_json_data):  # 判断主键数量是否匹配
        logger.info(f"Milvus主键回填：开始将{len(inserted_ids)}个自增chunk_id回填到切片")  # 记录回填日志
        for idx, item in enumerate(chunks_json_data):  # 遍历原始切片
            item['chunk_id'] = str(inserted_ids[idx])  # 将主键转为字符串并回填
        logger.info("Milvus主键回填完成：所有切片已绑定chunk_id")  # 记录回填完成日志
    else:
        logger.warning(f"Milvus主键回填失败：生成ID数量({len(inserted_ids)})与切片数量({len(chunks_json_data)})不一致")  # 记录回填失败警告

    return chunks_json_data  # 返回回填后的切片列表


if __name__ == '__main__':
    import sys
    import os
    from dotenv import load_dotenv

    current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件目录
    project_root = os.path.dirname(os.path.dirname(current_dir))  # 计算项目根目录
    load_dotenv(os.path.join(project_root, ".env"))  # 加载环境变量

    dim = 1024  # 设置测试向量维度
    test_state = {
        "task_id": "test_milvus_task",  # 设置测试任务ID
        "chunks": [  # 构造测试切片数据
            {
                "content": "Milvus 测试文本 1",
                "title": "测试标题",
                "item_name": "测试项目_Milvus",
                "parent_title": "test.pdf",
                "part": 1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,
                "sparse_vector": {1: 0.5, 10: 0.8}
            }
        ]
    }

    print("正在执行 Milvus 导入节点测试...")  # 打印测试开始信息
    try:
        if not os.getenv("MILVUS_URL"):  # 判断环境变量是否配置
            print("❌ 未设置 MILVUS_URL，无法连接 Milvus")  # 打印缺少URL提示
        elif not os.getenv("CHUNKS_COLLECTION"):  # 判断集合环境变量是否配置
            print("❌ 未设置 CHUNKS_COLLECTION")  # 打印缺少集合提示
        else:
            result_state = node_import_milvus(test_state)  # 执行节点函数

            chunks = result_state.get("chunks", [])  # 获取结果切片
            if chunks and chunks[0].get("chunk_id"):  # 判断是否生成chunk_id
                print(f"✅ Milvus 导入测试通过，生成 ID: {chunks[0]['chunk_id']}")  # 打印测试通过信息
            else:
                print("❌ 测试失败：未能获取 chunk_id")  # 打印测试失败信息

    except Exception as e:
        print(f"❌ 测试失败: {e}")  # 打印异常信息
