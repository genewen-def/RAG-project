import os
from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker
from app.conf.milvus_config import milvus_config
from app.core.logger import logger

_milvus_client = None  # 全局Milvus客户端单例


def get_milvus_client():
    """获取Milvus客户端单例实例。"""
    try:
        global _milvus_client  # 声明使用全局变量
        if _milvus_client is None:  # 判断单例是否未初始化
            milvus_uri = milvus_config.milvus_url  # 读取Milvus连接地址
            if not milvus_uri:  # 判断连接地址是否为空
                logger.error("Milvus客户端连接失败：缺少MILVUS_URL环境变量配置")  # 记录错误日志
                return None  # 返回空表示连接失败
            _milvus_client = MilvusClient(uri=milvus_uri)  # 初始化Milvus客户端
            logger.info("Milvus客户端连接成功")  # 记录连接成功日志
        return _milvus_client  # 返回客户端实例
    except Exception as e:  # 捕获连接异常
        logger.error(f"Milvus客户端连接异常：{str(e)}", exc_info=True)  # 记录异常日志
        return None  # 返回空表示异常


def _coerce_int64_ids(ids):
    """将ID转换为INT64类型并分离无效ID。"""
    ok, bad = [], []  # 初始化有效和无效ID列表
    for x in (ids or []):  # 遍历输入ID列表
        if x is None:  # 跳过空值
            continue  # 进入下一次循环
        try:
            ok.append(int(x))  # 尝试转换为整数并加入有效列表
        except Exception:  # 捕获转换异常
            bad.append(x)  # 将无效ID加入无效列表
    return ok, bad  # 返回有效ID和无效ID


def fetch_chunks_by_chunk_ids(
        client,
        collection_name: str,
        chunk_ids,
        *,
        output_fields=None,
        batch_size: int = 100,
):
    """通过chunk_id批量查询Milvus切片数据。"""
    if client is None:  # 判断客户端是否为空
        return []  # 返回空列表
    if not collection_name:  # 判断集合名是否为空
        return []  # 返回空列表
    if output_fields is None:  # 判断返回字段是否未指定
        output_fields = ["chunk_id", "content", "title", "parent_title", "item_name"]  # 设置默认返回字段

    ok_ids, bad_ids = _coerce_int64_ids(chunk_ids)  # 转换并校验ID类型
    if bad_ids:  # 判断是否存在无效ID
        logger.warning(f"存在无法转换为INT64的chunk_id，将跳过查询：{bad_ids}")  # 记录警告日志

    if not ok_ids:  # 判断有效ID是否为空
        return []  # 返回空列表

    results = []  # 初始化查询结果列表
    for i in range(0, len(ok_ids), batch_size):  # 按批次遍历有效ID
        batch = ok_ids[i: i + batch_size]  # 截取当前批次ID

        if hasattr(client, "get"):  # 判断客户端是否支持get方法
            try:
                got = client.get(collection_name=collection_name, ids=batch, output_fields=output_fields)  # 使用主键get方法查询
                if got:  # 判断查询结果是否非空
                    results.extend(got)  # 合并结果到总列表
                continue  # 进入下一批次
            except Exception as e:  # 捕获get查询异常
                logger.warning(f"Milvus get方法查询失败，将回退至query方法：{str(e)}")  # 记录回退日志

        try:
            expr = f"chunk_id in [{', '.join(str(x) for x in batch)}]"  # 构建过滤表达式
            q = client.query(collection_name=collection_name, filter=expr, output_fields=output_fields)  # 使用query方法查询
            if q:  # 判断查询结果是否非空
                results.extend(q)  # 合并结果到总列表
        except Exception as e:  # 捕获query查询异常
            logger.error(f"Milvus query方法批量查询chunk_id失败：{str(e)}", exc_info=True)  # 记录错误日志

    return results  # 返回查询结果


def create_hybrid_search_requests(dense_vector, sparse_vector, dense_params=None, sparse_params=None, expr=None,
                                  limit=5):
    """构建稠密向量和稀疏向量的混合搜索请求。"""
    if dense_params is None:  # 判断稠密向量参数是否未指定
        dense_params = {"metric_type": "COSINE"}  # 设置默认余弦相似度
    if sparse_params is None:  # 判断稀疏向量参数是否未指定
        sparse_params = {"metric_type": "IP"}  # 设置默认内积相似度

    dense_req = AnnSearchRequest(  # 构建稠密向量搜索请求
        data=[dense_vector],  # 传入稠密向量
        anns_field="dense_vector",  # 指定稠密向量字段
        param=dense_params,  # 传入搜索参数
        expr=expr,  # 传入过滤表达式
        limit=limit  # 设置返回数量
    )

    sparse_req = AnnSearchRequest(  # 构建稀疏向量搜索请求
        data=[sparse_vector],  # 传入稀疏向量
        anns_field="sparse_vector",  # 指定稀疏向量字段
        param=sparse_params,  # 传入搜索参数
        expr=expr,  # 传入过滤表达式
        limit=limit  # 设置返回数量
    )

    return [dense_req, sparse_req]  # 返回搜索请求列表


def hybrid_search(client, collection_name, reqs, ranker_weights=(0.5, 0.5), norm_score=False, limit=5,
                  output_fields=None, search_params=None):
    """执行Milvus稠密和稀疏向量混合搜索。"""
    try:
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)  # 初始化加权排序器

        if output_fields is None:  # 判断返回字段是否未指定
            output_fields = ["item_name"]  # 设置默认返回字段

        res = client.hybrid_search(  # 执行混合搜索
            collection_name=collection_name,  # 指定集合名称
            reqs=reqs,  # 传入搜索请求列表
            ranker=rerank,  # 传入加权排序器
            limit=limit,  # 设置返回数量
            output_fields=output_fields,  # 传入返回字段
            search_params=search_params  # 传入搜索参数
        )

        logger.info(f"Milvus混合搜索完成，集合[{collection_name}]共检索到{len(res[0])}条结果")  # 记录搜索完成日志
        return res  # 返回搜索结果
    except Exception as e:  # 捕获搜索异常
        logger.error(f"Milvus混合搜索执行失败，集合[{collection_name}]：{str(e)}", exc_info=True)  # 记录错误日志
        return None  # 返回空表示失败
