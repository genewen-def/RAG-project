import sys
import os
from typing import Any, List, Dict

from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
from app.utils.task_utils import add_running_task,add_done_task
from app.core.logger import logger


def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """BGE-M3文本向量化节点：校验输入、生成双向量并更新状态。"""
    current_node = sys._getframe().f_code.co_name  # 获取当前函数名
    logger.info(f">>> 开始执行LangGraph节点：{current_node}")  # 记录节点启动日志

    add_running_task(state.get("task_id", ""), current_node)  # 标记当前节点为运行中
    logger.info("--- BGE-M3 文本向量化处理启动 ---")  # 记录处理启动日志

    try:
        texts_to_embed = step_1_validate_input(state)  # 校验并获取待向量化的切片
        bge_m3_ef = step_2_init_model()  # 初始化BGE-M3模型实例
        output_data = step_3_generate_embeddings(texts_to_embed, bge_m3_ef)  # 批量生成双向量

        state['chunks'] = output_data  # 将带向量的切片写回状态
        logger.info(f"--- BGE-M3 向量化处理完成，共处理 {len(output_data)} 条文本切片 ---")  # 记录完成日志
        add_done_task(state.get("task_id", ""), current_node)  # 标记当前节点为已完成
    except Exception as e:
        logger.error(f"BGE-M3向量化节点执行失败：{str(e)}", exc_info=True)  # 记录节点异常日志

    return state  # 返回更新后的状态


def step_1_validate_input(state: ImportGraphState) -> List[Dict[str, Any]]:
    """校验state中chunks字段是否有效，返回切片列表。"""
    texts_to_embed = state.get("chunks")  # 从状态中提取切片数据
    if not isinstance(texts_to_embed, list) or not texts_to_embed:  # 校验是否为非空列表
        logger.error("向量化输入校验失败：chunks字段为空或非有效列表")  # 记录校验失败日志
        raise ValueError("错误: 无有效文本切片数据，无法执行向量化处理")  # 抛出异常终止节点

    logger.info(f"向量化输入校验通过，待处理文本切片数量：{len(texts_to_embed)}")  # 记录校验通过日志
    return texts_to_embed  # 返回校验通过的切片列表


def step_2_init_model():
    """获取BGE-M3单例模型实例，失败则抛出异常。"""
    try:
        ef = get_bge_m3_ef()  # 获取单例模型实例
        if ef is None:  # 校验模型实例是否为空
            raise ValueError("BGE-M3模型实例为None：pymilvus.model模块未找到或模型加载失败")  # 抛出模型加载异常

        logger.info("BGE-M3模型实例初始化成功（单例模式）")  # 记录模型初始化成功日志
        return ef  # 返回模型实例
    except Exception as e:
        error_msg = f"BGE-M3模型初始化失败：{e}，请检查模型路径/环境变量配置是否正确"  # 构造错误信息
        logger.error(error_msg)  # 记录模型初始化失败日志
        raise ValueError(error_msg)  # 抛出包装后的异常


def step_3_generate_embeddings(texts_to_embed: List[Dict[str, Any]], bge_m3_ef: Any) -> List[Dict[str, Any]]:
    """分批为切片生成稠密/稀疏双向量，异常批次保留原数据。"""
    output_data = []  # 初始化结果列表
    batch_size = 5  # 设置每批处理数量

    total = len(texts_to_embed)  # 获取待处理切片总数
    for i in range(0, total, batch_size):  # 按批次遍历切片
        batch_texts = texts_to_embed[i:i + batch_size]  # 截取当前批次切片
        start_idx, end_idx = i + 1, min(i + len(batch_texts), total)  # 计算当前批次起止索引

        try:
            input_texts = []  # 初始化当前批次输入文本列表
            for doc in batch_texts:  # 遍历当前批次切片
                item_name = doc["item_name"]  # 提取商品名称
                content = doc["content"]  # 提取切片内容
                text = f"商品：{item_name}，介绍：{content}" if item_name else content  # 拼接增强文本
                input_texts.append(text)  # 加入当前批次输入

            docs_embeddings = generate_embeddings(input_texts)  # 调用工具生成批量双向量
            if not docs_embeddings:  # 校验向量生成结果是否为空
                logger.warning(f"第{start_idx}-{end_idx}条切片：向量生成返回空，保留原数据")  # 记录空结果警告
                output_data.extend(batch_texts)  # 保留原切片数据
                continue  # 跳过当前批次

            for j, doc in enumerate(batch_texts):  # 遍历当前批次切片绑定向量
                item = doc.copy()  # 复制切片数据避免修改原数据
                item["dense_vector"] = docs_embeddings["dense"][j]  # 绑定稠密向量
                item["sparse_vector"] = docs_embeddings["sparse"][j]  # 绑定稀疏向量
                output_data.append(item)  # 加入结果列表

            logger.info(f"第{start_idx}-{end_idx}条切片：双向量生成成功")  # 记录批次成功日志

        except Exception as e:
            logger.error(
                f"第{start_idx}-{end_idx}条切片：向量生成失败，保留原数据 | 错误原因：{str(e)}",
                exc_info=True
            )  # 记录批次异常日志
            output_data.extend(batch_texts)  # 异常批次保留原数据
            continue  # 继续处理下一批次

    return output_data  # 返回带向量的切片列表


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件目录
    project_root = os.path.dirname(os.path.dirname(current_dir))  # 计算项目根目录
    test_state = ImportGraphState({  # 构造测试状态
        "task_id": "test_task_embedding_001",  # 测试任务ID
        "chunks": [  # 模拟文本切片
            {
                "content": "这是一个测试文档的内容，用于验证向量化是否成功。",
                "title": "测试文档标题",
                "item_name": "测试项目",
                "file_title": "测试文件.pdf"
            },
            {
                "content": "这是第二个测试文档的内容，用于验证批量处理逻辑。",
                "title": "测试文档标题2",
                "item_name": "测试项目",
                "file_title": "测试文件.pdf"
            }
        ]
    })

    logger.info("=== BGE-M3向量化节点本地单元测试启动 ===")  # 记录测试启动日志
    try:
        result_state = node_bge_embedding(test_state)  # 调用节点函数
        result_chunks = result_state.get("chunks", [])  # 获取结果切片

        logger.info(f"=== 向量化节点本地测试完成 ===")  # 记录测试完成日志
        logger.info(f"测试任务ID：{test_state.get('task_id')}")  # 打印测试任务ID
        logger.info(f"待处理切片数：2 | 实际处理切片数：{len(result_chunks)}")  # 打印处理数量
        logger.info(f"向量维度：{result_chunks}")  # 打印结果切片

        for idx, chunk in enumerate(result_chunks):  # 遍历结果切片
            has_dense = "dense_vector" in chunk  # 检查是否存在稠密向量
            has_sparse = "sparse_vector" in chunk  # 检查是否存在稀疏向量
            logger.info(
                f"第{idx + 1}条切片：稠密向量生成{'' if has_dense else '未'}成功 | 稀疏向量生成{'' if has_sparse else '未'}成功")  # 打印向量生成状态

    except Exception as e:
        logger.error(f"=== 向量化节点本地测试失败 ===" f"错误原因：{str(e)}", exc_info=True)  # 记录测试异常日志
        logger.warning("排查提示：请检查BGE-M3模型路径、显存是否充足、环境变量配置是否正确")  # 打印排查提示
