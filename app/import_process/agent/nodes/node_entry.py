import os
import sys
from os.path import splitext

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.format_utils import format_state
from app.utils.task_utils import add_running_task, add_done_task


def node_entry(state: ImportGraphState) -> ImportGraphState:
    """工作流入口节点：校验参数、判断文件类型并提取业务标识。"""
    func_name = sys._getframe().f_code.co_name  # 获取当前函数名

    logger.debug(f"【{func_name}】节点启动，\n当前工作流状态：{format_state(state)}")  # 记录节点启动日志

    add_running_task(state["task_id"], func_name)  # 标记当前节点为运行中

    document_path = state.get("local_file_path", "")  # 获取文件路径
    if not document_path:  # 校验文件路径是否为空
        logger.error(f"【{func_name}】核心参数缺失：工作流状态中未配置local_file_path，文件路径为空")  # 记录参数缺失错误
        return state  # 返回原状态

    if document_path.endswith(".pdf"):  # 判断是否为PDF文件
        logger.info(f"【{func_name}】文件类型校验通过：{document_path} → PDF格式，开启PDF解析流程")  # 记录PDF类型日志
        state["is_pdf_read_enabled"] = True  # 开启PDF解析开关
        state["pdf_path"] = document_path  # 设置PDF路径
    elif document_path.endswith(".md"):  # 判断是否为MD文件
        logger.info(f"【{func_name}】文件类型校验通过：{document_path} → MD格式，开启MD解析流程")  # 记录MD类型日志
        state["is_md_read_enabled"] = True  # 开启MD解析开关
        state["md_path"] = document_path  # 设置MD路径
    else:
        logger.warning(f"【{func_name}】文件类型校验失败：{document_path} → 不支持的格式，仅支持.pdf/.md")  # 记录不支持的格式警告

    file_name = os.path.basename(document_path)  # 提取文件名
    state["file_title"] = splitext(file_name)[0]  # 去除后缀设置为文件标题
    logger.info(f"【{func_name}】文件业务标识提取完成：file_title = {state['file_title']}")  # 记录标识提取日志

    add_done_task(state["task_id"], func_name)  # 标记当前节点为已完成

    logger.debug(f"【{func_name}】节点执行完成，\n更新后工作流状态：{format_state(state)}")  # 记录节点完成日志

    return state  # 返回更新后的状态


if __name__ == '__main__':
    logger.info("===== 开始node_entry节点单元测试 =====")  # 记录测试开始日志

    test_state1 = create_default_state(  # 构造不支持的TXT测试状态
        task_id="test_task_001",
        local_file_path="联想海豚用户手册.txt"
    )
    node_entry(test_state1)  # 执行TXT场景测试

    test_state2 = create_default_state(  # 构造MD测试状态
        task_id="test_task_002",
        local_file_path="小米用户手册.md"
    )
    node_entry(test_state2)  # 执行MD场景测试

    test_state3 = create_default_state(  # 构造PDF测试状态
        task_id="test_task_003",
        local_file_path="万用表的使用.pdf"
    )
    node_entry(test_state3)  # 执行PDF场景测试

    logger.info("===== 结束node_entry节点单元测试 =====")  # 记录测试结束日志
