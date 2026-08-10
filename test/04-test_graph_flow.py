import json

from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import create_default_state
import sys
from app.core.logger import logger

logger.info("===== 开始测试 =====")  # 记录测试开始日志

initial_state = create_default_state(local_file_path="万用表RS-12的使用.pdf")  # 创建默认图状态并指定测试 PDF 路径
final_state = None  # 初始化最终状态变量

for event in kb_import_app.stream(initial_state):  # 流式执行知识库导入图
    for key, value in event.items():  # 遍历当前事件中的节点与状态
        logger.info(f"节点: {key}")  # 记录当前执行节点名称
        final_state = value  # 保存当前节点返回的状态

logger.info(f"最终状态: \n {json.dumps(final_state, indent=4, ensure_ascii=False)}")  # 格式化输出最终状态

logger.info("图结构:")  # 记录即将打印图结构
kb_import_app.get_graph().print_ascii()  # 以 ASCII 形式打印图结构

logger.info("===== 测试结束 =====")  # 记录测试结束日志
