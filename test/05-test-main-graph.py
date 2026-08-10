from app.core.logger import logger
from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import ImportGraphState

if __name__ == "__main__":  # 仅在直接运行该脚本时执行测试流程
    from app.utils.path_util import PROJECT_ROOT  # 导入项目根目录路径常量
    import os  # 导入操作系统模块用于路径操作

    logger.info("===== 开始执行知识图谱导入全流程测试 =====")  # 记录全流程测试开始日志

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")  # 拼接测试 PDF 相对路径
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)  # 拼接测试 PDF 绝对路径

    test_output_dir = os.path.join(PROJECT_ROOT, "output")  # 拼接中间文件输出目录路径
    os.makedirs(test_output_dir, exist_ok=True)  # 若输出目录不存在则创建

    if not os.path.exists(test_pdf_path):  # 判断测试 PDF 文件是否存在
        logger.error(f"全流程测试失败：测试PDF文件不存在，路径：{test_pdf_path}")  # 记录文件不存在错误
        logger.info("请检查文件路径，或手动将测试文件放入项目根目录的doc文件夹中")  # 提示用户检查路径
    else:
        test_state = ImportGraphState({  # 构造导入图初始状态字典
            "task_id": "test_kg_import_workflow_001",  # 设置测试任务唯一标识
            "user_id": "test_user",  # 设置测试用户标识
            "local_file_path": test_pdf_path,  # 设置本地测试 PDF 文件路径
            "local_dir": test_output_dir,  # 设置中间文件输出目录
            "is_pdf_read_enabled": False,  # 关闭 PDF 读取开关
            "is_md_read_enabled": False  # 关闭 MD 读取开关
        })  # 完成初始状态构造

        try:  # 开始执行全流程并捕获异常
            logger.info(f"测试任务启动，PDF文件路径：{test_pdf_path}")  # 记录任务启动与 PDF 路径
            logger.info(f"中间文件输出目录：{test_output_dir}")  # 记录中间输出目录
            logger.info("开始执行全流程节点，依次执行：entry→pdf2md→md_img→split→item_name→embedding→milvus→kg")  # 记录节点执行顺序

            final_state = None  # 初始化最终状态变量

            for step in kb_import_app.stream(test_state, stream_mode="values"):  # 流式执行 LangGraph 全流程
                current_node = list(step.keys())[-1] if step else "未知节点"  # 获取当前执行完成的节点名称
                logger.info(f"✅ 节点执行完成：{current_node}")  # 记录节点执行完成
                final_state = step  # 保存当前步骤返回的状态

            if final_state:  # 判断是否成功获取最终状态
                logger.info("-" * 80)  # 打印分隔线
                logger.info("===== 全流程测试执行成功，核心结果预览 =====")  # 记录执行成功日志

                chunks = final_state.get("chunks", [])  # 从最终状态获取切片列表
                chunk_count = len(chunks)  # 计算切片总数
                md_content = final_state.get("md_content", "")[:150]  # 获取 MD 内容前 150 字符
                has_embedding = all("dense_vector" in c and "sparse_vector" in c for c in chunks) if chunks else False  # 判断是否所有切片均含向量
                has_chunk_id = all("chunk_id" in c for c in chunks) if chunks else False  # 判断是否所有切片均含 chunk_id
                kg_id = final_state.get("kg_id", "未生成")  # 获取知识图谱导入 ID

                logger.info(f"📄 PDF转MD内容预览（前150字符）：{md_content}...")  # 打印 MD 内容预览
                logger.info(f"📝 文档切分总切片数：{chunk_count}")  # 打印切片总数
                logger.info(f"🔍 所有切片是否完成向量化：{'是' if has_embedding else '否'}")  # 打印向量化状态
                logger.info(f"🗄️  所有切片是否完成Milvus入库（含chunk_id）：{'是' if has_chunk_id else '否'}")  # 打印 Milvus 入库状态
                logger.info(f"🧠 知识图谱导入ID：{kg_id}")  # 打印 KG 导入 ID
                logger.info(f"📂 最终状态包含的核心键：{list(final_state.keys())}")  # 打印最终状态键名
                logger.info("-" * 80)  # 打印分隔线
        except Exception as e:  # 捕获全流程执行中的异常
            logger.error(f"===== 全流程测试运行失败 =====", exc_info=True)  # 记录测试失败及堆栈
            logger.error(f"异常原因：{str(e)}")  # 记录异常原因

    logger.info("===== 知识图谱导入全流程测试结束 =====")  # 记录全流程测试结束日志
