from typing import TypedDict
import copy
from app.core.logger import logger


class ImportGraphState(TypedDict):  # 定义导入工作流状态类型
    """导入工作流状态定义。"""

    task_id: str          # 任务唯一标识

    is_md_read_enabled: bool   # 是否启用 Markdown 读取
    is_pdf_read_enabled: bool  # 是否启用 PDF 读取

    is_normal_split_enabled: bool   # 是否启用普通分块
    is_silicon_flow_api_enabled: bool  # 是否启用 SiliconFlow API
    is_advanced_split_enabled: bool  # 是否启用高级分块
    is_vllm_enabled: bool  # 是否启用 vLLM

    local_dir: str        # 本地工作目录
    local_file_path: str  # 本地输入文件路径
    file_title: str       # 文件标题
    pdf_path: str         # PDF 文件路径
    md_path: str          # Markdown 文件路径
    split_path: str       # 分块结果路径
    embeddings_path: str  # 向量文件路径

    md_content: str       # Markdown 全文内容
    chunks: list          # 文本分块列表
    item_name: str        # 识别出的主体名称

    embeddings_content: list  # 向量数据列表


graph_default_state: ImportGraphState = {  # 定义默认状态字典
    "task_id":"",  # 默认任务 ID
    "is_pdf_read_enabled": False,  # 默认不启用 PDF 读取
    "is_md_read_enabled": False,  # 默认不启用 Markdown 读取
    "is_normal_split_enabled": True,  # 默认启用普通分块
    "is_silicon_flow_api_enabled": True,  # 默认启用 SiliconFlow API
    "is_advanced_split_enabled": False,  # 默认不启用高级分块
    "is_vllm_enabled": False,  # 默认不启用 vLLM
    "local_dir": "",  # 默认本地目录
    "local_file_path": "",  # 默认本地文件路径
    "pdf_path": "",  # 默认 PDF 路径
    "md_path": "",  # 默认 Markdown 路径
    "file_title": "",  # 默认文件标题
    "split_path": "",  # 默认分块路径
    "embeddings_path": "",  # 默认向量文件路径
    "md_content": "",  # 默认 Markdown 内容
    "chunks": [],  # 默认空分块列表
    "item_name": "",  # 默认主体名称
    "embeddings_content": []  # 默认空向量列表
}


def create_default_state(**overrides) -> ImportGraphState:  # 创建默认状态并支持覆盖
    """创建默认状态，支持用传入参数覆盖默认值。"""
    state = copy.deepcopy(graph_default_state)  # 深拷贝默认状态
    state.update(overrides)  # 使用传入参数更新状态
    return state  # 返回新状态实例


def get_default_state() -> ImportGraphState:  # 获取新的默认状态实例
    """返回深拷贝后的默认状态，避免全局状态被污染。"""
    return copy.deepcopy(graph_default_state)  # 深拷贝并返回默认状态


if __name__ == "__main__":  # 脚本直接执行入口
    state = create_default_state(local_file_path="万用表RS-12的使用.pdf")  # 创建测试状态
    logger.info(state)  # 输出状态内容
