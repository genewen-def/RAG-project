from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.import_process.agent.nodes.node_entry import node_entry
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.import_process.agent.nodes.node_md_img import node_md_img
from app.import_process.agent.nodes.node_document_split import node_document_split
from app.import_process.agent.nodes.node_item_name_recognition import node_item_name_recognition
from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus


load_dotenv()  # 加载环境变量配置

workflow = StateGraph(ImportGraphState)  # 初始化导入工作流状态图

workflow.add_node("node_entry", node_entry)  # 注册流程入口节点
workflow.add_node("node_pdf_to_md", node_pdf_to_md)  # 注册 PDF 转 Markdown 节点
workflow.add_node("node_md_img", node_md_img)  # 注册 Markdown 图片处理节点
workflow.add_node("node_document_split", node_document_split)  # 注册文档分块节点
workflow.add_node("node_item_name_recognition", node_item_name_recognition)  # 注册项目名识别节点
workflow.add_node("node_bge_embedding", node_bge_embedding)  # 注册 BGE 向量化节点
workflow.add_node("node_import_milvus", node_import_milvus)  # 注册 Milvus 入库节点

workflow.set_entry_point("node_entry")  # 设置工作流入口节点


def route_after_entry(state: ImportGraphState) -> str:  # 入口节点后的条件路由函数
    """根据状态配置选择后续执行路径。"""
    if state.get("is_md_read_enabled"):  # 判断是否启用 Markdown 直接导入
        return "node_md_img"  # 跳转至 Markdown 图片处理节点
    elif state.get("is_pdf_read_enabled"):  # 判断是否启用 PDF 导入
        return "node_pdf_to_md"  # 跳转至 PDF 转 Markdown 节点
    else:  # 未启用任何导入方式
        return END  # 结束工作流


workflow.add_conditional_edges(
    "node_entry",  # 源节点名称
    route_after_entry,  # 条件路由函数
    {"node_md_img": "node_md_img", "node_pdf_to_md": "node_pdf_to_md", END: END}  # 路由目标映射
)

workflow.add_edge("node_pdf_to_md", "node_md_img")  # 连接 PDF 转换与图片处理节点
workflow.add_edge("node_md_img", "node_document_split")  # 连接图片处理与文档分块节点
workflow.add_edge("node_document_split", "node_item_name_recognition")  # 连接文档分块与项目名识别节点
workflow.add_edge("node_item_name_recognition", "node_bge_embedding")  # 连接项目名识别与向量化节点
workflow.add_edge("node_bge_embedding", "node_import_milvus")  # 连接向量化与 Milvus 入库节点
workflow.add_edge("node_import_milvus", END)  # 连接 Milvus 入库节点到结束节点

kb_import_app = workflow.compile()  # 编译工作流为可执行应用
