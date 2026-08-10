from langgraph.graph import StateGraph, END
from app.query_process.agent.state import QueryGraphState
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_query_kg import node_query_kg
from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp

builder = StateGraph(QueryGraphState)  # 初始化状态图构建器

builder.add_node("node_item_name_confirm", node_item_name_confirm)  # 注册商品确认节点
builder.add_node("node_multi_search", lambda x: x)  # 注册多路搜索分叉虚拟节点
builder.add_node("node_search_embedding", node_search_embedding)  # 注册向量搜索节点
builder.add_node("node_search_embedding_hyde", node_search_embedding_hyde)  # 注册HyDE搜索节点
builder.add_node("node_query_kg", node_query_kg)  # 注册图谱查询节点
builder.add_node("node_web_search_mcp", node_web_search_mcp)  # 注册网络搜索节点
builder.add_node("node_join", lambda x: {})  # 注册多路搜索合并虚拟节点
builder.add_node("node_rrf", node_rrf)  # 注册RRF排序节点
builder.add_node("node_rerank", node_rerank)  # 注册重排节点
builder.add_node("node_answer_output", node_answer_output)  # 注册答案生成节点

builder.set_entry_point("node_item_name_confirm")  # 设置流程入口节点


def route_after_item_confirm(state: QueryGraphState):  # 定义商品确认后的路由函数
    """根据状态决定跳转至输出节点或多路搜索节点。"""
    if state.get("answer"):  # 若已生成回答则直接输出
        return "node_answer_output"  # 返回答案输出节点名称
    return "node_multi_search"  # 返回多路搜索节点名称


builder.add_conditional_edges(  # 添加从商品确认节点出发的条件边
    "node_item_name_confirm",  # 起始节点名称
    route_after_item_confirm  # 路由判断函数
)

builder.add_edge("node_multi_search", "node_search_embedding")  # 连接多路搜索到向量搜索
builder.add_edge("node_multi_search", "node_search_embedding_hyde")  # 连接多路搜索到HyDE搜索
builder.add_edge("node_multi_search", "node_web_search_mcp")  # 连接多路搜索到网络搜索
builder.add_edge("node_multi_search", "node_query_kg")  # 连接多路搜索到图谱查询

builder.add_edge("node_search_embedding", "node_join")  # 连接向量搜索到合并节点
builder.add_edge("node_search_embedding_hyde", "node_join")  # 连接HyDE搜索到合并节点
builder.add_edge("node_web_search_mcp", "node_join")  # 连接网络搜索到合并节点
builder.add_edge("node_query_kg", "node_join")  # 连接图谱查询到合并节点

builder.add_edge("node_join", "node_rrf")  # 连接合并节点到排序节点
builder.add_edge("node_rrf", "node_rerank")  # 连接排序节点到重排节点
builder.add_edge("node_rerank", "node_answer_output")  # 连接重排节点到生成节点
builder.add_edge("node_answer_output", END)  # 连接生成节点到流程结束

query_app = builder.compile()  # 编译生成可执行查询图
