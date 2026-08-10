from typing import Dict, List
from .sse_utils import push_to_session

_tasks_running_list: Dict[str, List[str]] = {}  # 存储各任务正在运行的节点列表
_tasks_done_list: Dict[str, List[str]] = {}  # 存储各任务已完成的节点列表
_tasks_status: Dict[str, str] = {}  # 存储各任务状态
_tasks_result: Dict[str, Dict[str, str]] = {}  # 存储各任务结果字段

TASK_STATUS_PENDING = "pending"  # 任务待处理状态常量
TASK_STATUS_PROCESSING = "processing"  # 任务处理中状态常量
TASK_STATUS_COMPLETED = "completed"  # 任务完成状态常量
TASK_STATUS_FAILED = "failed"  # 任务失败状态常量

_NODE_NAME_TO_CN: Dict[str, str] = {  # 节点名到中文展示名的映射
    "upload_file": "开始上传文件",  # 上传文件节点
    "node_entry": "检查文件",  # 检查文件节点
    "node_pdf_to_md": "PDF转Markdown",  # PDF 转 Markdown 节点
    "node_md_img": "Markdown图片处理",  # Markdown 图片处理节点
    "node_item_name_recognition": "主体名称识别",  # 主体名称识别节点
    "node_document_split": "文档切分",  # 文档切分节点
    "node_bge_embedding": "向量生成",  # 向量生成节点
    "node_import_kg": "导入知识图谱",  # 导入知识图谱节点
    "node_import_milvus": "导入向量库",  # 导入向量库节点
    "__end__": "处理完成",  # 流程结束节点
    "END": "处理完成",  # 流程结束节点
    "node_item_name_confirm": "确认问题产品",  # 确认问题产品节点
    "node_answer_output": "生成答案",  # 生成答案节点
    "node_rerank": "重排序",  # 重排序节点
    "node_rrf": "倒排融合",  # 倒排融合节点
    "node_web_search_mcp": "网络搜索",  # 网络搜索节点
    "node_search_embedding": "切片搜索",  # 切片搜索节点
    "node_search_embedding_hyde": "切片搜索(假设性文档)",  # 假设性文档切片搜索节点
    "node_multi_search": "多路搜索",  # 多路搜索节点
    "node_query_kg": "查询知识图谱",  # 查询知识图谱节点
    "node_join": "多路搜索合并",  # 多路搜索合并节点
}


def _ensure_task(task_id: str) -> None:  # 定义任务数据结构初始化函数
    """确保指定 task_id 的内部数据结构已初始化。"""
    if task_id not in _tasks_running_list:  # 初始化正在运行列表
        _tasks_running_list[task_id] = []
    if task_id not in _tasks_done_list:  # 初始化已完成列表
        _tasks_done_list[task_id] = []
    if task_id not in _tasks_result:  # 初始化结果字典
        _tasks_result[task_id] = {}


def _to_cn(node_name: str) -> str:  # 定义节点名转中文函数
    """将节点名转换为中文展示名；无映射时返回原名。"""
    return _NODE_NAME_TO_CN.get(node_name, node_name)  # 查询映射表，不存在则返回原名


def add_running_task(task_id: str, node_name: str, is_stream: bool = False) -> None:  # 定义添加运行任务函数
    """添加正在运行的节点任务，可选推送进度更新。"""
    _ensure_task(task_id)  # 确保任务数据结构存在
    running = _tasks_running_list[task_id]  # 获取当前运行节点列表
    if node_name not in running:  # 避免重复追加
        running.append(node_name)  # 将节点加入运行列表

    if is_stream:  # 需要流式推送时更新队列
        task_push_queue(task_id)  # 推送任务进度


def add_done_task(task_id: str, node_name: str, is_stream: bool = False) -> None:  # 定义添加完成任务函数
    """添加已完成的节点任务，并从运行列表中移除同名节点。"""
    _ensure_task(task_id)  # 确保任务数据结构存在

    running = _tasks_running_list[task_id]  # 获取当前运行节点列表
    _tasks_running_list[task_id] = [n for n in running if n != node_name]  # 移除同名运行节点

    done = _tasks_done_list[task_id]  # 获取当前已完成节点列表
    if node_name not in done:  # 避免重复追加
        done.append(node_name)  # 将节点加入已完成列表

    if is_stream:  # 需要流式推送时更新队列
        task_push_queue(task_id)  # 推送任务进度


def set_task_result(task_id: str, key: str, value: str) -> None:  # 定义设置任务结果函数
    """设置任务结果字典中的指定字段。"""
    _ensure_task(task_id)  # 确保任务数据结构存在
    _tasks_result[task_id][key] = value  # 存储结果字段


def get_task_result(task_id: str, key: str, default: str = "") -> str:  # 定义获取任务结果函数
    """获取任务结果字典中的指定字段，不存在返回默认值。"""
    _ensure_task(task_id)  # 确保任务数据结构存在
    return _tasks_result.get(task_id, {}).get(key, default)  # 安全取值


def get_task_status(task_id: str) -> str:  # 定义获取任务状态函数
    """获取任务当前状态，未设置返回空字符串。"""
    return _tasks_status.get(task_id, "")  # 从状态字典取值


def get_done_task_list(task_id: str) -> List[str]:  # 定义获取已完成节点函数
    """获取已完成节点列表的中文展示名。"""
    _ensure_task(task_id)  # 确保任务数据结构存在
    done = _tasks_done_list.get(task_id, [])  # 获取已完成节点列表
    return [_to_cn(n) for n in done]  # 转换为中文展示名列表


def get_running_task_list(task_id: str) -> List[str]:  # 定义获取运行节点函数
    """获取正在运行节点列表的中文展示名。"""
    _ensure_task(task_id)  # 确保任务数据结构存在
    running = _tasks_running_list.get(task_id, [])  # 获取正在运行节点列表
    return [_to_cn(n) for n in running]  # 转换为中文展示名列表


def update_task_status(task_id: str, status_name: str, push_queue: bool = False) -> None:  # 定义更新任务状态函数
    """更新任务状态，可选推送进度更新。"""
    _tasks_status[task_id] = status_name  # 更新状态值
    if push_queue:  # 需要推送时更新队列
        task_push_queue(task_id)  # 推送任务进度


def task_push_queue(task_id: str):  # 定义任务进度推送函数
    """将任务进度推送到 SSE 会话队列。"""
    push_to_session(task_id, "progress", {  # 推送进度事件
        "status": get_task_status(task_id),  # 包含当前状态
        "done_list": get_done_task_list(task_id),  # 包含已完成节点
        "running_list": get_running_task_list(task_id),  # 包含正在运行节点
    })


def clear_task(task_id: str):  # 定义清空任务数据函数
    """清空指定任务的所有内存数据。"""
    _tasks_running_list.pop(task_id, None)  # 移除运行节点数据
    _tasks_done_list.pop(task_id, None)  # 移除已完成节点数据
    _tasks_status.pop(task_id, None)  # 移除状态数据
    _tasks_result.pop(task_id, None)  # 移除结果数据
