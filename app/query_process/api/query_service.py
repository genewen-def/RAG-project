from pathlib import Path
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_history_utils import *
from app.query_process.agent.main_graph import query_app


app = FastAPI(title="query service", description="掌柜智库查询服务！")  # 创建FastAPI应用实例
app.add_middleware(  # 添加跨域中间件
    CORSMiddleware,  # CORS中间件类
    allow_origins=["*"],  # 允许所有来源
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)


@app.get("/chat.html")  # 注册chat.html页面路由
async def chat():  # 定义页面返回处理函数
    """返回chat.html聊天页面。"""
    current_dir_parent_path = Path(__file__).absolute().parent.parent  # 计算query_process目录路径
    chat_html_path = current_dir_parent_path / "page" / "chat.html"  # 拼接chat.html文件路径
    if not chat_html_path.exists():  # 判断页面文件是否存在
        raise HTTPException(status_code=404, detail=f"没有查询到页面，地址为：{chat_html_path}！")  # 不存在则抛出404异常
    return FileResponse(chat_html_path)  # 返回页面文件响应


class QueryRequest(BaseModel):  # 定义查询请求数据模型
    """查询请求数据结构。"""
    query: str = Field(..., description="查询内容")  # 查询内容字段
    session_id: str = Field(None, description="会话ID")  # 会话ID字段
    is_stream: bool = Field(False, description="是否流式返回")  # 是否流式返回字段


@app.get("/health")  # 注册健康检查路由
async def health():  # 定义健康检查处理函数
    """检查服务健康状态。"""
    return {"ok": True}  # 返回服务正常标志


def run_query_graph(session_id: str, user_query: str, is_stream: bool = True):  # 定义执行查询图的函数
    """在后台运行查询流程图并更新任务状态。"""
    print(f"开始流程图处理...{session_id} {user_query} {is_stream}")  # 打印处理开始日志
    default_state = {"original_query": user_query, "session_id": session_id, "is_stream": is_stream}  # 构建默认状态
    try:  # 开始异常捕获
        query_app.invoke(default_state)  # 调用查询图执行流程
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)  # 更新任务为完成状态
    except Exception as e:  # 捕获执行异常
        print(f"流程执行异常: {e}")  # 打印异常日志
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)  # 更新任务为失败状态
        if is_stream:  # 判断是否为流式任务
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})  # 向会话推送错误事件


@app.post("/query")  # 注册查询接口路由
async def query(background_tasks: BackgroundTasks, request: QueryRequest):  # 定义查询接口处理函数
    """解析请求、启动查询流程并返回响应。"""
    user_query = request.query  # 获取用户查询内容
    session_id = request.session_id if request.session_id else str(uuid.uuid4())  # 生成或复用会话ID
    is_stream = request.is_stream  # 获取是否流式返回标志
    if is_stream:  # 判断是否流式返回
        create_sse_queue(session_id)  # 创建SSE结果队列
    update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)  # 更新任务为处理中状态
    print("开始处理流程... 是否流式:", is_stream, f"其他参数:{user_query}, session_id:{session_id}")  # 打印处理参数日志
    if is_stream:  # 流式分支
        background_tasks.add_task(run_query_graph, session_id, user_query, is_stream)  # 后台启动查询图任务
        print("开始处理结果....")  # 打印开始处理结果日志
        return {  # 返回流式处理中响应
            "message": "结果正在处理中...",  # 提示信息
            "session_id": session_id  # 返回会话ID
        }
    else:  # 非流式分支
        run_query_graph(session_id, user_query, is_stream)  # 同步运行查询图
        answer = get_task_result(session_id, "answer", "")  # 获取任务结果中的答案
        return {  # 返回同步处理完成响应
            "message": "处理完成！",  # 提示信息
            "session_id": session_id,  # 返回会话ID
            "answer": answer,  # 返回答案内容
            "done_list": []  # 返回空完成列表
        }


@app.get("/stream/{session_id}")  # 注册流式结果推送路由
async def stream(session_id: str, request: Request):  # 定义流式结果处理函数
    """通过SSE实时返回流式结果。"""
    print("调用流式/stream...")  # 打印流式接口调用日志
    return StreamingResponse(  # 返回SSE流式响应
        sse_generator(session_id, request),  # 传入SSE生成器
        media_type="text/event-stream",  # 设置媒体类型
        headers={  # 设置响应头
            "Cache-Control": "no-cache",  # 禁用缓存
            "Connection": "keep-alive",  # 保持长连接
            "X-Accel-Buffering": "no"  # 禁用代理缓冲
        }
    )


@app.get("/history/{session_id}")  # 注册历史记录查询路由
async def history(session_id: str, limit: int = 50):  # 定义历史记录处理函数
    """查询指定会话的历史消息记录。"""
    try:  # 开始异常捕获
        records = get_recent_messages(session_id, limit=limit)  # 获取近期消息记录
        items = []  # 初始化返回条目列表
        for r in records:  # 遍历每条记录
            items.append({  # 构建单条记录结构
                "_id": str(r.get("_id")) if r.get("_id") is not None else "",  # 格式化记录ID
                "session_id": r.get("session_id", ""),  # 获取会话ID
                "role": r.get("role", ""),  # 获取角色
                "text": r.get("text", ""),  # 获取消息文本
                "rewritten_query": r.get("rewritten_query", ""),  # 获取改写后的问题
                "item_names": r.get("item_names", []),  # 获取商品名称列表
                "ts": r.get("ts")  # 获取时间戳
            })
        return {"session_id": session_id, "items": items}  # 返回会话历史记录
    except Exception as e:  # 捕获查询异常
        raise HTTPException(status_code=500, detail=f"history error: {e}")  # 抛出500异常


@app.delete("/history/{session_id}")  # 注册清空历史记录路由
async def clear_chat_history(session_id: str):  # 定义清空历史记录函数
    """清空指定会话的聊天历史。"""
    count = clear_history(session_id)  # 调用清空历史方法并获取删除数量
    return {"message": "History cleared", "deleted_count": count}  # 返回删除结果


if __name__ == "__main__":  # 判断为主程序入口
    uvicorn.run("query_service:app", host="127.0.0.1", port=8001, reload=True)  # 启动Uvicorn服务
