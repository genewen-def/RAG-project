import json
import queue
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator
from fastapi import Request


class SSEEvent:  # 定义 SSE 事件类型常量类
    READY = "ready"  # 连接建立事件
    PROGRESS = "progress"  # 任务节点进度事件
    DELTA = "delta"  # LLM 流式输出增量事件
    FINAL = "final"  # 最终完整答案事件
    ERROR = "error"  # 错误信息事件
    CLOSE = "__close__"  # 关闭连接信号


_session_stream: Dict[str, queue.Queue] = {}  # 全局 SSE 会话队列存储


def get_sse_queue(session_id: str) -> Optional["queue.Queue"]:  # 定义获取 SSE 队列函数
    """获取指定 session 的 SSE 队列。"""
    return _session_stream.get(session_id)  # 从字典中查找队列


def create_sse_queue(session_id: str) -> "queue.Queue":  # 定义创建 SSE 队列函数
    """创建并注册一个新的 SSE 队列。"""
    print(f"[SSE] Creating queue for session: {session_id}")  # 打印创建日志
    q = queue.Queue()  # 实例化队列
    _session_stream[session_id] = q  # 注册到全局字典
    return q  # 返回新队列


def remove_sse_queue(session_id: str):  # 定义移除 SSE 队列函数
    """移除指定 session 的 SSE 队列。"""
    print(f"[SSE] Removing queue for session: {session_id}")  # 打印移除日志
    _session_stream.pop(session_id, None)  # 从字典中移除队列


def _sse_pack(event: str, data: Dict[str, Any]) -> str:  # 定义 SSE 消息打包函数
    """将事件和数据打包为 SSE 消息格式。"""
    payload = json.dumps(data, ensure_ascii=False)  # 序列化数据为 JSON 字符串
    return f"event: {event}\ndata: {payload}\n\n"  # 拼接 SSE 格式字符串


def push_to_session(session_id: str, event: str, data: Dict[str, Any]):  # 定义推送事件函数
    """向指定 session 推送 SSE 事件。"""
    stream_queue = get_sse_queue(session_id)  # 获取目标队列
    if stream_queue:  # 队列存在则入队
        stream_queue.put({"event": event, "data": data})  # 将事件放入队列
    else:  # 队列不存在则打印警告
        print(f"[SSE] Warning: No queue found for session {session_id} when pushing {event}")  # 打印警告日志


async def sse_generator(session_id: str, request: Request):  # 定义 SSE 异步生成器函数
    """SSE 异步生成器，用于 FastAPI 的 StreamingResponse。"""
    print(f"[SSE] Generator started for session: {session_id}")  # 打印生成器启动日志
    stream_queue = get_sse_queue(session_id)  # 获取目标队列
    if stream_queue is None:  # 队列不存在时结束生成器
        print(f"[SSE] Error: Queue not found for session {session_id}. Available sessions: {list(_session_stream.keys())}")  # 打印错误日志
        return  # 结束生成器

    loop = asyncio.get_running_loop()  # 获取当前事件循环
    try:  # 捕获生成器异常
        print(f"[SSE] Sending ready signal for {session_id}")  # 打印准备发送就绪信号日志
        yield _sse_pack("ready", {})  # 发送连接建立信号

        while True:  # 持续监听队列
            if await request.is_disconnected():  # 客户端断开则退出
                print(f"[SSE] Client disconnected: {session_id}")  # 打印断开日志
                print("-----------------------断开连接--------------------")  # 打印分隔线
                break  # 退出循环

            try:  # 非阻塞获取消息
                msg = await loop.run_in_executor(None, stream_queue.get, True, 1.0)  # 非阻塞取消息
            except queue.Empty:  # 队列空则继续轮询
                continue  # 进入下一次循环

            event = msg.get("event")  # 提取事件类型
            data = msg.get("data")  # 提取事件数据

            if event == "__close__":  # 收到关闭信号则退出
                print(f"[SSE] Closing signal received for {session_id}")  # 打印关闭信号日志
                break  # 退出循环

            yield _sse_pack(event, data)  # 推送事件到客户端
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):  # 客户端取消或连接断开
        print(f"[SSE] Client disconnected (Cancelled/Reset/Pipe): {session_id}")  # 打印断开日志
        return  # 静默退出
    except Exception as e:  # 捕获其他异常
        print(f"[SSE] Exception in generator for {session_id}: {e}")  # 打印异常日志
    finally:  # 生成器结束时清理
        print(f"[SSE] Generator finished for {session_id}")  # 打印生成器结束日志
        remove_sse_queue(session_id)  # 清理会话队列
