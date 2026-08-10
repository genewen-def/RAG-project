import os
import shutil
import uuid
from typing import List, Dict, Any
from datetime import datetime
import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.clients.minio_utils import get_minio_client
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import (
    add_running_task,
    add_done_task,
    get_done_task_list,
    get_running_task_list,
    update_task_status,
    get_task_status,
)
from app.import_process.agent.state import get_default_state
from app.import_process.agent.main_graph import kb_import_app
from app.core.logger import logger


app = FastAPI(  # 初始化 FastAPI 应用实例
    title="File Import Service",  # 设置 API 文档标题
    description="Web service for uploading files to Knowledge Base (PDF/MD → 解析 → 切分 → 向量化 → Milvus/KG入库)"  # 设置 API 文档描述
)

app.add_middleware(  # 添加跨域中间件
    CORSMiddleware,  # 指定中间件类
    allow_origins=["*"],  # 允许所有来源访问
    allow_credentials=True,  # 允许携带凭证
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"]  # 允许所有请求头
)


@app.get("/import.html", response_class=FileResponse)  # 注册获取导入页面接口
async def get_import_page():  # 返回文件导入前端页面
    """返回文件导入前端页面 import.html。"""
    html_abs_path = PROJECT_ROOT / "app/import_process/page/import.html"  # 拼接前端页面绝对路径
    logger.info(f"前端页面访问，文件绝对路径：{html_abs_path}")  # 记录页面访问日志

    if not os.path.exists(html_abs_path):  # 判断页面文件是否存在
        logger.error(f"前端页面文件不存在，路径：{html_abs_path}")  # 记录文件不存在错误日志
        raise HTTPException(status_code=404, detail="import.html page not found")  # 抛出 404 异常

    return FileResponse(  # 返回 HTML 文件响应
        path=html_abs_path,  # 指定文件绝对路径
        media_type="text/html"  # 指定媒体类型为 HTML
    )


def run_graph_task(task_id: str, local_dir: str, local_file_path: str):  # 执行 LangGraph 后台任务
    """在后台执行 LangGraph 导入全流程并更新任务状态。"""
    try:  # 开始异常捕获
        update_task_status(task_id, "processing")  # 更新任务状态为处理中
        logger.info(f"[{task_id}] 开始执行LangGraph全流程，本地文件路径：{local_file_path}")  # 记录流程启动日志

        init_state = get_default_state()  # 获取默认初始状态
        init_state["task_id"] = task_id  # 注入任务 ID
        init_state["local_dir"] = local_dir  # 注入本地目录
        init_state["local_file_path"] = local_file_path  # 注入本地文件路径

        for event in kb_import_app.stream(init_state):  # 流式执行 LangGraph 工作流
            for node_name, node_result in event.items():  # 遍历事件中的节点结果
                logger.info(f"[{task_id}] LangGraph节点执行完成：{node_name}")  # 记录节点完成日志
                add_done_task(task_id, node_name)  # 将节点加入已完成列表

        update_task_status(task_id, "completed")  # 更新任务状态为已完成
        logger.info(f"[{task_id}] LangGraph全流程执行完毕，任务完成")  # 记录流程完成日志

    except Exception as e:  # 捕获流程执行异常
        update_task_status(task_id, "failed")  # 更新任务状态为失败
        logger.error(f"[{task_id}] LangGraph全流程执行失败，异常信息：{str(e)}", exc_info=True)  # 记录错误日志


@app.post("/upload", summary="文件上传接口", description="支持多文件批量上传，自动触发知识库导入全流程")  # 注册文件上传接口
async def upload_files(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):  # 处理文件上传
    """接收多文件上传，保存本地并启动 LangGraph 后台导入任务。"""
    date_based_root_dir = os.path.join(PROJECT_ROOT / "output", datetime.now().strftime("%Y%m%d"))  # 构建按日期分层的输出根目录
    task_ids = []  # 初始化任务 ID 列表

    for file in files:  # 遍历处理每个上传文件
        task_id = str(uuid.uuid4())  # 生成唯一任务 ID
        task_ids.append(task_id)  # 收集任务 ID
        logger.info(f"[{task_id}] 开始处理上传文件，文件名：{file.filename}，文件类型：{file.content_type}")  # 记录文件处理日志

        add_running_task(task_id, "upload_file")  # 标记上传阶段运行中

        task_local_dir = os.path.join(date_based_root_dir, task_id)  # 构建任务本地目录
        os.makedirs(task_local_dir, exist_ok=True)  # 创建任务本地目录
        local_file_abs_path = os.path.join(task_local_dir, file.filename)  # 构建本地文件绝对路径

        with open(local_file_abs_path, "wb") as file_buffer:  # 以二进制写模式打开本地文件
            shutil.copyfileobj(file.file, file_buffer)  # 将上传文件内容写入本地
        logger.info(f"[{task_id}] 文件已保存至本地，路径：{local_file_abs_path}")  # 记录文件保存日志

        minio_pdf_base_dir = os.getenv("MINIO_PDF_DIR", "pdf_files")  # 获取 MinIO PDF 目录配置
        minio_object_name = f"{minio_pdf_base_dir}/{datetime.now().strftime('%Y%m%d')}/{file.filename}"  # 构建 MinIO 对象名
        try:  # 开始 MinIO 上传异常捕获
            minio_client = get_minio_client()  # 获取 MinIO 客户端
            if minio_client is None:  # 判断客户端是否获取失败
                raise HTTPException(status_code=500, detail="MinIO service connection failed, please check MinIO config")  # 抛出服务异常
            minio_bucket_name = os.getenv("MINIO_BUCKET_NAME", "kb-import-bucket")  # 获取 MinIO 桶名配置

            minio_client.fput_object(  # 上传文件到 MinIO
                bucket_name=minio_bucket_name,  # 指定目标桶名
                object_name=minio_object_name,  # 指定对象名称
                file_path=local_file_abs_path,  # 指定本地文件路径
                content_type=file.content_type  # 指定文件内容类型
            )
            logger.info(f"[{task_id}] 文件已成功上传至MinIO，桶名：{minio_bucket_name}，对象名：{minio_object_name}")  # 记录上传成功日志
        except Exception as e:  # 捕获上传异常
            logger.warning(f"[{task_id}] 文件上传MinIO失败，将继续执行本地处理流程，异常信息：{str(e)}", exc_info=True)  # 记录上传警告日志

        add_done_task(task_id, "upload_file")  # 标记上传阶段已完成

        background_tasks.add_task(run_graph_task, task_id, task_local_dir, local_file_abs_path)  # 添加 LangGraph 后台任务
        logger.info(f"[{task_id}] 已将LangGraph全流程加入后台任务，任务已启动")  # 记录后台任务启动日志

    logger.info(f"多文件上传处理完毕，共处理{len(files)}个文件，生成TaskID列表：{task_ids}")  # 记录批量处理完成日志
    return {  # 返回上传结果
        "code": 200,  # 设置响应状态码
        "message": f"Files uploaded successfully, total: {len(files)}",  # 设置响应消息
        "task_ids": task_ids  # 返回任务 ID 列表
    }


@app.get("/status/{task_id}", summary="任务状态查询", description="根据TaskID查询单个文件的处理进度和全局状态")  # 注册任务状态查询接口
async def get_task_progress(task_id: str):  # 查询任务处理进度
    """根据任务 ID 返回当前状态、已完成节点和运行中节点。"""
    task_status_info: Dict[str, Any] = {  # 构建任务状态返回字典
        "code": 200,  # 设置响应状态码
        "task_id": task_id,  # 设置任务 ID
        "status": get_task_status(task_id),  # 获取任务全局状态
        "done_list": get_done_task_list(task_id),  # 获取已完成节点列表
        "running_list": get_running_task_list(task_id)  # 获取运行中节点列表
    }
    logger.info(f"[{task_id}] 任务状态查询，当前状态：{task_status_info['status']}，已完成节点：{task_status_info['done_list']}")  # 记录状态查询日志
    return task_status_info  # 返回任务状态信息


if __name__ == "__main__":  # 脚本直接执行入口
    logger.info("File Import Service 服务启动中...")  # 记录服务启动日志
    uvicorn.run(  # 启动 uvicorn 服务
        "file_import_service:app",  # 指定应用入口模块
        host="127.0.0.1",  # 绑定本地地址
        port=8000,  # 绑定服务端口
        reload=True  # 启用开发环境自动重载
    )
