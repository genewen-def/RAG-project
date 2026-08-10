import os
import sys
import time
import requests
import zipfile
import shutil
from pathlib import Path
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.format_utils import format_state
from app.utils.task_utils import add_running_task, add_done_task
from app.conf.mineru_config import mineru_config
from app.core.logger import logger

MINERU_BASE_URL = mineru_config.base_url
MINERU_API_TOKEN = mineru_config.api_key


def step_1_validate_paths(state):
    """校验PDF文件路径和输出目录的有效性。"""
    log_prefix = "[step_1_validate_paths] "
    pdf_path = state.get("pdf_path", "").strip()  # 获取PDF路径
    local_dir = state.get("local_dir", "").strip()  # 获取输出目录

    if not pdf_path:  # 判断PDF路径是否为空
        raise ValueError(f"{log_prefix}工作流状态缺失有效参数：pdf_path，当前值：{repr(pdf_path)}")  # 抛出参数缺失异常
    if not local_dir:  # 判断输出目录是否为空
        raise ValueError(f"{log_prefix}工作流状态缺失有效参数：local_dir，当前值：{repr(local_dir)}")  # 抛出参数缺失异常

    pdf_path_obj = Path(pdf_path)  # 转换为Path对象
    output_dir_obj = Path(local_dir)  # 转换为Path对象

    if not pdf_path_obj.exists():  # 判断PDF文件是否存在
        raise FileNotFoundError(f"{log_prefix}PDF文件不存在，绝对路径：{pdf_path_obj.absolute()}")  # 抛出文件不存在异常
    if not pdf_path_obj.is_file():  # 判断路径是否为文件
        raise FileNotFoundError(f"{log_prefix}指定路径非文件（是目录），绝对路径：{pdf_path_obj.absolute()}")  # 抛出类型错误异常

    if not output_dir_obj.exists():  # 判断输出目录是否存在
        logger.info(f"{log_prefix}输出目录不存在，自动创建：{output_dir_obj.absolute()}")  # 记录自动创建日志
        output_dir_obj.mkdir(parents=True, exist_ok=True)  # 递归创建输出目录

    return pdf_path_obj, output_dir_obj  # 返回Path对象


def step_2_upload_and_poll(pdf_path_obj: Path, output_dir_obj: Path):
    """上传PDF到MinerU并轮询解析任务状态，返回ZIP下载链接。"""
    if not MINERU_BASE_URL or not MINERU_API_TOKEN:  # 判断MinerU配置是否缺失
        raise ValueError("MinerU配置缺失：请在.env中正确配置MINERU_BASE_URL和MINERU_API_TOKEN")  # 抛出配置缺失异常
    logger.info(f"[配置校验] MinerU基础配置加载成功，开始处理文件：{pdf_path_obj.name}")  # 记录配置校验日志

    request_headers = {
        "Content-Type": "application/json",  # 设置请求内容类型
        "Authorization": f"Bearer {MINERU_API_TOKEN}"  # 设置鉴权头
    }

    url_get_upload = f"{MINERU_BASE_URL}/file-urls/batch"  # 构造获取上传链接URL
    req_data = {
        "files": [{"name": pdf_path_obj.name}],  # 构造文件信息
        "model_version": "vlm"  # 设置解析模型版本
    }
    logger.debug(f"[获取上传链接] 调用接口：{url_get_upload}，请求参数：{req_data}")  # 记录请求日志
    resp = requests.post(url=url_get_upload, headers=request_headers, json=req_data, timeout=30)  # 请求上传链接

    if resp.status_code != 200:  # 判断HTTP状态码
        raise RuntimeError(f"[获取上传链接] 网络请求失败，状态码：{resp.status_code}，响应内容：{resp.text}")  # 抛出网络异常

    resp_data = resp.json()  # 解析响应JSON
    if resp_data["code"] != 0:  # 判断业务状态码
        raise RuntimeError(f"[获取上传链接] API业务错误，返回数据：{resp_data}")  # 抛出业务异常

    signed_url = resp_data["data"]["file_urls"][0]  # 提取上传链接
    batch_id = resp_data["data"]["batch_id"]  # 提取任务ID
    logger.info(f"[获取上传链接] 成功，batch_id：{batch_id}，上传链接已生成")  # 记录获取成功日志

    file_size = pdf_path_obj.stat().st_size  # 获取文件大小
    file_size_mb = file_size / 1024 / 1024  # 转换为MB
    logger.info(f"[文件上传] 开始读取PDF文件：{pdf_path_obj.name}，大小：{file_size_mb:.2f} MB")  # 记录文件信息
    with open(pdf_path_obj, "rb") as f:
        file_data = f.read()  # 读取PDF二进制数据

    upload_timeout = min(max(int(file_size_mb * 60) + 120, 120), 900)  # 计算动态上传超时
    logger.info(f"[文件上传] 本次上传超时时间设置为：{upload_timeout}s")  # 记录超时时间

    upload_session = requests.Session()  # 创建上传Session
    upload_session.trust_env = False  # 禁用环境代理

    max_upload_attempts = 3  # 设置最大重试次数
    pdf_headers = {}  # 初始化上传请求头
    last_error = None  # 初始化最后错误信息

    try:
        for attempt in range(1, max_upload_attempts + 1):  # 循环重试上传
            try:
                logger.info(f"[文件上传] 第 {attempt}/{max_upload_attempts} 次尝试上传...")  # 记录重试日志
                put_resp = upload_session.put(
                    url=signed_url,  # 上传URL
                    data=file_data,  # 上传数据
                    headers=pdf_headers,  # 请求头
                    timeout=(30, upload_timeout)  # 设置连接和读写超时
                )

                if put_resp.status_code == 200:  # 判断上传是否成功
                    logger.info(f"[文件上传] 成功，文件{pdf_path_obj.name}已存入云存储")  # 记录上传成功日志
                    break  # 跳出重试循环

                last_error = f"状态码：{put_resp.status_code}，响应内容：{put_resp.text[:500]}"  # 记录错误信息
                logger.warning(f"[文件上传] 第 {attempt} 次上传返回异常状态，{last_error}")  # 记录异常日志

            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)}"  # 记录异常类型和信息
                logger.warning(f"[文件上传] 第 {attempt} 次上传发生网络异常：{last_error}")  # 记录网络异常

            if attempt < max_upload_attempts:  # 判断是否还有重试次数
                backoff_seconds = 3 * attempt  # 计算退避时间
                logger.info(f"[文件上传] {backoff_seconds}秒后进行重试...")  # 记录重试等待日志
                time.sleep(backoff_seconds)  # 等待后重试
        else:
            raise RuntimeError(
                f"[文件上传] 连续{max_upload_attempts}次上传均失败，文件大小：{file_size_mb:.2f}MB，"
                f"最后一次错误：{last_error}。建议检查网络上行带宽或改用体积更小的文件。"
            )  # 所有重试失败抛出异常
    finally:
        upload_session.close()  # 关闭Session释放连接

    poll_url = f"{MINERU_BASE_URL}/extract-results/batch/{batch_id}"  # 构造轮询URL
    start_time = time.time()  # 记录轮询开始时间
    timeout_seconds = 600  # 设置最大轮询超时
    poll_interval = 3  # 设置轮询间隔
    logger.info(f"[任务轮询] 开始监控任务状态，batch_id：{batch_id}，最大超时：{timeout_seconds}s")  # 记录轮询开始日志

    while True:  # 循环轮询
        elapsed_time = time.time() - start_time  # 计算已耗时
        if elapsed_time > timeout_seconds:  # 判断是否超时
            raise TimeoutError(f"[任务轮询] 超时！任务处理超{int(timeout_seconds)}秒，batch_id：{batch_id}")  # 抛出超时异常

        try:
            poll_resp = requests.get(url=poll_url, headers=request_headers, timeout=10)  # 发起轮询请求
        except Exception as e:
            logger.warning(f"[任务轮询] 网络请求异常，{poll_interval}秒后重试：{str(e)}")  # 记录轮询异常
            time.sleep(poll_interval)  # 等待后重试
            continue

        if poll_resp.status_code != 200:  # 判断HTTP状态码
            if 500 <= poll_resp.status_code < 600:  # 判断是否为服务端错误
                logger.warning(f"[任务轮询] 服务端繁忙（状态码：{poll_resp.status_code}），{poll_interval}秒后重试")  # 记录服务端繁忙
                time.sleep(poll_interval)  # 等待后重试
                continue
            else:
                raise RuntimeError(f"[任务轮询] HTTP请求失败，状态码：{poll_resp.status_code}，响应内容：{poll_resp.text}")  # 抛出HTTP异常

        poll_data = poll_resp.json()  # 解析轮询响应
        if poll_data["code"] != 0:  # 判断业务状态码
            raise RuntimeError(f"[任务轮询] API业务错误，返回数据：{poll_data}")  # 抛出业务异常

        extract_results = poll_data["data"]["extract_result"]  # 提取解析结果
        if not extract_results:  # 判断结果是否为空
            logger.debug(f"[任务轮询] 结果暂为空，已耗时{int(elapsed_time)}s，继续等待")  # 记录等待日志
            time.sleep(poll_interval)  # 等待后继续
            continue

        result_item = extract_results[0]  # 获取首个结果项
        state_status = result_item["state"]  # 获取任务状态
        if state_status == "done":  # 判断任务是否完成
            logger.info(f"[任务轮询] 解析任务完成！总耗时：{int(elapsed_time)}s，batch_id：{batch_id}")  # 记录完成日志
            full_zip_url = result_item.get("full_zip_url")  # 获取ZIP下载链接
            if not full_zip_url:  # 判断链接是否为空
                raise RuntimeError("[任务轮询] 任务完成但未返回ZIP包下载链接，batch_id：{batch_id}")  # 抛出缺失链接异常
            logger.info(f"[任务轮询] 结果ZIP包下载链接：{full_zip_url}...")  # 记录下载链接
            return full_zip_url  # 返回ZIP下载链接
        elif state_status == "failed":  # 判断任务是否失败
            err_msg = result_item.get("err_msg", "未知错误，无具体信息")  # 获取错误信息
            raise RuntimeError(f"[任务轮询] 解析任务失败，batch_id：{batch_id}，错误信息：{err_msg}")  # 抛出任务失败异常
        else:
            logger.debug(
                f"[任务轮询] 处理中（已耗时{int(elapsed_time)}s），状态：{state_status} | 刷新间隔{poll_interval}s",
                end="\r"
            )  # 记录处理中状态
            time.sleep(poll_interval)  # 等待后继续轮询


def step_3_download_and_extract(zip_url: str, output_dir_obj: Path, pdf_stem: str) -> str:
    """下载ZIP包并解压，按优先级查找并重命名目标MD文件。"""
    logger.info(f"===== 开始处理[{pdf_stem}]的MinerU解析结果 =====")  # 记录开始处理日志

    logger.info(f"[步骤1/4] 开始下载ZIP包，链接：{zip_url}...")  # 记录下载开始日志
    resp = requests.get(zip_url, timeout=120)  # 下载ZIP包
    if resp.status_code != 200:  # 判断下载状态码
        raise RuntimeError(f"[步骤1/4] ZIP包下载失败，HTTP状态码：{resp.status_code}")  # 抛出下载异常

    zip_save_path = output_dir_obj / f"{pdf_stem}_result.zip"  # 构造ZIP保存路径
    with open(zip_save_path, "wb") as f:
        f.write(resp.content)  # 写入ZIP文件
    logger.info(f"[步骤1/4] ZIP包下载成功，保存路径：{zip_save_path}")  # 记录下载成功日志

    logger.info(f"[步骤2/4] 开始解压ZIP包...")  # 记录解压开始日志
    extract_target_dir = output_dir_obj / pdf_stem  # 构造解压目录

    if extract_target_dir.exists():  # 判断旧解压目录是否存在
        try:
            shutil.rmtree(extract_target_dir)  # 递归删除旧目录
            logger.info(f"[步骤2/4] 已清理旧的解压目录：{extract_target_dir}")  # 记录清理日志
        except Exception as e:
            logger.warning(f"[步骤2/4] 清理旧目录失败，可能不影响新文件解压：{str(e)}")  # 记录清理警告

    extract_target_dir.mkdir(parents=True, exist_ok=True)  # 创建解压目录

    with zipfile.ZipFile(zip_save_path, 'r') as zip_file_obj:
        zip_file_obj.extractall(extract_target_dir)  # 解压ZIP包
    logger.info(f"[步骤2/4] ZIP包解压完成，解压目录：{extract_target_dir}")  # 记录解压完成日志

    logger.info(f"[步骤3/4] 开始查找解压目录中的MD文件...")  # 记录查找开始日志
    md_file_list = list(extract_target_dir.rglob("*.md"))  # 递归查找所有MD文件
    if not md_file_list:  # 判断是否找到MD文件
        raise FileNotFoundError(f"[步骤3/4] 解压目录中未找到任何.md格式文件：{extract_target_dir}")  # 抛出未找到异常
    logger.info(f"[步骤3/4] 共找到{len(md_file_list)}个MD文件，按优先级匹配目标文件")  # 记录查找结果

    target_md_file = None  # 初始化目标MD文件
    for md_file in md_file_list:  # 遍历MD文件列表
        if md_file.stem == pdf_stem:  # 判断是否与PDF同名
            target_md_file = md_file  # 设置为优先级1目标
            logger.info(f"[步骤4/4] 匹配到优先级1目标：与PDF同名的MD文件 {target_md_file.name}")  # 记录匹配日志
            break

    if not target_md_file:  # 判断是否未匹配到同名文件
        for md_file in md_file_list:  # 遍历MD文件列表
            if md_file.name.lower() == "full.md":  # 判断是否为full.md
                target_md_file = md_file  # 设置为优先级2目标
                logger.info(f"[步骤4/4] 匹配到优先级2目标：MinerU默认文件 {target_md_file.name}")  # 记录匹配日志
                break

    if not target_md_file:  # 判断是否仍未匹配
        target_md_file = md_file_list[0]  # 兜底取第一个MD文件
        logger.info(f"[步骤4/4] 未匹配到前两级目标，兜底取第一个MD文件 {target_md_file.name}")  # 记录兜底日志

    if target_md_file.stem != pdf_stem:  # 判断是否需要重命名
        logger.info(f"[步骤4/4] 开始重命名MD文件，统一为PDF同名：{pdf_stem}.md")  # 记录重命名日志
        new_md_path = target_md_file.with_name(f"{pdf_stem}.md")  # 构造新路径
        try:
            target_md_file.rename(new_md_path)  # 重命名文件
            target_md_file = new_md_path  # 更新目标变量
            logger.info(f"[步骤4/4] MD文件重命名成功：{pdf_stem}.md")  # 记录重命名成功日志
        except OSError as e:
            logger.warning(f"[步骤4/4] MD文件重命名失败，将使用原文件名继续流程：{str(e)}")  # 记录重命名失败警告

    final_md_path = str(target_md_file.absolute())  # 转换为绝对路径字符串
    logger.info(f"===== [{pdf_stem}]解析结果处理完成，最终MD文件路径：{final_md_path} =====")  # 记录处理完成日志
    return final_md_path  # 返回最终MD路径


def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """PDF转MD核心节点：校验路径、上传解析、下载解压并更新状态。"""
    func_name = sys._getframe().f_code.co_name  # 获取当前函数名

    logger.debug(f"【{func_name}】节点启动，\n当前工作流状态：{format_state(state)}")  # 记录节点启动日志

    add_running_task(state["task_id"], func_name)  # 标记当前节点为运行中

    try:
        pdf_path_obj, output_dir_obj = step_1_validate_paths(state)  # 校验路径
        zip_url = step_2_upload_and_poll(pdf_path_obj, output_dir_obj)  # 上传并轮询
        md_path = step_3_download_and_extract(zip_url, output_dir_obj, pdf_path_obj.stem)  # 下载解压

        state["md_path"] = md_path  # 更新状态中的MD路径
        logger.info(f"【{func_name}】MD文件生成成功，路径：{md_path}")  # 记录生成成功日志

        try:
            with open(md_path, "r", encoding="utf-8") as f:
                state["md_content"] = f.read()  # 读取MD文件内容
            logger.debug(f"【{func_name}】MD文件内容读取成功，内容长度：{len(state['md_content'])}字符")  # 记录读取成功日志
        except Exception as e:
            logger.error(f"【{func_name}】读取MD文件内容失败：{str(e)}")  # 记录读取失败日志

        logger.info(f"【{func_name}】节点执行完成，更新后工作流状态键：{list(state.keys())}")  # 记录节点完成日志

    except Exception as e:
        logger.error(f"【{func_name}】PDF转MD流程执行失败：{str(e)}", exc_info=True)  # 记录流程失败日志
        raise  # 向上抛出异常
    finally:
        add_done_task(state["task_id"], func_name)  # 标记当前节点为已完成
        logger.debug(f"【{func_name}】节点执行完成，\n更新后工作流状态：{format_state(state)}")  # 记录最终状态日志

    return state  # 返回更新后的状态


if __name__ == "__main__":
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")  # 记录测试开始日志

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")  # 打印项目根目录

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")  # 构造测试PDF相对路径
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)  # 拼接测试PDF绝对路径

    test_state = create_default_state(  # 构造测试状态
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)  # 执行PDF转MD节点

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")  # 记录测试结束日志
