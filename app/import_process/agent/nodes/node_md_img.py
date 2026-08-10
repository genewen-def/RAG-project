import os
import re
import sys
import base64
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque
from minio import Minio
from minio.deleteobjects import DeleteObject
from app.clients.minio_utils import get_minio_client
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task
from app.lm.lm_utils import get_llm_client
from langchain.messages import HumanMessage
from langchain_core.exceptions import LangChainException
from app.conf.minio_config import minio_config
from app.conf.lm_config import lm_config
from app.core.logger import logger
from app.utils.rate_limit_utils import apply_api_rate_limit
from app.core.load_prompt import load_prompt

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def step_1_get_content(state: ImportGraphState) -> Tuple[str, Path, Path]:
    """从状态中提取MD内容、文件路径和图片文件夹路径。"""
    md_file_path = state["md_path"]  # 获取MD文件路径
    if not md_file_path:  # 判断路径是否为空
        raise FileNotFoundError(f"全局状态中无有效MD文件路径：{state['md_path']}")  # 抛出文件不存在异常

    path_obj = Path(md_file_path)  # 转换为Path对象
    if not state["md_content"]:  # 判断状态中是否已有MD内容
        with open(path_obj, "r", encoding="utf-8") as f:
            md_content = f.read()  # 从文件读取MD内容
        logger.debug(f"从文件读取MD内容完成，文件大小：{len(md_content)} 字符")  # 记录读取日志
    else:
        md_content = state["md_content"]  # 使用状态中的MD内容
        logger.debug(f"从全局状态获取MD内容完成，内容大小：{len(md_content)} 字符")  # 记录获取日志

    images_dir = path_obj.parent / "images"  # 构造图片文件夹路径
    return md_content, path_obj, images_dir  # 返回MD内容和路径对象


def is_supported_image(filename: str) -> bool:
    """判断文件后缀是否在支持的图片格式集合中。"""
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS  # 比较低写后缀是否在支持集合中


def find_image_in_md(md_content: str, image_filename: str, context_len: int = 100) -> List[Tuple[str, str]]:
    """查找MD中指定图片的所有引用位置，并返回每个位置的上下文。"""
    pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_filename) + r".*?\)")  # 编译图片引用匹配正则
    results = []  # 初始化结果列表

    for m in pattern.finditer(md_content):  # 遍历所有匹配
        start, end = m.span()  # 获取匹配起止位置
        pre_text = md_content[max(0, start - context_len):start]  # 截取上文
        post_text = md_content[end:min(len(md_content), end + context_len)]  # 截取下文
        logger.debug(f"图片[{image_filename}]匹配到引用，上文：{pre_text.strip()}")  # 记录上文日志
        logger.debug(f"图片[{image_filename}]匹配到引用，下文：{post_text.strip()}")  # 记录下文日志
        results.append((pre_text, post_text))  # 加入结果
    if not results:  # 判断是否无匹配
        logger.debug(f"MD内容中未找到图片[{image_filename}]的引用")  # 记录未找到日志
    return results  # 返回上下文列表


def step_2_scan_images(md_content: str, images_dir: Path) -> List[Tuple[str, str, Tuple[str, str]]]:
    """扫描图片文件夹，筛选MD中实际引用的支持格式图片。"""
    targets = []  # 初始化待处理图片列表
    for image_file in os.listdir(images_dir):  # 遍历图片文件夹
        if not is_supported_image(image_file):  # 判断是否支持该格式
            logger.debug(f"图片格式不支持，跳过：{image_file}")  # 记录跳过日志
            continue
        img_path = str(images_dir / image_file)  # 拼接图片完整路径
        context_list = find_image_in_md(md_content, image_file)  # 查找图片引用上下文
        if not context_list:  # 判断是否未在MD中引用
            logger.warning(f"图片未在MD中引用，跳过处理：{image_file}")  # 记录跳过日志
            continue
        targets.append((image_file, img_path, context_list[0]))  # 加入待处理列表
        logger.info(f"图片加入待处理列表：{image_file}")  # 记录加入日志
    logger.info(f"图片扫描完成，共筛选出待处理图片：{len(targets)} 张")  # 记录扫描完成日志
    return targets  # 返回待处理图片列表


def encode_image_to_base64(image_path: str) -> str:
    """将本地图片文件编码为Base64字符串。"""
    with open(image_path, "rb") as img_file:
        base64_str = base64.b64encode(img_file.read()).decode("utf-8")  # 读取并编码为Base64
    logger.debug(f"图片Base64编码完成，文件：{image_path}，编码后长度：{len(base64_str)}")  # 记录编码日志
    return base64_str  # 返回Base64字符串


def summarize_image(image_path: str, root_folder: str, image_content: Tuple[str, str]) -> str:
    """调用多模态大模型生成图片内容摘要。"""
    base64_image = encode_image_to_base64(image_path)  # 编码图片为Base64
    try:
        lvm_client = get_llm_client(model=lm_config.lv_model)  # 获取多模态大模型客户端

        prompt_text = load_prompt(
            name="image_summary",  # 提示词名称
            root_folder=root_folder,  # 传入文档主名
            image_content=image_content  # 传入图片上下文
        )

        messages = [
            HumanMessage(
                content=[
                    {
                        "type": "text",  # 文本类型
                        "text": prompt_text  # 文本提示词
                    },
                    {
                        "type": "image_url",  # 图片类型
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"  # Base64图片URL
                        }
                    }
                ]
            )
        ]

        response = lvm_client.invoke(messages)  # 调用大模型
        summary = response.content.strip().replace("\n", "")  # 提取并清洗摘要
        logger.info(f"图片摘要生成成功：{image_path}，摘要：{summary}")  # 记录生成成功日志
        return summary  # 返回摘要

    except LangChainException as e:
        logger.error(f"图片摘要生成失败（LangChain框架异常）：{image_path}，错误信息：{str(e)}")  # 记录框架异常
        return "图片描述"  # 返回默认摘要
    except Exception as e:
        logger.error(f"图片摘要生成失败（系统异常）：{image_path}，错误信息：{str(e)}")  # 记录系统异常
        return "图片描述"  # 返回默认摘要


def step_3_generate_summaries(doc_stem: str, targets: List[Tuple[str, str, Tuple[str, str]]],
                              requests_per_minute: int = 9) -> Dict[str, str]:
    """批量生成图片摘要，带API速率限制。"""
    summaries = {}  # 初始化摘要字典
    request_times = deque()  # 初始化请求时间队列

    for img_file, image_path, context in targets:  # 遍历待处理图片
        apply_api_rate_limit(request_times, requests_per_minute, window_seconds=60)  # 应用速率限制
        logger.debug(f"开始生成图片摘要：{image_path}")  # 记录开始日志
        summaries[img_file] = summarize_image(image_path, root_folder=doc_stem, image_content=context)  # 生成摘要

    logger.info(f"图片摘要批量生成完成，共处理{len(summaries)}张图片")  # 记录完成日志
    return summaries  # 返回摘要字典


def clean_minio_directory(minio_client: Minio, prefix: str) -> None:
    """清理MinIO指定前缀下的所有旧文件，实现幂等性。"""
    try:
        objects_to_delete = minio_client.list_objects(
            bucket_name=minio_config.bucket_name,  # 指定存储桶
            prefix=prefix,  # 指定目录前缀
            recursive=True  # 递归列出
        )
        delete_list = [DeleteObject(obj.object_name) for obj in objects_to_delete]  # 构造删除对象列表

        if delete_list:  # 判断是否存在待删除对象
            logger.info(f"开始清理MinIO旧文件，待删除文件数：{len(delete_list)}，目录：{prefix}")  # 记录清理日志
            errors = minio_client.remove_objects(minio_config.bucket_name, delete_list)  # 批量删除
            for error in errors:  # 遍历删除错误
                logger.error(f"MinIO文件删除失败：{error}")  # 记录删除错误
        else:
            logger.debug(f"MinIO目录无旧文件，无需清理：{prefix}")  # 记录无需清理日志
    except Exception as e:
        logger.error(f"MinIO目录清理失败：{prefix}，错误信息：{str(e)}")  # 记录清理失败日志


def upload_images_batch(minio_client: Minio, upload_dir: str, targets: List[Tuple[str, str, Tuple[str, str]]]) -> Dict[str, str]:
    """批量上传图片到MinIO，返回文件名与URL的映射。"""
    urls = {}  # 初始化URL字典
    for img_file, img_path, _ in targets:  # 遍历待上传图片
        object_name = f"{upload_dir}/{img_file}"  # 构造MinIO对象名
        logger.debug(f"构造MinIO对象名称完成：{object_name}")  # 记录对象名日志
        if img_url := upload_to_minio(minio_client, img_path, object_name):  # 上传并判断是否成功
            urls[img_file] = img_url  # 加入URL映射
    logger.info(f"图片批量上传完成，成功上传{len(urls)}/{len(targets)}张图片")  # 记录上传完成日志
    return urls  # 返回URL字典


def upload_to_minio(minio_client: Minio, local_path: str, object_name: str) -> str | None:
    """上传单张图片到MinIO并返回访问URL。"""
    try:
        logger.info(f"开始上传图片至MinIO：本地路径={local_path}，MinIO对象名={object_name}")  # 记录上传日志
        minio_client.fput_object(
            bucket_name=minio_config.bucket_name,  # 存储桶名
            object_name=object_name,  # 对象名
            file_path=local_path,  # 本地文件路径
            content_type=f"image/{os.path.splitext(local_path)[1][1:]}"  # 自动推断Content-Type
        )

        object_name = object_name.replace("\\", "%5C")  # 转义路径中的反斜杠
        protocol = "https" if minio_config.minio_secure else "http"  # 选择协议
        base_url = f"{protocol}://{minio_config.endpoint}/{minio_config.bucket_name}"  # 构造基础URL
        img_url = f"{base_url}{object_name}"  # 拼接完整图片URL
        logger.info(f"图片上传成功，访问URL：{img_url}")  # 记录上传成功日志
        return img_url  # 返回URL
    except Exception as e:
        logger.error(f"图片上传MinIO失败：{local_path}，错误信息：{str(e)}")  # 记录上传失败日志
        return None  # 返回空表示失败


def merge_summary_and_url(summaries: Dict[str, str], urls: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
    """合并摘要和URL，过滤上传失败的图片。"""
    image_info = {}  # 初始化图片信息字典
    for image_file, summary in summaries.items():  # 遍历摘要字典
        if url := urls.get(image_file):  # 判断是否上传成功
            image_info[image_file] = (summary, url)  # 加入图片信息
    logger.info(f"图片摘要与URL合并完成，有效图片信息{len(image_info)}条")  # 记录合并完成日志
    return image_info  # 返回合并后的字典


def process_md_file(md_content: str, image_info: Dict[str, Tuple[str, str]]) -> str:
    """替换MD内容中的本地图片引用为MinIO远程引用。"""
    for img_filename, (summary, new_url) in image_info.items():  # 遍历图片信息
        pattern = re.compile(
            r"!\[.*?\]\(.*?" + re.escape(img_filename) + r".*?\)",  # 编译图片引用正则
            re.IGNORECASE  # 忽略大小写
        )
        md_content = pattern.sub(f"![{summary}]({new_url})", md_content)  # 执行替换
        logger.debug(f"完成MD图片引用替换：{img_filename} → {new_url}")  # 记录替换日志

    logger.info(f"MD文件图片引用替换完成，共替换{len(image_info)}处图片引用")  # 记录替换完成日志
    logger.debug(f"替换后MD内容：{md_content[:500]}..." if len(md_content) > 500 else f"替换后MD内容：{md_content}")  # 记录替换后内容
    return md_content  # 返回替换后的MD内容


def step_4_upload_and_replace(minio_client: Minio, doc_stem: str, targets: List[Tuple[str, str, Tuple[str, str]]],
                              summaries: Dict[str, str], md_content: str) -> str:
    """清理旧文件、上传图片、合并摘要URL并替换MD引用。"""
    minio_img_dir = minio_config.minio_img_dir  # 获取MinIO图片根目录
    upload_dir = f"{minio_img_dir}/{doc_stem}".replace(" ", "")  # 构造上传目录并去除空格

    clean_minio_directory(minio_client, upload_dir)  # 清理旧目录
    urls = upload_images_batch(minio_client, upload_dir, targets)  # 批量上传图片
    image_info = merge_summary_and_url(summaries, urls)  # 合并摘要和URL
    if image_info:  # 判断是否存在有效图片信息
        md_content = process_md_file(md_content, image_info)  # 替换MD图片引用

    return md_content  # 返回处理后的MD内容


def step_5_backup_new_md_file(origin_md_path: str, md_content: str) -> str:
    """将处理后的MD内容保存为新文件，原文件保持不变。"""
    new_md_file_name = os.path.splitext(origin_md_path)[0] + "_new.md"  # 构造新文件路径

    with open(new_md_file_name, "w", encoding="utf-8") as f:
        f.write(md_content)  # 写入新MD内容

    logger.info(f"处理后MD文件已保存，新文件路径：{new_md_file_name}")  # 记录保存日志
    return new_md_file_name  # 返回新文件路径


def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """MD图片处理主节点：生成摘要、上传MinIO并替换MD图片引用。"""
    add_running_task(state["task_id"], sys._getframe().f_code.co_name)  # 标记当前节点为运行中

    md_content, path_obj, images_dir = step_1_get_content(state)  # 获取MD核心信息
    state["md_content"] = md_content  # 更新状态中的MD内容

    if not images_dir.exists():  # 判断图片文件夹是否存在
        logger.info(f"图片文件夹不存在，跳过图片处理：{images_dir.absolute()}")  # 记录跳过日志
        return state  # 返回原状态

    minio_client = get_minio_client()  # 获取MinIO客户端
    if not minio_client:  # 判断客户端是否初始化成功
        logger.warning("MinIO客户端初始化失败，已跳过图片处理全流程")  # 记录初始化失败日志
        return state  # 返回原状态

    targets = step_2_scan_images(md_content, images_dir)  # 扫描并筛选图片
    if not targets:  # 判断是否存在待处理图片
        logger.info("未检测到MD中引用的支持格式图片，跳过后续处理")  # 记录无图片日志
        return state  # 返回原状态

    summaries = step_3_generate_summaries(path_obj.stem, targets)  # 生成图片摘要
    new_md_content = step_4_upload_and_replace(minio_client, path_obj.stem, targets, summaries, md_content)  # 上传并替换
    state["md_content"] = new_md_content  # 更新状态中的MD内容

    new_md_file_name = step_5_backup_new_md_file(state['md_path'], new_md_content)  # 保存新MD文件
    state["md_path"] = new_md_file_name  # 更新状态中的MD路径
    logger.info(f"MD图片处理完成，新文件已保存：{new_md_file_name}")  # 记录处理完成日志

    return state  # 返回更新后的状态


if __name__ == "__main__":
    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")  # 打印项目根目录

    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")  # 构造测试MD相对路径
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)  # 拼接测试MD绝对路径

    if not os.path.exists(test_md_path):  # 判断测试文件是否存在
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")  # 记录文件不存在错误
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")  # 提示检查路径
    else:
        test_state = {
            "md_path": test_md_path,  # 设置MD路径
            "task_id": "test_task_123456",  # 设置测试任务ID
            "md_content": ""  # 初始化MD内容
        }
        logger.info("开始本地测试 - MD图片处理全流程")  # 记录测试启动日志
        result_state = node_md_img(test_state)  # 执行核心处理流程
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")  # 打印测试结果
