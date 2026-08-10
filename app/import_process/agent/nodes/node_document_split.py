import re
import json
import os
import sys
from typing import List, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils.task_utils import add_running_task
from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger

DEFAULT_MAX_CONTENT_LENGTH = 2000
MIN_CONTENT_LENGTH = 500


def step_1_get_inputs(state: ImportGraphState) -> Tuple[Any, str, int]:
    """从状态中提取MD内容、文件标题和最大Chunk长度。"""
    content = state.get("md_content")  # 获取MD原始内容
    if not content:  # 判断内容是否为空
        logger.warning("状态字典中无有效MD内容，终止文档切分")  # 记录空内容警告
        return None, None, None  # 返回空值终止后续处理

    content = content.replace("\r\n", "\n").replace("\r", "\n")  # 统一换行符
    file_title = state.get("file_title", "Unknown File")  # 获取文件标题
    max_len = DEFAULT_MAX_CONTENT_LENGTH  # 使用默认最大长度

    logger.info(f"步骤1：输入数据加载完成，文件标题：{file_title}，最大Chunk长度：{max_len}")  # 记录加载完成日志
    return content, file_title, max_len  # 返回标准化后的数据


def step_2_split_by_titles(content: str, file_title: str) -> Tuple[List[Dict[str, Any]], int, int]:
    """按Markdown标题初次切分，跳过代码块内标题。"""
    title_pattern = r"^\s*#{1,6}\s+.+"  # 定义标题匹配正则

    lines = content.split("\n")  # 按换行拆分为行列表
    sections = []  # 初始化章节列表
    current_title = ""  # 初始化当前章节标题
    current_lines = []  # 初始化当前章节行缓存
    title_count = 0  # 初始化有效标题计数
    in_code_block = False  # 初始化代码块状态标记

    def _flush_section():
        """将当前缓存的章节写入sections。"""
        if not current_lines:  # 空缓存则跳过
            return
        sections.append({
            "title": current_title,  # 写入章节标题
            "content": "\n".join(current_lines),  # 合并行缓存为内容
            "file_title": file_title,  # 写入文件标题
        })

    for line in lines:  # 逐行遍历MD内容
        stripped_line = line.strip()  # 去除行首尾空白
        if stripped_line.startswith("```") or stripped_line.startswith("~~~"):  # 判断代码块边界
            in_code_block = not in_code_block  # 翻转代码块状态
            current_lines.append(line)  # 将边界行加入当前章节
            continue  # 继续下一行

        is_valid_title = (not in_code_block) and re.match(title_pattern, line)  # 判断是否为有效标题
        if is_valid_title:  # 遇到新标题
            _flush_section()  # 写入上一个章节
            current_title = line.strip()  # 更新当前标题
            current_lines = [current_title]  # 新章节从标题开始
            title_count += 1  # 标题计数加一
            logger.debug(f"识别到MD标题：{current_title}")  # 记录标题识别日志
        else:
            current_lines.append(line)  # 普通行加入当前章节缓存

    _flush_section()  # 写入最后一个章节
    logger.info(f"步骤2：MD标题切分完成，识别到{title_count}个有效标题，原始文本共{len(lines)}行")  # 记录切分完成日志
    return sections, title_count, len(lines)  # 返回章节列表、标题数和行数


def step_3_handle_no_title(content: str, sections: List[Dict[str, Any]], title_count: int, file_title: str) -> List[Dict[str, Any]]:
    """无标题时将全文封装为单章节，否则返回原章节列表。"""
    if title_count == 0:  # 判断是否无标题
        logger.warning(f"步骤3：未识别到任何MD标题，将全文作为单个章节处理，文件：{file_title}")  # 记录无标题警告
        return [{"title": "无标题", "content": content, "file_title": file_title}]  # 返回单章节兜底
    logger.debug(f"步骤3：检测到{title_count}个有效标题，无需兜底处理")  # 记录无需兜底日志
    return sections  # 返回原章节列表


def _split_long_section(section: Dict[str, Any], max_length: int = DEFAULT_MAX_CONTENT_LENGTH) -> List[Dict[str, Any]]:
    """对超长章节按语义优先级二次切分，生成带元信息的子章节。"""
    content = section.get("content", "") or ""  # 获取章节内容
    if len(content) <= max_length:  # 判断长度是否超限
        return [section]  # 未超限直接返回原章节

    content = content.replace("\r\n", "\n").replace("\r", "\n")  # 统一换行符
    title = section.get("title", "") or ""  # 获取章节标题
    prefix = f"{title}\n\n" if title else ""  # 构造标题前缀
    available_len = max_length - len(prefix)  # 计算正文可用长度
    if available_len <= 0:  # 判断标题是否过长
        logger.warning(f"章节标题过长，无法切分：{title[:20]}...")  # 记录标题过长警告
        return [section]  # 返回原章节

    body = content  # 获取正文
    if title and body.lstrip().startswith(title):  # 判断正文开头是否重复标题
        body = body[body.find(title) + len(title):].lstrip()  # 去除重复标题

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=available_len,  # 设置正文最大长度
        chunk_overlap=0,  # 设置无重叠
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],  # 设置分隔符优先级
    )

    sub_sections = []  # 初始化子章节列表
    for idx, chunk in enumerate(splitter.split_text(body), start=1):  # 切分正文并带序号遍历
        text = chunk.strip()  # 去除子块首尾空白
        if not text:  # 跳过空内容
            continue
        full_text = (prefix + text).strip()  # 组装完整子块内容
        sub_sections.append({
            "title": f"{title}-{idx}" if title else f"chunk-{idx}",  # 生成子块标题
            "content": full_text,  # 写入完整内容
            "parent_title": title,  # 写入父标题
            "part": idx,  # 写入子块序号
            "file_title": section.get("file_title"),  # 写入文件标题
        })

    logger.debug(f"超长章节切分完成：{title} → 生成{len(sub_sections)}个子Chunk")  # 记录切分完成日志
    return sub_sections  # 返回子章节列表


def _merge_short_sections(sections: List[Dict[str, Any]], min_length: int = MIN_CONTENT_LENGTH) -> List[Dict[str, Any]]:
    """合并同父标题且长度不足阈值的相邻Chunk。"""
    if not sections:  # 判断列表是否为空
        logger.debug("待合并Chunk列表为空，直接返回")  # 记录空列表日志
        return []  # 返回空列表

    merged_sections = []  # 初始化合并结果列表
    current_chunk = None  # 初始化当前合并块

    for sec in sections:  # 遍历章节列表
        if current_chunk is None:  # 第一个块直接作为当前块
            current_chunk = sec
            continue

        is_current_short = len(current_chunk["content"]) < min_length  # 判断当前块是否过短
        is_same_parent = current_chunk.get("parent_title") == sec.get("parent_title")  # 判断是否同父标题

        if is_current_short and is_same_parent:  # 满足合并条件
            parent_title = sec.get("parent_title", "")  # 获取父标题
            next_content = sec["content"]  # 获取下一块内容
            if parent_title and next_content.startswith(parent_title):  # 判断开头是否重复父标题
                next_content = next_content[len(parent_title):].lstrip()  # 去除重复标题
            current_chunk["content"] += "\n\n" + next_content  # 合并内容
            if "part" in sec:  # 判断是否存在part字段
                current_chunk["part"] = sec["part"]  # 更新part为下一块序号
            logger.debug(f"合并短Chunk：{current_chunk.get('parent_title')} → 累计长度{len(current_chunk['content'])}")  # 记录合并日志
        else:
            merged_sections.append(current_chunk)  # 将当前块加入结果
            current_chunk = sec  # 切换为新的当前块

    if current_chunk is not None:  # 循环结束后仍有未写入的当前块
        merged_sections.append(current_chunk)  # 加入最后一个块

    logger.debug(f"短Chunk合并完成：原{len(sections)}个 → 合并后{len(merged_sections)}个")  # 记录合并完成日志
    return merged_sections  # 返回合并后的列表


def step_4_refine_chunks(sections: List[Dict[str, Any]], max_len: int) -> List[Dict[str, Any]]:
    """对章节列表执行长切短合，并补全parent_title等字段。"""
    if not max_len or max_len <= 0:  # 判断最大长度是否有效
        logger.warning(f"步骤4：Chunk最大长度配置无效（{max_len}），跳过精细化处理")  # 记录无效配置警告
        return sections  # 返回原章节

    refined_split = []  # 初始化切分结果列表
    for sec in sections:  # 遍历章节列表
        refined_split.extend(_split_long_section(sec, max_len))  # 切分并平铺结果
    logger.info(f"步骤4-1：超长章节切分完成，共生成{len(refined_split)}个初始子Chunk")  # 记录切分完成日志

    final_sections = _merge_short_sections(refined_split)  # 合并过短章节
    logger.info(f"步骤4-2：过短章节合并完成，最终得到{len(final_sections)}个Chunk")  # 记录合并完成日志

    for sec in final_sections:  # 遍历最终章节
        if not isinstance(sec, dict):  # 跳过非字典元素
            continue

        if "part" not in sec:  # 判断是否存在part字段
            sec["part"] = 0  # 补全默认part

        if not sec.get("parent_title"):  # 判断父标题是否缺失
            sec["parent_title"] = sec.get("title") or ""  # 用自身标题兜底
    logger.debug(f"步骤4-3：父标题兜底完成，所有Chunk均包含parent_title字段")  # 记录兜底完成日志

    return final_sections  # 返回最终章节列表


def step_5_print_stats(lines_count: int, sections: List[Dict[str, Any]]) -> None:
    """输出文档切分的核心统计信息。"""
    chunk_num = len(sections)  # 获取最终Chunk数量
    logger.info("-" * 50 + " 文档切分统计信息 " + "-" * 50)  # 打印分隔线
    logger.info(f"MD原始文本总行数：{lines_count}")  # 打印原始行数
    logger.info(f"最终生成Chunk数量：{chunk_num}")  # 打印Chunk数量
    if sections:  # 判断是否存在章节
        first_title = sections[0].get("title", "无标题")  # 获取首个标题
        logger.info(f"首个Chunk标题预览：{first_title}")  # 打印首个标题
    logger.info("-" * 110)  # 打印结束分隔线


def step_6_backup(state: ImportGraphState, sections: List[Dict[str, Any]]) -> None:
    """将Chunk结果以JSON格式备份到local_dir目录。"""
    local_dir = state.get("local_dir")  # 获取备份目录
    if not local_dir:  # 判断是否配置备份目录
        logger.warning("步骤6：未配置备份目录（local_dir），跳过Chunk结果备份")  # 记录跳过日志
        return  # 直接返回

    try:
        os.makedirs(local_dir, exist_ok=True)  # 创建备份目录
        backup_path = os.path.join(local_dir, "chunks.json")  # 拼接备份文件路径
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(
                sections,  # 写入切片数据
                f,  # 写入文件对象
                ensure_ascii=False,  # 保留中文
                indent=2  # 格式化缩进
            )
        logger.info(f"步骤6：Chunk结果备份成功，备份文件路径：{backup_path}")  # 记录备份成功日志
    except Exception as e:
        logger.error(f"步骤6：Chunk结果备份失败，错误信息：{str(e)}", exc_info=False)  # 记录备份失败日志


def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """文档切分主节点：将MD内容切分为长度适中的Chunk并更新状态。"""
    node_name = sys._getframe().f_code.co_name  # 获取当前节点名
    logger.info(f">>> 开始执行核心节点：【文档切分】{node_name}")  # 记录节点启动日志
    add_running_task(state["task_id"], node_name)  # 标记节点为运行中

    try:
        content, file_title, max_len = step_1_get_inputs(state)  # 获取并标准化输入
        if content is None:  # 判断是否有有效内容
            logger.info(f">>> 节点执行终止：{node_name}（无有效MD内容）")  # 记录终止日志
            return state  # 返回原状态

        sections, title_count, lines_count = step_2_split_by_titles(content, file_title)  # 按标题初切
        sections = step_3_handle_no_title(content, sections, title_count, file_title)  # 无标题兜底
        sections = step_4_refine_chunks(sections, max_len)  # 精细化切分合并
        step_5_print_stats(lines_count, sections)  # 输出统计信息

        state["chunks"] = sections  # 将Chunk写入状态
        step_6_backup(state, sections)  # 备份Chunk结果

        logger.info(f">>> 核心节点执行完成：【文档切分】{node_name}，已生成{len(sections)}个有效Chunk，结果已写入状态字典")  # 记录完成日志

    except Exception as e:
        logger.error(f">>> 核心节点执行失败：【文档切分】{node_name}，错误信息：{str(e)}", exc_info=True)  # 记录异常日志

    return state  # 返回更新后的状态


if __name__ == '__main__':
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img

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
            "md_content": "",  # 初始化MD内容
            "file_title": "hak180产品安全手册",  # 设置文件标题
            "local_dir": os.path.join(PROJECT_ROOT, "output"),  # 设置本地目录
        }
        logger.info("开始本地测试 - MD图片处理全流程")  # 记录测试启动日志
        result_state = node_md_img(test_state)  # 执行MD图片处理节点
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")  # 打印图片处理结果
        logger.info("\n=== 开始执行文档切分节点集成测试 ===")  # 打印切分测试开始

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")  # 记录切分节点启动
        final_state = node_document_split(result_state)  # 执行文档切分节点
        final_chunks = final_state.get("chunks", [])  # 获取最终Chunk列表
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")  # 打印测试结果
