import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pymongo import MongoClient, ASCENDING
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()  # 加载环境变量


class HistoryMongoTool:
    """MongoDB历史对话记录工具类。"""

    def __init__(self):
        """初始化MongoDB连接和集合。"""
        try:
            self.mongo_url = os.getenv("MONGO_URL")  # 读取MongoDB连接地址
            self.db_name = os.getenv("MONGO_DB_NAME")  # 读取数据库名称

            self.client = MongoClient(self.mongo_url)  # 创建MongoDB客户端
            self.db = self.client[self.db_name]  # 获取数据库对象
            self.chat_message = self.db["chat_message"]  # 获取对话集合

            self.chat_message.create_index([("session_id", 1), ("ts", -1)])  # 创建复合索引

            logging.info(f"Successfully connected to MongoDB: {self.db_name}")  # 记录连接成功日志
        except Exception as e:  # 捕获初始化异常
            logging.error(f"Failed to connect to MongoDB: {e}")  # 记录错误日志
            raise  # 抛出异常


_history_mongo_tool = None  # 全局单例

try:
    _history_mongo_tool = HistoryMongoTool()  # 模块加载时预初始化单例
except Exception as e:  # 捕获预初始化异常
    logging.warning(f"Could not initialize HistoryMongoTool on module load: {e}")  # 记录警告日志


def get_history_mongo_tool() -> HistoryMongoTool:
    """获取HistoryMongoTool单例实例。"""
    global _history_mongo_tool  # 声明使用全局变量
    if _history_mongo_tool is None:  # 判断单例是否为空
        _history_mongo_tool = HistoryMongoTool()  # 创建新实例
    return _history_mongo_tool  # 返回单例实例


def clear_history(session_id: str) -> int:
    """清空指定会话的历史记录。"""
    mongo_tool = get_history_mongo_tool()  # 获取单例
    try:
        result = mongo_tool.chat_message.delete_many({"session_id": session_id})  # 删除会话记录
        logging.info(f"Deleted {result.deleted_count} messages for session {session_id}")  # 记录删除日志
        return result.deleted_count  # 返回删除数量
    except Exception as e:  # 捕获删除异常
        logging.error(f"Error clearing history for session {session_id}: {e}")  # 记录错误日志
        return 0  # 返回0表示失败


def save_chat_message(
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        item_names: List[str] = None,
        image_urls: List[str] = None,
        message_id: str = None
) -> str:
    """写入或更新单条会话记录。"""
    ts = datetime.now().timestamp()  # 生成当前时间戳

    document = {  # 构建文档数据
        "session_id": session_id,  # 会话ID
        "role": role,  # 消息角色
        "text": text,  # 消息内容
        "rewritten_query": rewritten_query or "",  # 重写查询
        "item_names": item_names,  # 关联商品列表
        "image_urls": image_urls,  # 关联图片URL列表
        "ts": ts  # 时间戳
    }

    mongo_tool = get_history_mongo_tool()  # 获取单例
    if message_id:  # 判断是否为更新操作
        result = mongo_tool.chat_message.update_one(  # 执行更新
            {"_id": ObjectId(message_id)},  # 根据主键匹配
            {"$set": document}  # 更新文档字段
        )
        return message_id  # 返回message_id
    else:
        result = mongo_tool.chat_message.insert_one(document)  # 执行插入
        return str(result.inserted_id)  # 返回插入ID字符串


def update_message_item_names(ids: List[str], item_names: List[str]) -> int:
    """批量更新历史记录的关联商品名称。"""
    mongo_tool = get_history_mongo_tool()  # 获取单例
    try:
        object_ids = [ObjectId(i) for i in ids]  # 转换ID为ObjectId
        result = mongo_tool.chat_message.update_many(  # 执行批量更新
            {
                "_id": {"$in": object_ids}  # 主键在指定列表中
            },
            {"$set": {"item_names": item_names}}  # 设置商品名称
        )
        logging.info(f"Updated {result.modified_count} records to item_names: {item_names}")  # 记录更新日志
        return result.modified_count  # 返回更新数量
    except Exception as e:  # 捕获更新异常
        logging.error(f"Error updating history item_names: {e}")  # 记录错误日志
        return 0  # 返回0表示失败


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """查询指定会话最近N条记录。"""
    mongo_tool = get_history_mongo_tool()  # 获取单例
    try:
        query = {"session_id": session_id}  # 构建查询条件

        cursor = mongo_tool.chat_message.find(query).sort("ts", ASCENDING).limit(limit)  # 查询并排序限制条数
        messages = list(cursor)  # 转为列表
        return messages  # 返回记录列表
    except Exception as e:  # 捕获查询异常
        logging.error(f"Error getting recent messages: {e}")  # 记录错误日志
        return []  # 返回空列表


if __name__ == "__main__":  # 判断是否直接运行
    sid = "000015_hybrid"  # 定义测试会话ID
    save_chat_message(sid, "user", "你好 (Hybrid)")  # 保存用户消息
    save_chat_message(sid, "assistant", "你好！我是基于原生 Mongo + LangChain 对象的助手。")  # 保存助手回复
    save_chat_message(sid, "user", "这个万用表怎么换电池？", item_names=["混合万用表"])  # 保存带商品消息

    print("--- 查询 LangChain 对象记录 ---")  # 打印分隔标题
    messages = get_recent_messages(sid, limit=5)  # 查询最近记录
    print(f"查询到的记录数: {len(messages)}")  # 打印记录数
    for m in messages:  # 遍历记录
        print(f" {m}  ")  # 打印单条记录
