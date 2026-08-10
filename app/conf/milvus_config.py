from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()  # 加载.env环境变量


@dataclass  # 声明为数据类
class MilvusConfig:
    milvus_url: str  # Milvus服务连接地址
    chunks_collection: str  # 文本切片集合名称
    entity_name_collection: str  # 实体名称集合名称
    item_name_collection: str  # 文档实体类集合名称

milvus_config = MilvusConfig(  # 实例化Milvus配置对象
    milvus_url=os.getenv("MILVUS_URL"),  # 从环境变量读取Milvus连接地址
    chunks_collection=os.getenv("CHUNKS_COLLECTION"),  # 从环境变量读取切片集合名
    entity_name_collection=os.getenv("ENTITY_NAME_COLLECTION"),  # 从环境变量读取实体名称集合名
    item_name_collection=os.getenv("ITEM_NAME_COLLECTION")  # 从环境变量读取实体类集合名
)  # 完成Milvus配置对象实例化
