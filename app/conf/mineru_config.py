from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()  # 加载.env环境变量


@dataclass  # 声明为数据类
class MineruConfig:
    base_url: str  # MinerU服务基础URL
    api_key : str  # API访问令牌

mineru_config = MineruConfig(  # 实例化MinerU配置对象
    base_url=os.getenv("MINERU_BASE_URL"),  # 从环境变量读取MinerU基础URL
    api_key=os.getenv("MINERU_API_TOKEN")  # 从环境变量读取MinerU访问令牌
)  # 完成MinerU配置对象实例化
