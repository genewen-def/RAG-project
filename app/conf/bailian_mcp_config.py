from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()  # 加载.env环境变量


@dataclass  # 声明为数据类
class McpConfig:
    mcp_base_url: str  # MCP服务基础URL
    api_key : str  # API密钥

mcp_config = McpConfig(  # 实例化MCP配置对象
    mcp_base_url=os.getenv("MCP_DASHSCOPE_BASE_URL"),  # 从环境变量读取MCP基础URL
    api_key=os.getenv("OPENAI_API_KEY")  # 从环境变量读取API密钥
)  # 完成MCP配置对象实例化
