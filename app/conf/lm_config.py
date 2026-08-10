from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()  # 加载.env环境变量


@dataclass  # 声明为数据类
class LLMConfig:
    base_url: str  # 大模型服务基础URL
    api_key : str  # API密钥
    lv_model: str  # 视觉语言模型名称
    llm_model: str  # 大语言模型名称
    llm_temperature: float  # 大语言模型温度参数

lm_config = LLMConfig(  # 实例化LLM配置对象
    base_url=os.getenv("OPENAI_BASE_URL"),  # 从环境变量读取基础URL
    api_key=os.getenv("OPENAI_API_KEY"),  # 从环境变量读取API密钥
    lv_model=os.getenv("VL_MODEL"),  # 从环境变量读取视觉语言模型
    llm_model=os.getenv("LLM_DEFAULT_MODEL"),  # 从环境变量读取默认大模型
    llm_temperature=float(os.getenv("LLM_DEFAULT_TEMPERATURE"))  # 从环境变量读取温度并转为浮点数
)  # 完成LLM配置对象实例化
