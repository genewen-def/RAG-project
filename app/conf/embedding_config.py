from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()  # 加载.env环境变量


@dataclass  # 声明为数据类
class EmbeddingConfig:
    bge_m3_path: str  # 本地BGE-M3模型路径
    bge_m3: str  # BGE-M3模型仓库标识
    bge_device: str  # 模型运行设备
    bge_fp16: bool  # 是否开启半精度推理

embedding_config = EmbeddingConfig(  # 实例化Embedding配置对象
    bge_m3_path=os.getenv("BGE_M3_PATH"),  # 从环境变量读取模型路径
    bge_m3=os.getenv("BGE_M3"),  # 从环境变量读取模型标识
    bge_device=os.getenv("BGE_DEVICE"),  # 从环境变量读取运行设备
    bge_fp16=os.getenv("BGE_FP16") in ("1", "True", "true", 1)  # 将环境变量转换为布尔值
)  # 完成Embedding配置对象实例化
