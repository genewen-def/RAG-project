from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()  # 加载.env环境变量


@dataclass  # 声明为数据类
class RerankerConfig:
    bge_reranker_large: str  # 重排序模型路径
    bge_reranker_device: str  # 重排序模型运行设备
    bge_reranker_fp16: bool  # 是否开启半精度推理

reranker_config = RerankerConfig(  # 实例化重排序配置对象
    bge_reranker_large=os.getenv("BGE_RERANKER_LARGE"),  # 从环境变量读取模型路径
    bge_reranker_device=os.getenv("BGE_RERANKER_DEVICE"),  # 从环境变量读取运行设备
    bge_reranker_fp16=os.getenv("BGE_RERANKER_FP16") in ("1", "True", "true", 1)  # 将环境变量转换为布尔值
)  # 完成重排序配置对象实例化
