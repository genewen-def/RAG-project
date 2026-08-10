from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()  # 加载.env环境变量


@dataclass  # 声明为数据类
class MinIOConfig:
    endpoint: str  # MinIO服务端点地址
    access_key: str  # 访问密钥
    secret_key: str  # 密钥
    bucket_name: str  # 默认存储桶名称
    minio_img_dir: str  # 图片存储目录
    minio_secure: bool  # 是否启用SSL

minio_config = MinIOConfig(  # 实例化MinIO配置对象
    endpoint=os.getenv("MINIO_ENDPOINT"),  # 从环境变量读取服务端点
    access_key=os.getenv("MINIO_ACCESS_KEY"),  # 从环境变量读取访问密钥
    secret_key=os.getenv("MINIO_SECRET_KEY"),  # 从环境变量读取密钥
    bucket_name=os.getenv("MINIO_BUCKET_NAME"),  # 从环境变量读取存储桶名
    minio_img_dir=os.getenv("MINIO_IMG_DIR"),  # 从环境变量读取图片目录
    minio_secure=os.getenv("MINIO_SECURE") == "True"  # 将环境变量转换为布尔值
)  # 完成MinIO配置对象实例化
