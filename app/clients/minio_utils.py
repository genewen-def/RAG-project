import os
import json
from minio import Minio
from app.conf.minio_config import minio_config
from app.core.logger import logger

minio_client = None  # 全局MinIO客户端

try:
    minio_client = Minio(  # 初始化MinIO客户端
        endpoint=minio_config.endpoint,  # 设置服务端点
        access_key=minio_config.access_key,  # 设置访问密钥
        secret_key=minio_config.secret_key,  # 设置秘密密钥
        secure=False  # 使用HTTP连接
    )
    bucket_name = minio_config.bucket_name  # 读取存储桶名称

    if not minio_client.bucket_exists(bucket_name):  # 判断存储桶是否存在
        logger.info(f"MinIO存储桶[{bucket_name}]不存在，开始创建")  # 记录创建日志
        minio_client.make_bucket(bucket_name)  # 创建存储桶
        logger.info(f"MinIO存储桶[{bucket_name}]创建成功")  # 记录创建成功日志
    else:
        logger.info(f"MinIO存储桶[{bucket_name}]已存在，无需重复创建")  # 记录已存在日志

    bucket_policy = {  # 定义存储桶只读策略
        "Version": "2012-10-17",  # 设置策略版本
        "Statement": [{  # 定义策略声明
            "Effect": "Allow",  # 设置效果为允许
            "Principal": {"AWS": ["*"]},  # 允许所有用户
            "Action": ["s3:GetObject"],  # 允许获取对象操作
            "Resource": [f"arn:aws:s3:::{bucket_name}/*"]  # 指定资源范围
        }]
    }
    minio_client.set_bucket_policy(bucket_name, json.dumps(bucket_policy))  # 应用存储桶策略
    logger.info(f"MinIO存储桶[{bucket_name}]已配置公网只读策略，支持匿名URL访问")  # 记录策略配置日志

except Exception as e:  # 捕获初始化异常
    logger.error(f"MinIO客户端初始化失败，错误信息：{str(e)}", exc_info=True)  # 记录错误日志
    minio_client = None  # 将客户端置空


def get_minio_client():
    """获取全局MinIO客户端实例。"""
    return minio_client  # 返回客户端实例
