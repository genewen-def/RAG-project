import os
from neo4j import GraphDatabase

_neo4j_driver = None  # 全局Neo4j驱动单例


def get_neo4j_driver() -> GraphDatabase:
    """获取Neo4j驱动单例实例。"""
    global _neo4j_driver  # 声明使用全局变量
    if _neo4j_driver is None:  # 判断驱动是否未初始化
        _neo4j_driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")))  # 创建驱动连接
    return _neo4j_driver  # 返回驱动实例
