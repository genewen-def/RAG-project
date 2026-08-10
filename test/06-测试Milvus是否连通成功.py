from pymilvus import connections

connections.connect(host="127.0.0.1", port=19530)  # 连接到本地 Milvus 服务
print("✅本机Python连通Milvus成功")  # 打印连接成功提示
