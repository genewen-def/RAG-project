from modelscope.hub.snapshot_download import snapshot_download

local_dir = r"D:\ai_models\modelscope_cache\models\rerank"  # 设置本地缓存目录

snapshot_download(  # 下载重排序模型
    model_id="BAAI/bge-reranker-large",  # 指定模型 ID
    cache_dir=local_dir,  # 指定缓存目录
)  # 结束函数调用

print("下载完成，模型目录：", local_dir)  # 打印下载完成信息
