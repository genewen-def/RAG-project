try:  # 开始尝试加载 PyTorch 并检测 CUDA
    import torch  # 导入 PyTorch 库
    print(f"✅ PyTorch 加载成功！版本：{torch.__version__}")  # 打印 PyTorch 版本信息
    print(f"✅ CUDA 状态：{torch.cuda.is_available()}（CPU版显示False正常）")  # 打印 CUDA 是否可用
    print(f"✅ CUDA 设备数：{torch.cuda.device_count()}")  # 打印可用 CUDA 设备数量
    print(f"✅ CUDA 设备名称：{torch.cuda.get_device_name(0)}")  # 打印首个 CUDA 设备名称
except Exception as e:  # 捕获导入或 CUDA 检测过程中的异常
    print(f"❌ PyTorch 加载失败：{e}")  # 打印加载失败原因
