import time
from typing import Deque
from app.core.logger import logger


def apply_api_rate_limit(  # 定义 API 限流函数
        request_times: Deque[float],  # 请求时间戳队列参数
        max_requests: int,  # 窗口最大请求数参数
        window_seconds: int = 60  # 窗口时长参数，默认 60 秒
) -> None:  # 函数无返回值
    """使用滑动窗口限制 API 请求速率，超限则阻塞等待。"""
    current_time = time.time()  # 获取当前时间戳

    while request_times and current_time - request_times[0] >= window_seconds:  # 清理窗口外过期时间戳
        request_times.popleft()  # 移除最早的时间戳

    if len(request_times) >= max_requests:  # 窗口内请求数已达上限
        sleep_duration = window_seconds - (current_time - request_times[0])  # 计算需等待时长
        if sleep_duration > 0:  # 需要等待时阻塞
            logger.debug(f"触发API速率限制，窗口{window_seconds}秒内最多{max_requests}次，需等待：{sleep_duration:.2f} 秒")  # 记录限流日志
            time.sleep(sleep_duration)  # 阻塞等待
            current_time = time.time()  # 等待后更新时间戳
            while request_times and current_time - request_times[0] >= window_seconds:  # 再次清理过期时间戳
                request_times.popleft()  # 移除最早的时间戳

    request_times.append(current_time)  # 记录当前请求时间戳
    logger.debug(f"API请求时间戳已记录，当前{window_seconds}秒窗口内请求数：{len(request_times)}")  # 记录当前请求数日志
