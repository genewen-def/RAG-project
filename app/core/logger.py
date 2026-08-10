import sys
import inspect
from pathlib import Path
import os
from dotenv import load_dotenv
from loguru import logger


load_dotenv()  # 加载.env环境变量

LOG_CONSOLE_ENABLE = os.getenv("LOG_CONSOLE_ENABLE", "True").lower() == "true"  # 读取控制台日志开关
LOG_CONSOLE_LEVEL = os.getenv("LOG_CONSOLE_LEVEL", "INFO").upper()  # 读取控制台日志级别
LOG_FILE_ENABLE = os.getenv("LOG_FILE_ENABLE", "True").lower() == "true"  # 读取文件日志开关
LOG_FILE_LEVEL = os.getenv("LOG_FILE_LEVEL", "INFO").upper()  # 读取文件日志级别
LOG_FILE_RETENTION = os.getenv("LOG_FILE_RETENTION", "7 days")  # 读取日志文件保留时长

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # 根据当前文件位置推导项目根目录
LOG_DIR = PROJECT_ROOT / "logs"  # 定义日志目录路径
LOG_FILE_NAME = "app_{time:YYYYMMDD}.log"  # 定义日志文件名模板
LOG_FILE_PATH = LOG_DIR / LOG_FILE_NAME  # 拼接完整日志文件路径

LOG_FORMAT = (  # 定义日志输出格式
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name: <20}</cyan>:<cyan>{function: <15}</cyan>:<cyan>{line: <4}</cyan> - "
    "<level>{message}</level>"
)

def init_logger():
    """初始化并配置全局日志实例。"""
    logger.remove()  # 移除loguru默认控制台输出

    if LOG_CONSOLE_ENABLE:  # 判断是否开启控制台日志
        logger.add(  # 添加控制台日志输出
            sink=sys.stdout,  # 输出到标准输出
            level=LOG_CONSOLE_LEVEL,  # 设置控制台日志级别
            format=LOG_FORMAT,  # 设置日志格式
            colorize=True,  # 启用颜色输出
            enqueue=True  # 启用异步队列
        )

    if LOG_FILE_ENABLE:  # 判断是否开启文件日志
        LOG_DIR.mkdir(parents=True, exist_ok=True)  # 创建日志目录
        logger.add(  # 添加文件日志输出
            sink=LOG_FILE_PATH,  # 输出到日志文件
            level=LOG_FILE_LEVEL,  # 设置文件日志级别
            format=LOG_FORMAT,  # 设置日志格式
            rotation="00:00",  # 设置每日轮转时间
            retention=LOG_FILE_RETENTION,  # 设置日志保留策略
            encoding="utf-8",  # 设置文件编码为UTF-8
            enqueue=True,  # 启用异步队列
            backtrace=True,  # 启用异常回溯
            diagnose=True  # 启用诊断信息
        )

    return logger  # 返回配置好的日志实例

base_logger = init_logger()  # 初始化基础日志实例

def fix_log_position(record):
    """修正日志记录中的调用位置信息。"""
    for frame in inspect.stack():  # 遍历当前调用栈
        if ("_logger.py" in frame.filename or frame.function == "_log") or "logger.py" in frame.filename:  # 跳过loguru内部和当前模块帧
            continue  # 继续检查下一帧

        record.update(  # 更新日志记录的位置字段
            name=frame.filename.split("/")[-1].split("\\")[-1],  # 提取业务模块文件名
            function=frame.function,  # 提取业务模块函数名
            line=frame.lineno  # 提取业务模块行号
        )
        break  # 找到首个业务帧后退出循环

logger = base_logger.patch(fix_log_position)  # 应用位置修复补丁

if __name__ == '__main__':  # 当前模块直接运行时执行测试
    logger.info("【测试】logger.py内部调用（仅测试，业务模块调用会显示正确文件名）")  # 输出测试日志
    print(f"日志文件输出路径：{LOG_FILE_PATH}")  # 打印日志文件路径
