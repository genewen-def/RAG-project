from pathlib import Path
from dotenv import load_dotenv
import os
from pathlib import Path


def get_path_dir(ps: int = 0) -> Path:  # 定义获取父目录函数
    """获取当前文件向上第 ps 级父目录。"""
    dir_path = Path(__file__).parents[ps]  # 取当前文件的第 ps 级父目录
    return dir_path  # 返回目标目录路径


def get_project_root(identifier: str = ".env") -> Path:  # 定义获取项目根目录函数
    """获取项目根目录，优先使用环境变量，否则向上查找标识文件。"""
    env_root = os.getenv("PROJECT_ROOT")  # 读取项目根目录环境变量
    if env_root and Path(env_root).absolute().exists():  # 环境变量有效则直接返回
        return Path(env_root).absolute()

    current_dir = Path(__file__).absolute().parent  # 从当前文件所在目录开始查找
    while current_dir != current_dir.parent:  # 未到达文件系统根则继续
        if (current_dir / identifier).exists():  # 找到标识文件
            load_dotenv(dotenv_path=current_dir / identifier)  # 加载对应 .env 文件
            break  # 结束查找
        current_dir = current_dir.parent  # 向上移动一级

    current_dir = Path(__file__).absolute().parent  # 再次从当前目录开始查找
    while current_dir != current_dir.parent:  # 未到达文件系统根则继续
        if (current_dir / identifier).exists():  # 找到标识文件则返回该目录
            return current_dir
        current_dir = current_dir.parent  # 向上移动一级

    raise FileNotFoundError(f"未找到项目根目录标识「{identifier}」，且环境变量PROJECT_ROOT未配置")  # 查找失败抛出异常


PROJECT_ROOT = get_project_root(".env")  # 初始化项目根目录全局变量
