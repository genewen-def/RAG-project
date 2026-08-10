from pathlib import Path
from app.utils.path_util import PROJECT_ROOT
from app.core.logger import logger


def load_prompt(name: str, **kwargs) -> str:
    """加载提示词文件并渲染变量占位符。"""
    prompt_path = PROJECT_ROOT / 'prompts' / f'{name}.prompt'  # 拼接提示词文件路径

    if not prompt_path.exists():  # 判断提示词文件是否存在
        raise FileNotFoundError(f"提示词文件不存在：{prompt_path.absolute()}")  # 文件不存在时抛出异常

    raw_prompt = prompt_path.read_text(encoding='utf-8')  # 读取提示词文件内容

    if kwargs:  # 判断是否有需要渲染的变量
        rendered_prompt = raw_prompt.format(**kwargs)  # 使用传入变量渲染占位符
        logger.debug(f"提示词渲染成功，替换变量：{list(kwargs.keys())}")  # 记录渲染日志
        return rendered_prompt  # 返回渲染后的提示词

    return raw_prompt  # 无变量时返回原始提示词


if __name__ == '__main__':  # 当前模块直接运行时执行测试
    root_folder = "hl3070使用说明书"  # 定义测试用的文件夹名称
    image_content = ("这是图片的上文内容", "这是图片的下文内容")  # 定义测试用的图片上下文
    final_prompt = load_prompt(  # 调用函数加载并渲染提示词
        name='image_summary',  # 指定提示词文件名
        root_folder=root_folder,  # 传入文件夹名称变量
        image_content=image_content  # 传入图片上下文变量
    )
    print("✅ 渲染后的最终提示词：")  # 打印提示标题
    print(final_prompt)  # 输出渲染后的提示词
