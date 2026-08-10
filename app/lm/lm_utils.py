import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.exceptions import LangChainException
from typing import Optional

from app.conf.lm_config import lm_config
from app.core.logger import logger

_llm_client_cache = {}  # 全局缓存，避免重复初始化 LLM 客户端


def get_llm_client(model: Optional[str] = None, json_mode: bool = False) -> ChatOpenAI:  # 定义 LLM 客户端获取函数
    """获取带缓存的 ChatOpenAI 客户端实例。"""
    target_model = model or lm_config.llm_model or "qwen3-32b"  # 确定目标模型，按优先级取值
    cache_key = (target_model, json_mode)  # 构造缓存键

    if cache_key in _llm_client_cache:  # 缓存命中则直接返回
        logger.debug(f"[LLM客户端] 缓存命中，直接返回实例：模型={target_model}，JSON模式={json_mode}")
        return _llm_client_cache[cache_key]  # 返回缓存的客户端实例

    if not lm_config.api_key:  # 校验 API 密钥是否配置
        raise ValueError("[LLM客户端] 配置缺失：请在.env中配置OPENAI_API_KEY（大模型API密钥）")  # 抛出密钥缺失异常
    if not lm_config.base_url:  # 校验 API 基础地址是否配置
        raise ValueError("[LLM客户端] 配置缺失：请在.env中配置OPENAI_API_BASE（API接口基础地址）")  # 抛出地址缺失异常
    logger.info(f"[LLM客户端] 开始初始化新实例：模型={target_model}，JSON模式={json_mode}")  # 记录开始初始化日志

    extra_body = {"enable_thinking": False}  # 设置国产模型私有参数
    model_kwargs = {}  # 初始化 OpenAI 通用参数
    if json_mode:  # JSON 模式下设置响应格式
        model_kwargs["response_format"] = {"type": "json_object"}  # 设置 JSON 响应格式
        logger.debug(f"[LLM客户端] 已开启JSON输出模式，模型将返回标准JSON结构")  # 记录 JSON 模式日志

    try:  # 捕获初始化异常
        llm_client = ChatOpenAI(  # 初始化 ChatOpenAI 客户端
            model=target_model,  # 指定模型名称
            temperature=lm_config.llm_temperature or 0.1,  # 设置采样温度
            api_key=lm_config.api_key,  # 传入 API 密钥
            base_url=lm_config.base_url,  # 传入 API 基础地址
            extra_body=extra_body,  # 传入私有参数
            model_kwargs=model_kwargs,  # 传入通用参数
        )
    except LangChainException as e:  # 捕获初始化异常并转换
        raise Exception(f"[LLM客户端] 模型【{target_model}】初始化失败（LangChain层）：{str(e)}") from e  # 抛出友好异常

    _llm_client_cache[cache_key] = llm_client  # 将新实例存入缓存
    logger.info(f"[LLM客户端] 实例初始化成功并缓存：模型={target_model}，JSON模式={json_mode}")  # 记录初始化成功日志

    return llm_client  # 返回初始化完成的客户端


if __name__ == "__main__":  # 当前脚本直接运行时执行测试
    logger.info("===== 开始执行LLM客户端工具测试 =====")  # 打印测试开始日志
    try:  # 捕获测试异常
        client1 = get_llm_client()  # 测试默认配置创建客户端
        logger.info("✅ 测试1通过：默认配置客户端创建成功")  # 记录测试1结果

        client2 = get_llm_client(model="qwen-vl-plus")  # 测试指定模型创建客户端
        logger.info("✅ 测试2通过：指定多模态模型客户端创建成功")  # 记录测试2结果

        client3 = get_llm_client(model="qwen-vl-plus")  # 测试缓存命中
        logger.info(f"✅ 测试3通过：缓存机制验证成功，client2与client3为同一实例：{client2 is client3}")  # 记录测试3结果

        client4 = get_llm_client(model="qwen3-32b", json_mode=True)  # 测试 JSON 模式
        logger.info("✅ 测试4通过：JSON输出模式客户端创建成功")  # 记录测试4结果

    except Exception as e:  # 捕获测试异常
        logger.error(f"❌ LLM客户端工具测试失败：{str(e)}", exc_info=True)  # 记录错误日志
    finally:  # 测试结束清理
        logger.info("===== LLM客户端工具测试结束 =====")  # 打印测试结束日志
