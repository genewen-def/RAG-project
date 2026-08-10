import sys
import json
import asyncio
from app.utils.task_utils import add_done_task, add_running_task
from app.conf.bailian_mcp_config import mcp_config
from agents.mcp import MCPServerSse
from app.core.logger import logger

async def mcp_call(query):
    """异步调用百炼 MCP 搜索服务并返回原始结果。"""
    search_mcp = MCPServerSse(  # 初始化 MCP SSE 客户端
        name="search_mcp",
        params={
            "url": mcp_config.mcp_base_url,  # MCP 服务地址
            "headers": {"Authorization": mcp_config.api_key},  # 鉴权请求头
            "timeout": 300,  # 连接超时时间
            "sse_read_timeout": 300  # SSE 读取超时时间
        }
    )

    try:
        logger.info(f"[MCP] 正在连接百炼 WebSearch 服务: {mcp_config.mcp_base_url}")  # 打印连接日志
        await search_mcp.connect()  # 建立 SSE 连接

        logger.info(f"[MCP] 连接成功，正在调用工具 'bailian_web_search' 查询: {query}")  # 打印调用日志
        result = await search_mcp.call_tool(  # 调用搜索工具
            tool_name="bailian_web_search",
            arguments={"query": query, "count": 5}
        )
        logger.info("[MCP] 工具调用完成，已获取返回结果")  # 打印完成日志
        return result  # 返回原始结果

    except Exception as e:
        logger.error(f"[MCP] 调用过程中发生异常: {e}", exc_info=True)  # 打印调用异常
        return None  # 返回空结果

    finally:
        await search_mcp.cleanup()  # 关闭 MCP 连接释放资源


def node_web_search_mcp(state):
    """LangGraph 同步节点：调用 MCP 搜索并解析为结构化数据。"""
    logger.info("---node_web_search_mcp 开始处理---")  # 打印节点开始日志

    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))  # 标记任务开始

    query = state.get("rewritten_query", "")  # 获取改写后查询
    if not query:  # 若改写查询为空
        query = state.get("original_query", "")  # 降级使用原始查询

    docs = []  # 初始化结构化结果列表

    if query:  # 若查询非空
        try:
            logger.info(f"启动异步 MCP 调用，Query: {query}")  # 打印启动日志
            result = asyncio.run(mcp_call(query))  # 同步桥接执行异步 MCP 调用

            if result and not result.isError and result.content:  # 若结果有效
                raw_text = result.content[0].text  # 提取文本内容
                try:
                    data = json.loads(raw_text)  # 解析 JSON
                    pages = data.get("pages") or []  # 获取页面列表

                    logger.info(f"MCP 返回原始页面数量: {len(pages)}")  # 打印页面数量

                    for item in pages:  # 遍历页面结果
                        snippet = (item.get("snippet") or "").strip()  # 提取摘要
                        url = (item.get("url") or "").strip()  # 提取链接
                        title = (item.get("title") or "").strip()  # 提取标题

                        if not snippet:  # 过滤无摘要结果
                            continue

                        docs.append({"title": title, "url": url, "snippet": snippet})  # 加入结果列表

                except json.JSONDecodeError:
                    logger.error(f"MCP 返回结果解析 JSON 失败: {raw_text[:100]}...")  # 打印解析失败日志
            else:  # 结果无效或报错
                if result and result.isError:
                    logger.error(f"MCP 返回错误: {result}")  # 打印 MCP 错误
                else:
                    logger.warning("MCP 返回结果为空或无效")  # 打印警告

            logger.info(f"结构化搜索结果数量: {len(docs)}")  # 打印结构化结果数量

        except Exception as e:
            logger.error(f"MCP 搜索节点执行异常: {e}", exc_info=True)  # 打印节点异常
    else:
        logger.warning("查询词为空，跳过 MCP 搜索")  # 打印空查询警告

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))  # 标记任务结束

    logger.info("---node_web_search_mcp 处理结束---")  # 打印节点结束日志

    if docs:  # 若有有效搜索结果
        return {"web_search_docs": docs}  # 返回结构化结果
    return {}  # 无结果返回空字典


if __name__ == '__main__':
    print("\n" + "="*50)  # 打印分隔线
    print(">>> 启动 node_web_search_mcp 本地测试")  # 打印测试标题
    print("="*50)  # 打印分隔线

    test_state = {  # 构造测试状态
        "session_id": "test_mcp_session",
        "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，应在操作面板上如何设置",
        "is_stream": False
    }

    try:
        result_state = node_web_search_mcp(test_state)  # 调用节点函数

        print("\n" + "="*50)  # 打印分隔线
        print(">>> 测试结果摘要:")  # 打印结果标题
        search_results = result_state.get('web_search_docs', [])  # 获取搜索结果
        print(f"搜索结果数量: {len(search_results)}")  # 打印结果数量
        if search_results:  # 若有结果
            print("首条结果预览:")  # 打印预览标题
            print(json.dumps(search_results[0], indent=2, ensure_ascii=False))  # 格式化打印首条结果
        else:  # 若无结果
            print("未获取到搜索结果")  # 打印无结果提示
        print("="*50)  # 打印分隔线

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")  # 打印测试异常
