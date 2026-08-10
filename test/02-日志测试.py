from app.core.logger import logger

logger.trace("进入函数 calculate_complex_logic，参数 x=10, y=20")  # 输出 TRACE 级别跟踪日志
logger.trace("中间变量 state={'step': 1, 'val': 30}")  # 输出 TRACE 级别中间状态日志

logger.debug("数据库连接池当前大小：5")  # 输出 DEBUG 级别调试日志
logger.debug("正在尝试重试第 2 次请求...")  # 输出 DEBUG 级别重试日志

logger.info("用户 ID: 1001 登录成功")  # 输出 INFO 级别登录成功日志
logger.info("订单 #9527 已创建，金额：¥299.00")  # 输出 INFO 级别订单创建日志
logger.info("系统健康检查通过")  # 输出 INFO 级别健康检查日志

logger.success("数据备份完成！文件已保存至 /backup/2026-03-15.zip")  # 输出 SUCCESS 级别备份完成日志
logger.success("模型训练结束，准确率达到 98.5%")  # 输出 SUCCESS 级别训练完成日志

logger.warning("配置文件缺少 'TIMEOUT' 字段，使用默认值 30s")  # 输出 WARNING 级别配置缺失日志
logger.warning("检测到 API 响应时间超过 2s，性能可能下降")  # 输出 WARNING 级别性能告警日志
logger.warning("用户密码强度较弱，建议修改")  # 输出 WARNING 级别安全建议日志

logger.error("无法连接到 Redis 服务器：Connection refused")  # 输出 ERROR 级别连接失败日志
logger.error("用户 ID: 1002 的数据解析失败，跳过该记录")  # 输出 ERROR 级别解析失败日志

logger.critical("磁盘空间已满！无法写入任何新数据，系统即将停止服务")  # 输出 CRITICAL 级别磁盘满致命错误日志
logger.critical("核心加密密钥丢失，安全模块初始化失败")  # 输出 CRITICAL 级别密钥丢失致命错误日志

@logger.catch  # 使用 Loguru 自动捕获并记录函数异常
def divide(a, b):
    """返回两个数相除的结果，并由 Loguru 自动捕获记录异常。"""
    return a / b  # 返回 a 除以 b 的结果

divide(10, 0)  # 调用 divide 函数触发除零异常以验证自动捕获
