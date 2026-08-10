import os
from dotenv import load_dotenv

load_dotenv(override=True)  # 加载 .env 文件并允许覆盖系统环境变量

print(os.getenv("OPENAI_API_KEY"))  # 打印 OpenAI API 密钥验证加载结果

print(os.getenv("MY_KEY"))  # 打印自定义环境变量验证优先级效果
